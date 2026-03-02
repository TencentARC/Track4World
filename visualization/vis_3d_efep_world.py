import time
import argparse
from pathlib import Path
from typing import Tuple, List, Optional
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import viser
from tqdm.auto import tqdm
from vis_3d_efep import (
    remove_std_outlier_open3d, 
    process_trajectories, 
    fill_trajectory_gaps
)
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import matplotlib.colors as mcolors
# ==============================================================================
# Global Constants
# ==============================================================================
DEFAULT_POINT_DOWNSAMPLE_RATE = 20  # Downsample rate for trajectories
STATIC_SKIP_FRAMES = 5             # Skip frames for accumulating static points
STATIC_VOXEL_SIZE = 0.02           # Voxel size for downsampling static background
MAX_DISPLACEMENT = 5             # Maximum displacement for trajectory segments
DEFAULT_CAM_POS = (4.34, 0.34, 4.74)
DEFAULT_LOOK_AT = (0.0, 0.0, 0.0)
DEFAULT_UP = (0.33, 0.0, 0.4)
DEFAULT_WXYZ = (-0.05, 0.98, -0.17, -0.12)

# ==============================================================================
# Helper Functions
# ==============================================================================

def fade_color_saturation_batch(rgb_uint8, factor):
    """
    批量处理 RGB 数组 (N, 3)
    factor: 0.0 (灰度) 到 1.0 (原色)
    """
    if factor == 1.0:
        return rgb_uint8
    rgb_float = rgb_uint8 / 255.0
    hsv = mcolors.rgb_to_hsv(rgb_float)
    hsv[..., 1] *= factor  # 仅缩放饱和度通道
    new_rgb = mcolors.hsv_to_rgb(hsv)
    return (new_rgb * 255).astype(np.uint8)

def fade_color_saturation(rgb_uint8, factor):
    """
    rgb_uint8: (3,) uint8 array
    factor: 0.0 (褪色为灰) 到 1.0 (原色)
    """
    # 归一化到 0-1 并转为 HSV
    rgb_float = rgb_uint8 / 255.0
    hsv = mcolors.rgb_to_hsv(rgb_float)
    
    # 降低饱和度，保持亮度
    hsv[1] = hsv[1] * factor 
    
    # 转回 RGB
    new_rgb = mcolors.hsv_to_rgb(hsv)
    return (new_rgb * 255).astype(np.uint8)

def canonicalize_quaternions(quats):
    """
    解决四元数双倍覆盖问题 (q 和 -q 代表同一旋转)。
    确保相邻帧的四元数点积为正，保证插值路径最短。
    """
    canon_quats = quats.copy()
    for i in range(1, len(canon_quats)):
        prev = canon_quats[i - 1]
        curr = canon_quats[i]
        # 如果点积为负，说明走的是“长路径”，将当前四元数取反
        if np.dot(prev, curr) < 0:
            canon_quats[i] = -curr
    return canon_quats

def smooth_translation_spline(t, smoothing=0.5):
    """
    使用 UnivariateSpline 进行平移平滑。
    相比 Savitzky-Golay，Spline 更能保证全局的物理连续性 (C2 连续)。
    smoothing: 平滑因子，越大越平滑，越小越贴近原轨迹。
    """
    T = t.shape[0]
    x = np.arange(T)
    t_smooth = np.zeros_like(t)
    
    # 对 x, y, z 分别拟合
    weights = np.ones(T) 
    # (可选) 如果有置信度，可以降低某些帧的权重
    
    for i in range(3):
        # s 是平滑参数，需要根据数据噪声程度调整
        # s=0 插值经过所有点，s很大则变成直线
        spl = UnivariateSpline(x, t[:, i], w=weights, s=smoothing)
        t_smooth[:, i] = spl(x)
        
    return t_smooth

def smooth_rotation_savgol(rot_objs, win=21, poly=3):
    """
    对旋转进行平滑。
    1. 提取四元数
    2. 连续化 (Canonicalize)
    3. Savitzky-Golay 滤波
    4. 归一化
    """
    quats = rot_objs.as_quat()
    
    # 关键步骤：解决符号跳变
    quats = canonicalize_quaternions(quats)
    
    # 确保窗口是奇数
    if win % 2 == 0: win += 1
    
    # 滤波
    quats_smooth = savgol_filter(quats, window_length=win, polyorder=poly, axis=0, mode='interp')
    
    # 归一化 (滤波后模长不为1)
    quats_smooth /= np.linalg.norm(quats_smooth, axis=1, keepdims=True)
    
    return R.from_quat(quats_smooth)

def smooth_c2w(c2w, 
                        trans_smoothing=1.0, # 平移平滑度 (Spline s参数)
                        rot_window=21,       # 旋转窗口大小
                        rot_poly=3):         # 旋转多项式阶数
    """
    c2w: (T, 4, 4) 或 (T, 3, 4)
    """
    T = c2w.shape[0]
    c2w_smooth = c2w.copy()
    
    # 1. 分离旋转和平移
    raw_t = c2w[:, :3, 3]
    raw_R = c2w[:, :3, :3]
    
    # 2. 平滑平移 (Spline 方法)
    # Spline 的 s 参数取决于数据的数值范围。
    # 如果数据是米制(scale~1.0)，s=0.1~1.0 比较合适。
    # 如果平移抖动严重，增大 s。
    print(f"Smoothing Translation (Spline, s={trans_smoothing})...")
    smooth_t = smooth_translation_spline(raw_t, smoothing=trans_smoothing)
    
    # 3. 平滑旋转 (Savitzky-Golay on Unwrapped Quaternions)
    print(f"Smoothing Rotation (SavGol, win={rot_window}, poly={rot_poly})...")
    rot_objs = R.from_matrix(raw_R)
    smooth_r_objs = smooth_rotation_savgol(rot_objs, win=rot_window, poly=rot_poly)
    smooth_R = smooth_r_objs.as_matrix()
    
    # 4. 重组
    c2w_smooth[:, :3, :3] = smooth_R
    c2w_smooth[:, :3, 3] = smooth_t
    
    return c2w_smooth

def smooth_trajectories_temporal(trajs, mask, sigma=2.0):
    """
    对轨迹进行时间维度的平滑，使用高斯滤波以获得极高的平滑度。
    
    Args:
        trajs: (T, N, 3) 轨迹数据
        mask: (T, N) 可见性掩码
        sigma: 平滑强度。
               sigma=1.0: 轻微平滑
               sigma=2.0-3.0: 非常平滑 (推荐)
               sigma=5.0+: 极度平滑 (可能会导致快速转弯处出现“切角”)
    """
    print(f"Smoothing point trajectories (Gaussian, sigma={sigma})...")
    T, N, D = trajs.shape
    smoothed_trajs = trajs.copy()
    
    # 定义最小片段长度，短于此长度的片段不进行平滑（避免过度拟合噪声）
    min_segment_len = max(int(sigma * 3), 5) 

    for i in tqdm(range(N), desc="Smoothing Trajectories"):
        # 1. 获取当前点的所有有效帧索引
        valid_indices = np.where(mask[:, i])[0]
        
        if len(valid_indices) == 0:
            continue
            
        # 2. 识别连续片段 (Split into continuous segments)
        # 如果索引跳跃超过 1，说明中间有断层 (Gap)
        # 例如: [1, 2, 3, 10, 11, 12] -> [1, 2, 3] 和 [10, 11, 12]
        splits = np.where(np.diff(valid_indices) > 1)[0] + 1
        segments = np.split(valid_indices, splits)
        
        for seg_idx in segments:
            # 如果片段太短，高斯滤波没有意义，跳过
            if len(seg_idx) < min_segment_len:
                continue
            
            # 3. 提取该片段的 3D 数据 (L, 3)
            raw_data = trajs[seg_idx, i, :]
            
            # 4. 应用高斯滤波
            # axis=0 表示沿时间轴平滑
            # mode='nearest' 使得边界处平滑过渡，不会发散
            smooth_data = gaussian_filter1d(
                raw_data, 
                sigma=sigma, 
                axis=0, 
                mode='nearest'
            )
            
            # 5. 写回结果
            smoothed_trajs[seg_idx, i, :] = smooth_data
            
    return smoothed_trajs

def cam_points_to_world(points_cam, c2w):
    N = points_cam.shape[0]
    if N == 0:
        return points_cam
    points_h = np.concatenate([points_cam, np.ones((N, 1))], axis=1)
    points_world = (c2w @ points_h.T).T[:, :3]
    return points_world

# ==============================================================================
# Main Execution
# ==============================================================================

def main(
    max_frames: int = 400,
    share: bool = False,
) -> None:
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ply_dir', type=str, 
        default='results/cat/3d_efep_output', 
        help='Directory containing frame_xx.ply files and masks'
    )
    parser.add_argument(
        '--save_dir', type=str, 
        default='recordings/cat', 
        help='Directory to save output files'
    )
    args = parser.parse_args()

    ply_dir = Path(args.ply_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if not ply_dir.exists():
        raise FileNotFoundError(f"{ply_dir} not found")

    # --- Viser Server Setup ---
    server = viser.ViserServer()
    
    # [Fix 1]: Use callback to set initial camera
    # Automatically set camera position when a client connects
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        print(f"New client connected! Setting camera for {client.client_id}")
        client.camera.position = DEFAULT_CAM_POS
        client.camera.look_at = DEFAULT_LOOK_AT
        client.camera.up_direction = DEFAULT_UP
        client.camera.wxyz = DEFAULT_WXYZ

    if share:
        server.request_share_url()

    # --- Load File Paths ---
    ply_files = sorted([f for f in ply_dir.glob("frame_*.ply")], key=lambda x: int(x.stem.split("_")[-1]))
    num_frames = min(max_frames, len(ply_files))
    if num_frames == 0:
        raise RuntimeError(f"No valid frame_*.ply files found in {ply_dir}")

    # --- 1. Load Trajectory Data & Mask ---
    print("Loading trajectory data...")
    traj_path = ply_dir / 'trajectory_all_pointmap.npy'
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    
    trajectory_all_raw = np.load(str(traj_path))

    traj_mask_path = ply_dir / 'trajectory_all_pointmap_dyn_mask.npy'
    if not traj_mask_path.exists():
        print(f"Warning: Trajectory mask not found. Assuming all dynamic.")
        traj_dyn_mask_raw = np.ones(
            (trajectory_all_raw.shape[0], trajectory_all_raw.shape[1]), 
            dtype=bool
        )
    else:
        print(f"Loading trajectory mask from {traj_mask_path}...")
        traj_dyn_mask_raw = np.load(str(traj_mask_path))
        if traj_dyn_mask_raw.ndim == 3:
            traj_dyn_mask_raw = traj_dyn_mask_raw.squeeze(-1)

    # --- 2. Load Camera Poses & Transform Trajectories ---
    c2w_path = ply_dir / 'c2w.npy'
    if not c2w_path.exists():
        raise FileNotFoundError(f"Camera pose file not found: {c2w_path}")
    c2w = np.load(str(c2w_path))
    # c2w = smooth_c2w(c2w, rot_window=c2w.shape[0]-1)
    trajectories_3d = trajectory_all_raw.copy()
    for i in tqdm(range(trajectory_all_raw.shape[0]), desc="Transforming Trajectories"):
        world_pts = cam_points_to_world(trajectory_all_raw[i], c2w[i])
        trajectories_3d[i] = world_pts * [1, -1, -1]
    
    visibility_mask = ~np.isnan(trajectory_all_raw).any(axis=-1)
    # Process trajectories (outlier removal, etc.)
    trajectories_3d, visibility_mask, traj_dyn_mask_raw = process_trajectories(
        trajectories_3d, visibility_mask, traj_dyn_mask_raw, 
        k_consecutive=5, jump_threshold=12, acc_threshold=10
    )
    # trajectories_3d, visibility_mask, traj_dyn_mask_raw = process_trajectories(
    #     trajectories_3d, visibility_mask, traj_dyn_mask_raw, 
    #     k_consecutive=5, jump_threshold=0.3, acc_threshold=0.3
    # )
    print("Filtering trajectories...")

    # =========================================================
    # 1. Global Filtering: Keep only trajectories that are 
    #    "dynamic at all visible moments"
    # =========================================================
    dyn = traj_dyn_mask_raw.astype(bool)
    vis = visibility_mask.astype(bool)
    
    # Logic: (Not Visible) OR (Is Dynamic) = Valid
    # If a frame is (Visible AND Static), it evaluates to False, 
    # and the whole trajectory is discarded.
    per_frame_condition = (~vis) | dyn
    keep_pure_dynamic_mask = np.all(per_frame_condition, axis=0)  # shape (N,)

    # Slice arrays directly to remove invalid trajectories
    trajectories_3d = trajectories_3d[:, keep_pure_dynamic_mask]
    visibility_mask = visibility_mask[:, keep_pure_dynamic_mask]
    traj_dyn_mask_raw = traj_dyn_mask_raw[:, keep_pure_dynamic_mask]
    print(f"Remaining trajectories after dynamic filter: {trajectories_3d.shape[1]}")

    # =========================================================
    # 2. Per-frame Spatial Denoising
    # =========================================================
    # Note: If trajectories are very sparse, this might remove good points.
    # If lines disappear completely, comment out this loop.
    for i in tqdm(range(trajectories_3d.shape[0]), desc="Spatial Outlier Removal"):
        mask = visibility_mask[i]
        if np.sum(mask) == 0:
            continue
            
        pts = trajectories_3d[i][mask]
        
        # Only perform statistical filtering if there are enough points
        if pts.shape[0] > 10: 
            _, ind = remove_std_outlier_open3d(pts)
            
            # Update mask
            valid_indices = np.where(mask)[0]
            new_mask = np.zeros_like(mask, dtype=bool)
            if ind.shape[0] > 0:
                filtered_indices = valid_indices[ind]
                new_mask[filtered_indices] = True
            visibility_mask[i] = new_mask

    # =========================================================
    # 3. [Critical] Repair Breaks (Gap Filling)
    # =========================================================
    # Must run after denoising to fill short-term breaks
    print("Repairing broken trajectories...")
    trajectories_3d, visibility_mask = fill_trajectory_gaps(
        trajectories_3d, 
        visibility_mask, 
        max_gap=5  
    )
    trajectories_3d = smooth_trajectories_temporal(
        trajectories_3d, 
        visibility_mask, 
        sigma=3.0 
    )
    # --- 3. Load Point Clouds (Split Static/Dynamic) ---
    dynamic_point_nodes: List[viser.PointCloudHandle] = []
    dynamic_colors_original: List[np.ndarray] = []  # 新增：存储原始颜色副本
    static_points_accumulator = []
    static_colors_accumulator = []

    print(f"Loading {num_frames} frames and splitting static/dynamic...")
    
    for i, ply_file in enumerate(tqdm(ply_files[:num_frames], desc="Processing Frames")):
        pcd = o3d.io.read_point_cloud(str(ply_file))
        points_local = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        mask_path = ply_dir / f"pc_dyn_mask_{i:03d}.npy"
        if not mask_path.exists():
            mask_path_alt = ply_dir / f"pc_dyn_mask_{i}.npy"
            if mask_path_alt.exists():
                mask_path = mask_path_alt
            else:
                is_dynamic = np.ones(points_local.shape[0], dtype=bool)
        
        if mask_path.exists():
            is_dynamic = np.load(str(mask_path))
            if is_dynamic.ndim > 1:
                is_dynamic = is_dynamic.squeeze()
            is_dynamic = (is_dynamic > 0)
        
        min_len = min(len(is_dynamic), len(points_local))
        is_dynamic = is_dynamic[:min_len]
        points_local = points_local[:min_len]
        colors = colors[:min_len]

        points_world = cam_points_to_world(points_local, c2w[i])
        points_world = 100 * points_world * [1, -1, -1]
        
        pts_dyn = points_world[is_dynamic]
        col_dyn_uint8 = (colors[is_dynamic] * 255).astype(np.uint8) # 转换为 uint8
        pts_stat = points_world[~is_dynamic]
        col_stat = colors[~is_dynamic]

        # Add dynamic points to the scene
        node = server.scene.add_point_cloud(
            name=f"/dynamic/t{i}",
            points=pts_dyn,
            colors=col_dyn_uint8,
            point_size=0.01,
            point_shape="rounded",
            visible=False 
        )
        dynamic_point_nodes.append(node)
        dynamic_colors_original.append(col_dyn_uint8) # 保存副本

        # Accumulate static points
        if i % STATIC_SKIP_FRAMES == 0 and len(pts_stat) > 0:
            static_points_accumulator.append(pts_stat)
            static_colors_accumulator.append(col_stat)

    # --- 4. Process Static Background ---
    print("Merging and downsampling static background...")
    static_node: Optional[viser.PointCloudHandle] = None
    
    if static_points_accumulator:
        all_static_pts = np.concatenate(static_points_accumulator, axis=0)
        all_static_cols = np.concatenate(static_colors_accumulator, axis=0)
        
        pcd_static = o3d.geometry.PointCloud()
        pcd_static.points = o3d.utility.Vector3dVector(all_static_pts)
        pcd_static.colors = o3d.utility.Vector3dVector(all_static_cols)
        
        # Remove statistical outliers and downsample
        pcd_static, ind = pcd_static.remove_statistical_outlier(
            nb_neighbors=128, std_ratio=4.0
        )
        pcd_static_down = pcd_static.voxel_down_sample(voxel_size=STATIC_VOXEL_SIZE) 
        
        final_static_pts = np.asarray(pcd_static_down.points)
        final_static_cols = (np.asarray(pcd_static_down.colors) * 255).astype(np.uint8)
        
        print(f"Static background: {len(all_static_pts)} -> {len(final_static_pts)} points")

        # Add static points to the scene
        static_node = server.scene.add_point_cloud(
            name="/static_background",
            points=final_static_pts,
            colors=final_static_cols,
            point_size=0.01,
            point_shape="rounded",
            visible=True
        )
    else:
        print("Warning: No static points found.")

    # --- 5. Prepare Dynamic Trajectory Colors ---
    trajectories_3d_down = trajectories_3d[:, 3::DEFAULT_POINT_DOWNSAMPLE_RATE]
    visibility_mask_down = visibility_mask[:, 3::DEFAULT_POINT_DOWNSAMPLE_RATE]
    
    N_traj = trajectories_3d_down.shape[1]
    initial_colors = np.zeros((N_traj, 3), dtype=np.uint8)
    
    if N_traj > 0:
        first_visible_idx = np.argmax(visibility_mask_down, axis=0)
        never_visible = ~np.any(visibility_mask_down, axis=0)
        
        first_visible_idx[never_visible] = 0
        indices = np.arange(N_traj)
        first_visible_xyz = trajectories_3d_down[first_visible_idx, indices]
        first_visible_xyz[never_visible] = np.nan
        
        xyz_min = np.nanmin(first_visible_xyz, axis=0)
        xyz_max = np.nanmax(first_visible_xyz, axis=0)
        xyz_norm = (first_visible_xyz - xyz_min) / (xyz_max - xyz_min + 1e-6)
        scalar = np.nansum(xyz_norm, axis=1)
        
        scalar = (scalar - np.nanmin(scalar)) / (np.nanmax(scalar) - np.nanmin(scalar) + 1e-6)
        sort_idx = np.argsort(scalar)
        colors_hsv = plt.cm.hsv(np.linspace(0, 1, 5 * N_traj))[:, :3]
    
        # Assign colors sorted by spatial position
        sorted_hsv = colors_hsv[3 * N_traj + np.argsort(sort_idx)]
        initial_colors = (sorted_hsv * 255).astype(np.uint8)
    else:
        initial_colors = np.zeros((0, 3), dtype=np.uint8)

    # --- 6. GUI Controls ---
    with server.gui.add_folder("Playback"):
        # --- Point Cloud Appearance ---
        gui_static_point_size = server.gui.add_slider(
            "Static Point Size", min=0.0001, max=10, step=0.0001, initial_value=0.01
        )
        gui_dynamic_point_size = server.gui.add_slider(
            "Dynamic Point Size", min=0.0001, max=10, step=0.0001, initial_value=0.01
        )
        gui_line_width = server.gui.add_slider(
            "Line width", min=0.1, max=5.0, step=0.1, initial_value=0.5
        )

        # --- Playback Logic ---
        gui_timestep = server.gui.add_slider(
            "Timestep", min=0, max=num_frames - 1, step=1, initial_value=0
        )
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=24
        )

        # --- Visibility Toggles ---
        gui_show_static = server.gui.add_checkbox("Show Static Background", True)
        gui_show_dynamic = server.gui.add_checkbox("Show Dynamic Points", True)
        gui_show_traj = server.gui.add_checkbox("Show Dynamic Trajectories", True)

        # --- Motion Analysis ---
        gui_max_traj_length = server.gui.add_slider(
            "Trail Length", min=1, max=50, step=1, initial_value=10
        )
        gui_max_displacement = server.gui.add_slider(
            "Max Displacement", 
            min=0.1, max=20.0, step=0.1, 
            initial_value=MAX_DISPLACEMENT
        )

        # --- Camera Controls ---
        gui_cam_pos = server.gui.add_vector3(
            "Position", initial_value=DEFAULT_CAM_POS, step=0.05
        )
        gui_cam_look = server.gui.add_vector3(
            "Look At", initial_value=DEFAULT_LOOK_AT, step=0.05
        )
        gui_cam_up = server.gui.add_vector3(
            "Up Direction", initial_value=DEFAULT_UP, step=0.05
        )
        
        btn_reset_cam = server.gui.add_button("Reset to Default")
        btn_sync_from_view = server.gui.add_button("Sync from View")

        @gui_cam_pos.on_update
        def _(_):
            for client in server.get_clients().values():
                client.camera.position = gui_cam_pos.value

        @gui_cam_look.on_update
        def _(_):
            for client in server.get_clients().values():
                client.camera.look_at = gui_cam_look.value

        @gui_cam_up.on_update
        def _(_):
            for client in server.get_clients().values():
                client.camera.up_direction = gui_cam_up.value

        @btn_reset_cam.on_click
        def _(_):
            gui_cam_pos.value = DEFAULT_CAM_POS
            gui_cam_look.value = DEFAULT_LOOK_AT
            gui_cam_up.value = DEFAULT_UP
            for client in server.get_clients().values():
                client.camera.position = DEFAULT_CAM_POS
                client.camera.look_at = DEFAULT_LOOK_AT
                client.camera.up_direction = DEFAULT_UP
                client.camera.wxyz = DEFAULT_WXYZ

        @btn_sync_from_view.on_click
        def _(_):
            clients = server.get_clients()
            if clients:
                client = list(clients.values())[0]
                gui_cam_pos.value = client.camera.position
                gui_cam_look.value = client.camera.look_at
                gui_cam_up.value = client.camera.up_direction

        gui_dyn_saturation = server.gui.add_slider(
            "Dyn. Saturation", min=0.0, max=1.0, step=0.01, initial_value=1.0
        )

        # 实时滑动反馈：当滑动条变动时，立刻更新当前正在显示的帧
        @gui_dyn_saturation.on_update
        def _(_):
            t_curr = gui_timestep.value
            if t_curr < len(dynamic_point_nodes):
                # 立刻更新当前帧的视觉效果
                dynamic_point_nodes[t_curr].colors = fade_color_saturation_batch(
                    dynamic_colors_original[t_curr], 
                    gui_dyn_saturation.value
                )
    line_node = server.scene.add_line_segments(
        name="/trajectories",
        points=np.zeros((0, 2, 3)),
        colors=np.zeros((0, 2, 3), dtype=np.uint8),
        line_width=gui_line_width.value,
        visible=True,
    )

    # ==========================================================================
    # Core Update Logic (Modified)
    # ==========================================================================
    def update_scene_state(
        t_curr: int, 
        t_prev: int, 
        history_pos: List[np.ndarray], 
        history_col: List[np.ndarray],
        history_ind: List[np.ndarray]  # Stores trajectory IDs for segments
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        
        # Update static point visibility and size
        if static_node is not None:
            static_node.visible = gui_show_static.value
            static_node.point_size = gui_static_point_size.value

        # Update dynamic points visibility
        if t_curr != t_prev:
            if t_prev >= 0 and t_prev < len(dynamic_point_nodes):
                dynamic_point_nodes[t_prev].visible = False
            
            if gui_show_dynamic.value:
                node = dynamic_point_nodes[t_curr]
                node.visible = True
                node.point_size = gui_dynamic_point_size.value
                
                # --- 关键修改：每一帧显示前应用滑动条的饱和度 ---
                node.colors = fade_color_saturation_batch(
                    dynamic_colors_original[t_curr], 
                    gui_dyn_saturation.value
                )
            else:
                dynamic_point_nodes[t_curr].visible = False

        # Update trajectory lines
        if gui_show_traj.value and N_traj > 0:
            line_node.visible = True
            line_node.line_width = gui_line_width.value

            # If looped back to frame 0, clear history
            if t_curr == 0 and t_prev != 0:
                history_pos.clear()
                history_col.clear()
                history_ind.clear() # Clear indices
                line_node.points = np.zeros((0, 2, 3))
                return history_pos, history_col, history_ind

            # --- 1. Calculate new segments for the current frame ---
            current_active_indices = np.array([], dtype=int) 

            if t_curr < num_frames and t_curr > 0:
                pos_curr = 100 * trajectories_3d_down[t_curr - 1]
                pos_next = 100 * trajectories_3d_down[t_curr]
                
                # Get visibility mask
                valid_mask = (
                    visibility_mask_down[t_curr - 1] & 
                    visibility_mask_down[t_curr]
                )
                
                if np.any(valid_mask):
                    # Get original indices (0 to N_traj-1)
                    all_indices = np.arange(N_traj)
                    
                    # Initial filtering
                    p1 = pos_curr[valid_mask]
                    p2 = pos_next[valid_mask]
                    curr_inds = all_indices[valid_mask] # Corresponding trajectory IDs
                    
                    # Calculate displacement and filter jumps
                    dist = np.linalg.norm(p2 - p1, axis=1)
                    jump_mask = dist < gui_max_displacement.value
                    
                    if np.any(jump_mask):
                        final_p1 = p1[jump_mask]
                        final_p2 = p2[jump_mask]
                        segments = np.stack([final_p1, final_p2], axis=1)
                        
                        cols = initial_colors[valid_mask][jump_mask]
                        segment_colors = np.stack([cols, cols], axis=1)
                        
                        final_indices = curr_inds[jump_mask] # Finally retained IDs
                        
                        # Store in history
                        history_pos.append(segments)
                        history_col.append(segment_colors)
                        history_ind.append(final_indices) # Store IDs
                        
                        # Record active IDs for the current frame
                        current_active_indices = final_indices

            # Maintain maximum trajectory length
            while len(history_pos) > gui_max_traj_length.value:
                history_pos.pop(0)
                history_col.pop(0)
                history_ind.pop(0)
            
            # --- 2. Rendering Logic (Critical Modification) ---
            # Only historical segments of trajectories present in 
            # current_active_indices will be displayed
            if history_pos and len(current_active_indices) > 0:
                
                # Create a boolean lookup table to quickly check if historical 
                # segments belong to currently active trajectories
                active_lookup = np.zeros(N_traj, dtype=bool)
                active_lookup[current_active_indices] = True
                
                render_pos_list = []
                render_col_list = []
                num_history_steps = len(history_pos)
                # Iterate through history, keeping only segments of active trajectories
                for i, (h_pos, h_col, h_ind) in enumerate(zip(history_pos, history_col, history_ind)):
                    keep_mask = active_lookup[h_ind]
                    
                    if np.any(keep_mask):
                        # 1. 计算 Alpha 因子 (从 0.0 到 1.0)
                        # i 越小（越旧），alpha 越低，线条越透明
                        alpha_factor = 1 - i / num_history_steps
                        
                        # 2. 获取原始 RGB
                        rgb_cols = h_col[keep_mask] # 形状 (M, 2, 3)
                        if rgb_cols.shape[0] > 1:
                            faded_col = fade_color_saturation(rgb_cols, alpha_factor)
                        else:
                            faded_col = rgb_cols
                        # 拼接成 (M, 2, 4)
                        rgba_cols = faded_col
                        
                        render_pos_list.append(h_pos[keep_mask])
                        render_col_list.append(rgba_cols)
                
                if render_pos_list:
                    line_node.points = np.concatenate(render_pos_list, axis=0)
                    line_node.colors = np.concatenate(render_col_list, axis=0)
                else:
                    line_node.points = np.zeros((0, 2, 3))
            else:
                # If no active trajectories or history is empty, do not show lines
                line_node.points = np.zeros((0, 2, 3))
        else:
            line_node.visible = False
            
        return history_pos, history_col, history_ind

    # ==========================================================================
    # Serialization
    # ==========================================================================
    with server.gui.add_folder("Export"):
        save_btn = server.gui.add_button("Save .viser file")

        @save_btn.on_click
        def _(_):
            print("Initializing Viser Serialization...")
            was_playing = gui_playing.value
            gui_playing.value = False 
            
            for node in dynamic_point_nodes:
                node.visible = False
            line_node.points = np.zeros((0, 2, 3))
            
            serializer = server.get_scene_serializer()
            
            rec_history_pos = []
            rec_history_col = []
            rec_history_ind = [] 
            rec_prev_t = -1
            
            try:
                for t in tqdm(range(num_frames), desc="Recording .viser"):
                    # <--- Update call
                    rec_history_pos, rec_history_col, rec_history_ind = update_scene_state(
                        t_curr=t, 
                        t_prev=rec_prev_t, 
                        history_pos=rec_history_pos, 
                        history_col=rec_history_col,
                        history_ind=rec_history_ind
                    )
                    rec_prev_t = t
                    serializer.insert_sleep(1.0 / gui_framerate.value)
                
                print("Serializing data...")
                data = serializer.serialize()
                
                if gui_show_dynamic.value and gui_show_traj.value:
                    output_filename = save_dir / "pc_line.viser"
                elif (not gui_show_dynamic.value) and gui_show_traj.value:
                    output_filename = save_dir / "line.viser"
                elif gui_show_dynamic.value and (not gui_show_traj.value):
                    output_filename = save_dir / "pc.viser"
                else:
                    output_filename = save_dir / "output.viser"

                Path(output_filename).write_bytes(data)
                print(f"Saved {output_filename} ({len(data)/1024/1024:.2f} MB)")
                
            except Exception as e:
                print(f"Serialization failed: {e}")
            finally:
                gui_playing.value = was_playing
                gui_timestep.value = 0

    # ==========================================================================
    # Main Loop
    # ==========================================================================
    prev_timestep = -1
    live_history_pos = []
    live_history_col = []
    live_history_ind = [] 
    print_counter = 0

    while True:
        # Update timestep if playing
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        
        t_curr = gui_timestep.value

        # <--- Update call parameters, receiving 3 return values
        live_history_pos, live_history_col, live_history_ind = update_scene_state(
            t_curr=t_curr,
            t_prev=prev_timestep,
            history_pos=live_history_pos,
            history_col=live_history_col,
            history_ind=live_history_ind 
        )
        
        # [Fix 2]: Print camera parameters periodically
        print_counter += 1
        if print_counter % 60 == 0:
            clients = server.get_clients()
            if len(clients) > 0:
                # Get first connected client
                client = list(clients.values())[0]
                cam = client.camera
                print(f"\n--- Camera State (Frame {t_curr}) ---")
                print(f"Position: {np.round(cam.position, 2)}")
                print(f"LookAt:   {np.round(cam.look_at, 2)}")
                print(f"Up:       {np.round(cam.up_direction, 2)}")
                print(f"Wxyz:     {np.round(cam.wxyz, 2)}")
                print(f"-------------------------------------")

        prev_timestep = t_curr
        time.sleep(1.0 / gui_framerate.value)
 
if __name__ == "__main__":
    main()