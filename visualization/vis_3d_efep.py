import time
import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import open3d as o3d
import viser
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter1d

# Try importing faiss for GPU acceleration, fallback if not available
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: 'faiss' not found. GPU outlier removal will not be available.")

# Global Constants
# Sampling rate: Downsample points to maintain performance during visualization.
DEFAULT_POINT_DOWNSAMPLE_RATE = 10
MAX_DISPLACEMENT = 5.0

# ==============================================================================
# Helper Functions
# ==============================================================================

def smooth_trajectories_temporal(trajs, mask, sigma=2.0):
    """
    Smooth trajectories in the temporal dimension using Gaussian filtering for extremely high smoothness.
    
    Args:
        trajs: (T, N, 3) Trajectory data
        mask: (T, N) Visibility mask
        sigma: Smoothing strength.
               sigma=1.0: Slight smoothing
               sigma=2.0-3.0: Very smooth (recommended)
               sigma=5.0+: Extremely smooth (may cause "corner cutting" in fast turns)
    """
    print(f"Smoothing point trajectories (Gaussian, sigma={sigma})...")
    T, N, D = trajs.shape
    smoothed_trajs = trajs.copy()
    
    # Define minimum segment length, segments shorter than this are not smoothed (to avoid overfitting noise)
    min_segment_len = max(int(sigma * 3), 5) 

    for i in tqdm(range(N), desc="Smoothing Trajectories"):
        # 1. Get all valid frame indices for the current point
        valid_indices = np.where(mask[:, i])[0]
        
        if len(valid_indices) == 0:
            continue
            
        # 2. Identify continuous segments (Split into continuous segments)
        # If index jump is greater than 1, it means there is a gap in between
        # For example: [1, 2, 3, 10, 11, 12] -> [1, 2, 3] and [10, 11, 12]
        splits = np.where(np.diff(valid_indices) > 1)[0] + 1
        segments = np.split(valid_indices, splits)
        
        for seg_idx in segments:
            # If segment is too short, Gaussian filtering is meaningless, skip
            if len(seg_idx) < min_segment_len:
                continue
            
            # 3. Extract 3D data for this segment (L, 3)
            raw_data = trajs[seg_idx, i, :]
            
            # 4. Apply Gaussian filtering
            # axis=0 means smooth along the time axis
            # mode='nearest' makes the boundary transition smooth, won't diverge
            smooth_data = gaussian_filter1d(
                raw_data, 
                sigma=sigma, 
                axis=0, 
                mode='nearest'
            )
            
            # 5. Write back the results
            smoothed_trajs[seg_idx, i, :] = smooth_data
            
    return smoothed_trajs

def process_trajectories(
    trajectories_3d, 
    visibility_mask, 
    traj_dyn_mask_raw, 
    k_consecutive=5, 
    jump_threshold=0.6, 
    acc_threshold=0.5,   # Acceleration threshold for filtering jittery trajectories
    smooth_sigma=1.5     # Smoothing parameter (reserved for optional post-processing)
):
    """
    Filter unstable 3D trajectories based on visibility continuity, motion jumps,
    and smoothness constraints, and optionally prepare them for smoothing.
    """
    T, N, _ = trajectories_3d.shape

    # 1. Consecutive Visibility Filtering
    mask_int = visibility_mask.astype(np.int32)
    cumsum_mask = np.cumsum(
        np.vstack([np.zeros((1, N)), mask_int]), axis=0
    )
    window_sums = cumsum_mask[k_consecutive:] - cumsum_mask[:-k_consecutive]
    valid_length_mask = np.any(window_sums == k_consecutive, axis=0)

    # 2. Motion Jump Filtering (Velocity Check)
    velocities = 100 * (trajectories_3d[1:] - trajectories_3d[:-1])
    vel_norms = np.linalg.norm(velocities, axis=2)

    valid_consecutive_frames = visibility_mask[1:] & visibility_mask[:-1]
    has_jumps = np.any(
        (vel_norms > jump_threshold) & valid_consecutive_frames,
        axis=0
    )

    # 3. Smoothness / Jitter Filtering (Acceleration Check)
    accelerations = velocities[1:] - velocities[:-1]
    acc_norms = np.linalg.norm(accelerations, axis=2)

    valid_acc_frames = (
        visibility_mask[2:] &
        visibility_mask[1:-1] &
        visibility_mask[:-2]
    )

    is_jittery = np.any(
        (acc_norms > acc_threshold) & valid_acc_frames,
        axis=0
    )

    # 4. Final Filtering and Application
    final_keep_mask = (
        valid_length_mask &
        (~has_jumps) &
        (~is_jittery)
    )

    if not np.any(final_keep_mask):
        print("Warning: No trajectories survived filtering.")
        return (
            trajectories_3d[:, []],
            visibility_mask[:, []],
            traj_dyn_mask_raw[:, []]
        )

    trajectories_3d = trajectories_3d[:, final_keep_mask]
    visibility_mask = visibility_mask[:, final_keep_mask]
    traj_dyn_mask_raw = traj_dyn_mask_raw[:, final_keep_mask]

    return trajectories_3d, visibility_mask, traj_dyn_mask_raw

def fill_trajectory_gaps(trajectories, mask, max_gap=3):
    """
    Linearly interpolate short missing segments in trajectories.
    """
    T, N, _ = trajectories.shape
    filled_trajectories = trajectories.copy()
    filled_mask = mask.copy()

    for i in tqdm(range(N), desc="Filling gaps"):
        valid_indices = np.where(mask[:, i])[0]
        
        if len(valid_indices) < 2:
            continue

        diffs = valid_indices[1:] - valid_indices[:-1]
        gap_indices = np.where((diffs > 1) & (diffs <= max_gap + 1))[0]

        for gap_idx in gap_indices:
            start_t = valid_indices[gap_idx]
            end_t = valid_indices[gap_idx + 1]
            
            p_start = filled_trajectories[start_t, i]
            p_end = filled_trajectories[end_t, i]

            for t in range(start_t + 1, end_t):
                alpha = (t - start_t) / (end_t - start_t)
                filled_trajectories[t, i] = (1 - alpha) * p_start + alpha * p_end
                filled_mask[t, i] = True 

    return filled_trajectories, filled_mask

def remove_radius_outlier_gpu(points: np.ndarray, 
    nb_points: int = 40, 
    radius: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """Removes outliers using GPU-accelerated Faiss (K-Nearest Neighbors)."""
    if not HAS_FAISS:
        raise RuntimeError("Faiss is not installed.")
        
    points_torch = torch.from_numpy(points).float().cuda()
    n, d = points_torch.shape
    
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    
    gpu_index.add(points) 
    D, I = gpu_index.search(points, nb_points)
    
    mask = D[:, -1] < radius**2
    return points[mask].cpu().numpy(), np.nonzero(mask.cpu().numpy())[0]

def remove_radius_outlier_open3d(points: np.ndarray, 
    nb_points: int = 20, 
    radius: float = 0.001
) -> Tuple[np.ndarray, np.ndarray]:
    """Removes outliers using Open3D (CPU implementation)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_filtered, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    points_filtered = np.asarray(pcd_filtered.points)
    return points_filtered, np.array(ind)

def remove_std_outlier_open3d(points: np.ndarray, 
    nb_neighbors: int = 30, 
    std_ratio: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Removes outliers using Open3D (CPU implementation)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_filtered, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    points_filtered = np.asarray(pcd_filtered.points)
    return points_filtered, np.array(ind)

# ==============================================================================
# Main Execution
# ==============================================================================

def main(
    max_frames: int = 400,
    share: bool = False,
) -> None:
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_dir', type=str, 
                        default='results/cat/3d_efep_output', 
                        help='Directory containing frame_xx.ply files and trajectory_all_pointmap.npy')
    args = parser.parse_args()

    ply_dir = Path(args.ply_dir)
    if not ply_dir.exists():
        raise FileNotFoundError(f"{ply_dir} not found")

    # --- Viser Server Setup ---
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    # --- Load File Paths ---
    ply_files = sorted([f for f in ply_dir.glob("frame_*.ply")], key=lambda x: int(x.stem.split("_")[-1]))
    
    num_frames = min(max_frames, len(ply_files))
    if num_frames == 0:
        raise RuntimeError(f"No valid frame_*.ply files found in {ply_dir}")

    # --- Load Data ---
    traj_path = ply_dir / 'trajectory_all_pointmap.npy'
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
        
    trajectory_all_raw = np.load(str(traj_path))

    initial_colors = None
    point_nodes: List[viser.PointCloudHandle] = []

    print(f"Loading {num_frames} point cloud frames...")
    
    # Load and preprocess point clouds
    for i, ply_file in enumerate(tqdm(ply_files[:num_frames], desc="Loading point clouds")):
        pcd = o3d.io.read_point_cloud(str(ply_file))
        
        points_all = 100 * np.asarray(pcd.points)
        colors_all = np.asarray(pcd.colors)

        if i == 0:
            num_tracks = colors_all.shape[0]
            track_colors = plt.cm.get_cmap('hsv', num_tracks)
            colors_track = track_colors(np.arange(num_tracks))[:, :3]
            initial_colors = (colors_track * 255).astype(np.uint8)

        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/point_cloud",
                points=points_all,
                colors=(colors_all * 255).astype(np.uint8),
                point_size=0.01,
                point_shape="rounded",
                visible=(i == 0)
            )
        )

    # --- Process Trajectory Data ---
    trajectories_3d = trajectory_all_raw.copy()
    visibility_mask = ~np.isnan(trajectory_all_raw).any(axis=-1)  # [T, N]

    trajectories_3d, visibility_mask, _ = process_trajectories(
        trajectories_3d, visibility_mask, visibility_mask, 
        k_consecutive=5, jump_threshold=8, acc_threshold=6
    )
    print("Filtering trajectories...")
    for i in tqdm(range(trajectories_3d.shape[0])):
        pts = trajectories_3d[i][visibility_mask[i]]
        if pts.shape[0] == 0:
            continue
            
        _, ind = remove_std_outlier_open3d(pts)
        
        mask = visibility_mask[i].copy()
        valid_indices = np.where(mask)[0]
        new_mask = np.zeros_like(mask, dtype=bool)
        if ind.shape[0] > 0:
            filtered_indices = valid_indices[ind]
            new_mask[filtered_indices] = True
        visibility_mask[i] = new_mask

    # Apply initial downsampling
    trajectories_3d = trajectories_3d[:, ::DEFAULT_POINT_DOWNSAMPLE_RATE]
    visibility_mask = visibility_mask[:, ::DEFAULT_POINT_DOWNSAMPLE_RATE]
    
    # 1. Fill short-term breakpoints
    trajectories_3d, visibility_mask = fill_trajectory_gaps(
        trajectories_3d, 
        visibility_mask, 
        max_gap=5  
    )
    
    # 2. [New] Temporal Gaussian smoothing to make trajectories smooth like a ribbon
    trajectories_3d = smooth_trajectories_temporal(
        trajectories_3d, 
        visibility_mask, 
        sigma=3.0 
    )
    
    T, N, _ = trajectories_3d.shape

    # --- Compute Trajectory Colors ---
    first_visible_idx = np.argmax(visibility_mask, axis=0)
    never_visible = ~np.any(visibility_mask, axis=0)
    first_visible_idx[never_visible] = 0
    
    indices = np.arange(N)
    first_visible_xyz = trajectories_3d[first_visible_idx, indices]
    first_visible_xyz[never_visible] = np.nan
    
    xyz_min = np.nanmin(first_visible_xyz, axis=0)
    xyz_max = np.nanmax(first_visible_xyz, axis=0)
    xyz_norm = (first_visible_xyz - xyz_min) / (xyz_max - xyz_min + 1e-6)
    
    scalar = np.nansum(xyz_norm, axis=1)
    scalar = (scalar - np.nanmin(scalar)) / (np.nanmax(scalar) - np.nanmin(scalar) + 1e-6)
    
    sort_idx = np.argsort(scalar)
    colors_hsv = plt.cm.hsv(np.linspace(0, 1, N))[:, :3]
    initial_colors = (colors_hsv[np.argsort(sort_idx)] * 255).astype(np.uint8)

    # --- GUI Controls ---
    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.00001, max=2, step=1e-3, initial_value=0.008
        )
        gui_line_width = server.gui.add_slider(
            "Line width", min=0.01, max=10, step=0.01, initial_value=0.3
        )
        gui_timestep = server.gui.add_slider(
            "Timestep", min=0, max=num_frames - 1, step=1, initial_value=0, disabled=True
        )
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=24
        )
        gui_show_trajectories = server.gui.add_checkbox("Show Trajectories", True)
        gui_max_traj_length = server.gui.add_slider(
            "Max trajectory length", min=1, max=200, step=1, initial_value=10
        )
        gui_downsample = server.gui.add_slider(
            "Downsample rate", min=1, max=500, step=1, 
            initial_value=DEFAULT_POINT_DOWNSAMPLE_RATE
        )
        gui_max_displacement = server.gui.add_slider(
            "Displacement value", min=0.01, max=100, step=0.01, 
            initial_value=MAX_DISPLACEMENT
        )
        gui_vis_mode = server.gui.add_button_group(
            "Visualization Mode", ("PointCloud", "Tracking", "Both")
        )
        gui_vis_mode.value = "Both"

    # --- Trajectory Line Node Initialization ---
    line_node = server.scene.add_line_segments(
        name="/trajectories",
        points=np.zeros((1, 2, 3)), 
        colors=np.zeros((1, 2, 3), dtype=np.uint8),
        line_width=gui_line_width.value,
        visible=True,
    )

    # --- Visualization Logic ---
    def apply_vis_mode(mode: str, timestep: int):
        if mode == "PointCloud":
            for i, node in enumerate(point_nodes):
                node.visible = (i == timestep)
            line_node.visible = False
        elif mode == "Tracking":
            for node in point_nodes:
                node.visible = False
            line_node.visible = True
        else:  # "Both"
            for i, node in enumerate(point_nodes):
                node.visible = (i == timestep)
            line_node.visible = True

    try:
        apply_vis_mode(gui_vis_mode.value, gui_timestep.value)
    except Exception:
        pass

    try:
        @gui_vis_mode.on_update
        def _(_):
            apply_vis_mode(gui_vis_mode.value, gui_timestep.value)
    except Exception:
        pass

    # --- Trajectory Update Logic ---
    # Use history buffers with strict tracking IDs to avoid connecting wrong lines due to point flickering
    live_history_pos = []
    live_history_col = []
    live_history_ind = []

    def update_trajectories(t_curr: int, t_prev: int, show_lines: bool, line_width: float):
        line_node.visible = show_lines
        line_node.line_width = line_width

        if not show_lines:
            return

        # Clear historical trajectories if the timeline loops back to the start
        if t_curr == 0 and t_prev != 0:
            live_history_pos.clear()
            live_history_col.clear()
            live_history_ind.clear()
            line_node.points = np.zeros((0, 2, 3))
            return

        current_active_indices = np.array([], dtype=int)

        if t_curr > 0 and t_curr < trajectories_3d.shape[0]:
            pos_curr = 100 * trajectories_3d[t_curr - 1]
            pos_next = 100 * trajectories_3d[t_curr]
            
            # Get the visibility mask of the current frame
            valid_mask = visibility_mask[t_curr - 1] & visibility_mask[t_curr]
            
            if np.any(valid_mask):
                all_indices = np.arange(N)
                
                p1 = pos_curr[valid_mask]
                p2 = pos_next[valid_mask]
                curr_inds = all_indices[valid_mask]
                
                # Filter abnormal jumps (teleportation artifacts)
                dist = np.linalg.norm(p2 - p1, axis=1)
                jump_mask = dist < gui_max_displacement.value
                
                if np.any(jump_mask):
                    final_p1 = p1[jump_mask]
                    final_p2 = p2[jump_mask]
                    segments = np.stack([final_p1, final_p2], axis=1)
                    
                    cols = initial_colors[valid_mask][jump_mask]
                    segment_colors = np.stack([cols, cols], axis=1)
                    final_indices = curr_inds[jump_mask]
                    
                    # Store in history
                    live_history_pos.append(segments)
                    live_history_col.append(segment_colors)
                    live_history_ind.append(final_indices)
                    
                    current_active_indices = final_indices

        # Maintain maximum tail length
        max_len = int(gui_max_traj_length.value)
        while len(live_history_pos) > max_len:
            live_history_pos.pop(0)
            live_history_col.pop(0)
            live_history_ind.pop(0)

        # Rendering logic: only show the history of trajectories that are [currently alive], and add fading trail effect
        if live_history_pos and len(current_active_indices) > 0:
            active_lookup = np.zeros(N, dtype=bool)
            active_lookup[current_active_indices] = True
            
            render_pos_list = []
            render_col_list = []
            
            for i, (h_pos, h_col, h_ind) in enumerate(zip(live_history_pos, live_history_col, live_history_ind)):
                keep_mask = active_lookup[h_ind]
                
                if np.any(keep_mask):
                    
                    rgb_cols = h_col[keep_mask]
                    render_pos_list.append(h_pos[keep_mask])
                    render_col_list.append(rgb_cols)
            
            if render_pos_list:
                line_node.points = np.concatenate(render_pos_list, axis=0)
                line_node.colors = np.concatenate(render_col_list, axis=0)
            else:
                line_node.points = np.zeros((0, 2, 3))
        else:
            line_node.points = np.zeros((0, 2, 3))

    # Initialize the first call
    update_trajectories(gui_timestep.value, -1, gui_show_trajectories.value, gui_line_width.value)

    # --- Main Loop ---
    prev_timestep = gui_timestep.value
    last_vis_mode = gui_vis_mode.value
    current_downsample = gui_downsample.value

    while True:
        # 1. Handle Playback
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        current_timestep = gui_timestep.value

        # 2. Reset history if looping back to start
        if current_timestep == 0:
            live_history_pos.clear()
            live_history_col.clear()
            live_history_ind.clear()

        # 3. Handle Downsample Rate Change
        if gui_downsample.value != current_downsample:
            current_downsample = gui_downsample.value
            print(f"[Downsample] Updating rate to: 1/{current_downsample}")
            
            trajectories_3d = trajectory_all_raw[:, ::current_downsample]
            visibility_mask = (~np.isnan(trajectory_all_raw).any(axis=-1))[:, ::current_downsample]
            
            N = trajectories_3d.shape[1]
            
            # Clear cache
            live_history_pos.clear()
            live_history_col.clear()
            live_history_ind.clear()
            line_node.points = np.zeros((0, 2, 3))
            
            first_visible_idx = np.argmax(visibility_mask, axis=0)
            never_visible = ~np.any(visibility_mask, axis=0)
            first_visible_idx[never_visible] = 0
            
            indices = np.arange(N)
            first_visible_xyz = trajectories_3d[first_visible_idx, indices]
            first_visible_xyz[never_visible] = np.nan
            
            xyz_min = np.nanmin(first_visible_xyz, axis=0)
            xyz_max = np.nanmax(first_visible_xyz, axis=0)
            xyz_norm = (first_visible_xyz - xyz_min) / (xyz_max - xyz_min + 1e-6)
            
            scalar = np.nansum(xyz_norm, axis=1)
            scalar = (scalar - np.nanmin(scalar)) / (np.nanmax(scalar) - np.nanmin(scalar) + 1e-6)
            
            sort_idx = np.argsort(scalar)
            colors_hsv = plt.cm.hsv(np.linspace(0, 1, N))[:, :3]
            initial_colors = (colors_hsv[np.argsort(sort_idx)] * 255).astype(np.uint8)

        # 4. Update Point Size
        if gui_point_size.value != point_nodes[current_timestep].point_size:
            for node in point_nodes:
                node.point_size = gui_point_size.value

        # 5. Update Frame Visibility & Trajectories
        if current_timestep != prev_timestep:
            if gui_vis_mode.value in ("PointCloud", "Both"):
                point_nodes[prev_timestep].visible = False
                point_nodes[current_timestep].visible = True
            else:
                point_nodes[prev_timestep].visible = False
                point_nodes[current_timestep].visible = False

            if gui_vis_mode.value in ("Tracking", "Both"):
                update_trajectories(current_timestep, prev_timestep, True, gui_line_width.value)
            
            prev_timestep = current_timestep

        # 6. Handle Mode Change
        if gui_vis_mode.value != last_vis_mode:
            apply_vis_mode(gui_vis_mode.value, current_timestep)
            last_vis_mode = gui_vis_mode.value

        time.sleep(1.0 / gui_framerate.value)

if __name__ == "__main__":
    main()
