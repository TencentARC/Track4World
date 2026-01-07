import time
import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import viser
from tqdm.auto import tqdm
from vis_3d_efep import remove_std_outlier_open3d
# ==============================================================================
# Global Constants
# ==============================================================================
DEFAULT_POINT_DOWNSAMPLE_RATE = 10  # Downsample rate for trajectories
STATIC_SKIP_FRAMES = 5             # Skip frames for accumulating static points
STATIC_VOXEL_SIZE = 0.02           # Voxel size for downsampling static background
MAX_DISPLACEMENT = 0.5            # Maximum displacement for trajectory segments
DEFAULT_CAM_POS = (4.34, 0.34, 4.74)
DEFAULT_LOOK_AT = (0.0, 0.0, 0.0)
DEFAULT_UP = (0.33, 0.0, 0.4)
DEFAULT_WXYZ = (-0.05, 0.98, -0.17, -0.12)
# ==============================================================================
# Helper Functions
# ==============================================================================

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
    parser.add_argument('--ply_dir', type=str, 
                        default='results/cat/3d_efep_output', 
                        help='Directory containing frame_xx.ply files and masks')
    args = parser.parse_args()

    ply_dir = Path(args.ply_dir)
    if not ply_dir.exists():
        raise FileNotFoundError(f"{ply_dir} not found")

    # --- Viser Server Setup ---
    server = viser.ViserServer()
    
    # 【Fix 1】: Use callback to set initial camera
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
        traj_dyn_mask_raw = np.ones((trajectory_all_raw.shape[0], trajectory_all_raw.shape[1]), dtype=bool)
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

    trajectories_3d = trajectory_all_raw.copy()
    for i in tqdm(range(trajectory_all_raw.shape[0]), desc="Transforming Trajectories"):
        trajectories_3d[i] = cam_points_to_world(trajectory_all_raw[i], c2w[i]) * [1, -1, -1]
    
    visibility_mask = ~np.isnan(trajectory_all_raw).any(axis=-1)

    print("Filtering trajectories...")
    for i in tqdm(range(trajectories_3d.shape[0])):
        mask = visibility_mask[i]
        if i < traj_dyn_mask_raw.shape[0]:
            dyn_mask = traj_dyn_mask_raw[i] == 1
            visibility_mask[i] = mask & dyn_mask
        pts = trajectories_3d[i][visibility_mask[i]]
        _, ind = remove_std_outlier_open3d(pts)
        # Update mask based on inliers
        mask = visibility_mask[i].copy()
        valid_indices = np.where(mask)[0]
        new_mask = np.zeros_like(mask, dtype=bool)
        if ind.shape[0] > 0:
            filtered_indices = valid_indices[ind]
            new_mask[filtered_indices] = True
        visibility_mask[i] = new_mask

    # --- 3. Load Point Clouds (Split Static/Dynamic) ---
    dynamic_point_nodes: List[viser.PointCloudHandle] = []
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

        points_world = 100 * cam_points_to_world(points_local, c2w[i]) * [1, -1, -1]
        
        pts_dyn = points_world[is_dynamic]
        col_dyn = colors[is_dynamic]
        pts_stat = points_world[~is_dynamic]
        col_stat = colors[~is_dynamic]

        # Add dynamic points to the scene
        node = server.scene.add_point_cloud(
            name=f"/dynamic/t{i}",
            points=pts_dyn,
            colors=(col_dyn * 255).astype(np.uint8),
            point_size=0.01,
            point_shape="rounded",
            visible=False 
        )
        dynamic_point_nodes.append(node)

        # Accumulate static points (skip some frames to reduce redundancy)
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
        pcd_static, ind = pcd_static.remove_statistical_outlier(nb_neighbors=128, std_ratio=4.0)
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
    trajectories_3d_down = trajectories_3d[:, ::DEFAULT_POINT_DOWNSAMPLE_RATE]
    visibility_mask_down = visibility_mask[:, ::DEFAULT_POINT_DOWNSAMPLE_RATE]
    
    N_traj = trajectories_3d_down.shape[1]
    initial_colors = np.zeros((N_traj, 3), dtype=np.uint8)
    
    if N_traj > 0:
        first_visible_idx = np.argmax(visibility_mask_down, axis=0)
        never_visible = ~np.any(visibility_mask_down, axis=0)
        visible_mask = ~never_visible
        n_visible = np.sum(visible_mask)

        first_visible_idx[never_visible] = 0
        indices = np.arange(N_traj)
        first_visible_xyz = trajectories_3d_down[first_visible_idx, indices]
        first_visible_xyz[never_visible] = np.nan
        
        xyz_min = np.nanmin(first_visible_xyz, axis=0)
        xyz_max = np.nanmax(first_visible_xyz, axis=0)
        xyz_norm = (first_visible_xyz - xyz_min) / (xyz_max - xyz_min + 1e-6)
        scalar = np.nansum(xyz_norm, axis=1)
        
        if n_visible > 0:
            vis_scalar = scalar[visible_mask]
            sort_idx_vis = np.argsort(vis_scalar)
            colors_hsv_vis = plt.cm.hsv(np.linspace(0, 1, n_visible))[:, :3]
            colors_rgb_vis = (colors_hsv_vis * 255).astype(np.uint8)
            mapped_vis_colors = np.zeros((n_visible, 3), dtype=np.uint8)
            mapped_vis_colors[sort_idx_vis] = colors_rgb_vis
            initial_colors[visible_mask] = mapped_vis_colors
    else:
        initial_colors = np.zeros((0, 3), dtype=np.uint8)

    # --- 6. GUI Controls ---
    with server.gui.add_folder("Playback"):
        # --- Point Cloud Appearance ---
        # Controls size for stationary elements (e.g., walls, floors)
        gui_static_point_size = server.gui.add_slider(
            "Static Point Size", min=0.001, max=10, step=0.001, initial_value=0.01
        )
        # Controls size for moving elements (e.g., people, vehicles)
        gui_dynamic_point_size = server.gui.add_slider(
            "Dynamic Point Size", min=0.001, max=10, step=0.001, initial_value=0.01
        )
        # Controls the thickness of the 3D trajectory lines
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
        # Number of historical frames to draw for trails
        gui_max_traj_length = server.gui.add_slider(
            "Trail Length", min=1, max=50, step=1, initial_value=5
        )
        # Threshold to filter out noise or limit motion visualization
        gui_max_displacement = server.gui.add_slider(
            "Max Displacement", min=0.1, max=10.0, step=0.1, initial_value=MAX_DISPLACEMENT
        )

        # --- Camera Controls ---
        # Vector inputs for manual camera positioning
        gui_cam_pos = server.gui.add_vector3(
            "Position", initial_value=DEFAULT_CAM_POS, step=0.05
        )
        gui_cam_look = server.gui.add_vector3(
            "Look At", initial_value=DEFAULT_LOOK_AT, step=0.05
        )
        gui_cam_up = server.gui.add_vector3(
            "Up Direction", initial_value=DEFAULT_UP, step=0.05
        )
        
        # Action buttons for view management
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

    line_node = server.scene.add_line_segments(
        name="/trajectories",
        points=np.zeros((0, 2, 3)),
        colors=np.zeros((0, 2, 3), dtype=np.uint8),
        line_width=gui_line_width.value,
        visible=True,
    )

    # ==========================================================================
    # Core Update Logic
    # ==========================================================================
    def update_scene_state(
        t_curr: int, 
        t_prev: int, 
        history_pos: List[np.ndarray], 
        history_col: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        
        # Update static point visibility and size
        if static_node is not None:
            static_node.visible = gui_show_static.value
            static_node.point_size = gui_static_point_size.value

        # Update dynamic points visibility
        if t_curr != t_prev:
            if t_prev >= 0 and t_prev < len(dynamic_point_nodes):
                dynamic_point_nodes[t_prev].visible = False
            
            if gui_show_dynamic.value:
                dynamic_point_nodes[t_curr].visible = True
                dynamic_point_nodes[t_curr].point_size = gui_dynamic_point_size.value
            else:
                dynamic_point_nodes[t_curr].visible = False

        # Update trajectory lines
        if gui_show_traj.value and N_traj > 0:
            line_node.visible = True
            line_node.line_width = gui_line_width.value

            if t_curr == 0 and t_prev != 0:
                history_pos.clear()
                history_col.clear()
                line_node.points = np.zeros((0, 2, 3))
                return history_pos, history_col

            if t_curr < num_frames:
                pos_curr = 100 * trajectories_3d_down[t_curr - 1]
                pos_next = 100 * trajectories_3d_down[t_curr]
                valid_mask = visibility_mask_down[t_curr - 1] & visibility_mask_down[t_curr]
                
                if np.any(valid_mask):
                    p1 = pos_curr[valid_mask]
                    p2 = pos_next[valid_mask]
                    dist = np.linalg.norm(p2 - p1, axis=1)
                    jump_mask = dist < gui_max_displacement.value
                    
                    if np.any(jump_mask):
                        final_p1 = p1[jump_mask]
                        final_p2 = p2[jump_mask]
                        segments = np.stack([final_p1, final_p2], axis=1)
                        cols = initial_colors[valid_mask][jump_mask]
                        segment_colors = np.stack([cols, cols], axis=1)
                        history_pos.append(segments)
                        history_col.append(segment_colors)

            # Maintain max trail length
            while len(history_pos) > gui_max_traj_length.value:
                history_pos.pop(0)
                history_col.pop(0)
            
            if history_pos:
                line_node.points = np.concatenate(history_pos, axis=0)
                line_node.colors = np.concatenate(history_col, axis=0)
            else:
                line_node.points = np.zeros((0, 2, 3))
        else:
            line_node.visible = False
            
        return history_pos, history_col

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
            rec_prev_t = -1
            
            try:
                for t in tqdm(range(num_frames), desc="Recording .viser"):
                    rec_history_pos, rec_history_col = update_scene_state(
                        t_curr=t, 
                        t_prev=rec_prev_t, 
                        history_pos=rec_history_pos, 
                        history_col=rec_history_col
                    )
                    rec_prev_t = t
                    serializer.insert_sleep(1.0 / gui_framerate.value)
                
                print("Serializing data...")
                data = serializer.serialize()
                output_filename = "track_recording.viser"
                Path(output_filename).write_bytes(data)
                print(f"Successfully saved {output_filename} ({len(data)/1024/1024:.2f} MB)")
                
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
    print_counter = 0

    while True:
        # Update timestep if playing
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        
        t_curr = gui_timestep.value

        live_history_pos, live_history_col = update_scene_state(
            t_curr=t_curr,
            t_prev=prev_timestep,
            history_pos=live_history_pos,
            history_col=live_history_col
        )

        # 【Fix 2】: Print camera parameters periodically
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
