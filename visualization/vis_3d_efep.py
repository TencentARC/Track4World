import time
import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import open3d as o3d
import viser
import viser.transforms as tf
from tqdm.auto import tqdm

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


# ==============================================================================
# Helper Functions
# ==============================================================================

def remove_radius_outlier_gpu(points: np.ndarray, nb_points: int = 40, radius: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes outliers using GPU-accelerated Faiss (K-Nearest Neighbors).
    
    Args:
        points: Input point cloud data (N, 3).
        nb_points: Number of neighbors to check.
        radius: Search radius.
        
    Returns:
        Tuple containing filtered points and indices of kept points.
    """
    if not HAS_FAISS:
        raise RuntimeError("Faiss is not installed.")
        
    points_torch = torch.from_numpy(points).float().cuda()
    n, d = points_torch.shape
    
    # Initialize Faiss GPU resources
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # Add points and search
    gpu_index.add(points) # Note: Faiss handles numpy/torch conversion usually
    D, I = gpu_index.search(points, nb_points)
    
    # Filter based on radius (D is squared L2 distance)
    mask = D[:, -1] < radius**2
    
    return points[mask].cpu().numpy(), np.nonzero(mask.cpu().numpy())[0]


def remove_radius_outlier_open3d(points: np.ndarray, nb_points: int = 40, radius: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes outliers using Open3D (CPU implementation).
    
    Args:
        points: Input point cloud data (N, 3).
        nb_points: Number of neighbors to check.
        radius: Search radius.
        
    Returns:
        Tuple containing filtered points and indices of kept points.
    """
    # 1. Convert numpy array to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 2. Call Open3D's efficient C++ implementation
    #    Returns: (filtered_pcd, indices_list)
    pcd_filtered, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    
    # 3. Convert back to numpy
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
                        default='/group/40075/jiahaolu/cleaned_code/Holi4D/huggingfacespace/Holi4d_demo/parkour', 
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
    # Sort files by frame number
    ply_files = sorted([f for f in ply_dir.glob("frame_*.ply")], key=lambda x: int(x.stem.split("_")[-1]))#[:-1]
    
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
        
        # Basic cleanup
        _, ind = pcd.remove_radius_outlier(nb_points=10, radius=0.001)
        
        # Scale and flip coordinates
        points_all = 100 * np.asarray(pcd.points) * [1, -1, -1]
        colors_all = np.asarray(pcd.colors)

        # Initialize colors based on the first frame (Logic from original code)
        if i == 0:
            num_tracks = colors_all.shape[0]
            track_colors = plt.cm.get_cmap('hsv', num_tracks)
            colors_track = track_colors(np.arange(num_tracks))[:, :3]
            initial_colors = (colors_track * 255).astype(np.uint8)

        # Add to Viser scene (initially hidden except frame 0)
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
    # Use the raw loaded trajectory data
    trajectories_3d = trajectory_all_raw
    
    # Create visibility mask (True where data is not NaN)
    visibility_mask = ~np.isnan(trajectory_all_raw).any(axis=-1)  # [T, N]

    # Filter trajectories using radius outlier removal
    print("Filtering trajectories...")
    for i in tqdm(range(trajectories_3d.shape[0])):
        pts = trajectories_3d[i][visibility_mask[i]]
        if pts.shape[0] == 0:
            continue
            
        _, ind = remove_radius_outlier_open3d(pts, nb_points=20, radius=0.001)
        
        # Update mask based on inliers
        mask = visibility_mask[i].copy()
        valid_indices = np.where(mask)[0]
        filtered_indices = valid_indices[ind]
        new_mask = np.zeros_like(mask, dtype=bool)
        new_mask[filtered_indices] = True
        visibility_mask[i] = new_mask

    # Apply initial downsampling
    trajectories_3d = trajectories_3d[:, ::DEFAULT_POINT_DOWNSAMPLE_RATE]
    visibility_mask = visibility_mask[:, ::DEFAULT_POINT_DOWNSAMPLE_RATE]

    T, N, _ = trajectories_3d.shape

    # --- Compute Trajectory Colors ---
    # Determine color based on the position at the first visible frame
    first_visible_idx = np.argmax(visibility_mask, axis=0)
    never_visible = ~np.any(visibility_mask, axis=0)
    first_visible_idx[never_visible] = 0
    
    indices = np.arange(N)
    first_visible_xyz = trajectories_3d[first_visible_idx, indices]
    first_visible_xyz[never_visible] = np.nan
    
    # Normalize positions for color mapping
    xyz_min = np.nanmin(first_visible_xyz, axis=0)
    xyz_max = np.nanmax(first_visible_xyz, axis=0)
    xyz_norm = (first_visible_xyz - xyz_min) / (xyz_max - xyz_min + 1e-6)
    
    scalar = np.nansum(xyz_norm, axis=1)
    scalar = (scalar - np.nanmin(scalar)) / (np.nanmax(scalar) - np.nanmin(scalar) + 1e-6)
    
    sort_idx = np.argsort(scalar)
    colors_hsv = plt.cm.hsv(np.linspace(0, 1, N))[:, :3]
    
    # Assign colors sorted by spatial position
    initial_colors = (colors_hsv[np.argsort(sort_idx)] * 255).astype(np.uint8)

    # --- GUI Controls ---
    with server.gui.add_folder("Playback"):
        # Appearance
        gui_point_size = server.gui.add_slider("Point size", min=0.00001, max=0.02, step=1e-3, initial_value=0.008)
        gui_line_width = server.gui.add_slider("Line width", min=0.01, max=10, step=0.01, initial_value=0.3)
        
        # Playback
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=num_frames - 1, step=1, initial_value=0, disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", False)
        gui_framerate = server.gui.add_slider("FPS", min=1, max=60, step=0.1, initial_value=24)
        
        # Trajectory Settings
        gui_show_trajectories = server.gui.add_checkbox("Show Trajectories", True)
        gui_max_traj_length = server.gui.add_slider("Max trajectory length", min=1, max=200, step=1, initial_value=5)
        gui_downsample = server.gui.add_slider("Downsample rate", min=1, max=500, step=1, initial_value=DEFAULT_POINT_DOWNSAMPLE_RATE)
        
        # Visualization Mode
        gui_vis_mode = server.gui.add_button_group("Visualization Mode", ("PointCloud", "Tracking", "Both"))
        gui_vis_mode.value = "Both"

    # --- Trajectory Line Node Initialization ---
    line_node = server.scene.add_line_segments(
        name="/trajectories",
        points=np.zeros((1, 2, 3)),  # Start with 1 empty segment to avoid errors
        colors=np.zeros((1, 2, 3), dtype=np.uint8),
        line_width=gui_line_width.value,
        visible=True,
    )

    # --- Visualization Logic ---

    def apply_vis_mode(mode: str, timestep: int):
        """
        Updates visibility of point clouds and lines based on the selected mode.
        """
        if mode == "PointCloud":
            # Show only current frame's point cloud
            for i, node in enumerate(point_nodes):
                node.visible = (i == timestep)
            line_node.visible = False
        elif mode == "Tracking":
            # Hide all point clouds, show trajectory lines
            for node in point_nodes:
                node.visible = False
            line_node.visible = True
        else:  # "Both"
            for i, node in enumerate(point_nodes):
                node.visible = (i == timestep)
            line_node.visible = True

    # Initial application of visualization mode
    try:
        apply_vis_mode(gui_vis_mode.value, gui_timestep.value)
    except Exception:
        pass

    # Callback for mode change (if supported by Viser version)
    try:
        @gui_vis_mode.on_update
        def _(_):
            apply_vis_mode(gui_vis_mode.value, gui_timestep.value)
    except Exception:
        pass

    # --- Trajectory Update Logic ---
    
    # History buffers for the trail effect
    all_line_positions = []
    all_line_colors = []

    def update_trajectories(t_curr: int, show_lines: bool, line_width: float):
        """
        Calculates and updates the trajectory lines for the current timestep.
        """
        line_node.visible = show_lines
        line_node.line_width = line_width

        if not show_lines or t_curr == 0:
            return

        # Get positions for current and next frame (scaled)
        # Note: t_curr is used as index for 'prev' because we draw line from t to t+1
        prev_positions = 100 * trajectories_3d[t_curr][visibility_mask[t_curr]]
        
        # Ensure we don't go out of bounds
        if t_curr + 1 >= trajectories_3d.shape[0]:
            return
            
        curr_positions = 100 * trajectories_3d[t_curr+1][visibility_mask[t_curr]]
        
        if prev_positions.shape[0] == 0:
            return

        # Prepare colors
        new_colors = np.repeat(initial_colors[visibility_mask[t_curr]][:, None, :], 2, axis=1)

        # Filter out large jumps (teleportation artifacts)
        displacement = np.linalg.norm(curr_positions - prev_positions, axis=1)
        MAX_DISPLACEMENT = 1.0
        valid_mask = displacement < MAX_DISPLACEMENT
        
        prev_positions = prev_positions[valid_mask]
        curr_positions = curr_positions[valid_mask]
        new_colors = new_colors[valid_mask]

        if prev_positions.shape[0] == 0:
            return

        # Create new line segments
        new_lines = np.stack([prev_positions, curr_positions], axis=1)
        all_line_positions.append(new_lines)
        all_line_colors.append(new_colors)

        # Trim history to max length
        max_len = int(gui_max_traj_length.value)
        if len(all_line_positions) > max_len:
            all_line_positions.pop(0)
            all_line_colors.pop(0)

        # Flatten and update Viser node
        full_lines = np.concatenate(all_line_positions, axis=0)
        full_colors = np.concatenate(all_line_colors, axis=0)

        line_node.points = full_lines
        line_node.colors = full_colors

    # Initial trajectory update
    update_trajectories(gui_timestep.value, gui_show_trajectories.value, gui_line_width.value)

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
            all_line_positions = []
            all_line_colors = []

        # 3. Handle Downsample Rate Change
        if gui_downsample.value != current_downsample:
            current_downsample = gui_downsample.value
            print(f"[Downsample] Updating rate to: 1/{current_downsample}")
            
            # Re-slice data
            trajectories_3d = trajectory_all_raw[:, ::current_downsample]
            visibility_mask = (~np.isnan(trajectory_all_raw).any(axis=-1))[:, ::current_downsample]
            
            # Re-calculate colors for new subset
            N = trajectories_3d.shape[1]
            all_line_positions = []
            all_line_colors = []
            
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
            # Toggle point cloud visibility
            if gui_vis_mode.value in ("PointCloud", "Both"):
                point_nodes[prev_timestep].visible = False
                point_nodes[current_timestep].visible = True
            else:
                point_nodes[prev_timestep].visible = False
                point_nodes[current_timestep].visible = False

            # Update trajectory lines (only if mode includes Tracking)
            if gui_vis_mode.value in ("Tracking", "Both"):
                update_trajectories(current_timestep, True, gui_line_width.value)
            
            prev_timestep = current_timestep

        # 6. Handle Mode Change (Fallback check)
        if gui_vis_mode.value != last_vis_mode:
            apply_vis_mode(gui_vis_mode.value, current_timestep)
            last_vis_mode = gui_vis_mode.value

        # Sleep to maintain framerate
        time.sleep(1.0 / gui_framerate.value)

if __name__ == "__main__":
    main()