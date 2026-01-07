import time
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import viser
import viser.transforms as tf
from tqdm.auto import tqdm

# ==============================================================================
# Configuration
# ==============================================================================

# Global downsample rate for trajectory lines to improve performance
POINT_DOWNSAMPLE_RATE = 50 

def main(
    max_frames: int = 400, 
    share: bool = False
) -> None:
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Holi4D 3D-FF (First Frame) Visualization")
    parser.add_argument('--ply_dir', type=str, 
                        default='results/cat/3d_ff_output', 
                        help='Directory containing flow_xx.ply and vis_xx.npy files')
    args = parser.parse_args()

    ply_dir = Path(args.ply_dir)
    if not ply_dir.exists():
        raise FileNotFoundError(f"Directory not found: {ply_dir}")

    # --- Viser Server Setup ---
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    # --- File Discovery ---
    # Sort files by frame index
    ply_files = sorted(ply_dir.glob("flow_*.ply"), key=lambda x: int(x.stem.split("_")[-1]))
    vis_files = sorted(ply_dir.glob("vis_*.npy"), key=lambda x: int(x.stem.split("_")[-1]))
    
    num_frames = min(max_frames, len(ply_files))
    if num_frames == 0:
        raise RuntimeError(f"No valid flow_*.ply files found in {ply_dir}")
    
    if len(vis_files) < num_frames:
        print("Warning: Mismatch between PLY and NPY file counts. Visualization might be incomplete.")

    # --- Data Loading & Processing ---
    # We need to find points that are valid (not outliers) across ALL frames 
    # to draw consistent trajectories.
    
    raw_pcds = []
    raw_vis = []
    common_indices = None

    print(f"Loading {num_frames} frames...")
    
    for i in tqdm(range(num_frames), desc="Processing Frames"):
        # 1. Load Point Cloud
        pcd = o3d.io.read_point_cloud(str(ply_files[i]))
        
        # 2. Load Visibility/Confidence Map
        # Shape is usually (N, 1) or (N,), we ensure it's 1D
        vis = np.load(vis_files[i]).reshape(-1)
        
        # 3. Outlier Removal
        # We calculate indices of "good" points for this specific frame
        _, ind_list = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
        current_indices = np.array(ind_list)
        
        # 4. Update Common Intersection
        # We only want to track points that survive filtering in EVERY frame (or the frames processed so far)
        if common_indices is None:
            common_indices = current_indices
        else:
            # Intersection of current valid points and previous common points
            common_indices = np.intersect1d(common_indices, current_indices)

        raw_pcds.append(pcd)
        raw_vis.append(vis)

    if common_indices is None or len(common_indices) == 0:
        raise RuntimeError("No common points found across frames after filtering! Try increasing the outlier radius.")

    print(f"Tracking {len(common_indices)} common points across {num_frames} frames.")

    # --- Final Data Assembly ---
    all_points_list = []
    all_colors_list = []
    all_vis_list = []
    
    point_nodes: List[viser.PointCloudHandle] = []

    # Process filtered data for visualization
    for i in range(num_frames):
        pcd = raw_pcds[i]
        vis = raw_vis[i]
        
        # Extract only the common points
        pts = np.asarray(pcd.points)[common_indices] * 100 # Scale up for visualization
        cols = np.asarray(pcd.colors)[common_indices]
        v = vis[common_indices]
        
        all_points_list.append(pts)
        all_colors_list.append(cols)
        all_vis_list.append(v)

        # Add Point Cloud Node to Viser (Hidden by default)
        # We filter by visibility score > 0.1 for the point cloud display
        mask = v > 0.1
        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/point_cloud",
                points=pts[mask],
                colors=(cols[mask] * 255).astype(np.uint8),
                point_size=0.01,
                point_shape="rounded",
                visible=(i == 0) # Only show first frame initially
            )
        )

    # Stack into (T, N, 3) arrays
    all_points = np.stack(all_points_list, axis=0)
    all_vis = np.stack(all_vis_list, axis=0)
    
    # Generate consistent colors for trajectories based on the first frame
    # Using HSV colormap for distinct tracking lines
    num_tracks = len(common_indices)
    track_colors_hsv = plt.cm.get_cmap('hsv', num_tracks)
    initial_colors = (track_colors_hsv(np.arange(num_tracks))[:, :3] * 255).astype(np.uint8)
    
    # Subsample colors for the downsampled trajectory lines
    initial_colors_sampled = initial_colors[::POINT_DOWNSAMPLE_RATE]

    # ===================== GUI Controls =====================
    with server.gui.add_folder("Playback Controls"):
        # --- Playback Logic ---
        # Toggles the animation state (on/off)
        gui_playing = server.gui.add_checkbox("Playing", True)
        
        # Controls the speed of the temporal update
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=1, initial_value=24
        )
        
        # Manual scrub control for the current video/sequence frame
        gui_timestep = server.gui.add_slider(
            "Timestep", min=0, max=num_frames - 1, step=1, initial_value=0
        )
        
        # --- Appearance & Aesthetics ---
        # Controls the radius of individual 3D points
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=10, step=0.001, initial_value=0.01
        )
        
        # Controls the thickness of tracking lines/trajectories
        gui_line_width = server.gui.add_slider(
            "Line width", min=0.1, max=5.0, step=0.1, initial_value=0.5
        )
        
        # Sets how many historical frames of motion are visible behind a point
        gui_max_traj_length = server.gui.add_slider(
            "Trail Length (Frames)", min=1, max=50, step=1, initial_value=5
        )

        # --- Visualization Mode ---
        # Switch between viewing static geometry, motion paths, or both
        gui_vis_mode = server.gui.add_button_group(
            "Vis Mode", ("PointCloud", "Tracking", "Both")
        )
        gui_vis_mode.value = "Both"
        
    # ===================== Trajectory Setup =====================
    # Line node for drawing trails
    line_node = server.scene.add_line_segments(
        name="/trajectories",
        points=np.zeros((0, 2, 3)),
        colors=np.zeros((0, 2, 3), dtype=np.uint8),
        line_width=gui_line_width.value,
        visible=True,
    )

    # History buffers for the trail effect
    all_line_positions = []
    all_line_colors = []

    def update_trajectories(t_curr: int, show_lines: bool):
        """Updates the trajectory lines based on current timestep."""
        if not show_lines:
            line_node.visible = False
            return
        
        line_node.visible = True
        line_node.line_width = gui_line_width.value

        if t_curr == 0:
            return

        # Get positions for t-1 and t
        # Apply downsampling to reduce rendering load
        prev_pts = all_points[t_curr - 1, ::POINT_DOWNSAMPLE_RATE]
        curr_pts = all_points[t_curr, ::POINT_DOWNSAMPLE_RATE]
        
        # Check visibility for both frames
        prev_vis = all_vis[t_curr - 1, ::POINT_DOWNSAMPLE_RATE] > 0.1
        curr_vis = all_vis[t_curr, ::POINT_DOWNSAMPLE_RATE] > 0.1
        valid_mask = prev_vis & curr_vis

        if not np.any(valid_mask):
            return

        # Create line segments
        p1 = prev_pts[valid_mask]
        p2 = curr_pts[valid_mask]
        new_lines = np.stack([p1, p2], axis=1) # (M, 2, 3)

        # Create colors (repeated for start/end of segment)
        c = initial_colors_sampled[valid_mask]
        new_colors = np.stack([c, c], axis=1) # (M, 2, 3)

        # Update history
        all_line_positions.append(new_lines)
        all_line_colors.append(new_colors)

        # Trim history
        MAX_TRAJECTORY_LENGTH = gui_max_traj_length.value
        if len(all_line_positions) > MAX_TRAJECTORY_LENGTH:
            all_line_positions.pop(0)
            all_line_colors.pop(0)

        # Update Viser Node
        if all_line_positions:
            line_node.points = np.concatenate(all_line_positions, axis=0)
            line_node.colors = np.concatenate(all_line_colors, axis=0)

    # Initial update
    update_trajectories(gui_timestep.value, True)

    # ===================== Main Loop =====================
    prev_timestep = gui_timestep.value

    while True:
        # 1. Handle Playback
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        current_timestep = gui_timestep.value

        # Reset trails if looping back to start
        if current_timestep == 0:
            all_line_positions = []
            all_line_colors = []
            line_node.points = np.zeros((0, 2, 3))

        # 2. Determine Visibility based on Mode
        show_points = gui_vis_mode.value in ("PointCloud", "Both")
        show_lines = gui_vis_mode.value in ("Tracking", "Both")

        # 3. Update Point Clouds
        # Only update if timestep changed or visibility toggled
        if current_timestep != prev_timestep or True: # Logic simplified for robustness
            for i, node in enumerate(point_nodes):
                is_current = (i == current_timestep)
                node.visible = is_current and show_points
                if is_current:
                    node.point_size = gui_point_size.value

        # 4. Update Trajectories
        if current_timestep != prev_timestep:
            update_trajectories(current_timestep, show_lines)
            prev_timestep = current_timestep
        else:
            # Just toggle visibility if paused
            line_node.visible = show_lines

        time.sleep(1.0 / gui_framerate.value)

if __name__ == "__main__":
    main()

