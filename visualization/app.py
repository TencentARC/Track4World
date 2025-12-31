import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import viser
import gradio as gr
from pathlib import Path
from typing import Tuple

# ==============================================================================
# Global State
# ==============================================================================

# 启动 Viser 服务器 (端口 8080)
global_server = viser.ViserServer(port=8080)
current_thread = None
stop_event = threading.Event()


# ==============================================================================
# Helper Functions (保持不变)
# ==============================================================================
# ... (此处省略 remove_radius_outlier_gpu, remove_radius_outlier_open3d, compute_trajectory_colors 函数，与之前一致) ...

def remove_radius_outlier_open3d(points: np.ndarray, nb_neighbors: int = 30, std_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_filtered, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    points_filtered = np.asarray(pcd_filtered.points)
    return points_filtered, np.array(ind)

def compute_trajectory_colors(trajectories: np.ndarray, mask: np.ndarray) -> np.ndarray:
    N = trajectories.shape[1]
    first_visible_idx = np.argmax(mask, axis=0)
    never_visible = ~np.any(mask, axis=0)
    first_visible_idx[never_visible] = 0
    indices = np.arange(N)
    first_visible_xyz = trajectories[first_visible_idx, indices]
    first_visible_xyz[never_visible] = np.nan
    xyz_min = np.nanmin(first_visible_xyz, axis=0)
    xyz_max = np.nanmax(first_visible_xyz, axis=0)
    xyz_norm = (first_visible_xyz - xyz_min) / (xyz_max - xyz_min + 1e-6)
    scalar = np.nansum(xyz_norm, axis=1)
    scalar = (scalar - np.nanmin(scalar)) / (np.nanmax(scalar) - np.nanmin(scalar) + 1e-6)
    sort_idx = np.argsort(scalar)
    colors_hsv = plt.cm.hsv(np.linspace(0, 1, N))[:, :3]
    final_colors = np.zeros((N, 3))
    final_colors[sort_idx] = colors_hsv
    return (final_colors * 255).astype(np.uint8)

# ==============================================================================
# Visualization Logic (后台线程)
# ==============================================================================

def visualization_loop(ply_dir_str: str, max_frames: int, default_downsample: int):
    server = global_server
    server.scene.reset() 
    
    ply_dir = Path(ply_dir_str)
    if not ply_dir.exists():
        print(f"Error: {ply_dir} not found")
        return

    # --- Load Data ---
    print("Scanning directory for PLY files...")
    ply_files = sorted([f for f in ply_dir.glob("frame_*.ply")], key=lambda x: int(x.stem.split("_")[-1]))
    num_frames = min(max_frames, len(ply_files))
    
    if num_frames == 0:
        print("No frames found.")
        return

    traj_path = ply_dir / 'trajectory_all_pointmap.npy'
    if not traj_path.exists():
        print("Trajectory file not found.")
        return
    
    print("Loading trajectory data...")
    trajectory_all_raw = np.load(str(traj_path))

    # --- Pre-load Point Clouds ---
    point_nodes = []
    print(f"Loading {num_frames} point cloud frames...")
    
    for i, ply_file in enumerate(ply_files[:num_frames]):
        if stop_event.is_set(): return 
        
        pcd = o3d.io.read_point_cloud(str(ply_file))
        _, ind = pcd.remove_radius_outlier(nb_points=10, radius=0.001)
        points_all = 100 * np.asarray(pcd.points) * [1, -1, -1]
        colors_all = np.asarray(pcd.colors)

        node = server.scene.add_point_cloud(
            name=f"/frames/t{i}/point_cloud",
            points=points_all,
            colors=(colors_all * 255).astype(np.uint8),
            point_size=0.01,
            point_shape="rounded",
            visible=(i == 0)
        )
        point_nodes.append(node)

    # --- GUI Controls ---
    with server.gui.add_folder(f"Controls ({ply_dir.name})"):
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider("FPS", min=1, max=60, step=0.1, initial_value=24)
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=num_frames - 1, step=1, initial_value=0)
        gui_point_size = server.gui.add_slider("Point size", min=0.001, max=0.05, step=0.001, initial_value=0.01)
        gui_line_width = server.gui.add_slider("Line width", min=0.1, max=5.0, step=0.1, initial_value=0.5)
        gui_max_traj_length = server.gui.add_slider("Trail Length", min=1, max=50, step=1, initial_value=5)
        gui_downsample = server.gui.add_slider("Downsample", min=1, max=100, step=1, initial_value=default_downsample)
        gui_vis_mode = server.gui.add_button_group("Vis Mode", ("PointCloud", "Tracking", "Both"))
        gui_vis_mode.value = "Both"

    line_node = server.scene.add_line_segments(
        name="/trajectories",
        points=np.zeros((0, 2, 3)),
        colors=np.zeros((0, 2, 3), dtype=np.uint8),
        line_width=gui_line_width.value,
        visible=True,
    )

    visibility_mask_raw = ~np.isnan(trajectory_all_raw).any(axis=-1)

    prev_timestep = -1
    current_downsample = -1
    trajectories_3d = None
    visibility_mask = None
    initial_colors = None
    history_lines_pos = [] 
    history_lines_col = []

    print("Visualization loop started.")

    while not stop_event.is_set():
        # 1. Playback
        if gui_playing.value:
            next_step = (gui_timestep.value + 1) % num_frames
            gui_timestep.value = next_step
        
        t_curr = gui_timestep.value
        
        # 2. Downsample Update
        if gui_downsample.value != current_downsample:
            current_downsample = gui_downsample.value
            trajectories_3d = trajectory_all_raw[:, ::current_downsample]
            visibility_mask = visibility_mask_raw[:, ::current_downsample]
            initial_colors = compute_trajectory_colors(trajectories_3d, visibility_mask)
            history_lines_pos = []
            history_lines_col = []
            prev_timestep = -1 

        # 3. Vis Mode
        show_points = gui_vis_mode.value in ("PointCloud", "Both")
        show_lines = gui_vis_mode.value in ("Tracking", "Both")

        # 4. Update Points
        if t_curr != prev_timestep:
            if prev_timestep >= 0 and prev_timestep < len(point_nodes):
                point_nodes[prev_timestep].visible = False
            if show_points:
                point_nodes[t_curr].visible = True
                point_nodes[t_curr].point_size = gui_point_size.value
            else:
                point_nodes[t_curr].visible = False

        # 5. Update Lines
        if show_lines:
            line_node.visible = True
            line_node.line_width = gui_line_width.value
            
            if t_curr == 0:
                history_lines_pos = []
                history_lines_col = []
                line_node.points = np.zeros((0, 2, 3))
            elif t_curr > 0 and t_curr < num_frames:
                t_prev = t_curr - 1
                pos_prev = 100 * trajectories_3d[t_prev]
                pos_curr = 100 * trajectories_3d[t_curr]
                valid_mask = visibility_mask[t_prev] & visibility_mask[t_curr]
                
                if np.any(valid_mask):
                    p1 = pos_prev[valid_mask]
                    p2 = pos_curr[valid_mask]
                    dist = np.linalg.norm(p2 - p1, axis=1)
                    jump_mask = dist < 1.0
                    
                    if np.any(jump_mask):
                        final_p1 = p1[jump_mask]
                        final_p2 = p2[jump_mask]
                        segments = np.stack([final_p1, final_p2], axis=1)
                        cols = initial_colors[valid_mask][jump_mask]
                        segment_colors = np.stack([cols, cols], axis=1)
                        history_lines_pos.append(segments)
                        history_lines_col.append(segment_colors)
            
            while len(history_lines_pos) > gui_max_traj_length.value:
                history_lines_pos.pop(0)
                history_lines_col.pop(0)
            
            if history_lines_pos:
                line_node.points = np.concatenate(history_lines_pos, axis=0)
                line_node.colors = np.concatenate(history_lines_col, axis=0)
            else:
                line_node.points = np.zeros((0, 2, 3))
        else:
            line_node.visible = False

        prev_timestep = t_curr
        time.sleep(1.0 / gui_framerate.value)

# ==============================================================================
# Gradio Interface
# ==============================================================================

def launch_visualization(ply_dir, max_frames, downsample, custom_url):
    global current_thread, stop_event, global_server
    
    # 1. 停止旧线程
    if current_thread is not None and current_thread.is_alive():
        stop_event.set()
        current_thread.join()
    
    stop_event.clear()
    
    # 2. 启动新线程
    current_thread = threading.Thread(
        target=visualization_loop, 
        args=(ply_dir, int(max_frames), int(downsample)),
        daemon=True
    )
    current_thread.start()
    
    # 3. 确定使用的 URL
    # 如果用户提供了代理 URL，直接使用；否则尝试使用本地地址
    if custom_url and custom_url.strip():
        viser_url = custom_url.strip()
    else:
        viser_port = global_server.get_port()
        viser_url = f"http://127.0.0.1:{viser_port}"
    
    print(f"Embedding Viser URL: {viser_url}")
    
    # 4. 返回 iframe
    return f"""
    <div style="width: 100%; height: 600px; border: 1px solid #ccc; border-radius: 8px; overflow: hidden;">
        <iframe src="{viser_url}" width="100%" height="100%" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>
    </div>
    <p style="text-align: center; color: #666; font-size: 0.9em; margin-top: 5px;">
        Viewer Source: <a href="{viser_url}" target="_blank">{viser_url}</a>
    </p>
    """

with gr.Blocks(title="Holi4D Visualizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 Holi4D Interactive Visualizer (Remote Mode)")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_dir = gr.Textbox(
                label="Data Directory Path", 
                placeholder="/path/to/your/data", 
                value="./assets/epef"
            )
            
            # === 新增：代理 URL 输入框 ===
            viser_url_input = gr.Textbox(
                label="Viser Proxy URL (Required for Remote Server)", 
                placeholder="Paste the https://.../proxy/8080/ link here",
                value="",
                info="如果你在远程服务器，请粘贴 8080 端口的代理链接（包含 ?websocket=...）"
            )
            
            with gr.Row():
                max_frames_input = gr.Number(label="Max Frames", value=400, precision=0)
                downsample_input = gr.Number(label="Initial Downsample", value=10, precision=0)
            
            btn_launch = gr.Button("🚀 Launch Visualization", variant="primary")
            
            gr.Markdown("""
            ### 使用说明:
            1. 输入数据文件夹路径。
            2. **关键步骤**：将你得到的 `https://.../proxy/8080/?websocket=...` 链接粘贴到 **Viser Proxy URL** 框中。
            3. 点击 Launch。
            """)

        with gr.Column(scale=3):
            viser_output = gr.HTML(label="Viser Viewer")

    btn_launch.click(
        fn=launch_visualization,
        inputs=[input_dir, max_frames_input, downsample_input, viser_url_input],
        outputs=viser_output
    )

if __name__ == "__main__":
    print("Starting Gradio...")
    # 允许所有 IP 访问 Gradio，以便你能打开网页
    demo.launch(server_name="0.0.0.0", server_port=7860)