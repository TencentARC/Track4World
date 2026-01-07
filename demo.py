import os
import sys
import time
import json
import shutil
import argparse
import logging
from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
import cv2
import PIL.Image

# Custom modules (Assuming these exist in your project structure)
import holi4d.utils.basic
import holi4d.utils.improc
import holi4d.utils.saveload
import utils3d

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# I/O Helper Functions
# ==============================================================================

def read_mp4(name_path: str) -> Tuple[List[np.ndarray], int]:
    """
    Reads all frames from an MP4 video file.

    Args:
        name_path: Path to the input video file.

    Returns:
        A tuple containing:
        - List of frames (RGB numpy arrays).
        - The framerate (FPS) of the video.
    """
    vidcap = cv2.VideoCapture(name_path)
    if not vidcap.isOpened():
        raise IOError(f"Cannot open video file: {name_path}")
        
    framerate = int(round(vidcap.get(cv2.CAP_PROP_FPS)))
    logger.info(f'Video FPS: {framerate}')
    
    frames = []
    while True:
        ret, frame = vidcap.read()
        if not ret:
            break
        # Convert BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
    vidcap.release()
    logger.info(f"Read {len(frames)} frames from {name_path}")
    return frames, framerate


def save_ply(
    save_path: Union[str, os.PathLike], 
    vertices: np.ndarray, 
    faces: np.ndarray, 
    vertex_colors: np.ndarray,
    vertex_normals: Optional[np.ndarray] = None,
):
    """
    Saves a mesh or point cloud to a .ply file using trimesh.

    Args:
        save_path: Destination path.
        vertices: (N, 3) array of vertex coordinates.
        faces: (M, 3) array of face indices (empty for point clouds).
        vertex_colors: (N, 3) or (N, 4) array of colors.
        vertex_normals: (N, 3) array of normals (optional).
    """
    try:
        import trimesh
    except ImportError:
        logger.error(
            "Trimesh is required for saving PLY files. "
            "Please install it via `pip install trimesh`."
        )
        return

    mesh = trimesh.Trimesh(
        vertices=vertices, 
        faces=faces, 
        vertex_colors=vertex_colors,
        vertex_normals=vertex_normals,
        process=False  # Disable auto-processing to keep data raw
    )
    mesh.export(save_path)

# ==============================================================================
# Visualization Tools
# ==============================================================================

def draw_pts_gpu(
    rgbs: torch.Tensor, 
    trajs: torch.Tensor, 
    visibs: torch.Tensor, 
    colormap: np.ndarray, 
    rate: int = 1, 
    bkg_opacity: float = 0.5
) -> np.ndarray:
    """
    Draws 2D trajectories onto RGB frames using GPU acceleration.
    
    This function renders points as "icons" (circles with soft edges) directly 
    on the GPU tensor to avoid slow CPU loops.

    Args:
        rgbs: (T, C, H, W) Video tensor.
        trajs: (T, N, 2) Trajectory coordinates.
        visibs: (T, N) Visibility mask.
        colormap: (N, 3) Colors for each trajectory.
        rate: Sampling rate for point size calculation.
        bkg_opacity: Opacity of the background video (0.0 to 1.0).

    Returns:
        (T, H, W, 3) Numpy array of visualized frames.
    """
    device = rgbs.device
    T, C, H, W = rgbs.shape
    
    # Permute dimensions to match drawing logic: (N, T, ...)
    trajs = trajs.permute(1, 0, 2)  # N, T, 2
    visibs = visibs.permute(1, 0)   # N, T
    N = trajs.shape[0]
    
    colors = torch.tensor(colormap, dtype=torch.float32, device=device)  # [N, 3]

    # Dim background to highlight trajectories
    rgbs = rgbs * bkg_opacity
    
    # Determine point radius and opacity based on sampling rate
    opacity = 1.0
    if rate == 1:
        radius = 1
        opacity = 0.9
    elif rate == 2:
        radius = 1
    elif rate == 4:
        radius = 2
    elif rate == 8:
        radius = 4
    else:
        radius = 6
    
    # Sharpness controls the anti-aliasing of the circle icon
    sharpness = 0.15 + 0.05 * np.log2(rate)
    
    # --- Create Drawing Icon (Soft Circle) ---
    D = radius * 2 + 1
    y = torch.arange(D, device=device).float()[:, None] - radius
    x = torch.arange(D, device=device).float()[None, :] - radius
    dist2 = x**2 + y**2
    # Formula: I(r) = clamp(1 - (r^2 - R^2/2) / (2R * sigma), 0, 1)
    icon = torch.clamp(
        1 - (dist2 - (radius**2) / 2.0) / (radius * 2 * sharpness), 
        0, 1
    )
    icon = icon.view(1, D, D)
    
    # Offsets for icon placement
    dx = torch.arange(-radius, radius + 1, device=device)
    dy = torch.arange(-radius, radius + 1, device=device)
    disp_y, disp_x = torch.meshgrid(dy, dx, indexing="ij")
    
    # --- Main Drawing Loop ---
    for t in range(T):
        mask = visibs[:, t]  # [N]
        if mask.sum() == 0:
            continue
            
        # Get coordinates for visible points
        xy = trajs[mask, t] + 0.5  # [N_vis, 2]
        xy[:, 0] = xy[:, 0].clamp(0, W - 1)
        xy[:, 1] = xy[:, 1].clamp(0, H - 1)
        colors_now = colors[mask]  # [N_vis, 3]
        
        N_vis = xy.shape[0]
        cx = xy[:, 0].long()
        cy = xy[:, 1].long()
        
        # Calculate pixel grid for icons centered at (cx, cy)
        x_grid = cx[:, None, None] + disp_x  # [N_vis, D, D]
        y_grid = cy[:, None, None] + disp_y  # [N_vis, D, D]
        
        # Keep only valid pixels within canvas bounds
        valid = (x_grid >= 0) & (x_grid < W) & (y_grid >= 0) & (y_grid < H)
        x_valid = x_grid[valid]
        y_valid = y_grid[valid]
        icon_weights = icon.expand(N_vis, D, D)[valid]
        
        # Expand colors to match valid pixels
        colors_valid = colors_now[:, :, None, None].expand(N_vis, 3, D, D)
        colors_valid = colors_valid.permute(1, 0, 2, 3)[:, valid]
        
        # Flatten indices for scatter operation
        idx_flat = (y_valid * W + x_valid).long()

        # Use scatter_add_ for efficient GPU drawing (accumulate colors and weights)
        accum = torch.zeros_like(rgbs[t])  # [3, H, W]
        weight = torch.zeros(1, H * W, device=device)  # [1, H*W]
        img_flat = accum.view(C, -1)
        
        weighted_colors = colors_valid * icon_weights
        img_flat.scatter_add_(
            1, idx_flat.unsqueeze(0).expand(C, -1), weighted_colors
        )
        weight.scatter_add_(1, idx_flat.unsqueeze(0), icon_weights.unsqueeze(0))
        weight = weight.view(1, H, W)

        # Alpha blending: Image = Background * (1 - alpha) + Foreground * alpha
        alpha = weight.clamp(0, 1) * opacity
        accum = accum / (weight + 1e-6)  # Normalize accumulated colors
        rgbs[t] = rgbs[t] * (1 - alpha) + accum * alpha
        
    # Convert back to CPU numpy for saving
    # Clamp to 0-255, convert to byte, permute to (T, H, W, 3), move to CPU
    rgbs = rgbs.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    
    # Enhance saturation if background is black (pure visualization mode)
    if bkg_opacity == 0.0:
        for t in range(T):
            hsv_frame = cv2.cvtColor(rgbs[t], cv2.COLOR_RGB2HSV)
            saturation_factor = 1.5
            hsv_frame[..., 1] = np.clip(
                hsv_frame[..., 1] * saturation_factor, 0, 255
            )
            rgbs[t] = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
            
    return rgbs

# ==============================================================================
# Model Loading & Utils
# ==============================================================================

def load_model(args, config: Dict) -> torch.nn.Module:
    """
    Initializes the Holi4D model and loads pretrained weights.
    """
    from holi4d.nets.model import Holi4D
    
    logger.info("Initializing Holi4D Model...")

    model = Holi4D(
        **config['model'],
        seqlen=16,
        use_3d=True,
        use_model=args.coordinate.split('_')[-1]
    )

    if args.ckpt_init and os.path.exists(args.ckpt_init):
        logger.info(f'Loading weights from local file: {args.ckpt_init}...')
        state_dict = torch.load(args.ckpt_init, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    else:
        # Fallback to Hub download
        url = "https://huggingface.co/cyun9286/holi4d/resolve/main/holi4d.pth"
        logger.info(f'Local checkpoint not found. Downloading from {url}...')
        state_dict = torch.hub.load_state_dict_from_url(
            url, map_location='cpu', check_hash=False
        )
        model.load_state_dict(state_dict, strict=False)
    
    model.cuda()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    
    logger.info('Model loaded and set to evaluation mode.')
    return model

# ==============================================================================
# Inference Functions
# ==============================================================================

def forward_video(rgbs: torch.Tensor, framerate: int, model: torch.nn.Module, args):
    """
    Runs 2D tracking inference and generates a visualization video.
    """
    B, T, C, H, W = rgbs.shape
    assert C == 3 and B == 1
    device = rgbs.device

    # Create 2D grid coordinates for flow calculation
    # Shape: 1, H*W, 2
    grid_xy = holi4d.utils.basic.gridcloud2d(
        1, H, W, norm=False, device='cuda:0'
    ).float() 
    grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W) # 1, 1, 2, H, W

    torch.cuda.empty_cache()
    logger.info('Starting 2D forward pass...')
    f_start_time = time.time()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # 1. Forward tracking (Time t -> t+n)
        flows_e, visconf_maps_e, _, _, _ = model.forward_sliding(
            rgbs[:, args.query_frame:], 
            iters=args.inference_iters, 
            sw=None, 
            is_training=False
        )

        traj_maps_e = flows_e.to(torch.float32).cuda() + grid_xy
        visconf_maps_e = visconf_maps_e.to(torch.float32)

        # 2. Backward tracking (Time t -> 0) if query_frame is not the start
        if args.query_frame > 0:
            backward_flows_e, backward_visconf_maps_e, _, _, _ = model.forward_sliding(
                rgbs[:, :args.query_frame + 1].flip([1]),
                iters=args.inference_iters,
                sw=None,
                is_training=False
            )
            backward_traj_maps_e = backward_flows_e.to(torch.float32).cuda() + grid_xy
            
            # Flip back to chronological order
            backward_traj_maps_e = backward_traj_maps_e.flip([1])[:, :-1]
            backward_visconf_maps_e = backward_visconf_maps_e.flip([1])[:, :-1]

            # Merge trajectories
            traj_maps_e = torch.cat([backward_traj_maps_e, traj_maps_e], dim=1)
            visconf_maps_e = torch.cat(
                [backward_visconf_maps_e, visconf_maps_e], dim=1
            ).to(torch.float32)
            
    ftime = time.time() - f_start_time
    logger.info(
        f'Forward pass finished; {ftime:.2f}s / {T} frames; '
        f'{round(T / ftime)} FPS'
    )
    
    # 3. Visualization Preparation
    rate = args.rate
    # Subsample trajectories for visualization
    trajs_e = traj_maps_e[:, :, :, ::rate, ::rate].reshape(B, T, 2, -1)
    trajs_e = trajs_e.permute(0, 1, 3, 2)  # B, T, N, 2
    
    visconfs_e = visconf_maps_e[:, :, :, ::rate, ::rate].reshape(B, T, 2, -1)
    visconfs_e = visconfs_e.permute(0, 1, 3, 2)  # B, T, N, 2

    xy0 = trajs_e[0, 0].cpu().numpy()
    colors = holi4d.utils.improc.get_2d_colors(xy0, H, W)

    fn = os.path.basename(args.mp4_path).split('.')[0]
    rgb_out_f = os.path.join(
        args.save_base_dir, 
        f"{args.mode}_output", 
        f"pt_vis_{fn}_rate{rate}_q{args.query_frame}.mp4"
    )
    temp_dir = os.path.join(
        args.save_base_dir, 
        f"{args.mode}_output", 
        f"temp_pt_vis_{fn}_rate{rate}_q{args.query_frame}"
    )
    holi4d.utils.basic.mkdir(temp_dir)

    # Draw frames
    frames = draw_pts_gpu(
        rgbs[0].to('cuda:0'), 
        trajs_e[0], 
        visconfs_e[0, :, :, 1] > args.conf_thr,
        colors, 
        rate=rate, 
        bkg_opacity=args.bkg_opacity
    )

    # Stack Input and Output frames for comparison
    if args.vstack:
        frames_input = rgbs[0].clamp(0, 255).byte()
        frames_input = frames_input.permute(0, 2, 3, 1).cpu().numpy()
        frames = np.concatenate([frames_input, frames], axis=1)
    elif args.hstack:
        frames_input = rgbs[0].clamp(0, 255).byte()
        frames_input = frames_input.permute(0, 2, 3, 1).cpu().numpy()
        frames = np.concatenate([frames_input, frames], axis=2)
    
    # 4. Save frames and generate MP4
    logger.info('Writing frames to disk...')
    for ti in range(T):
        temp_out_f = f'{temp_dir}/{ti:03d}.jpg'
        im = PIL.Image.fromarray(frames[ti])
        im.save(temp_out_f)
        
    logger.info(f'Generating MP4: {rgb_out_f}')
    os.system(
        f'ffmpeg -y -hide_banner -loglevel error -f image2 -framerate {framerate} '
        f'-pattern_type glob -i "./{temp_dir}/*.jpg" -c:v libx264 -crf 20 '
        f'-pix_fmt yuv420p {rgb_out_f}'
    )

    # shutil.rmtree(temp_dir, ignore_errors=True) 
    return None


def forward_video3d_pair(rgbs: torch.Tensor, model: torch.nn.Module, args) -> Dict:
    """
    Runs 3D tracking in 'Pair' mode (allows skipping frames).
    """
    B, T_full, C, H, W = rgbs.shape
    
    torch.cuda.empty_cache()
    logger.info('Starting 3D (Pair/Skip Frames) forward pass...')
    f_start_time = time.time()

    # Select frames (e.g., every 5th frame)
    if args.Ts == -1:
        select_views = range(0, T_full, 1)
    else:
        # Default stride logic (can be adjusted)
        select_views = range(0, min(args.Ts, T_full), 1)
        # select_views = range(0, T_full, 4)
    
    select_views = list(select_views)
    rgbs_selected = rgbs[:, select_views]
    T_selected = len(select_views)
    logger.info(f"Selected {T_selected} frames from {T_full} for 3D Pair inference.")

    with torch.autocast(device_type='cuda', dtype=torch.float32):
        output = model.infer_pair(
            rgbs_selected, 
            iters=args.inference_iters, 
            sw=None, 
            is_training=False, 
            tracking3d=True, 
            force_projection=True
        )
        
        # Unpack results
        traj_maps_2d = output[1]['flow_2d']
        visconf_maps = output[1]['visconf_maps_e']
        traj_maps_3d = output[1]['flow_3d']
        world_points = output[0]['world_points']
        camera_poses = output[0]['camera_poses']
        points = output[0]['points']
        masks = output[0]['mask']

    ftime = time.time() - f_start_time
    logger.info(f'3D (Pair) Forward pass finished; {ftime:.2f}s')

    return {
        'traj_3d': traj_maps_3d[0],      # (T, 3, H, W)
        'traj_2d': traj_maps_2d[0],      # (T, 2, H, W)
        'visconf': visconf_maps[0],      # (T, 2, H, W)
        'rgbs': rgbs_selected[0],        # (T, 3, H, W)
        'points': points[0],             # (T, H, W, 3)
        'masks': masks[0],               # (T, H, W)
        'world_points': world_points[0], # (T, H, W, 3)
        'camera_poses': camera_poses     # (T, 4, 4)
    }, select_views


def forward_video3d_ff(rgbs: torch.Tensor, model: torch.nn.Module, args) -> Dict:
    """
    Runs 3D tracking in 'Full' mode (processes all selected frames sequentially).
    """
    B, T_full, C, H, W = rgbs.shape
    
    torch.cuda.empty_cache()
    logger.info('Starting 3D (Full Frames) forward pass...')
    f_start_time = time.time()

    if args.Ts == -1:
        select_views = range(0, T_full, 1)
    else:
        select_views = range(0, min(args.Ts, T_full), 1)
        
    select_views = list(select_views)
    rgbs_selected = rgbs[:, select_views]
    T_selected = len(select_views)

    with torch.autocast(device_type='cuda', dtype=torch.float32):
        output = model.infer(
            rgbs_selected, 
            iters=args.inference_iters, 
            sw=None, 
            is_training=False, 
            tracking3d=True
        )

        traj_maps_2d = output[1]['flow_2d']
        visconf_maps = output[1]['visconf_maps_e']
        traj_maps_3d = output[1]['flow_3d']
        points = output[0]['points']
        masks = output[0]['mask']
        world_points = output[0]['world_points']
        camera_poses = output[0]['camera_poses']

    ftime = time.time() - f_start_time
    logger.info(f'3D (Full) Forward pass finished; {ftime:.2f}s')

    return {
        'traj_3d': traj_maps_3d[0],
        'traj_2d': traj_maps_2d[0],
        'visconf': visconf_maps[0],
        'rgbs': rgbs_selected[0],
        'points': points[0],
        'masks': masks[0],
        'world_points': world_points[0], # (T, H, W, 3)
        'camera_poses': camera_poses     # (T, 4, 4)
    }, select_views

# ==============================================================================
# Saving Logic: Long-term Trajectories & Point Clouds
# ==============================================================================

def save_efep(
    results: Dict, 
    masks_tensor, 
    save_dir: str, 
    W: int, 
    H: int, 
    vis_mode: str,
    coordinate: str,
):
    """
    Computes and saves long-term 3D trajectories and visibility masks.
    
    This function tracks pixels from frame t to t+1 using 2D flow, 
    then samples the 3D coordinates at the new location.
    
    Optimization:
        To prevent GPU OOM (Out of Memory) on long sequences, the 
        `trajectory_storage_3d` tensor is kept on the CPU.
    """
    logger.info(f"Saving Long Trajectories (CPU Optimized) to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Extract Data (GPU) ---
    all_pairwise_flows_3d = results['traj_3d'].permute(0, -1, 1, 2) # (T, 3, H, W)
    all_pairwise_flows_2d = results['traj_2d']                      # (T, 2, H, W)
    
    # Calculate visibility (Forward * Backward confidence)
    all_visconf_maps = (
        results['visconf'][:, 0].cuda() * results['visconf'][:, 1].cuda()
    )[:, None]
    
    rgbs = results['rgbs']
    points = results['points'].permute(0, -1, 1, 2)
    
    if vis_mode == 'geometry':
        points_vis = results['points'].permute(0, -1, 1, 2).clone()
    elif vis_mode == 'flow':
        points_vis = results['traj_3d'].permute(0, -1, 1, 2).clone()
    else:
        raise ValueError(f"Unknown visualization mode: {vis_mode}")
    
    camera_poses = results['camera_poses']
    masks = results['masks'][:, None]

    T_pairs = all_pairwise_flows_3d.shape[0]
    NumFrames = T_pairs + 1
    
    device = rgbs.device
    cpu_device = torch.device('cpu') 

    # --- 2. Initialize Storage (CPU) ---
    # Stores 3D coordinates for all pixels for all frames relative to t_start
    # Shape: (T, H*W, 3)
    trajectory_storage_3d = torch.full(
        (NumFrames, H * W, 3), float('nan'),
        device=cpu_device, dtype=torch.float32
    )
    trajectory_storage_dyn_mask = torch.full(
        (NumFrames, H * W), 0.0,
        device=cpu_device, dtype=torch.float32
    )
    # Initialize UV coordinates (Fixed reference frame)
    u, v = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    uv_coords_init = torch.stack([v, u], dim=-1).reshape(-1, 2) # (H*W, 2)

    # Initialize current tracking UV (on GPU)
    current_uv_map = uv_coords_init.clone().float()
    
    # Initialize mask for the first iteration
    mask_cleaned_t_start_np_pre = torch.ones(
        (H, W), device=device, dtype=torch.bool
    )

    # --- 3. Iterate per frame ---
    for t_start in range(NumFrames):
        logger.info(f"  > Processing Frame {t_start} / {NumFrames - 1}...")

        # --- Clean Mask and Extract RGB ---
        mask_t_start_bool = masks[t_start].squeeze().cpu().numpy()
        depth_t_start_np = points[t_start, -1].cpu().numpy()
        
        # Remove depth edges from mask to avoid flying pixels at object boundaries
        kernel_size = 4
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dyn_mask = masks_tensor[t_start].cpu().numpy().astype(bool)  # (H, W, 1)
        static_mask = ~dyn_mask
        
        dyn_mask_eroded = cv2.erode(
            dyn_mask.astype(np.uint8), kernel, iterations=2
        ).astype(bool)
        static_mask_eroded = cv2.erode(
            static_mask.astype(np.uint8), kernel, iterations=10
        ).astype(bool) 
        
        mask_eroded = dyn_mask_eroded | static_mask_eroded
        
        # Combine masks: 1. Model mask, 2. Depth edges (removed), 3. Eroded boundary
        is_not_depth_edge = ~utils3d.numpy.depth_edge(depth_t_start_np, rtol=0.04)
        
        if coordinate.split('_')[0] == 'camera':
            mask_cleaned_t_start_np = (
                mask_t_start_bool & is_not_depth_edge
            )
        else:
            mask_cleaned_t_start_np = (
                mask_t_start_bool & is_not_depth_edge & mask_eroded
            )

        rgb_t_start_np = rgbs[t_start].permute(1, 2, 0).cpu().numpy()
        rgb_t_start_np = rgb_t_start_np.astype(np.float32) / 255.0
        points_t_start_torch = points[t_start].permute(1, 2, 0) # GPU

        # --- Save Raw Frame PLY ---
        faces_frame, vertices_frame, colors_frame, uvs_frame = utils3d.numpy.image_mesh(
            points_t_start_torch.cpu().numpy(),
            rgb_t_start_np,
            utils3d.numpy.image_uv(width=W, height=H),
            mask=mask_cleaned_t_start_np,
            tri=True
        )
        
        save_ply(
            f"{save_dir}/frame_{t_start:03d}.ply",
            vertices_frame,
            np.zeros((0, 3), dtype=np.int32),
            colors_frame,
            None
        )
        
        _, visconf_flow, _, _ = utils3d.numpy.image_mesh(                    
            masks_tensor[t_start].cpu().numpy()[..., None],
            rgb_t_start_np,
            utils3d.numpy.image_uv(width=W, height=H),
            mask=mask_cleaned_t_start_np,
            tri=True
        )

        np.save(f"{save_dir}/pc_dyn_mask_{t_start:03d}.npy", visconf_flow)
        
        # Stop here for last frame
        if t_start == NumFrames - 1:
            continue

        # --- Save Flow PLY (Projected Points) ---
        flows_t_start_torch = all_pairwise_flows_3d[t_start].permute(1, 2, 0)
        faces_flow, vertices_flow, colors_flow, uvs_flow = utils3d.numpy.image_mesh(
            flows_t_start_torch.cpu().numpy(),
            rgb_t_start_np,
            utils3d.numpy.image_uv(width=W, height=H),
            mask=mask_cleaned_t_start_np,
            tri=True
        )
        
        # Update mask for next iteration
        mask_cleaned_t_start_np_pre = torch.from_numpy(
            mask_cleaned_t_start_np
        ).to(device)
        
        save_ply(
            f"{save_dir}/flow_{t_start:03d}.ply",
            vertices_flow,
            np.zeros((0, 3), dtype=np.int32),
            colors_flow,
            None
        )

        # Initial Frame Logic: Store t0 points
        if t_start == 0:
            points_t0_cpu = points_vis[t_start].permute(1, 2, 0)
            points_t0_cpu = points_t0_cpu.reshape(-1, 3).to(cpu_device)
            trajectory_storage_3d[t_start] = points_t0_cpu
            
            masks_t0_cpu = masks_tensor[t_start].reshape(-1).to(cpu_device)
            trajectory_storage_dyn_mask[t_start] = masks_t0_cpu
            continue
        
        # --- Tracking Logic (GPU) ---
        flow_2d_px_t_torch = all_pairwise_flows_2d[t_start - 1]
        
        # Visibility check: Confidence > 0.6 AND was valid in previous frame
        vis_t_torch = (
            (all_visconf_maps[t_start - 1] > 0.6)[0] & mask_cleaned_t_start_np_pre
        )
        flow3d_torch = points_vis[t_start]
        dyn_mask_torch = masks_tensor[t_start]
        
        # Step 1: Get previous frame UVs
        uv_prev = current_uv_map.clone()
        uv_prev_int = uv_prev.round().long()
        
        # Check bounds
        valid_mask = (
            (uv_prev_int[:, 0] >= 0) & (uv_prev_int[:, 0] < W) &
            (uv_prev_int[:, 1] >= 0) & (uv_prev_int[:, 1] < H)
        )
        uv_prev_int_inbounds = uv_prev_int[valid_mask]
        
        if uv_prev_int_inbounds.shape[0] == 0:
            continue
            
        # Filter by visibility mask
        vis_mask = vis_t_torch[
            uv_prev_int_inbounds[:, 1], uv_prev_int_inbounds[:, 0]
        ]
        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)[vis_mask]

        if len(valid_idx) == 0:
            continue

        # Step 2: Propagate UVs via Flow: UV_{t} = UV_{t-1} + Flow(UV_{t-1})
        uv_t = torch.full_like(uv_prev, float('nan'))
        uv_from_flow = flow_2d_px_t_torch[
            :, uv_prev_int[valid_idx, 1], uv_prev_int[valid_idx, 0]
        ].permute(1, 0)
        uv_t[valid_idx] = uv_from_flow

        # Step 3: Sample 3D Points at new UVs
        uv_t_int = uv_t.round().long()
        valid_mask2 = (
            (uv_t_int[:, 0] >= 0) & (uv_t_int[:, 0] < W) &
            (uv_t_int[:, 1] >= 0) & (uv_t_int[:, 1] < H)
        )
        valid_idx2 = valid_mask2.nonzero(as_tuple=False).squeeze(1)

        if len(valid_idx2) == 0:
            continue

        P_valid_gpu = flow3d_torch[
            :, uv_t_int[valid_idx2, 1], uv_t_int[valid_idx2, 0]
        ].permute(1, 0)
        dyn_mask_valid_gpu = dyn_mask_torch[
            uv_t_int[valid_idx2, 1], uv_t_int[valid_idx2, 0]
        ]
        
        # Offload to CPU storage
        trajectory_storage_3d[t_start, valid_idx2] = P_valid_gpu.to(cpu_device)
        trajectory_storage_dyn_mask[t_start, valid_idx2] = dyn_mask_valid_gpu.to(cpu_device)

        # Step 4: Update current UV map for next iteration
        uv_t_int1 = uv_t_int.clone().float()
        uv_t_int1[~valid_mask2] = float('nan')
        current_uv_map = uv_t_int1

        # Step 5: Handle new points (disocclusions / entry)
        # Identify points in the original grid that are not currently being tracked
        full_idx = uv_coords_init[:, 1] * W + uv_coords_init[:, 0]
        uv_idx = uv_t_int[valid_idx2, 1] * W + uv_t_int[valid_idx2, 0]

        mask = ~torch.isin(full_idx, uv_idx)
        uv_rest = uv_coords_init[mask]

        if uv_rest.shape[0] > 0:
            P_new_gpu = flow3d_torch[
                :, uv_rest[:, 1], uv_rest[:, 0]
            ].permute(1, 0)
            dyn_mask_new_gpu = dyn_mask_torch[uv_rest[:, 1], uv_rest[:, 0]]
            
            # Expand storage on CPU to accommodate new tracks
            new_storage_block_cpu = torch.full(
                (NumFrames, uv_rest.shape[0], 3), float('nan'),
                device=cpu_device, dtype=torch.float32
            )
            new_storage_block_cpu[t_start] = P_new_gpu.to(cpu_device)
            
            new_storage_block_cpu_dyn = torch.full(
                (NumFrames, uv_rest.shape[0]), float('nan'),
                device=cpu_device, dtype=torch.float32
            )
            new_storage_block_cpu_dyn[t_start] = dyn_mask_new_gpu.to(cpu_device)
            
            trajectory_storage_3d = torch.cat(
                [trajectory_storage_3d, new_storage_block_cpu], dim=1
            )
            trajectory_storage_dyn_mask = torch.cat(
                [trajectory_storage_dyn_mask, new_storage_block_cpu_dyn], dim=1
            )
            # Update UV map on GPU
            current_uv_map = torch.cat([current_uv_map, uv_rest], dim=0)

    logger.info(f"Saved {NumFrames} frames to {save_dir}")
    np.save(
        f"{save_dir}/trajectory_all_pointmap.npy", 
        trajectory_storage_3d.numpy()
    )
    np.save(
        f"{save_dir}/trajectory_all_pointmap_dyn_mask.npy", 
        trajectory_storage_dyn_mask.numpy()
    )
    np.save(f"{save_dir}/c2w.npy", camera_poses.cpu().numpy())


def save_ff(results: Dict, save_dir: str, W: int, H: int):
    """
    (Full Mode) Saves point clouds specific to 'Full' inference output.
    Saves geometry relative to the first frame.
    """
    logger.info(f"Saving Full Mode Point Clouds to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    
    traj_3dmaps_e = results['traj_3d']
    rgbs = results['rgbs']
    points = results['points']
    camera_poses = results['camera_poses']
    masks = results['masks']
    visconf = (results['visconf'][:, 0].cuda() * results['visconf'][:, 1].cuda())
    T = traj_3dmaps_e.shape[0]

    # Save raw points
    np.save(f"{save_dir}/all_points", points.cpu().numpy())

    # Create a clean mask based on the first frame and depth edges across all frames
    mask_cleaned = masks[0].cpu().numpy() & ~utils3d.numpy.depth_edge(
        points[0, ..., -1].cpu().numpy(), rtol=0.04
    )
    for t in range(T):
        mask_cleaned_t = ~utils3d.numpy.depth_edge(
            traj_3dmaps_e[t, ..., -1].cpu().numpy(), rtol=0.04
        )
        mask_cleaned = mask_cleaned & mask_cleaned_t
        
    rgb_frame_t0 = rgbs[0].permute(1, 2, 0).cpu().numpy()
    rgb_frame_t0 = rgb_frame_t0.astype(np.float32) / 255.0
    uv_coords = utils3d.numpy.image_uv(width=W, height=H)

    for t in range(T):
        # 1. Save Frame Geometry (Reconstructed)
        mask_cleaned_frame = masks[t].cpu().numpy() & ~utils3d.numpy.depth_edge(
            points[t, ..., -1].cpu().numpy(), rtol=0.04
        )
        rgb_frame = rgbs[t].permute(1, 2, 0).cpu().numpy()
        rgb_frame = rgb_frame.astype(np.float32) / 255.0
        
        faces_frame, vertices_frame, colors_frame, uvs_frame = utils3d.numpy.image_mesh(                    
            points[t].cpu().numpy(),
            rgb_frame,
            uv_coords,
            mask=mask_cleaned_frame,
            tri=True
        )
        
        save_ply(
            f"{save_dir}/frame_{t:03d}.ply", 
            vertices_frame, 
            np.zeros((0, 3), dtype=np.int32), 
            colors_frame, 
            None
        )

        # 2. Save Flow Geometry (Projected from T0)
        # Colored by first frame to visualize tracking consistency
        faces_flow, vertices_flow, colors_flow, uvs_flow = utils3d.numpy.image_mesh(                    
            traj_3dmaps_e[t].cpu().numpy(),
            rgb_frame_t0, 
            uv_coords,
            mask=mask_cleaned, 
            tri=True
        )
        
        # Save visibility map for debugging
        _, visconf_flow, _, _ = utils3d.numpy.image_mesh(                    
            visconf[t].cpu().numpy()[..., None],
            rgb_frame_t0,
            uv_coords,
            mask=mask_cleaned, 
            tri=True
        )
        np.save(f"{save_dir}/vis_{t:03d}.npy", visconf_flow)
        
        save_ply(
            f"{save_dir}/flow_{t:03d}.ply", 
            vertices_flow, 
            np.zeros((0, 3), dtype=np.int32), 
            colors_flow, 
            None
        )

    logger.info(f"Saved {T} frames to {save_dir}")
    np.save(f"{save_dir}/c2w.npy", camera_poses.cpu().numpy())

# ==============================================================================
# Main Logic
# ==============================================================================

def run_demo(model, args):
    """
    Main orchestration function.
    """
    # 1. Load Video
    logger.info(f"Loading video: {args.mp4_path}")
    rgbs, framerate = read_mp4(args.mp4_path)
    if not rgbs:
        logger.error("Error: Could not read video frames.")
        return

    H_orig, W_orig = rgbs[0].shape[:2]
    
    # --- New: Load Dynamic Masks ---
    mask_dir = Path(args.save_base_dir).joinpath("mask")
    logger.info(f"Loading masks from: {mask_dir}")

    # Get all mask files and sort them (assuming png or jpg format)
    mask_files = sorted(
        list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.jpg"))
    )

    if len(mask_files) < len(rgbs):
        logger.warning(
            f"Warning: Fewer masks ({len(mask_files)}) than video frames ({len(rgbs)})."
        )

    # 2. Preprocessing (Crop and Resize)
    if args.max_frames and len(rgbs) > args.max_frames:
        logger.info(f"Clipping video to first {args.max_frames} frames.")
        rgbs = rgbs[:args.max_frames]
        mask_files = mask_files[:args.max_frames]

    # Calculate scale to fit image_size while maintaining aspect ratio
    scale = min(int(args.image_size) / H_orig, int(args.image_size) / W_orig)
    H, W = int(H_orig * scale), int(W_orig * scale)

    # Ensure dimensions are divisible by 64 (common requirement for UNet architectures)
    H, W = (H // 64) * 64, (W // 64) * 64

    logger.info(f"Resizing video from ({H_orig}, {W_orig}) to ({H}, {W})")
    rgbs_resized = [
        cv2.resize(rgb, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
        for rgb in rgbs
    ]

    masks_processed = []
    for m_path in mask_files:
        # Read mask image
        mask_img = cv2.imread(str(m_path))  # (H_orig, W_orig, 3)

        if mask_img is None:
            logger.error(f"Failed to read mask: {m_path}")
            continue

        # Resize mask
        # NOTE: Use INTER_NEAREST for masks to avoid interpolation artifacts
        # and preserve binary (0/1) values along object boundaries
        mask_resized = cv2.resize(
            mask_img, dsize=(W, H), interpolation=cv2.INTER_NEAREST
        )

        # Binarization logic:
        # Color (0, 0, 0) indicates static regions;
        # any non-zero value in any channel indicates dynamic regions
        is_dynamic = np.any(mask_resized > 0, axis=-1).astype(np.float32)

        masks_processed.append(is_dynamic)

    # 3. Convert to Tensor
    masks_tensor = torch.stack(
        [torch.from_numpy(m) for m in masks_processed], dim=0
    )
    rgbs_tensor = [
        torch.from_numpy(rgb).permute(2, 0, 1) for rgb in rgbs_resized
    ]
    # Shape: 1, T, C, H, W
    rgbs_tensor = torch.stack(rgbs_tensor, dim=0).unsqueeze(0).float() 
    logger.info(f"Input Tensor Shape: {rgbs_tensor.shape}")
    
    # 4. Inference
    results = None
    with torch.no_grad():
        if args.mode == '2d':
            logger.info("--- Running 2D Tracking Mode ---")
            forward_video(rgbs_tensor.cuda(), framerate, model, args)
        
        elif args.mode == '3d_efep':
            logger.info("--- Running 3D Pair Mode (Every Frame Every Pixel) ---")
            results, select_views = forward_video3d_pair(
                rgbs_tensor.cuda(), model, args
            )
        
        elif args.mode == '3d_ff':
            logger.info("--- Running 3D First Frame Tracking Mode ---")
            results, select_views = forward_video3d_ff(
                rgbs_tensor.cuda(), model, args
            )
        
        else:
            logger.warning("No valid mode selected ('2d', '3d_efep' or '3d_ff').")
    
    masks_tensor = masks_tensor[select_views]
    rgbs_tensor = rgbs_tensor[:, select_views]

    save_rgb_dir = os.path.join(args.save_base_dir, "final_rgb")
    save_mask_dir = os.path.join(args.save_base_dir, "final_dyn_mask")

    os.makedirs(save_rgb_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    for i, idx in enumerate(select_views):
        # RGB: shape (C, H, W) -> convert to HWC for cv2
        rgb_img = np.transpose(rgbs_tensor[0, i].numpy(), (1, 2, 0))
        rgb_img = rgb_img.astype(np.uint8)
        rgb_path = os.path.join(save_rgb_dir, f"frame_{i:04d}.png")
        # cv2 uses BGR
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        # Mask: 0=static, 1=dynamic -> scale to 0/255
        mask_img = (masks_tensor[i].numpy() * 255).astype(np.uint8)
        mask_path = os.path.join(save_mask_dir, f"mask_{i:04d}.png")
        cv2.imwrite(mask_path, mask_img)

    logger.info(f"Saved {len(select_views)} frames to:")
    logger.info(f"  RGBs: {save_rgb_dir}")
    logger.info(f"  Masks: {save_mask_dir}")
    
    # 5. Post-processing / Saving 3D results
    if results:
        # Determine output directory
        if args.save_base_dir:
            save_dir = os.path.join(args.save_base_dir, f"{args.mode}_output")
        else:
            save_dir = f"./output_{args.mode}"
            logger.info(f"No save_base_dir provided, saving to {save_dir}")

        if args.mode == '3d_efep':
            save_efep(
                results, masks_tensor.cuda(), save_dir, W, H, args.vis_mode, args.coordinate
            )
        elif args.mode == '3d_ff':
            save_ff(results, save_dir, W, H)

    logger.info("Demo execution complete.")


def main():
    parser = argparse.ArgumentParser(description="Holi4D Demo Script")
    
    # --- Paths and Model Config ---
    parser.add_argument(
        "--ckpt_init", type=str, default='checkpoints/holi4d.pth', 
        help="Path to local checkpoint file (optional)"
    )
    parser.add_argument(
        "--mp4_path", type=str, default='demo_data/cat.mp4', 
        help="Input MP4 video file path"
    )
    parser.add_argument(
        "--config_path", type=str, default='holi4d/config/eval/v1.json', 
        help="Path to model config json"
    )
    
    # --- Output ---
    parser.add_argument(
        "--save_base_dir", type=str, default='results/horsejump-high', 
        help="Root directory for output."
    )

    # --- Mode Selection ---
    parser.add_argument(
        "--mode", type=str, default='3d_ff', 
        choices=['2d', '3d_ff', '3d_efep'],
        help="'2d': 2d tracking, '3d_ff': 3d first frame tracking, "
            "'3d_efep' 3d tracking every pixel of every frame"
    )
    parser.add_argument(
        "--vis_mode", type=str, default='geometry', 
        choices=['flow', 'geometry'],
        help="'flow': visualize flow, 'geometry': visualize geometry"
    )
    parser.add_argument(
        "--coordinate", type=str, default='camera_base', 
        choices=['camera_base', 'world_pi3', 'world_depthanythingv3'],
        help="'camera': camera centric, 'world': world centric"
    )
    
    # --- Inference Params ---
    parser.add_argument(
        "--query_frame", type=int, default=0, 
        help="Start frame index for tracking"
    )
    parser.add_argument(
        "--image_size", type=int, default=640, 
        help="Max image dimension for resize"
    )
    parser.add_argument(
        "--max_frames", type=int, default=400, 
        help="Max frames to process"
    )
    parser.add_argument(
        "--inference_iters", type=int, default=4, 
        help="Model inference iterations"
    )
    parser.add_argument(
        "--Ts", type=int, default=-1, 
        help="Frame stride/selection count for 3D modes"
    )
    
    # --- 2D Vis Params ---
    parser.add_argument(
        "--rate", type=int, default=1, 
        help="[VIS] Sampling rate"
    )
    parser.add_argument(
        "--conf_thr", type=float, default=0.5, 
        help="[VIS] Confidence threshold"
    )
    parser.add_argument(
        "--bkg_opacity", type=float, default=0.5, 
        help="[VIS] Background opacity"
    )
    parser.add_argument(
        "--vstack", action='store_true', default=False, 
        help="[VIS] Vertically stack output"
    )
    parser.add_argument(
        "--hstack", action='store_true', default=True, 
        help="[VIS] Horizontally stack output"
    )

    args = parser.parse_args()

    # Create output directory
    if args.save_base_dir:
        os.makedirs(args.save_base_dir, exist_ok=True)
        # Copy input video for reference
        dst_path = os.path.join(args.save_base_dir, "input_copy.mp4")
        shutil.copy(args.mp4_path, dst_path)
        logger.info(f"Copied input video to {dst_path}")

    # Load Config
    logger.info(f"Loading config: {args.config_path}")
    try:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Config file not found at {args.config_path}")
        return
    
    # Setup PyTorch
    torch.set_grad_enabled(False)
    
    # Load Model
    model = load_model(args, config)
    
    # Run
    run_demo(model, args)

if __name__ == "__main__":
    main()