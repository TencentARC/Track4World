import sys
import os
import io
import json
import logging
import argparse
import os.path as osp
from glob import glob
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

# Add root to path
ROOT = Path(__file__).resolve().parents[2]  # Holi4D/
sys.path.insert(0, str(ROOT))

# Custom modules
import holi4d.utils.basic
from holi4d.nets.model import Holi4D
from holi4d.utils.alignment import align_points_scale_xyz_shift
from tapvid3d_metrics import compute_tapvid3d_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Data Processing Utilities
# ==============================================================================

def decode_jpeg_bytes(images_jpeg_bytes: List[bytes]) -> np.ndarray:
    """
    Decodes a list of JPEG byte strings into a numpy array of images.
    
    Args:
        images_jpeg_bytes: List of byte strings representing JPEG images.
        
    Returns:
        np.ndarray: Stacked images with shape (T, H, W, 3).
    """
    images = []
    for jpeg_bytes in images_jpeg_bytes:
        # Convert bytes to stream, open with PIL, and convert to RGB
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        images.append(np.array(img))
    return np.stack(images, axis=0)


def project_points_to_video_frame(
    camera_pov_points3d: np.ndarray, 
    camera_intrinsics: np.ndarray, 
    height: int, 
    width: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects 3D points from camera coordinate space to the 2D image plane.
    
    Args:
        camera_pov_points3d: (..., 3) 3D points in camera frame.
        camera_intrinsics: (4,) [fx, fy, cx, cy].
        height: Image height.
        width: Image width.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Projected 2D coordinates (..., 2).
            - Boolean mask indicating valid points inside the image frustum.
    """
    # Extract depth (Z) and normalize X, Y
    z_depth = camera_pov_points3d[..., 2] + 1e-8
    u_d = camera_pov_points3d[..., 0] / z_depth
    v_d = camera_pov_points3d[..., 1] / z_depth

    f_u, f_v, c_u, c_v = camera_intrinsics

    # Apply intrinsics
    u_d = u_d * f_u + c_u
    v_d = v_d * f_v + c_v

    # Create mask for points within image boundaries
    # Note: Z > 0 check is implicit usually, but can be added if needed.
    masks = (u_d >= 0) & (u_d <= width - 1) & (v_d >= 0) & (v_d <= height - 1)
    
    return np.stack([u_d, v_d], axis=-1), masks


def resize_all(
    images: np.ndarray, 
    tracks_cam: np.ndarray, 
    tracks_uv: Optional[np.ndarray], 
    vis: np.ndarray, 
    intrinsics: np.ndarray, 
    new_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    Resizes images and adjusts tracking data and intrinsics consistently.
    
    Args:
        images: (T, H, W, 3) Video frames.
        tracks_cam: (T, N, 3) 3D tracks in camera frame (usually unchanged by 2D resize).
        tracks_uv: (T, N, 2) 2D tracks in pixel coordinates.
        vis: (T, N) Visibility flags.
        intrinsics: Camera intrinsics (supports various shapes: 1D, 2D, 3D).
        new_size: Target (Height, Width).
    
    Returns:
        Resized tuple: (images, tracks_cam, tracks_uv, vis, intrinsics)
    """
    assert images.ndim == 4, f"images should be (T, H, W, 3), got {images.shape}"
    old_H, old_W = images.shape[1:3]
    new_H, new_W = new_size
    
    scale_x = new_W / old_W
    scale_y = new_H / old_H

    # 1. Resize images
    images_resized = np.stack([
        cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_AREA)
        for img in images
    ], axis=0)

    # 2. Resize 2D tracks (u, v)
    if tracks_uv is not None:
        tracks_uv_resized = tracks_uv.copy().astype(np.float32)
        tracks_uv_resized[..., 0] *= scale_x
        tracks_uv_resized[..., 1] *= scale_y
    else:
        tracks_uv_resized = None

    # 3. Resize intrinsics
    # Handles different shapes of intrinsic matrices/vectors
    if intrinsics is not None:
        intrinsics = np.array(intrinsics, dtype=np.float32)
        
        if intrinsics.ndim == 1 and intrinsics.shape[0] == 4:
            # Format: [fx, fy, cx, cy]
            fx, fy, cx, cy = intrinsics
            intrinsics_resized = np.array(
                [fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y], 
                dtype=np.float32
            )
        elif intrinsics.ndim == 2 and intrinsics.shape[1] == 4:
            # Format: (T, 4) -> Batch of vectors
            intrinsics_resized = intrinsics.copy().astype(np.float32)
            intrinsics_resized[:, 0] *= scale_x  # fx
            intrinsics_resized[:, 1] *= scale_y  # fy
            intrinsics_resized[:, 2] *= scale_x  # cx
            intrinsics_resized[:, 3] *= scale_y  # cy
        elif intrinsics.ndim == 2 and intrinsics.shape == (3, 3):
            # Format: (3, 3) -> Single Matrix
            K = intrinsics.copy()
            K[0, 0] *= scale_x
            K[1, 1] *= scale_y
            K[0, 2] *= scale_x
            K[1, 2] *= scale_y
            intrinsics_resized = K
        elif intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3):
            # Format: (T, 3, 3) -> Batch of Matrices
            intrinsics_resized = np.stack([
                np.array([
                    [K[0,0]*scale_x, K[0,1],         K[0,2]*scale_x],
                    [K[1,0],         K[1,1]*scale_y, K[1,2]*scale_y],
                    [K[2,0],         K[2,1],         K[2,2]]
                ], dtype=np.float32)
                for K in intrinsics
            ])
        else:
            raise ValueError(f"Unsupported intrinsics shape: {intrinsics.shape}")
    else:
        intrinsics_resized = None

    # 4. Visibility and 3D Camera tracks usually remain unchanged by 2D resizing
    vis_resized = vis
    tracks_cam_resized = tracks_cam

    return images_resized, tracks_cam_resized, tracks_uv_resized, vis_resized, intrinsics_resized


# ==============================================================================
# Dataset Definition
# ==============================================================================

class TrackingEvalDataset(data.Dataset):
    """
    Dataset class for loading tracking evaluation data (ADT, DS, PO, etc.).
    Loads .npz files containing images, tracks, and camera parameters.
    """
    def __init__(self, dataset_name: str = 'adt', root: str = './adt_mini', num_frames: int = 16):
        self.root = root
        self.dataset_name = dataset_name
        self.num_frames = num_frames
        # Collect all .npz files
        self.data_list = sorted(glob(osp.join(root, "*.npz")))
        logger.info(f"Found {len(self.data_list)} sequences in {root}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # Load data
        sample = np.load(self.data_list[index])
        images = decode_jpeg_bytes(sample['images_jpeg_bytes'])
        tracks_cam = sample['tracks_XYZ']
        vis = sample['visibility']
        intrinsics = sample['fx_fy_cx_cy']
        
        # Project 3D points to get 2D ground truth tracks
        h, w = images.shape[-3], images.shape[-2]
        tracks_uv, masks = project_points_to_video_frame(tracks_cam, intrinsics, h, w)
        
        # Update visibility with projection mask (points must be inside image)
        vis = masks & vis

        # Slice to desired number of frames
        num_frames = self.num_frames
        images = images[:num_frames]
        tracks_cam = tracks_cam[:num_frames]
        tracks_uv = tracks_uv[:num_frames]
        vis = vis[:num_frames]

        # Filter points: Keep points that appear at least twice and are visible in the first frame
        # Note: This logic assumes we only track points visible at start (t=0) or handles them later?
        # The logic `vis[0]` implies we filter based on visibility in the *first frame of the slice*.
        count = vis[:num_frames].sum(axis=0)
        valid_points = (count >= 2) & (vis[0])
        
        tracks_cam = tracks_cam[:, valid_points]
        tracks_uv = tracks_uv[:, valid_points]
        vis = vis[:, valid_points]

        # Resize if not ADT dataset (ADT is assumed to be the canonical resolution)
        if self.dataset_name != 'adt':
            new_size = (360, 640)
            images, tracks_cam, tracks_uv, vis, intrinsics = \
                resize_all(images, tracks_cam, tracks_uv, vis, intrinsics, new_size)
                
        return images, tracks_cam, tracks_uv, vis, intrinsics


# ==============================================================================
# Evaluation Logic
# ==============================================================================

@torch.inference_mode()
def test_track(model: torch.nn.Module, args: argparse.Namespace):
    """
    Main evaluation loop for tracking.
    """
    # 1. Setup DataLoader
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 4, 'drop_last': False}
    
    # Map dataset names to paths
    dataset_paths = {
        'adt': 'evaluation/track/adt_mini',
        'ds': 'evaluation/track/ds_mini',
        'po': 'evaluation/track/po_mini',
        'pstudio': 'evaluation/track/pstudio_mini'
    }
    dataset_dir = dataset_paths.get(args.dataset, 'evaluation/track/adt_mini')
    
    test_dataset = TrackingEvalDataset(
        dataset_name=args.dataset, 
        root=dataset_dir, 
        num_frames=args.num_frames
    )
    test_loader = DataLoader(test_dataset, **loader_args)

    # 2. Initialize Metrics
    count_all = 0
    metrics_all = {
        'occlusion_accuracy': 0.0, 
        'average_jaccard': 0.0, 
        'average_pts_within_thresh': 0.0, 
        'average_pts_within_thresh_with_occ': 0.0
    }

    logger.info(f"Starting evaluation on {args.dataset} with {len(test_loader)} batches...")

    # 3. Evaluation Loop
    for i_batch, test_data_blob in enumerate(tqdm(test_loader)):
        # Move data to GPU
        rgbs, trajs_g, tracks_uv, vis_g, intrinsics = [item.cuda() for item in test_data_blob]
        
        # Skip if no valid points
        if vis_g.shape[-1] == 0:
            continue

        # Prepare input format: (B, T, C, H, W)
        rgbs = rgbs.permute(0, 1, -1, 2, 3).float()
        B, T, C, H, W = rgbs.shape
        B, T, N, D = trajs_g.shape

        # Identify the first frame where each point is visible (query frame)
        __, first_positive_inds = torch.max(vis_g, dim=1)

        # Create coordinate grid
        grid_xy = holi4d.utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float()
        grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W) # (1, 1, 2, H, W)

        # Initialize containers for estimated trajectories
        trajs_e = torch.zeros([B, T, N, 2], device='cuda:0')
        trajs3d_e = torch.zeros([B, T, N, 3], device='cuda:0')
        visconfs_e = torch.zeros([B, T, N, 2], device='cuda:0')
        
        query_points_all = []
        eval_dict = None
        pre_first_positive_ind = 0

        # Mixed precision context
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            
            # Group points by their starting frame (Chunking strategy)
            # This allows processing points that appear at t=0, t=1, etc. separately
            unique_start_frames = torch.unique(first_positive_inds)
            
            for first_positive_ind in unique_start_frames:
                # Get indices of points starting at this specific frame
                chunk_pt_idxs = torch.nonzero(first_positive_inds[0] == first_positive_ind, as_tuple=False)[:, 0]
                
                # Get query points (B, K, 2)
                chunk_pts = tracks_uv[:, first_positive_ind[None].repeat(chunk_pt_idxs.shape[0]), chunk_pt_idxs]
                
                # Store query info: [Frame_Index, X, Y]
                query_points_all.append(torch.cat([first_positive_inds[:, chunk_pt_idxs, None], chunk_pts], dim=2))
                
                # Initialize maps for this chunk
                traj_maps_e = grid_xy.repeat(1, T, 1, 1, 1).clone()
                traj_maps3d_e = torch.zeros((B, T, H, W, 3)).cuda()
                visconf_maps_e = torch.zeros_like(traj_maps_e).clone()

                # Update internal state (eval_dict) if moving forward in time
                if eval_dict is not None:
                    offset = first_positive_ind - pre_first_positive_ind
                    # Shift features/maps to align with the new start time
                    keys_to_shift = ['fmaps', 'ctxfeats', 'fmaps3d_detail', 'pms', 'points', 'masks']
                    # Iterate through specific feature keys to maintain temporal alignment
                    for key in keys_to_shift:
                        if key in eval_dict:
                            # If the feature has a temporal dimension (ndim > 1), 
                            # slice starting from the offset
                            if eval_dict[key].ndim > 1:
                                eval_dict[key] = eval_dict[key][:, offset:]
                            else:
                                # For 1D tensors, slice directly along the first axis
                                eval_dict[key] = eval_dict[key][offset:]

                # Run Model Inference
                if first_positive_ind < T - 1:
                    output, eval_dict = model.evaluation(
                        rgbs[:, first_positive_ind:], 
                        iters=4, 
                        sw=None, 
                        is_training=False, 
                        tracking3d=True, 
                        force_projection=True, 
                        eval_dict=None 
                        # Note: In original code, eval_dict was passed as None here. 
                        # If state persistence is needed, check model implementation.
                    )
                    
                    # Extract outputs
                    pred_2d_motion = output[1]['flow_2d']
                    visconf = output[1]['visconf_maps_e']
                    pred_3dmotion = output[1]['flow_3d']

                    # Fill result maps
                    traj_maps_e[:, first_positive_ind:] = pred_2d_motion
                    traj_maps3d_e[:, first_positive_ind:] = pred_3dmotion
                    visconf_maps_e[:, first_positive_ind:] = visconf
                    
                    # Cleanup
                    del output, pred_2d_motion, pred_3dmotion, visconf
                    torch.cuda.empty_cache()
                
                pre_first_positive_ind = first_positive_ind

                # Sample the dense flow maps at the query point locations
                # Get integer coordinates of query points
                xyt = tracks_uv[:, first_positive_ind].round().long()[0, chunk_pt_idxs] # (K, 2)
                
                # Gather 2D Trajectories
                trajs_e_chunk = traj_maps_e[:, :, :, xyt[:, 1], xyt[:, 0]] # (B, T, 2, K)
                trajs_e_chunk = trajs_e_chunk.permute(0, 1, 3, 2) # (B, T, K, 2)
                # Scatter back to global tensor
                scatter_idx_2d = chunk_pt_idxs[None, None, :, None].repeat(1, trajs_e_chunk.shape[1], 1, 2)
                trajs_e.scatter_add_(2, scatter_idx_2d, trajs_e_chunk)

                # Gather 3D Trajectories
                trajs3d_e_chunk = traj_maps3d_e[:, :, xyt[:, 1], xyt[:, 0]]
                scatter_idx_3d = chunk_pt_idxs[None, None, :, None].repeat(1, trajs3d_e_chunk.shape[1], 1, 3)
                trajs3d_e.scatter_add_(2, scatter_idx_3d, trajs3d_e_chunk)

                # Gather Visibility Confidence
                visconfs_e_chunk = visconf_maps_e[:, :, :, xyt[:, 1], xyt[:, 0]]
                visconfs_e_chunk = visconfs_e_chunk.permute(0, 1, 3, 2)
                visconfs_e.scatter_add_(2, scatter_idx_2d, visconfs_e_chunk)

        # Post-process visibility confidence
        visconfs_e[..., 0] *= visconfs_e[..., 1] # Combine scores if needed
        assert (torch.all(visconfs_e >= 0) and torch.all(visconfs_e <= 1))

        # Format query points for metrics: (B, N, 3) -> [t, y, x]
        query_points_all = torch.cat(query_points_all, dim=1)[..., [0, 2, 1]] 

        # Prepare Ground Truth and Predictions for Metric Calculation
        gt_occluded = (vis_g < 0.5).bool().transpose(1, 2) # (B, N, T)
        gt_tracks3d = trajs_g
        pred_tracks3d = trajs3d_e

        # 3D Alignment (Scale & Shift)
        # Since monocular depth is up to scale, we align predictions to GT
        # using visible points.
        valid_mask = gt_occluded.transpose(1, 2) # (B, T, N)
        
        # Select points for alignment (subsampling for speed)
        if gt_tracks3d[valid_mask][::4].shape[0] != 0:
            align_src = pred_tracks3d[valid_mask][::2]
            align_tgt = gt_tracks3d[valid_mask][::2]
            align_weights = 1 / torch.ones_like(align_tgt.norm(dim=-1))
        else:
            # Fallback if few valid points
            align_src = pred_tracks3d.reshape(-1, 3)[::4]
            align_tgt = gt_tracks3d.reshape(-1, 3)[::4]
            align_weights = 1 / torch.ones_like(align_tgt.norm(dim=-1))
            
        scale, shift = align_points_scale_xyz_shift(
            align_src, align_tgt, align_weights, exp=20
        )
        
        pred_tracks3d_aligned = pred_tracks3d * scale + shift

        # Compute Metrics
        out_metrics, scaled_pred_traj_3d = compute_tapvid3d_metrics(
            gt_occluded=gt_occluded.transpose(1, 2).cpu().numpy(),
            gt_tracks=gt_tracks3d.cpu().numpy(),
            pred_occluded=gt_occluded.transpose(1, 2).cpu().numpy(),
            pred_tracks=pred_tracks3d_aligned.cpu().numpy(),
            intrinsics_params=intrinsics[0].cpu().numpy(),
            scaling="median",
            query_points=query_points_all.cpu().numpy(),
            order="b t n",
            use_fixed_metric_threshold=True,
            return_scaled_pred=True,
        )

        # Accumulate Metrics
        count_all += 1
        for key in metrics_all:
            metrics_all[key] += out_metrics[key] 
        for key in metrics_all:
            print(f"{key}: {metrics_all[key] / count_all}")
    # 4. Print Final Results
    logger.info("Evaluation Complete. Results:")
    for key in metrics_all:
        print(f"{key}: {metrics_all[key] / count_all}")


# ==============================================================================
# Model Loading
# ==============================================================================

def load_model(args: argparse.Namespace, config: Dict) -> torch.nn.Module:
    """
    Initializes the Holi4D model and loads pretrained weights.
    """
    logger.info("Initializing Holi4D Model...")

    model = Holi4D(
        **config['model'],
        seqlen=16,
        use_3d=True,
    )

    if args.ckpt_init and os.path.exists(args.ckpt_init):
        logger.info(f'Loading weights from local file: {args.ckpt_init}...')
        state_dict = torch.load(args.ckpt_init, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    else:
        # Fallback to Hub download
        url = "https://huggingface.co/cyun9286/holi4d/resolve/main/holi4d.pth"
        logger.info(f'Local checkpoint not found. Downloading from {url}...')
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=False)
        model.load_state_dict(state_dict, strict=False)
    
    model.cuda()
    model.eval()
    
    # Freeze parameters
    for p in model.parameters():
        p.requires_grad = False
    
    logger.info('Model loaded and set to evaluation mode.')
    return model


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    # Disable gradient computation globally for evaluation
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Holi4D Evaluation / Tracking Script")

    parser.add_argument(
        "--ckpt_init",
        type=str,
        default="./checkpoints/holi4d.pth",
        help="Path to model checkpoint file"
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="./holi4d/config/eval/v1.json",
        help="Path to model configuration JSON file"
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        choices=[16, 50],
        help="Number of frames used for tracking"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="adt",
        choices=["adt", "ds", "po", "pstudio"],
        help="Dataset name for evaluation"
    )

    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config_path):
        logger.error(f"Config file not found: {args.config_path}")
        sys.exit(1)
        
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # Build model and run evaluation
    model = load_model(args, config)
    test_track(model, args)

