import os
import sys
import itertools
import json
import logging 
import warnings
import h5py
import re
import io
import argparse
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

# Third-party libraries
import cv2
import numpy as np
import torch
import pandas as pd  # Added for data summarization
from tqdm import tqdm
from PIL import Image
import imageio.v2 as imageio 

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]  # Track4World/
sys.path.insert(0, str(ROOT))

# Custom Project Imports
import utils3d
from track4world.nets.model import Track4World
from track4world.utils.geometry_torch import mask_aware_nearest_resize
from track4world.utils.alignment import align_points_scale_xyz_shift
from utils import *
from demo import load_model
# ==============================================================================
# Configuration & Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Helper Functions: Evaluation
# ==============================================================================

def point_evaluation(
    pred_points: Union[np.ndarray, torch.Tensor], 
    gt_points: Union[np.ndarray, torch.Tensor], 
    max_depth: float = 80.0, 
    use_gpu: bool = False, 
    mask_pre: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates predicted point clouds against ground truth.
    
    Metrics:
    1. Abs Rel: Mean absolute relative error.
    2. Threshold (d1): % of pixels where max(ratio, 1/ratio) < 1.25.
    3. Valid Pixels: Count of pixels used for evaluation.
    """
    # Convert to Tensor
    if isinstance(pred_points, np.ndarray): pred_points = torch.from_numpy(pred_points)
    if isinstance(gt_points, np.ndarray): gt_points = torch.from_numpy(gt_points)
    if isinstance(mask_pre, np.ndarray): mask_pre = torch.from_numpy(mask_pre)

    if use_gpu and torch.cuda.is_available():
        pred_points = pred_points.cuda()
        gt_points = gt_points.cuda()
        if mask_pre is not None: mask_pre = mask_pre.cuda()

    # 1. Create Validity Mask
    # Filter out invalid GT (depth close to 0) and distant points
    mask = (gt_points[..., -1] > 1e-5)
    if max_depth is not None:
        mask = mask & (gt_points[..., -1] < max_depth)
    
    if mask_pre is not None:
        mask = mask & mask_pre
    
    predicted_masked = pred_points[mask]
    ground_truth_masked = gt_points[mask]

    if predicted_masked.numel() == 0:
        warnings.warn("No valid pixels for evaluation.")
        return np.array([0.0, 0.0, 0.0]), np.zeros_like(gt_points.cpu().numpy())

    torch.cuda.empty_cache()

    # 2. Alignment (Scale & Shift)
    # We downsample (resize) the mask to find a coarse alignment first to save compute
    _, lr_mask, lr_index = mask_aware_nearest_resize(None, mask, (10, 10), return_index=True)
    
    pred_points_lr = pred_points[lr_index][lr_mask]
    gt_points_lr = gt_points[lr_index][lr_mask]
    
    # Calculate alignment parameters based on low-res points
    scale, shift = align_points_scale_xyz_shift(
        pred_points_lr, 
        gt_points_lr, 
        1 / (gt_points_lr.norm(dim=-1) + 1e-6), 
        exp=10
    )
    
    # Apply alignment to the full-resolution masked predictions
    predicted_aligned = predicted_masked * scale + shift
    torch.cuda.empty_cache()

    # 3. Calculate Metrics
    dist_gt = torch.norm(ground_truth_masked, dim=-1)
    dist_pred = torch.norm(predicted_aligned, dim=-1)
    dist_err = torch.norm(predicted_aligned - ground_truth_masked, dim=-1)

    # Metric: Absolute Relative Error
    abs_rel = (dist_err / (dist_gt + 1e-6)).mean().item()

    # Metric: Threshold Accuracy (delta < 1.25)
    threshold_1 = (dist_err < 0.25 * torch.minimum(dist_gt, dist_pred)).float().mean().item()
    
    num_valid_pixels = predicted_masked.shape[0]

    results = np.array([abs_rel, threshold_1, num_valid_pixels])
    
    # 4. Generate Error Map for Visualization
    error_map_full = torch.zeros_like(gt_points)
    # Store relative error in the map
    error_map_full[mask] = torch.abs(predicted_aligned - ground_truth_masked) / (ground_truth_masked + 1e-6)

    return results, error_map_full.cpu().numpy()

def load_scene_gt(
    input_path: Path, 
    args: argparse.Namespace, 
    selected_views: np.ndarray, 
    start_frame: int, 
    end_frame: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads Ground Truth Point Cloud/Depth and Mask.
    Returns:
        gt_points: (N, H, W, 3)
        gt_mask: (N, H, W) or None
    """
    input_name = input_path.stem
    gt_points = None
    gt_mask = None

    if args.gt_dataset_type == 'Kubric-3D':
        depth_path = input_path / 'depths'
        if depth_path.exists():
            depth_files = sorted([
                os.path.join(depth_path, f) for f in os.listdir(depth_path) 
                if f.endswith(('.png', '.jpg'))
            ])
            # Load depth range metadata
            meta_path = input_path / f"{input_name}_sparse_tracking.npy"
            depth_range = np.load(meta_path, allow_pickle=True).item()["depth_range"]
            
            # Load depths corresponding to selected views
            gt_depths = np.stack([
                load_depth_kubric(depth_files[i], depth_range)[0] for i in selected_views
            ], axis=0)
            
            # Intrinsics for Kubric (Hardcoded as per original script)
            gt_intrinsic = np.array([[560/512, 0, 0.5], [0, 560/512, 0.5], [0, 0, 1]])
            
            # Back-project to 3D points
            gt_points = utils3d.torch.depth_to_points(
                torch.tensor(gt_depths).double(), 
                intrinsics=torch.tensor(gt_intrinsic).double(), 
                use_ray=True
            ).numpy()

    elif args.gt_dataset_type in ['Sintel', 'Scannet', 'Monkaa', 'KITTI', 'GMUKitchens']:
        # Load from HDF5
        hdf5_files = sorted(input_path.glob("*.hdf5"))
        if hdf5_files:
            with h5py.File(hdf5_files[0], "r") as f:
                gt_points = np.array(f['point_map'], dtype=np.float32)[selected_views]
                gt_mask = np.array(f['valid_mask'], dtype=np.float32)[selected_views]

    else:
        # Fallback for numpy datasets
        input_data = np.load(input_path)
        if 'depth_map' in input_data:
            gt_depths = input_data['depth_map']
            fx_fy_cx_cy = input_data['fx_fy_cx_cy']
            
            # Construct intrinsic matrix
            gt_intrinsic = np.array([
                [fx_fy_cx_cy[0]/640, 0, fx_fy_cx_cy[2]/640],
                [0, fx_fy_cx_cy[1]/480, fx_fy_cx_cy[3]/480], 
                [0, 0, 1]
            ])
            
            gt_points = utils3d.torch.depth_to_points(
                torch.tensor(gt_depths).double(), 
                intrinsics=torch.tensor(gt_intrinsic).double(), 
                use_ray=False
            ).numpy()[selected_views]

    return gt_points, gt_mask

# ==============================================================================
# Main Execution
# ==============================================================================

def main_point():
    """Entry point for Point Cloud Evaluation"""
    args = get_shared_args()
    
    # Configuration for Point Task
    point_config = {
        'target_key': 'points',
        'gt_loader': load_scene_gt,           # Function from your original script
        'eval_fn': point_evaluation,          # Function from your original script
        'metric_header': 'Point Evaluation Summary'
    }
    
    run_evaluation_pipeline(args, point_config)

if __name__ == '__main__':
    main_point()

