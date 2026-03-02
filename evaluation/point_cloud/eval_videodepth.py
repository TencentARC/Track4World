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
from track4world.utils.alignment import align_depth_affine
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


def depth_evaluation(
    predicted_depth: Union[np.ndarray, torch.Tensor],
    ground_truth_depth: Union[np.ndarray, torch.Tensor],
    max_depth: float = 80.0,
    use_gpu: bool = False,
    mask_pre: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates predicted depth maps against ground truth.

    Metrics:
    1. Abs Rel: Mean absolute relative error.
    2. Threshold (d1): % of pixels where max(pred/gt, gt/pred) < 1.25.
    3. Valid Pixels: Count of pixels used for evaluation.
    """
    # Convert to Tensor
    if isinstance(predicted_depth, np.ndarray): predicted_depth = torch.from_numpy(predicted_depth)
    if isinstance(ground_truth_depth, np.ndarray): ground_truth_depth = torch.from_numpy(ground_truth_depth)
    if isinstance(mask_pre, np.ndarray): mask_pre = torch.from_numpy(mask_pre)

    if use_gpu and torch.cuda.is_available():
        predicted_depth = predicted_depth.cuda()
        ground_truth_depth = ground_truth_depth.cuda()
        if mask_pre is not None: mask_pre = mask_pre.cuda()

    # 1. Create Validity Mask
    # Filter out invalid GT (depth close to 0) and distant points
    mask = (ground_truth_depth > 1e-5)
    if max_depth is not None:
        mask = mask & (ground_truth_depth < max_depth)

    if mask_pre is not None:
        mask = mask & mask_pre

    predicted_masked = predicted_depth[mask]
    ground_truth_masked = ground_truth_depth[mask]

    if predicted_masked.numel() == 0:
        warnings.warn("No valid pixels for evaluation.")
        return np.array([0.0, 0.0, 0.0]), np.zeros_like(ground_truth_depth.cpu().numpy())

    torch.cuda.empty_cache()

    # 2. Alignment (Scale & Shift)
    # We downsample (resize) the mask to find a coarse alignment first to save compute
    _, lr_mask, lr_index = mask_aware_nearest_resize(None, mask, (16, 16), return_index=True)
    
    pred_depth_lr = predicted_depth[lr_index][lr_mask]
    gt_depth_lr = ground_truth_depth[lr_index][lr_mask]

    # Calculate affine alignment parameters based on low-res depths
    scale, shift = align_depth_affine(
        pred_depth_lr, 
        gt_depth_lr, 
        1 / (gt_depth_lr + 1e-6)
    )

    # Apply alignment to the full-resolution masked predictions
    predicted_aligned = predicted_masked * scale + shift
    
    # Clip aligned predictions to valid range
    predicted_aligned = torch.clamp(predicted_aligned, min=1e-8, max=max_depth)
    torch.cuda.empty_cache()

    # 3. Calculate Metrics
    # Metric: Absolute Relative Error
    abs_rel = torch.mean(torch.abs(predicted_aligned - ground_truth_masked) / ground_truth_masked).item()

    # Metric: Threshold Accuracy (delta < 1.25)
    max_ratio = torch.maximum(predicted_aligned / ground_truth_masked, ground_truth_masked / predicted_aligned)
    threshold_1 = torch.mean((max_ratio < 1.25).float()).item()
    
    num_valid_pixels = predicted_masked.numel()

    results = np.array([abs_rel, threshold_1, num_valid_pixels])

    # 4. Generate Error Map for Visualization
    error_map_full = torch.zeros_like(ground_truth_depth)
    error_map_full[mask] = torch.abs(predicted_aligned - ground_truth_masked) / ground_truth_masked

    return results, error_map_full.cpu().numpy()

def load_scene_ground_truth(
    data_dir: Path, 
    args: argparse.Namespace, 
    view_indices: np.ndarray, 
    start_frame: int, 
    end_frame: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads Ground Truth (GT) Depth maps and Validity Masks from various dataset formats.
    
    Args:
        data_dir: Path to the sequence directory.
        args: Namespace containing 'gt_dataset_type'.
        view_indices: Array of indices specifying which frames to load.
        
    Returns:
        gt_depths: (N, H, W) depth values in metric units.
        gt_masks: (N, H, W) binary validity masks.
    """
    sequence_name = data_dir.stem
    gt_depths = None
    gt_masks = None

    # --- Case 1: Kubric-3D Dataset ---
    if args.gt_dataset_type == 'Kubric-3D':
        depth_folder = data_dir / 'depths'
        if depth_folder.exists():
            # Gather all depth image files
            all_depth_files = sorted([
                os.path.join(depth_folder, f) for f in os.listdir(depth_folder) 
                if f.endswith(('.png', '.jpg'))
            ])
            # Metadata contains the normalization range for 16-bit depth
            metadata_file = data_dir / f"{sequence_name}_sparse_tracking.npy"
            depth_metadata = np.load(metadata_file, allow_pickle=True).item()
            range_val = depth_metadata["depth_range"]
            
            # Load and un-normalize depth for the specific views
            gt_depths = np.stack([
                load_depth_kubric(all_depth_files[i], range_val)[0] 
                for i in view_indices
            ], axis=0)

    # --- Case 2: HDF5 Standardized Datasets (Sintel, Scannet, etc.) ---
    elif args.gt_dataset_type in ['Sintel', 'Scannet', 'Monkaa', 'KITTI', 'GMUKitchens']:
        hdf5_files = sorted(data_dir.glob("*.hdf5"))
        if hdf5_files:
            with h5py.File(hdf5_files[0], "r") as h5_file:
                # 'point_map' is (T, H, W, 3); we extract the Z-channel for depth
                point_map = np.array(h5_file['point_map'], dtype=np.float32)
                gt_masks = np.array(h5_file['valid_mask'], dtype=np.float32)
                
                # Filter by selected views and take the Z component
                gt_depths = point_map[view_indices, ..., -1]
                if gt_masks is not None:
                    gt_masks = gt_masks[view_indices]

    # --- Case 3: Bonn RGB-D Dataset ---
    elif args.gt_dataset_type == 'Bonn':
        depth_folder = data_dir / 'depth_110'
        if depth_folder.exists():
            all_depth_files = sorted([
                os.path.join(depth_folder, f) for f in os.listdir(depth_folder) 
                if f.endswith(('.png', '.jpg'))
            ])
            # Bonn depth is stored in mm; divide by 5000 to get meters (standard)
            gt_depths = np.stack([
                np.asarray(Image.open(all_depth_files[i])).astype(np.float64) / 5000.0 
                for i in view_indices
            ], axis=0)

    # --- Case 4: Default Numpy Archive Loader (TUM/PO) ---
    else:
        # Assumes data is packed in a single .npz or .npy file
        archive_data = np.load(data_dir)
        if 'depth_map' in archive_data:
            gt_depths = archive_data['depth_map'][view_indices]

    return gt_depths, gt_masks

# ==============================================================================
# Main Execution
# ==============================================================================

def main_depth():
    """Entry point for Depth Evaluation"""
    args = get_shared_args()
    
    # Configuration for Depth Task
    depth_config = {
        'target_key': 'depth',
        'gt_loader': load_scene_ground_truth, # Function from your original script
        'eval_fn': depth_evaluation,          # Function from your original script
        'metric_header': 'Depth Evaluation Summary'
    }
    
    run_evaluation_pipeline(args, depth_config)

if __name__ == '__main__':
    main_depth()

