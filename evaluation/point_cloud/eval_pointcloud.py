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
ROOT = Path(__file__).resolve().parents[2]  # Holi4D/
sys.path.insert(0, str(ROOT))

# Custom Project Imports
import utils3d
from holi4d.nets.model import Holi4D
from holi4d.utils.geometry_torch import mask_aware_nearest_resize
from holi4d.utils.alignment import align_points_scale_xyz_shift
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

def main():
    parser = argparse.ArgumentParser(description="Holi4D Inference and Evaluation Script")

    # --- Model & Environment Configuration ---
    parser.add_argument(
        "--ckpt_init", type=str, default="./checkpoints/holi4d.pth", 
        help="Path to pretrained model checkpoint"
    )
    parser.add_argument(
        "--config_path", type=str, default="./holi4d/config/eval/v1.json", 
        help="Path to model configuration JSON"
    )
    parser.add_argument(
        "--output", "-o", dest="output_path", type=str, default='./output_sintel/23', 
        help="Directory to save predictions and visualizations"
    )
    parser.add_argument(
        "--device", dest="device_name", type=str, default='cuda', 
        help="Computation device (e.g., 'cuda', 'cpu', 'mps')"
    )
    parser.add_argument(
        "--fp16", action='store_true', 
        help="Enable half-precision (float16) for faster inference"
    )
    
    # --- Data & Input Configuration ---
    parser.add_argument(
        "--resize-to", type=int, default=512, 
        help="Resolution to which the image's short edge is resized"
    )
    parser.add_argument(
        "--frames", type=str, default='0-150', 
        help="Target frame range in 'start-end' format"
    )
    parser.add_argument(
        "--gt-dataset-type", type=str, default='Sintel', 
        choices=['Tum', 'Sintel', 'Scannet', 'Monkaa', 'Kubric-3D', 'KITTI', 'GMUKitchens'], 
        help="Ground truth dataset format for evaluation"
    )
    
    # --- Inference & Geometric Hyperparameters ---
    parser.add_argument(
        "--fov_x", dest="fov_x_", type=float, default=None, 
        help="Horizontal Field of View in degrees. If None, the model recovers it."
    )
    parser.add_argument(
        "--resolution_level", type=int, default=0, 
        help="Detail level [0-9], where higher is more detailed but slower"
    )
    parser.add_argument(
        "--num_tokens", type=int, default=None, 
        help="Explicit Transformer token count (overrides resolution_level)"
    )
    parser.add_argument(
        "--max-depth", type=float, default=70.0, 
        help="Maximum depth threshold (meters) for evaluation metrics"
    )
    parser.add_argument(
        "--coordinate", type=str, default='camera_base', 
        choices=['camera_base', 'world_pi3', 'world_depthanythingv3'],
        help="'camera': camera centric, 'world': world centric"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=130, 
        help="Temporal chunk size for processing long videos to manage VRAM"
    )


    args = parser.parse_args()

    # 1. Setup
    torch.set_grad_enabled(False)

    if not os.path.exists(args.config_path):
        logger.error(f"Config file not found: {args.config_path}")
        sys.exit(1)
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # 2. Load Model
    model = load_model(args, config)
    device = torch.device(args.device_name)

    # 3. Prepare Data Paths
    logger.info(f"Output will be saved to: {args.output_path}")
    input_paths = get_scene_paths(args.gt_dataset_type)
    if not input_paths:
        logger.error("No scenes found. Exiting.")
        sys.exit(1)

    # 4. Process Scenes
    for input_path in tqdm(input_paths, desc="Processing scenes"):
        input_name = input_path.stem
        
        # Create unique output directory
        base_path = Path(args.output_path) / input_name
        save_path = base_path
        counter = 1
        while save_path.exists():
            save_path = Path(f"{str(base_path)}_{counter}")
            counter += 1
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Parse frame range
        try:
            start_frame, end_frame = map(int, args.frames.split('-'))
        except ValueError:
            logger.error(f"Invalid frame range format: {args.frames}. Use 'start-end'.")
            continue

        # --- Load Input ---
        try:
            video_tensor, selected_views = load_scene_input(input_path, args, device, start_frame, end_frame)
        except Exception as e:
            logger.error(f"Failed to load input for {input_name}: {e}")
            continue

        # --- Run Inference ---
        logger.info(f"Running inference on {input_name} ({video_tensor.shape[0]} frames)...")
        
        all_pred_points = []
        
        # Process in chunks to manage GPU memory
        for i in range(0, video_tensor.shape[0], args.chunk_size):
            chunk_video = video_tensor[i : i + args.chunk_size]
            
            output = model.infer_pure_point(
                chunk_video,
                fov_x=args.fov_x_,
                resolution_level=args.resolution_level,
                num_tokens=args.num_tokens,
                use_fp16=args.fp16,
                current_batch_size=1
            )
            
            # Move to CPU immediately to free GPU memory
            all_pred_points.append(output[0]['points'].cpu().numpy())
            torch.cuda.empty_cache()

        # Concatenate results: (N, H, W, 3)
        pred_points = np.concatenate(all_pred_points, axis=1).squeeze(0)

        # --- Evaluation ---
        logger.info("Loading ground truth for evaluation...")
        try:
            gt_points, gt_mask = load_scene_gt(input_path, args, selected_views, start_frame, end_frame)
        except Exception as e:
            logger.warning(f"Failed to load GT for {input_name}: {e}. Skipping evaluation.")
            gt_points = None

        if gt_points is not None:
            gt_h, gt_w = gt_points.shape[1], gt_points.shape[2]
            
            # Resize predictions to match GT resolution
            pred_points_resized = np.array([
                cv2.resize(pred, (gt_w, gt_h), cv2.INTER_NEAREST) for pred in pred_points
            ])
            
            logger.info("Calculating metrics...")
            
            # Determine mask based on dataset type
            mask_pre = (gt_mask == 1) if gt_mask is not None else None
            
            # Adjust max_depth based on dataset
            eval_max_depth = 30.0 if args.gt_dataset_type == 'Kubric-3D' else args.max_depth

            eval_results, _ = point_evaluation(
                pred_points_resized, 
                gt_points, 
                max_depth=eval_max_depth, 
                use_gpu=True, 
                mask_pre=mask_pre
            )
            
            # Save Results
            results_file = save_path / "evaluation_results.txt"
            with open(results_file, "w") as f:
                f.write("--- Point Evaluation Summary ---\n")
                f.write("-" * 35 + "\n")
                f.write(f"{'Metric':<20}{'Value':<12}\n")
                f.write("-" * 35 + "\n")
                f.write(f"{'Abs_Rel':<20}{eval_results[0]:<12.6f}\n")
                f.write(f"{'d1 < 1.25':<20}{eval_results[1]:<12.6f}\n")
                f.write(f"{'Valid Pixels':<20}{int(eval_results[2])}\n")
            
            logger.info(f"Results saved to {results_file}")
        else:
            logger.warning("Skipping evaluation due to missing Ground Truth.")

    # 5. Summarize All Results
    summarize_evaluation_results(args.output_path)

if __name__ == '__main__':
    main()

