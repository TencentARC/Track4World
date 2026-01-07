import sys
import os
import json
import math
import logging
import argparse
import os.path as osp
from glob import glob
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable

import torch
from torch.utils.data import DataLoader

# ==============================================================================
# Path Setup & Custom Imports
# ==============================================================================

# Add root to path
ROOT = Path(__file__).resolve().parents[2]  # Holi4D/
sys.path.insert(0, str(ROOT))

# Custom Project Imports
import utils3d
from holi4d.nets.model import Holi4D
import holi4d.utils.basic
from holi4d.utils.geometry_torch import mask_aware_nearest_resize
from holi4d.utils.alignment import align_points_scale_xyz_shift

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
# Geometry Helper Functions
# ==============================================================================

def backproject(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Back-projects depth map to 3D points in the camera coordinate system.
    
    Args:
        depth: (H, W) Depth map.
        K: (3, 3) Intrinsic matrix.
        
    Returns:
        points_3d: (H, W, 3) 3D coordinates.
    """
    h, w = depth.shape
    device = depth.device
    
    # Create meshgrid
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    ones = torch.ones_like(x)
    
    # Stack to homogeneous coordinates (3, N)
    pixels = torch.stack((x, y, ones), dim=-1).float().reshape(-1, 3).T
    
    # Back-projection: P = K_inv * pixel * depth
    depth_flat = depth.reshape(-1)
    K_inv = torch.inverse(K)
    points_3d = (K_inv @ pixels) * depth_flat
    
    return points_3d.T.reshape(h, w, 3)


def backproject_w_flow(depth: torch.Tensor, K: torch.Tensor, flow_2d: torch.Tensor) -> torch.Tensor:
    """
    Back-projects depth map to 3D points using 2D flow to shift pixels.
    Used to compute the 3D position of points in the next frame (P2).
    
    Args:
        depth: (H, W) Depth map at t+1.
        K: (3, 3) Intrinsic matrix.
        flow_2d: (H, W, 2) Optical flow (u_shift, v_shift).
        
    Returns:
        points_3d: (H, W, 3) 3D coordinates.
    """
    h, w = depth.shape
    device = depth.device
    
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    
    # Apply flow to grid coordinates
    grid_x2 = (x.float() + flow_2d[..., 0])
    grid_y2 = (y.float() + flow_2d[..., 1])
    ones = torch.ones_like(x)
    
    pixels = torch.stack((grid_x2, grid_y2, ones), dim=-1).float().reshape(-1, 3).T
    
    depth_flat = depth.reshape(-1)
    K_inv = torch.inverse(K)
    points_3d = (K_inv @ pixels) * depth_flat
    
    return points_3d.T.reshape(h, w, 3)


# ==============================================================================
# Dataset Definition
# ==============================================================================

class KITTIDataset(torch.utils.data.Dataset):
    """
    Dataset loader for KITTI Scene Flow data stored in .npz format.
    """
    def __init__(self, root: str):
        logger.info(f'Loading KITTI dataset from {root}...')
        self.root = root
        self.data_list = sorted(glob(osp.join(root, "*.npz")))
        logger.info(f"Found {len(self.data_list)} sequences.") 

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int):
        # Load compressed data
        data = np.load(self.data_list[index])
        
        # Extract arrays
        image1 = data["image1"]
        image2 = data["image2"]
        disp1 = data["disp1"]
        disp2 = data["disp2"]
        flow = data["flow"]
        valid = data["valid"]
        K = data["K"]
        extrinsics = data["extrinsics"]
        
        # Extract focal length for depth calculation
        fx = K[0, 0]

        # Convert to Tensors
        # Images: (H, W, 3) -> (3, H, W)
        image1 = torch.from_numpy(image1).float().permute(2, 0, 1)
        image2 = torch.from_numpy(image2).float().permute(2, 0, 1)
        
        disp1 = torch.from_numpy(disp1).float()
        disp2 = torch.from_numpy(disp2).float()
        flow = torch.from_numpy(flow).float()
        valid = torch.from_numpy(valid).float()
        K = torch.from_numpy(K).float()
        extrinsics = torch.from_numpy(extrinsics).float()
        
        # Refine validity mask: point must be valid in GT and have valid depth in frame 2
        valid = valid * (disp2 > 0).float()

        # Compute Depth from Disparity
        # Depth = baseline * focal_length / disparity
        baseline = 0.54 # Standard KITTI baseline is approx 0.54m, code used 0.1? 
        # Note: If the .npz data was pre-scaled or normalized, 0.1 might be correct. 
        # Keeping original value 0.1 to match user logic.
        baseline_used = 0.1 
        depth1 = baseline_used * fx / (disp1 + 1e-6)
        depth2 = baseline_used * fx / (disp2 + 1e-6)

        # Compute 3D Points (Scene Flow Ground Truth)
        # P1: Points in frame 1
        # P2: Points in frame 2 (projected using flow and depth2)
        P1 = backproject(depth1, K)
        P2 = backproject_w_flow(depth2, K, flow)

        return image1, image2, depth1, depth2, flow, valid, K, extrinsics, P1, P2


# ==============================================================================
# Metric Computation Functions
# ==============================================================================

def compute_optical_flow_metrics(
    pred_flow: torch.Tensor, 
    gt_flow: torch.Tensor, 
    valid_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Computes 2D Optical Flow metrics (EPE, Accuracy, Outliers).
    """
    if pred_flow.dim() == 3:
        pred_flow = pred_flow.unsqueeze(0)
        gt_flow = gt_flow.unsqueeze(0)
        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(0)

    B, H, W, _ = pred_flow.shape

    if valid_mask is None:
        valid_mask = torch.ones((B, H, W), dtype=torch.bool, device=pred_flow.device)
    else:
        valid_mask = valid_mask.squeeze(1).bool()

    # End-Point Error
    epe = torch.norm(pred_flow - gt_flow, dim=-1)

    epe_valid = epe[valid_mask]
    gt_valid = gt_flow[valid_mask]
    gt_norm = torch.norm(gt_valid, dim=-1)

    relative_err = epe_valid / (gt_norm + 1e-8)

    return {
        "EPE2D": epe_valid.sum().item(),
        "ACC1_2D": (epe_valid < 1.0).float().sum().item(),
        "ACC3_2D": (epe_valid < 3.0).float().sum().item(),
        "Outlier_2D": ((epe_valid > 3.0) & (relative_err > 0.05)).float().sum().item(),
    }


def compute_pc_metrics(
    pred: torch.Tensor, 
    gt: torch.Tensor, 
    valid_mask: torch.Tensor
) -> Dict[str, float]:
    """
    Computes Point Cloud reconstruction metrics.
    """
    dist_gt = torch.norm(gt[valid_mask], dim=-1)
    dist_err = torch.norm(pred[valid_mask] - gt[valid_mask], dim=-1)
    
    # Absolute Relative Error
    abs_rel = (dist_err / (dist_gt + 1e-6)).sum().item()

    dist_pred = torch.norm(pred[valid_mask], dim=-1)
    
    # Threshold Accuracy
    threshold_1 = (dist_err < 0.25 * torch.minimum(dist_gt, dist_pred)).float().sum().item()
    
    return {
        'abs_rel': abs_rel,
        'threshold_1': threshold_1
    }


def compute_scene_flow_metrics(
    pred_flow3d: torch.Tensor, 
    gt_flow3d: torch.Tensor, 
    valid_mask: torch.Tensor
) -> Dict[str, float]:
    """
    Computes 3D Scene Flow metrics.
    """
    diff = (pred_flow3d - gt_flow3d)[valid_mask]
    epe3d = torch.norm(diff, dim=-1)
    gt_norm = torch.norm(gt_flow3d[valid_mask], dim=-1)
    relative_err = epe3d / (gt_norm + 1e-6)

    return {
        'EPE3D': epe3d.sum().item(),
        'Acc3D_strict': ((epe3d < 0.05) | (relative_err < 0.05)).float().sum().item(),
        'Acc3D_relax': ((epe3d < 0.1) | (relative_err < 0.1)).float().sum().item(),
        'Outlier': ((epe3d > 0.3) | (relative_err > 0.1)).float().sum().item(),
    }


# ==============================================================================
# Evaluation Loop
# ==============================================================================

@torch.inference_mode()
def test_flow(model: torch.nn.Module, args: argparse.Namespace):
    """
    Main evaluation loop for KITTI Scene Flow.
    """
    # 1. Setup DataLoader
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 4, 'drop_last': False}
    test_dataset = KITTIDataset(root='evaluation/flow/kitti')
    test_loader = DataLoader(test_dataset, **loader_args)

    # 2. Initialize Metrics
    count_all = 0
    metrics_all = {
        'EPE3D': 0.0, 'Acc3D_strict': 0.0, 'Acc3D_relax': 0.0, 'Outlier': 0.0, 
        "EPE2D": 0.0, "ACC1_2D": 0.0, "ACC3_2D": 0.0, "Outlier_2D": 0.0,
        "abs_rel": 0.0, "threshold_1": 0.0
    }

    logger.info(f"Starting evaluation with {len(test_loader)} batches...")

    # 3. Iterate over dataset
    for i_batch, test_data_blob in enumerate(tqdm(test_loader)):
        
        # Unpack data
        image1, image2, depth1, depth2, flow_gt, valid, intrinsics, extrinsics, P1, P2 = \
            [data_item.cuda() for data_item in test_data_blob]

        valid_mask = valid.unsqueeze(-1) > 0.5
        
        # Prepare Ground Truth Flow (Absolute Coordinates)
        # flow_gt is relative (u_shift, v_shift). We convert it to absolute (u_new, v_new)
        # because the model output might be in absolute coordinates.
        H, W = flow_gt.shape[1], flow_gt.shape[2]
        grid_xy = holi4d.utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float()
        grid_xy = grid_xy.reshape(1, H, W, 2) 
        
        flow2d_absolute = flow_gt + grid_xy 
        gt_3dmotion = P2 - P1

        # 4. Model Inference
        rgb_images = torch.cat([image1, image2], dim=0) # Stack frames
        
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            output = model.infer_pair(
                rgb_images[None], 
                iters=4, 
                sw=None, 
                is_training=False, 
                tracking3d=True, 
                force_projection=True, 
                apply_mask=False
            )
        
        # 5. Extract Predictions
        # 2D Motion (Absolute coordinates)
        pred_2d_motion = output[1]['flow_2d'][:, 0].permute(0, 2, 3, 1)
        
        # 3D Motion & Points
        pred_points = output[0]['points'][:, 0]
        pred_flow3d = output[1]['flow_3d'][:, 0]
        pred_3dmotion = pred_flow3d - pred_points

        # 6. Alignment (Scale & Shift)
        # Since monocular depth is scale-ambiguous, we align predictions to GT.
        # We concatenate points and flow to align them together.
        gt_concat = torch.cat([P1, P2], dim=0)
        pred_concat = torch.cat([pred_points, pred_flow3d], dim=0)
        valid_concat = torch.cat([valid_mask[..., 0], valid_mask[..., 0]], dim=0)

        # Use a low-resolution mask-aware resize for efficient alignment calculation
        _, lr_mask, lr_index = mask_aware_nearest_resize(None, valid_concat, (32, 32), return_index=True)
        
        pred_points_lr = pred_concat[lr_index][lr_mask]
        gt_points_lr = gt_concat[lr_index][lr_mask]
        
        # Calculate scale and shift
        scale, shift = align_points_scale_xyz_shift(
            pred_points_lr, 
            gt_points_lr, 
            1 / (gt_points_lr.norm(dim=-1) + 1e-6), 
            exp=10
        )
        
        # Apply alignment
        pred_aligned = pred_concat * scale + shift
        pred_3dmotion_aligned = pred_3dmotion * scale # Motion only needs scaling

        # 7. Compute Metrics
        count_all += valid_mask.sum()
        
        metrics_sf = compute_scene_flow_metrics(pred_3dmotion_aligned, gt_3dmotion, valid_mask[..., 0])
        metrics_of = compute_optical_flow_metrics(pred_2d_motion, flow2d_absolute, valid_mask[..., 0])
        metrics_pc = compute_pc_metrics(pred_aligned, gt_concat, valid_concat)

        # Accumulate results
        for key, val in metrics_of.items():    
            metrics_all[key] += 0 if math.isnan(val) else val
        for key, val in metrics_sf.items():
            metrics_all[key] += 0 if math.isnan(val) else val
        for key, val in metrics_pc.items():
            metrics_all[key] += 0 if math.isnan(val) else val

    # 8. Print Final Results
    logger.info("Evaluation Complete. Results:")
    table = PrettyTable(['Metric', 'Value'])
    
    for key in metrics_all:
        if key in ["abs_rel", "threshold_1"]:
            # These metrics were calculated on concatenated (points + flow), so divide by 2*count
            val = metrics_all[key] / (2 * count_all)
        else:
            val = metrics_all[key] / count_all
        table.add_row([key, f"{val:.4f}"])
    
    print(table)


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
        base='base'
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

    parser = argparse.ArgumentParser(description="Holi4D KITTI Flow Evaluation")

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

    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config_path):
        logger.error(f"Config file not found: {args.config_path}")
        sys.exit(1)
        
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # Build model and run evaluation
    model = load_model(args, config)
    test_flow(model, args)

