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
# Dataset Definition
# ==============================================================================

class BlinkvisionflowDataset(torch.utils.data.Dataset):
    """
    Dataset for loading BlinkVision flow data.
    Loads RGB images, depths, masks, and flow particles from .npz files.
    """
    def __init__(self, root: str):
        logger.info(f'Loading Blinkvisionflow dataset from {root}...')
        self.root = root
        self.data_list = sorted(glob(osp.join(root, "*.npz")))
        logger.info(f"Found {len(self.data_list)} sequences.")

    def getitem_helper(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Internal helper to load and preprocess a single sample.
        """
        data = np.load(self.data_list[index])
        
        # Extract data arrays
        depths = data["depths"]
        rgb_images = data["rgb_images"]
        depths_mask = data["depths_mask"]
        particle = data["particle"]
        intrinsics = data["intrinsics"]

        # Prepare forward motion and depth
        # particle shape: (..., 2+1+1) -> [x, y, z, valid_flag]
        forward_2dmotion = np.tile(particle[..., :2][None], (2, 1, 1, 1))   # (2, H, W, 2)
        forward_depth = np.tile(particle[..., 2][None], (2, 1, 1))          # (2, H, W)
        
        # Validate coordinates
        coords_x = forward_2dmotion[..., 0]
        coords_y = forward_2dmotion[..., 1]
        valid_x = (coords_x >= 0) & (coords_x < rgb_images.shape[-2])
        valid_y = (coords_y >= 0) & (coords_y < rgb_images.shape[-3])
        
        # Create validity masks
        # particle[..., -1] == 1 indicates the point is valid in the dataset
        valid_particle = particle[..., -1] == 1
        valid = (depths_mask) & valid_particle
        visibility = (depths_mask) & valid_particle & valid_x & valid_y 

        # Normalize intrinsics (if needed, seems to be normalizing by height/width)
        # Note: This modifies the intrinsics in place for this sample
        intrinsics[0][0] /= rgb_images.shape[-2] # fx / W ?
        intrinsics[0][2] /= rgb_images.shape[-2] # cx / W ?
        intrinsics[1][1] /= rgb_images.shape[-3] # fy / H ?
        intrinsics[1][2] /= rgb_images.shape[-3] # cy / H ?

        # Convert to Tensors
        rgb_images = torch.from_numpy(rgb_images).permute(0, 3, 1, 2) # (T, C, H, W)
        depths = torch.from_numpy(depths).float()
        depths_mask = torch.from_numpy(depths_mask)
        forward_2dmotion = torch.from_numpy(forward_2dmotion)
        forward_depth = torch.from_numpy(forward_depth)
        visibility = torch.from_numpy(visibility)
        valid = torch.from_numpy(valid)
        intrinsics = torch.from_numpy(intrinsics).float()

        # Heuristic Depth Scaling
        # Adjusts forward_depth units (e.g., m vs cm) to match mean depth scale
        mean_depth = depths[depths_mask].mean()
        valid_forward_depth_mean = forward_depth[visibility].mean()
        
        if valid_forward_depth_mean > 10 * mean_depth:
            forward_depth = forward_depth / 100.0
        elif valid_forward_depth_mean * 10 < mean_depth:
            forward_depth = forward_depth * 100.0

        sample = {
            'rgb_images': rgb_images,       # (2, 3, H, W)
            'depths': depths,               # (2, H, W)
            'depths_mask': depths_mask,     # (2, H, W)
            'forward_2dmotion': forward_2dmotion, # (2, H, W, 2)
            'forward_depth': forward_depth, # (2, H, W)
            'visibility': visibility,       # (2, H, W)
            'intrinsics': intrinsics,       # (2, 3, 3)
            'valid': valid                  # (2, H, W)
        }
        return sample

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], bool]:
        """
        Retrieves a sample. Retries with random indices if the current sample is invalid.
        """
        gotit = False
        fails = 0
        max_retries = 8
        samp = None

        while not gotit and fails < max_retries:
            samp = self.getitem_helper(index)
            # Check if there are any valid points in the sample
            if torch.sum(samp['valid']) > 0:
                gotit = True
            else:
                fails += 1
                # Pick a random index to retry
                index = np.random.randint(len(self.data_list))
        
        if fails > 4:
            logger.warning(f'Sampling failed {fails} times before finding valid data.')
            
        return samp, True

    def __len__(self):
        return len(self.data_list)


# ==============================================================================
# Metric Computation Functions
# ==============================================================================

def compute_pc_metrics(
    pred: torch.Tensor, 
    gt: torch.Tensor, 
    valid_mask: torch.Tensor
) -> Dict[str, float]:
    """
    Computes Point Cloud metrics (Absolute Relative Error, Threshold Accuracy).
    
    Args:
        pred: (N, 3) Predicted points.
        gt: (N, 3) Ground truth points.
        valid_mask: (N,) Boolean mask of valid points.
    """
    # Filter valid points
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    # Calculate distances
    dist_gt = torch.norm(gt_valid, dim=-1)
    dist_err = torch.norm(pred_valid - gt_valid, dim=-1)
    dist_pred = torch.norm(pred_valid, dim=-1)

    # Absolute Relative Error
    abs_rel = (dist_err / (dist_gt + 1e-6)).sum().item()

    # Threshold Accuracy (Error < 25% of min(GT, Pred))
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
    Computes 3D Scene Flow metrics (EPE, Accuracy, Outliers).
    """
    # Filter valid flow
    diff = (pred_flow3d - gt_flow3d)[valid_mask]
    epe3d = torch.norm(diff, dim=-1)
    gt_norm = torch.norm(gt_flow3d[valid_mask], dim=-1)
    
    relative_err = epe3d / (gt_norm + 1e-6)

    metrics = {
        'EPE3D': epe3d.sum().item(),
        'Acc3D_strict': ((epe3d < 0.05) | (relative_err < 0.05)).float().sum().item(),
        'Acc3D_relax': ((epe3d < 0.1) | (relative_err < 0.1)).float().sum().item(),
        'Outlier': ((epe3d > 0.3) | (relative_err > 0.1)).float().sum().item(),
    }
    return metrics


def compute_optical_flow_metrics(
    pred_flow: torch.Tensor, 
    gt_flow: torch.Tensor, 
    valid_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Computes 2D Optical Flow metrics (EPE, Accuracy, Outliers).
    """
    # Ensure dimensions are (B, H, W, 2)
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

    # Calculate End-Point Error (EPE)
    epe = torch.norm(pred_flow - gt_flow, dim=-1)  # [B, H, W]

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


# ==============================================================================
# Evaluation Loop
# ==============================================================================

@torch.inference_mode()
def test_flow(model: torch.nn.Module, args: argparse.Namespace):
    """
    Main evaluation loop for Scene Flow and Optical Flow.
    """
    # 1. Setup Data Loader
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 4, 'drop_last': False}
    test_dataset = BlinkvisionflowDataset(root='evaluation/flow/blinkvision')
    test_loader = DataLoader(test_dataset, **loader_args)

    # 2. Initialize Metrics Accumulator
    count_all = 0
    metrics_all = {
        'EPE3D': 0.0, 'Acc3D_strict': 0.0, 'Acc3D_relax': 0.0, 'Outlier': 0.0, 
        "EPE2D": 0.0, "ACC1_2D": 0.0, "ACC3_2D": 0.0, "Outlier_2D": 0.0,
        "abs_rel": 0.0, "threshold_1": 0.0
    }

    logger.info(f"Starting evaluation with {len(test_loader)} batches...")

    # 3. Iterate over dataset
    for i_batch, test_data_blob in enumerate(tqdm(test_loader)):
        # Unpack and move to GPU
        # Note: test_data_blob is a tuple (dict, bool) due to dataset implementation
        batch_data = test_data_blob[0]
        
        rgb_images = batch_data['rgb_images'].float().cuda()        # (B, 2, 3, H, W)
        depths = batch_data['depths'].float().cuda()                # (B, 2, H, W)
        forward_depth = batch_data['forward_depth'].float().cuda()  # (B, 2, H, W)
        forward_2dmotion = batch_data['forward_2dmotion'].float().cuda() # (B, 2, H, W, 2)
        visibility = batch_data['visibility'].float().cuda()        # (B, 2, H, W)
        intrinsics = batch_data['intrinsics'].float().cuda()        # (B, 2, 3, 3)
        flow_valid = batch_data['valid'].cuda()                     # (B, 2, H, W)

        # Prepare masks for the first frame (t=0)
        flow_valid = flow_valid[:, 0]
        visibility = visibility[:, 0]
        flow_valid = flow_valid & (visibility == 1)
        
        H, W = rgb_images.shape[-2], rgb_images.shape[-1]
        
        # 4. Model Inference
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            output = model.infer_pair(
                rgb_images, 
                iters=4, 
                sw=None, 
                is_training=False, 
                tracking3d=True, 
                force_projection=True, 
                apply_mask=False
            )
        
        # 5. Process Predictions
        # Normalize GT 2D motion to pixel coordinates (if it was normalized)
        # Note: The dataset normalized intrinsics, so we might need to check if motion needs scaling.
        # Based on original code:
        forward_2dmotion[..., 0] /= W
        forward_2dmotion[..., 1] /= H
        
        # Extract predictions
        pred_2d_motion = output[1]['flow_2d'][:, 0].permute(0, 2, 3, 1) # (B, H, W, 2)
        pred_points = output[0]['points'][:, 0]                         # (B, 3, H, W) -> (B, H, W, 3) implicitly? Check shape.
        pred_flow3d = output[1]['flow_3d'][:, 0]
        pred_3dmotion = output[1]['flow_3d'][:, 0] - output[0]['points'][:, 0]

        # 6. Process Ground Truth
        # Unproject GT flow to 3D
        gt_3dflow = utils3d.torch.unproject_cv(
            forward_2dmotion, 
            forward_depth, 
            intrinsics=intrinsics[..., None, :, :], 
            use_ray=False
        )
        
        # Restore 2D motion to pixel units for metric calculation
        forward_2dmotion[..., 0] *= W
        forward_2dmotion[..., 1] *= H 
        forward_2dmotion = forward_2dmotion[:, 0]
        
        # Get GT 3D points from depth
        gt_points = utils3d.torch.depth_to_points(depths, intrinsics=intrinsics, use_ray=False)

        # 7. Prepare Data for Alignment
        # Concatenate points and flow for unified alignment
        gt = torch.cat([gt_points[:, 0], gt_3dflow[:, 0]], dim=0)
        pred = torch.cat([pred_points, pred_flow3d], dim=0)
        valid = torch.cat([flow_valid, flow_valid], dim=0)
        
        gt_3dmotion = gt_3dflow[:, 0] - gt_points[:, 0]

        # 8. Alignment (Scale & Shift)
        # Monocular methods often lack absolute scale, so we align predictions to GT.
        # We use a low-resolution mask-aware resize to speed up alignment calculation.
        _, lr_mask, lr_index = mask_aware_nearest_resize(None, valid, (32, 32), return_index=True)
        
        pred_points_lr_masked = pred[lr_index][lr_mask]
        gt_points_lr_masked = gt[lr_index][lr_mask]
        
        # Calculate alignment parameters
        scale, shift = align_points_scale_xyz_shift(
            pred_points_lr_masked, 
            gt_points_lr_masked, 
            1 / (gt_points_lr_masked.norm(dim=-1) + 1e-6), 
            exp=20
        )
        
        # Apply alignment
        pred_aligned = pred * scale + shift
        pred_3dmotion_aligned = pred_3dmotion * scale

        # 9. Compute Metrics
        count_all += flow_valid.sum()
        
        metrics_sf = compute_scene_flow_metrics(pred_3dmotion_aligned, gt_3dmotion, flow_valid)
        metrics_of = compute_optical_flow_metrics(pred_2d_motion, forward_2dmotion, flow_valid)
        metrics_pc = compute_pc_metrics(pred_aligned, gt, valid)

        # Accumulate (handling NaNs)
        for key, val in metrics_of.items():    
            metrics_all[key] += 0 if math.isnan(val) else val
            
        for key, val in metrics_sf.items():
            metrics_all[key] += 0 if math.isnan(val) else val
            
        for key, val in metrics_pc.items():
            metrics_all[key] += 0 if math.isnan(val) else val

    # 10. Print Final Results
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

    parser = argparse.ArgumentParser(description="Holi4D Flow Evaluation Script")

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
