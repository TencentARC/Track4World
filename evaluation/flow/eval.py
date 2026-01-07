import sys
import os
import json
import math
import logging
import argparse
import os.path as osp
from glob import glob
from pathlib import Path

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
import holi4d.utils.basic
from holi4d.utils.geometry_torch import mask_aware_nearest_resize
from holi4d.utils.alignment import align_points_scale_xyz_shift
from demo import load_model
from frame_utils import *
import utils3d
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

class Kubric3DflowDataset(torch.utils.data.Dataset):
    """
    Dataset loader for Kubric 3D Flow data stored in .npz format.
    """
    def __init__(self, root: str):
        logger.info(f'Loading Kubric dataset from {root}...')
        self.root = root
        self.data_list = sorted(glob(osp.join(root, "*.npz")))
        logger.info(f"Found {len(self.data_list)} sequences.") 

    def getitem_helper(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Internal helper to load and preprocess a single sample.
        """
        data = np.load(self.data_list[index])
        
        # Extract arrays
        rgb_images = data["rgb_images"]
        depths = data["depths"]
        depths_mask = data["depths_mask"]
        forward_2dmotion = data["forward_2dmotion"]
        forward_depth = data["forward_depth"]
        visibility = data["visibility"]
        valid = data["valid"]
        intrinsics = data["intrinsics"]
        c2w = data["c2w"]

        # Convert to Tensors
        rgb_images = torch.from_numpy(rgb_images).permute(0, 3, 1, 2) # (T, C, H, W)
        depths = torch.from_numpy(depths).float() # (T, H, W)
        depths_mask = torch.from_numpy(depths_mask) # (T, H, W)

        # Calculate mean depth for potential scaling (though not actively used in final logic)
        if depths_mask.sum() == 0:
            mean_depth = torch.tensor(1.0)
        else:
            mean_depth = depths[depths_mask].mean().clone()

        forward_2dmotion = torch.from_numpy(forward_2dmotion) # (T, H, W, 2)
        forward_depth = torch.from_numpy(forward_depth)
        visibility = torch.from_numpy(visibility)
        valid = torch.from_numpy(valid)
        valid = (valid == 1.0)
        
        # Normalize Intrinsics
        # Assuming intrinsics are [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        intrs = torch.from_numpy(intrinsics).float()
        intrs[:, 0, 0] /= rgb_images.shape[-1] # fx / W
        intrs[:, 0, 2] /= rgb_images.shape[-1] # cx / W
        intrs[:, 1, 1] /= rgb_images.shape[-2] # fy / H
        intrs[:, 1, 2] /= rgb_images.shape[-2] # cy / H
        
        c2w = torch.from_numpy(c2w)

        sample = {
            'rgb_images': rgb_images,       # (T, 3, H, W)
            'depths': depths,               # (T, H, W)
            'depths_mask': depths_mask,     # (T, H, W)
            'metric_scale': mean_depth,     # Scalar
            'forward_2dmotion': forward_2dmotion, # (T, H, W, 2)
            'forward_depth': forward_depth, # (T, H, W)
            'visibility': visibility,       # (T, H, W)
            'intrinsics': intrs,            # (T, 3, 3)
            'c2w': c2w,                     # (T, 4, 4)
            'valid': valid                  # (T, H, W)
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
            if torch.sum(samp['valid']) > 0:
                gotit = True
            else:
                fails += 1
                index = np.random.randint(len(self.data_list))
        
        if fails > 4:
            logger.warning(f'Sampling failed {fails} times before finding valid data.')
            
        return samp, True

    def __len__(self):
        return len(self.data_list)

# ==============================================================================
# Evaluation Loop (Refactored)
# ==============================================================================

@torch.inference_mode()
def _evaluate_model(model, loader, batch_processor_fn, alignment_exp=10, aligned_scene_flow=True):
    """
    Generic evaluation loop shared across datasets.
    
    Args:
        model: The model to evaluate.
        loader: The dataloader.
        batch_processor_fn: A function that takes (batch, model) and returns:
            - rgb_images: Input tensor for the model.
            - gt_data: Dict containing 'points', 'flow3d', 'flow2d', 'valid'.
            - meta: Dict containing 'H', 'W'.
        alignment_exp: Exponent used for alignment scaling (default 10, BlinkVision uses 20).
        aligned_scene_flow: Whether to align.
    """
    # 1. Initialize Metrics
    count_all = 0
    metrics_all = {
        'EPE3D': 0.0, 'Acc3D_strict': 0.0, 'Acc3D_relax': 0.0, 'Outlier': 0.0, 
        "EPE2D": 0.0, "ACC1_2D": 0.0, "ACC3_2D": 0.0, "Outlier_2D": 0.0,
        "abs_rel": 0.0, "threshold_1": 0.0
    }

    logger.info(f"Starting evaluation with {len(loader)} batches...")

    # 2. Iterate over dataset
    for i_batch, batch_blob in enumerate(tqdm(loader)):
        
        # --- A. Pre-process Batch (Dataset Specific) ---
        rgb_images, gt_data, meta = batch_processor_fn(batch_blob)
        
        # --- B. Model Inference ---
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            output = model.infer_pair(
                rgb_images, 
                iters=4, 
                sw=None, 
                is_training=False, 
                tracking3d=True, 
                force_projection=True, 
                apply_mask=False,
                aligned_scene_flow=aligned_scene_flow
            )

        # --- C. Extract Predictions ---
        # 2D Motion (Absolute coordinates)
        pred_2d_motion = output[1]['flow_2d'][:, 0].permute(0, 2, 3, 1) # (B, H, W, 2)
        
        # 3D Motion & Points
        pred_points = output[0]['points'][:, 0]
        pred_flow3d = output[1]['flow_3d'][:, 0]
        pred_3dmotion = pred_flow3d - pred_points

        # --- D. Alignment (Scale & Shift) ---
        # Concatenate points and flow for unified alignment
        # GT Data
        gt_concat = torch.cat([gt_data['points'], gt_data['flow3d']], dim=0)
        gt_3dmotion = gt_data['flow3d'] - gt_data['points']
        valid_mask = gt_data['valid']
        valid_concat = torch.cat([valid_mask, valid_mask], dim=0)

        # Pred Data
        pred_concat = torch.cat([pred_points, pred_flow3d], dim=0)

        # Low-resolution mask-aware resize for efficient alignment calculation
        _, lr_mask, lr_index = mask_aware_nearest_resize(None, valid_concat, (32, 32), return_index=True)
        
        pred_points_lr = pred_concat[lr_index][lr_mask]
        gt_points_lr = gt_concat[lr_index][lr_mask]
        
        # Calculate scale and shift
        scale, shift = align_points_scale_xyz_shift(
            pred_points_lr, 
            gt_points_lr, 
            1 / (gt_points_lr.norm(dim=-1) + 1e-6), 
            exp=alignment_exp
        )
        
        # Apply alignment
        pred_aligned = pred_concat * scale + shift
        pred_3dmotion_aligned = pred_3dmotion * scale

        # --- E. Compute Metrics ---
        count_all += valid_mask.sum()
        
        metrics_sf = compute_scene_flow_metrics(pred_3dmotion_aligned, gt_3dmotion, valid_mask)
        metrics_of = compute_optical_flow_metrics(pred_2d_motion, gt_data['flow2d'], valid_mask)
        metrics_pc = compute_pc_metrics(pred_aligned, gt_concat, valid_concat)

        # Accumulate results
        for key, val in metrics_of.items():    
            metrics_all[key] += 0 if math.isnan(val) else val
        for key, val in metrics_sf.items():
            metrics_all[key] += 0 if math.isnan(val) else val
        for key, val in metrics_pc.items():
            metrics_all[key] += 0 if math.isnan(val) else val

    # 3. Print Final Results
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
# Batch Processors (Dataset Specific Logic)
# ==============================================================================

def _process_kitti_batch(batch_blob):
    """Processor for KITTI dataset structure."""
    image1, image2, depth1, depth2, flow_gt, valid, intrinsics, extrinsics, P1, P2 = \
        [data_item.cuda() for data_item in batch_blob]

    valid_mask = (valid.unsqueeze(-1) > 0.5)[..., 0] # (B, H, W)
    
    # Prepare Ground Truth Flow (Absolute Coordinates)
    H, W = flow_gt.shape[1], flow_gt.shape[2]
    grid_xy = holi4d.utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float()
    grid_xy = grid_xy.reshape(1, H, W, 2) 
    
    flow2d_absolute = flow_gt + grid_xy 
    
    # Stack frames for inference
    rgb_images = torch.cat([image1, image2], dim=0) 
    # (B*2, 3, H, W) or (1, 2, 3, H, W) depending on model input expectation
    # Note: The original code did `rgb_images[None]` inside infer_pair for KITTI, 
    # but `rgb_images` passed to infer_pair for Kubric was (B, 2, 3, H, W).
    # Assuming standard input is (B, 2, 3, H, W).
    if rgb_images.dim() == 4:
        rgb_images = rgb_images.unsqueeze(0)

    return rgb_images, {
        'points': P1,
        'flow3d': P2, # Used to calculate motion P2-P1
        'flow2d': flow2d_absolute,
        'valid': valid_mask
    }, {'H': H, 'W': W}


def _process_kubric_blink_batch(batch_blob, use_ray, check_visibility):
    """Processor for Kubric and BlinkVision dataset structures."""
    batch_data = batch_blob[0]
    
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
    
    if check_visibility:
        flow_valid = flow_valid & (visibility == 1)
    
    H, W = rgb_images.shape[-2], rgb_images.shape[-1]

    # Normalize GT 2D motion to pixel coordinates for unprojection
    forward_2dmotion[..., 0] /= W
    forward_2dmotion[..., 1] /= H
    
    # Unproject GT flow to 3D
    gt_3dflow = utils3d.torch.unproject_cv(
        forward_2dmotion, 
        forward_depth, 
        intrinsics=intrinsics[..., None, :, :], 
        use_ray=use_ray
    )
    
    # Restore 2D motion to pixel units for metric calculation
    forward_2dmotion[..., 0] *= W
    forward_2dmotion[..., 1] *= H 
    gt_flow2d = forward_2dmotion[:, 0]
    
    # Get GT 3D points from depth
    gt_points = utils3d.torch.depth_to_points(depths, intrinsics=intrinsics, use_ray=use_ray)

    return rgb_images, {
        'points': gt_points[:, 0],
        'flow3d': gt_3dflow[:, 0],
        'flow2d': gt_flow2d,
        'valid': flow_valid
    }, {'H': H, 'W': W}


# ==============================================================================
# Entry Points
# ==============================================================================

def test_kitti(model: torch.nn.Module, args: argparse.Namespace):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 4, 'drop_last': False}
    test_dataset = KITTIDataset(root='evaluation/flow/kitti')
    test_loader = DataLoader(test_dataset, **loader_args)
    
    _evaluate_model(
        model,
        test_loader,
        _process_kitti_batch,
        alignment_exp=20,
        aligned_scene_flow=True
    )


def test_kubric(model: torch.nn.Module, args: argparse.Namespace):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 4, 'drop_last': False}
     
    if args.len_level == 0:
        data_dir = 'evaluation/flow/kubric_short'
    elif args.len_level == 1:
        data_dir = 'evaluation/flow/kubric_long'
    else:
        raise ValueError(f"Invalid len_level: {args.len_level}")
        
    test_dataset = Kubric3DflowDataset(root=data_dir)
    test_loader = DataLoader(test_dataset, **loader_args)
    
    # Kubric: use_ray=True, check_visibility=False (based on original code comments), exp=10
    processor = lambda b: _process_kubric_blink_batch(b, use_ray=True, check_visibility=False)
    _evaluate_model(
        model, 
        test_loader, 
        processor, 
        alignment_exp=20, 
        aligned_scene_flow=False
    )


def test_blinkvision(model: torch.nn.Module, args: argparse.Namespace):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 4, 'drop_last': False}
    test_dataset = BlinkvisionflowDataset(root='evaluation/flow/blinkvision')
    test_loader = DataLoader(test_dataset, **loader_args)
    
    # BlinkVision: use_ray=False, check_visibility=True, exp=20
    processor = lambda b: _process_kubric_blink_batch(b, use_ray=False, check_visibility=True)
    _evaluate_model(model, 
        test_loader, 
        processor, 
        alignment_exp=20, 
        aligned_scene_flow=True
    )

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

    parser.add_argument(
        "--coordinate", type=str, default='camera_base', 
        choices=['camera_base', 'world_pi3', 'world_depthanythingv3'],
        help="'camera': camera centric, 'world': world centric"
    )

    parser.add_argument(
        "--dataset", type=str, default='kitti', 
        choices=['kitti', 'kubric_short', 'kubric_long', 'blinkvision'],
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
    if args.dataset == 'kitti':
        test_kitti(model, args)
    elif args.dataset == 'kubric_short':
        args.len_level = 0
        test_kubric(model, args)
    elif args.dataset == 'kubric_long':
        args.len_level = 1
        test_kubric(model, args)
    elif args.dataset == 'blinkvision':
        test_blinkvision(model, args)

