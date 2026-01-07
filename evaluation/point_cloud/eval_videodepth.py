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
from holi4d.utils.alignment import align_depth_affine

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
# Helper Functions: Image & Video I/O
# ==============================================================================

def decode_jpeg_bytes(images_jpeg_bytes: List[bytes]) -> np.ndarray:
    """Decodes a list of JPEG bytes into a numpy array of images."""
    images = []
    for jpeg_bytes in images_jpeg_bytes:
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        images.append(np.array(img))
    return np.stack(images, axis=0)

def read_images_or_video(
    input_path: Union[str, Path], 
    resize_to: Optional[int], 
    device: torch.device, 
    gt_dataset_type: str = 'Sintel'
) -> Tuple[List[np.ndarray], List[torch.Tensor], np.ndarray]:
    """
    Reads frames from a directory of images or a video file.

    Args:
        input_path: Path to image directory or video file.
        resize_to: Short edge size to resize to (maintains aspect ratio).
        device: Torch device.
        selected_views: Indices of frames to load.
        gt_dataset_type: Dataset type (affects sorting logic).

    Returns:
        video: List of numpy images (H, W, 3).
        video_list: List of torch tensors (3, H, W).
        selected_views: Array of loaded frame indices.
    """
    input_path = Path(input_path)
    video = []
    video_list = []

    # --- Case 1: Input is a Directory of Images ---
    if input_path.is_dir():
        include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
        
        # Sorting logic depends on dataset naming convention
        if gt_dataset_type == 'Kubric-3D':
            # Sort by number in filename
            image_paths = sorted(
                itertools.chain(*(input_path.glob(f'*.{suffix}') for suffix in include_suffices)),
                key=lambda p: int(re.findall(r'\d+', p.stem)[-1]) if re.findall(r'\d+', p.stem) else -1
            )
        else:
            # Sort alphabetically
            image_paths = sorted(
                itertools.chain(*(input_path.glob(f'*.{suffix}') for suffix in include_suffices)),
                key=lambda p: p.name
            )

        if not image_paths:
            raise FileNotFoundError(f'No image files found in {input_path}')
        
        for i in tqdm(np.arange(len(image_paths)), desc=f'Loading images from {input_path.name}', leave=False):
            if i < 0 or i >= len(image_paths):
                continue

            image = cv2.cvtColor(cv2.imread(str(image_paths[i])), cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Resize logic
            if resize_to is not None:
                scale = resize_to / min(h, w)
                w_new, h_new = int(w * scale), int(h * scale)
                image = cv2.resize(image, (w_new, h_new), cv2.INTER_AREA)

            video.append(image)
            image_tensor = torch.tensor(image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
            video_list.append(image_tensor)

    # --- Case 2: Input is a Video File ---
    elif input_path.is_file() and input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {input_path}")
        
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices_to_read = np.arange(num_frames) 

        for frame_idx in tqdm(frame_indices_to_read, desc=f'Loading video {input_path.name}', leave=False):
            if frame_idx >= num_frames: break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            if resize_to is not None:
                scale = resize_to / min(h, w)
                w_new, h_new = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (w_new, h_new), cv2.INTER_AREA)

            video.append(frame)
            image_tensor = torch.tensor(frame / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
            video_list.append(image_tensor)

        cap.release()

    else:
        raise ValueError(f"Unsupported input: {input_path}")

    return video, video_list

def load_depth_kubric(path: str, depth_range: Tuple[float, float]) -> Tuple[np.ndarray, None]:
    """Loads 16-bit PNG depth from Kubric dataset and converts to meters."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Depth PNG not found: {path}")
    
    depth_uint16 = imageio.imread(path).astype(np.uint16)
    mask_nan, mask_inf = depth_uint16 == 0, depth_uint16 == 65535
    
    depth_min, depth_max = map(float, depth_range)
    # Normalize to [0, 1] then scale to [min, max]
    depth_float = depth_min + depth_uint16.astype(np.float32) * (depth_max - depth_min) / np.iinfo(np.uint16).max
    
    depth_float[mask_nan] = np.nan
    depth_float[mask_inf] = np.inf
    return depth_float, None

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


def summarize_evaluation_results(root_dir: str):
    """
    Walks through the output directory, parses evaluation_results.txt,
    and calculates weighted averages for the metrics.
    """
    logger.info(f"Summarizing results from: {root_dir}")
    
    # Regex patterns to extract metrics
    pattern_absrel = re.compile(r"Abs_Rel\s+([\d.]+)")
    pattern_d1 = re.compile(r"d1 < 1.25\s+([\d.]+)")
    pattern_valid = re.compile(r"Valid Pixels\s+(\d+)")

    records = []

    if not os.path.exists(root_dir):
        logger.error(f"Output directory {root_dir} does not exist.")
        return

    # Iterate through all scene subdirectories
    for scene in sorted(os.listdir(root_dir)):
        scene_path = os.path.join(root_dir, scene)
        if not os.path.isdir(scene_path):
            continue
            
        eval_file = os.path.join(scene_path, "evaluation_results.txt")
        if not os.path.isfile(eval_file):
            continue
        
        try:
            with open(eval_file, "r") as f:
                content = f.read()
            
            abs_rel_match = pattern_absrel.search(content)
            d1_match = pattern_d1.search(content)
            valid_match = pattern_valid.search(content)
            
            if abs_rel_match and d1_match and valid_match:
                abs_rel = float(abs_rel_match.group(1))
                d1 = float(d1_match.group(1))
                valid_pixels = int(valid_match.group(1))
                
                records.append({
                    "scene": scene,
                    "Abs_Rel": abs_rel,
                    "d1": d1,
                    "Valid_Pixels": valid_pixels
                })
        except Exception as e:
            logger.warning(f"Error reading {eval_file}: {e}")

    # Calculate and print statistics
    df = pd.DataFrame(records)
    if df.empty:
        logger.warning(f"No valid evaluation files found in {root_dir}")
    else:
        # Calculate weighted averages based on valid pixels
        total_pixels = df["Valid_Pixels"].sum()
        weighted_abs_rel = (df["Abs_Rel"] * df["Valid_Pixels"]).sum() / total_pixels
        weighted_d1 = (df["d1"] * df["Valid_Pixels"]).sum() / total_pixels

        print("\n" + "="*50)
        print("FINAL EVALUATION SUMMARY")
        print("="*50)
        print(df.to_string(index=False))
        print("-" * 50)
        print(f"📊 Weighted Averages (over {len(df)} scenes):")
        print(f"  Abs_Rel     = {weighted_abs_rel:.4f}")
        print(f"  d1 < 1.25   = {weighted_d1:.4f}")
        print("="*50 + "\n")
        
        # Optionally save the summary to a file
        summary_path = os.path.join(root_dir, "final_summary.txt")
        with open(summary_path, "w") as f:
            f.write(df.to_string(index=False))
            f.write(f"\n\nWeighted Abs_Rel: {weighted_abs_rel:.4f}\n")
            f.write(f"Weighted d1 < 1.25: {weighted_d1:.4f}\n")
        logger.info(f"Summary saved to {summary_path}")

# ==============================================================================
# Helper Functions: Model & Data Loading Logic
# ==============================================================================

def load_model(args: argparse.Namespace, config: Dict) -> torch.nn.Module:
    """Initializes the Holi4D model and loads pretrained weights."""
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
        url = "https://huggingface.co/cyun9286/holi4d/resolve/main/holi4d.pth"
        logger.info(f'Local checkpoint not found. Downloading from {url}...')
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=False)
        model.load_state_dict(state_dict, strict=False)
    
    device = torch.device(args.device_name)
    model.to(device)
    model.eval()
    
    # Freeze parameters
    for p in model.parameters():
        p.requires_grad = False
    
    return model

def get_scene_paths(dataset_type: str) -> List[Path]:
    """Returns a list of scene paths based on the dataset type."""
    # NOTE: These paths are hardcoded based on the environment. 
    root_path = Path(f'evaluation/point_cloud/{dataset_type}')
    
    if not root_path.exists():
        logger.warning(f"Dataset root path does not exist: {root_path}")
        return []

    return sorted([p for p in root_path.iterdir() if p.is_dir() or p.suffix == '.npz'])

def load_scene_input(
    input_path: Path, 
    args: argparse.Namespace, 
    device: torch.device, 
    start_frame: int, 
    end_frame: int
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    """
    Loads the RGB input for a specific scene.
    Returns:
        video_tensor: (B, C, H, W) normalized tensor.
        selected_views: Indices of loaded frames.
    """
    if args.gt_dataset_type == 'Kubric-3D':
        rgb_path = input_path / 'video_frames'
        _, video_list = read_images_or_video(
            rgb_path, args.resize_to, device, gt_dataset_type=args.gt_dataset_type
        )
        video_tensor = torch.stack(video_list, dim=0)[start_frame:end_frame]
        selected_views = np.arange(start_frame, min(end_frame, len(video_list)))

    elif args.gt_dataset_type == 'Bonn':
        rgb_path = input_path / 'rgb_110'
        _, video_list = read_images_or_video(
            rgb_path, args.resize_to, device, gt_dataset_type=args.gt_dataset_type
        )
        video_tensor = torch.stack(video_list, dim=0)[start_frame:end_frame]
        selected_views = np.arange(start_frame, min(end_frame, len(video_list)))

    elif args.gt_dataset_type in ['Sintel', 'Scannet', 'Monkaa', 'KITTI', 'GMUKitchens']:
        # Find the first mp4 file in the directory
        mp4_files = sorted(input_path.glob("*.mp4"))
        if not mp4_files:
            raise FileNotFoundError(f"No .mp4 files found in {input_path}")
        rgb_path = mp4_files[0]
        
        _, video_list = read_images_or_video(
            rgb_path, args.resize_to, device, gt_dataset_type=args.gt_dataset_type
        )
        video_tensor = torch.stack(video_list, dim=0)[start_frame:end_frame]
        selected_views = np.arange(start_frame, min(end_frame, len(video_list)))

    else:
        # Fallback for numpy-based datasets (e.g., TUM, PO)
        input_data = np.load(input_path)
        video = decode_jpeg_bytes(input_data['images_jpeg_bytes'])
        h, w = video.shape[1], video.shape[2]
        
        video_list = []
        if args.resize_to is not None:
            scale = args.resize_to / min(h, w)
            w_new, h_new = int(w * scale), int(h * scale)
            for i in range(video.shape[0]):
                video_list.append(cv2.resize(video[i], (w_new, h_new), cv2.INTER_AREA))
        else:
            video_list = [v for v in video]
            
        video_np = np.stack(video_list)
        video_tensor = torch.from_numpy(video_np).to(device)[start_frame:end_frame]
        video_tensor = video_tensor.permute(0, -1, 1, 2) / 255.0
        selected_views = np.arange(start_frame, min(end_frame, len(video_list)))

    return video_tensor, selected_views

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

    elif args.gt_dataset_type in ['Sintel', 'Scannet', 'Monkaa', 'KITTI', 'GMUKitchens']:
        # Load from HDF5
        hdf5_files = sorted(input_path.glob("*.hdf5"))
        if hdf5_files:
            with h5py.File(hdf5_files[0], "r") as f:
                gt_points = np.array(f['point_map'], dtype=np.float32)
                gt_mask = np.array(f['valid_mask'], dtype=np.float32)
            gt_depths = gt_points[..., -1][selected_views]

    elif args.gt_dataset_type =='Bonn':
        depth_path = input_path / 'depth_110'
        if depth_path.exists():
            depth_files = sorted([
                os.path.join(depth_path, f) for f in os.listdir(depth_path) 
                if f.endswith(('.png', '.jpg'))
            ])
            gt_depths = np.stack([
                np.asarray(Image.open(depth_files[i])).astype(np.float64) / 5000.0 for i in selected_views
            ], axis=0)
    else:
        # Fallback for numpy-based datasets (e.g., TUM, PO)
        input_data = np.load(input_path)
        if 'depth_map' in input_data:
            gt_depths = input_data['depth_map'][selected_views]

    return gt_depths, gt_mask

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Holi4D Inference and Evaluation Script")

    # --- Model Configuration ---
    parser.add_argument("--ckpt_init", type=str, default="./checkpoints/holi4d.pth", help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, default="./holi4d/config/eval/v1.json", help="Path to model config JSON")
    parser.add_argument("--output", "-o", dest="output_path", type=str, default='./output_sintel/23', help="Output folder path")
    parser.add_argument("--device", dest="device_name", type=str, default='cuda', help="Device name")
    parser.add_argument("--fp16", action='store_true', help="Use fp16 precision")
    
    # --- Data Configuration ---
    parser.add_argument("--resize-to", type=int, default=512, help="Resize short edge to this size")
    parser.add_argument("--frames", type=str, default='0-150', help="Frame range 'start-end'")
    parser.add_argument("--gt-dataset-type", type=str, default='Sintel', 
                        choices=['Bonn', 'Sintel', 'Scannet', 'Monkaa', 'Kubric-3D', 'KITTI', 'GMUKitchens'], 
                        help="Dataset type")
    
    # --- Inference Parameters ---
    parser.add_argument("--fov_x", dest="fov_x_", type=float, default=None, help="Horizontal FOV (deg). None = Auto")
    parser.add_argument("--resolution_level", type=int, default=0, help="Resolution level [0-9]")
    parser.add_argument("--num_tokens", type=int, default=None, help="Explicit token count (overrides resolution_level)")
    parser.add_argument("--max-depth", type=float, default=70.0, help="Max depth for evaluation")
    parser.add_argument("--chunk_size", type=int, default=130, help="Process in chunks to avoid OOM")
    
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
        torch.cuda.empty_cache()
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
        
        all_pred_depths = []
        
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
            all_pred_depths.append(output[0]['depth'].cpu().numpy())
            torch.cuda.empty_cache()

        # Concatenate results: (N, H, W, 3)
        pred_depths = np.concatenate(all_pred_depths, axis=1).squeeze(0)

        # --- Evaluation ---
        logger.info("Loading ground truth for evaluation...")
        try:
            gt_depths, gt_mask = load_scene_gt(input_path, args, selected_views, start_frame, end_frame)
        except Exception as e:
            logger.warning(f"Failed to load GT for {input_name}: {e}. Skipping evaluation.")
            gt_depths = None

        if gt_depths is not None:
            gt_h, gt_w = gt_depths.shape[1], gt_depths.shape[2]
            
            # Resize predictions to match GT resolution
            pred_depths_resized = np.array([
                cv2.resize(pred, (gt_w, gt_h), cv2.INTER_NEAREST) for pred in pred_depths
            ])
            
            logger.info("Calculating metrics...")
            
            # Determine mask based on dataset type
            mask_pre = (gt_mask == 1) if gt_mask is not None else None
            
            # Adjust max_depth based on dataset
            eval_max_depth = 30.0 if args.gt_dataset_type == 'Kubric-3D' else args.max_depth

            eval_results, _ = depth_evaluation(
                pred_depths_resized, 
                gt_depths, 
                max_depth=eval_max_depth, 
                use_gpu=True, 
                mask_pre=mask_pre
            )
            
            # Save Results
            results_file = save_path / "evaluation_results.txt"
            with open(results_file, "w") as f:
                f.write("--- Depth Evaluation Summary ---\n")
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

