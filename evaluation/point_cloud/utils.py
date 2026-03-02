import os
import sys
import itertools
import re
import io
import json
import argparse
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import logging 

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
from track4world.nets.model import Track4World
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
            image_tensor = torch.tensor(frame, dtype=torch.float32, device=device).permute(2, 0, 1)
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

def get_shared_args():
    """
    Defines the superset of arguments required for both tasks.
    """
    parser = argparse.ArgumentParser(description="Track4World Inference and Evaluation Script")

    # --- Model & Environment ---
    parser.add_argument("--ckpt_init", type=str, default="./checkpoints/track4world_da3.pth", 
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--config_path", type=str, default="./track4world/config/eval/v1.json",
                        help="Path to model config JSON")
    parser.add_argument("--output", "-o", dest="output_path", type=str, default='./output_eval', 
                        help="Output directory")
    parser.add_argument("--device", dest="device_name", type=str, default='cuda', 
                        help="Computation device")
    parser.add_argument("--fp16", action='store_true', 
                        help="Enable half-precision (float16)")
    
    # --- Data & Input ---
    parser.add_argument("--resize-to", type=int, default=512, 
                        help="Resize short edge to this resolution")
    parser.add_argument("--frames", type=str, default='0-150', 
                        help="Frame range 'start-end'")
    # Merged choices from both scripts
    parser.add_argument("--gt-dataset-type", type=str, default='Sintel', 
                        choices=['Tum', 'Bonn', 'Sintel', 'Scannet', 'Monkaa', 'Kubric-3D', 'KITTI', 'GMUKitchens'], 
                        help="Ground truth dataset format")
    
    # --- Inference Parameters ---
    parser.add_argument("--fov_x", dest="fov_x_", type=float, default=None, 
                        help="Horizontal FOV")
    parser.add_argument("--resolution_level", type=int, default=0,
                        help="Detail level [0-9]")
    parser.add_argument("--num_tokens", type=int, default=None, 
                        help="Explicit token count")
    parser.add_argument("--max-depth", type=float, default=70.0, 
                        help="Max depth threshold for metrics")
    parser.add_argument("--coordinate", type=str, default='world_depthanythingv3', 
                        choices=['camera_base', 'world_pi3', 'world_depthanythingv3'])
    parser.add_argument(
        "--use_original_backbone",
        action="store_true",
        help="Use the original pretrained backbone instead of the modified one."
    )
    parser.add_argument("--chunk_size", type=int, default=130,
                        help="Temporal chunk size")

    return parser.parse_args()

def run_evaluation_pipeline(args, task_config):
    """
    Generic pipeline for running inference and evaluation.
    
    Args:
        args: Parsed arguments.
        task_config (dict): Configuration specific to the task (Point vs Depth).
            - target_key: Key to extract from model output ('points' or 'depth').
            - gt_loader: Function to load ground truth.
            - eval_fn: Function to calculate metrics.
            - metric_header: String for the results text file header.
    """
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
        
        all_preds = []
        
        # Process in chunks
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
            
            # Extract specific key (points or depth) and move to CPU
            pred_chunk = output[0][task_config['target_key']].cpu().numpy()
            all_preds.append(pred_chunk)
            torch.cuda.empty_cache()

        # Concatenate results: (N, H, W, C)
        pred_data = np.concatenate(all_preds, axis=1).squeeze(0)

        # --- Evaluation ---
        logger.info("Loading ground truth for evaluation...")
        try:
            # Call the specific GT loader
            gt_data, gt_mask = task_config['gt_loader'](input_path, args, selected_views, start_frame, end_frame)
        except Exception as e:
            logger.warning(f"Failed to load GT for {input_name}: {e}. Skipping evaluation.")
            gt_data = None

        if gt_data is not None:
            gt_h, gt_w = gt_data.shape[1], gt_data.shape[2]
            
            # Resize predictions to match GT resolution
            pred_data_resized = np.array([
                cv2.resize(pred, (gt_w, gt_h), cv2.INTER_NEAREST) for pred in pred_data
            ])
            
            logger.info("Calculating metrics...")
            
            mask_pre = (gt_mask == 1) if gt_mask is not None else None
            eval_max_depth = 30.0 if args.gt_dataset_type == 'Kubric-3D' else args.max_depth

            # Call the specific evaluation function
            eval_results, _ = task_config['eval_fn'](
                pred_data_resized, 
                gt_data, 
                max_depth=eval_max_depth, 
                use_gpu=True, 
                mask_pre=mask_pre
            )
            
            # Save Results
            results_file = save_path / "evaluation_results.txt"
            with open(results_file, "w") as f:
                f.write(f"--- {task_config['metric_header']} ---\n")
                f.write("-" * 35 + "\n")
                f.write(f"{'Metric':<20}{'Value':<12}\n")
                f.write("-" * 35 + "\n")
                f.write(f"{'Abs_Rel':<20}{eval_results[0]:<12.6f}\n")
                f.write(f"{'d1 < 1.25':<20}{eval_results[1]:<12.6f}\n")
                f.write(f"{'Valid Pixels':<20}{int(eval_results[2])}\n")
            
            logger.info(f"Results saved to {results_file}")
        else:
            logger.warning("Skipping evaluation due to missing Ground Truth.")

    # 5. Summarize
    summarize_evaluation_results(args.output_path)