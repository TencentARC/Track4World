import sys
import os

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "submodules", "Grounded-SAM-2")  # adjust if needed
sys.path.append(os.path.abspath(project_root))
import argparse
import cv2
import json
import torch
import numpy as np
import supervision as sv
from pathlib import Path
from supervision.draw.color import ColorPalette

sys.path.insert(0, 'submodules/Grounded-SAM-2')

from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torchvision 
from tqdm import tqdm

"""
Hyper parameters
"""
parser = argparse.ArgumentParser(description="Process a single video with Grounded SAM 2")
parser.add_argument('--video-path', required=True, 
help="Path to the input video file (mp4, avi) or directory of frames")
parser.add_argument('--text-prompt', required=True, 
help="Text prompts for detection, e.g., 'car. person. dog'")
parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-base")
parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
parser.add_argument("--output-dir", default="output_single_video", help="Directory to save results")
parser.add_argument("--force-cpu", action="store_true")
parser.add_argument("--box-threshold", type=float, default=0.1)
parser.add_argument("--text-threshold", type=float, default=0.1)
args = parser.parse_args()

# Constants
GROUNDING_MODEL = args.grounding_model
VIDEO_PATH = args.video_path
TEXT_PROMPT = args.text_prompt
SAM2_CHECKPOINT = args.sam2_checkpoint
SAM2_MODEL_CONFIG = args.sam2_model_config
DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
OUTPUT_DIR = Path(args.output_dir)

# Create Output Dirs
output_path_vis = OUTPUT_DIR / "vis"
output_path_mask = OUTPUT_DIR / "mask"
# --- CHANGE 1: Define color path one level up from OUTPUT_DIR ---
# output_path_color = OUTPUT_DIR.parent / "color"

output_path_vis.mkdir(parents=True, exist_ok=True)
output_path_mask.mkdir(parents=True, exist_ok=True)
# output_path_color.mkdir(parents=True, exist_ok=True)

# Environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"Loading models on {DEVICE}...")

# Build SAM2 image predictor
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# Build Grounding DINO
processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)

def id_to_colors(id): # id to color
    rgb = np.zeros((3, ), dtype=np.uint8)
    for i in range(3):
        rgb[i] = id % 256
        id = id // 256
    return rgb

# Generate random color mapping
idx_to_id = [i for i in range(256*256*256)]
np.random.shuffle(idx_to_id) 

def get_frames_generator(source_path):
    """
    Yields frames from a video file or a directory of images.
    Returns: (frame_name, frame_rgb_numpy, original_pil_image)
    """
    if os.path.isdir(source_path):
        print(f"Processing directory: {source_path}")
        files = sorted([f for f in os.listdir(source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for filename in files:
            full_path = os.path.join(source_path, filename)
            image_pil = Image.open(full_path).convert("RGB")
            yield filename, np.array(image_pil), image_pil
    else:
        print(f"Processing video file: {source_path}")
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {source_path}")
        
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm(total=total_frames, desc="Processing Frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR (OpenCV) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(frame_rgb)
                
                # Create a filename for saving
                frame_name = f"{frame_idx:05d}.jpg"
                
                yield frame_name, frame_rgb, image_pil
                
                frame_idx += 1
                pbar.update(1)
        cap.release()

# --- Main Processing Loop ---

print(f"Starting inference with prompt: '{TEXT_PROMPT}'")

frame_generator = get_frames_generator(VIDEO_PATH)

# Use tqdm if processing a directory (video file handles its own tqdm inside generator)
if os.path.isdir(VIDEO_PATH):
    frame_generator = tqdm(frame_generator, desc="Processing Frames")

for image_file, image, image_pil in frame_generator:
    
    # --- CHANGE 2: Convert to BGR and save original frame immediately ---
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(str(output_path_color / image_file), img_bgr)

    # 1. Prepare SAM2
    sam2_predictor.set_image(image)

    # 2. Run Grounding DINO
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[image_pil.size[::-1]]
    )

    # 3. Get box prompts and run SAM2
    input_boxes = results[0]["boxes"].cpu().numpy()

    if input_boxes.shape[0] != 0:
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Squeeze dim if needed (n, 1, H, W) -> (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))

        # 4. Supervision Visualization Logic
        # img_bgr is already defined at the top of the loop
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids,
            confidence=np.array(confidences)
        )

        # Non-Maximum Suppression (NMS)
        nms_idx = torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy).float(), 
                    torch.from_numpy(detections.confidence).float(), 
                    0.5
                ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.mask = detections.mask[nms_idx]

        # Annotate
        labels_vis = [
            f"{class_names[id]} {confidence:.2f}"
            for id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(scene=img_bgr.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels_vis)

        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        # 5. Save Output
        # Prepare Mask Data
        masks_final = detections.mask
        labels_final = detections.class_id
        
        color_mask = np.zeros(image.shape, dtype=np.uint8)
        obj_info_json = []

        # Sort masks by size (largest first) for better rendering
        mask_size = [np.sum(m) for m in masks_final]
        sorted_mask_idx = np.argsort(mask_size)[::-1]

        for idx in sorted_mask_idx:
            m = masks_final[idx]
            # Use random color map logic from original script
            color_mask[m] = id_to_colors(idx_to_id[idx])

            obj_info_json.append({
                "id": idx_to_id[idx],
                "label": class_names[labels_final[idx]],
                "score": float(detections.confidence[idx]),
            })

        color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

        # Save Visualizations
        cv2.imwrite(str(output_path_vis / image_file), annotated_frame)
        
        # Save Masks (PNG + JSON)
        png_name = os.path.splitext(image_file)[0] + ".png"
        json_name = os.path.splitext(image_file)[0] + ".json"
        
        cv2.imwrite(str(output_path_mask / png_name), color_mask_bgr)
        with open(output_path_mask / json_name, "w") as f:
            json.dump(obj_info_json, f)

    else:
        # No detections
        # img_bgr is already defined at the top of the loop
        cv2.imwrite(str(output_path_vis / image_file), img_bgr)
        
        png_name = os.path.splitext(image_file)[0] + ".png"
        json_name = os.path.splitext(image_file)[0] + ".json"
        
        cv2.imwrite(str(output_path_mask / png_name), np.zeros(image.shape, dtype=np.uint8))
        with open(output_path_mask / json_name, "w") as f:
            json.dump([], f)

print(f"Processing complete. Results saved to {OUTPUT_DIR}")