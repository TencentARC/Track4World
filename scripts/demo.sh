#!/bin/bash
# ============================================================
# Unified Path Settings
# ============================================================

VIDEO="demo_data/cat.mp4"
OUTPUT_DIR="results/cat"
SAM2_CKPT="checkpoints/sam2.1_hiera_large.pt"
TRACK4WORLD_CKPT="checkpoints/track4world_pi3.pth"

# ============================================================
# 1️⃣ DINO + SAM2 Segmentation
# ============================================================

python scripts/run_dino_sam2.py \
    --video-path "$VIDEO" \
    --sam2-checkpoint "$SAM2_CKPT" \
    --output-dir "$OUTPUT_DIR" \
    --text-prompt "cat."

# ============================================================
# 2️⃣ Track4World 3D EFEP Demo
# ============================================================

python demo.py \
    --mp4_path "$VIDEO" \
    --coordinate world_pi3 \
    --mode 3d_efep \
    --Ts -1 \
    --ckpt_init "$TRACK4WORLD_CKPT" \
    --save_base_dir "$OUTPUT_DIR"
