#!/bin/bash

# ============================================================
# Unified Path Settings
# ============================================================

VIDEO="/group/40075/public_datasets/GeometryCrafter/datasets/DAVIS/horsejump-high.mp4"
OUTPUT_DIR="results/horsejump-high"
SAM2_CKPT="/group/40075/jiahaolu/cleaned_code/DELTA_densetrack3d/checkpoints/sam2.1_hiera_large.pt"
HOLI4D_CKPT="/group/40075/jiahaolu/MoGe/alltracker/checkpoints/cleaned_model.pth"

# ============================================================
# 1️⃣ DINO + SAM2 Segmentation
# ============================================================

# CUDA_VISIBLE_DEVICES=1 \
# /data/miniconda3/envs/holi4d/bin/python scripts/run_dino_sam2.py \
#     --video-path "$VIDEO" \
#     --sam2-checkpoint "$SAM2_CKPT" \
#     --output-dir "$OUTPUT_DIR" \
#     --text-prompt "car, black car."

# ============================================================
# 2️⃣ Holi4D 3D EFEP Demo
# ============================================================

CUDA_VISIBLE_DEVICES=1 \
/data/miniconda3/envs/holi4d/bin/python demo.py \
    --mp4_path "$VIDEO" \
    --mode 3d_efep \
    --coordinate world_depthanythingv3 \
    --Ts -1 \
    --ckpt_init "$HOLI4D_CKPT" \
    --save_base_dir "$OUTPUT_DIR"
