# **Holi4D: Holistic Feedforward 4D Motion Reconstruction of All Pixels from Monocular Videos**

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue)
[![arXiv](https://img.shields.io/badge/arXiv-2412.03079-b31b1b.svg)](https://arxiv.org/abs/2412.03079)
[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://igl-hkust.github.io/TrackingWorld.github.io/)

**Authors**
`<br>`
[Jiahao Lu](https://github.com/jiah-cloud)`<sup>`1`</sup>` &nbsp;•&nbsp; [Jiayi Xu](https://openreview.net/profile?id=~Jiayi_Xu10)`<sup>`1`</sup>` &nbsp;•&nbsp; [Wenbo Hu](https://wbhu.github.io/)`<sup>`2&dagger;`</sup>` &nbsp;•&nbsp; [Ruijie Zhu](https://ruijiezhu94.github.io/ruijiezhu/)`<sup>`2`</sup>`
`<br>`
[Sai-Kit Yeung](https://saikit.org/index.html)`<sup>`1`</sup>` &nbsp;•&nbsp; [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)`<sup>`2`</sup>` &nbsp;•&nbsp; [Yuan Liu](https://liuyuan-pal.github.io/)`<sup>`1&dagger;`</sup>`

</div>

---

### 📖 Abstract

**Holi4D** is a foundation 4D motion model that enables holistic 4D reconstruction from arbitrary monocular videos. Given a single video input of a dynamic scene, Holi4D predicts 3D points and scene flows for **all pixels** in a feedforward manner. This capability supports a variety of downstream tasks, including dense tracking and camera pose estimation.

---

### 🖼️ Teaser

<div align="center">
  <img src="assets/teaser.png" width="100%" alt="Holi4D Teaser">
</div>

---

## ⚙️ Setup and Installation

### 1. Clone the Repository

Clone the repository with submodules to ensure all dependencies are included:

```bash
git clone --recursive https://github.com/TencentARC/Holi4D.git
cd Holi4D
```

### 2. Environment Setup

We provide an installation script tested with **CUDA 12.1** and **Python 3.11**.

```bash
conda create -n holi4d python=3.11
conda activate holi4d
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

git clone https://github.com/jiah-cloud/utils3d.git 
git clone --no-checkout https://github.com/yyfz/Pi3.git holi4d/nets/external/pi3_repo
cd holi4d/nets/external/pi3_repo
git sparse-checkout init
git sparse-checkout set pi3
git checkout main
find . -maxdepth 1 -type f -exec rm -f {} \;
mv pi3 ../pi3
cd ../../../..

git clone --no-checkout https://github.com/ByteDance-Seed/Depth-Anything-3.git holi4d/nets/external/dad3_repo
cd holi4d/nets/external/dad3_repo
git sparse-checkout init
git sparse-checkout set src/depth_anything_3
git checkout main
find . -maxdepth 1 -type f -exec rm -f {} \;
mv src/depth_anything_3 ../depth_anything_3
cd ../../../..

git clone https://github.com/IDEA-Research/Grounded-SAM-2.git submodules
cd submodules
pip install -e .
pip install --no-build-isolation -e grounding_dino
cd ..

mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O ./checkpoints/sam2.1_hiera_large.pt
wget https://huggingface.co/cyun9286/holi4d/resolve/main/holi4d.pth
```

### 3. Download Weights

Download the pre-trained model weights and place them in the project root (or your preferred checkpoint directory).

* **Holi4D Weights:** [Download from HuggingFace](https://huggingface.co/cyun9286/holi4d/blob/main/holi4d.pth)

---

## 🚀 Demo

Run the following commands to perform tracking and reconstruction on the provided demo video (`demo_data/cat.mp4`).

### 1. First Frame 3D Tracking (`3d_ff`)

Reconstructs 3D motion based on the first frame.

```bash
python demo.py \
    --mp4_path demo_data/cat.mp4 \
    --mode 3d_ff \
    --Ts -1 \
    --save_base_dir results/cat
```

### 2. Dense Tracking: Every Pixel, Every Frame (`3d_efep`)

Performs dense 3D tracking for every pixel across all frames.

```bash
python demo.py \
    --mp4_path demo_data/cat.mp4 \
    --mode 3d_efep \
    --Ts -1 \
    --save_base_dir results/cat
```

### 3. 2D Tracking (`2d`)

Performs standard 2D tracking in image space.

```bash
python demo.py \
    --mp4_path demo_data/cat.mp4 \
    --mode 2d \
    --Ts -1 \
    --save_base_dir results/cat
```

---

## ✨ Visualization

Visualize the dense 4D trajectories and reconstructed scenes using the generated output files.

**Visualize First Frame 3D Tracking:**

```bash
python visualization/vis_3d_ff.py --ply_dir results/cat/3d_ff_output
```

**Visualize Dense Tracking (Every Pixel):**

```bash
python visualization/vis_3d_efep.py --ply_dir results/cat/3d_efep_output
```

<div align="center">
  <img src="assets/demo.gif" width="100%" alt="Visualization Demo">
</div>

---

## 📊 Evaluation

For detailed instructions on how to evaluate the model on standard benchmarks (Sintel, KITTI, Kubric, etc.), please refer to the evaluation guide:

👉 **[Evaluation Guide (evaluation/eval.md)](evaluation/eval.md)**

---

## 📝 Citation

If you find **Holi4D** useful for your research or applications, please consider citing our paper:

```bibtex

```

---

## 🤝 Acknowledgements

Our codebase is built upon [MoGe](https://github.com/microsoft/MoGe) and [Alltracker](https://github.com/aharley/alltrackerh). We also gratefully acknowledge [Trackingworld](https://github.com/IGL-HKUST/TrackingWorld), [VGGT](https://github.com/facebookresearch/vggt) for their excellent work!
