## 📊 Evaluation

To reproduce the results reported in our paper, please follow the steps below.

### 0. Data Preparation

**Step 1: General Evaluation Datasets**
Download the main evaluation datasets and unzip them into the project directory.

[**📥 Download Evaluation Datasets**](https://huggingface.co/TencentARC/Track4World/resolve/main/evaluation_datasets.zip)

**Step 2: 2D Tracking Datasets (TAP-Vid)**
For 2D tracking benchmarks, please refer to the [**DeepMind TAP-Vid repository**](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid) for instructions on downloading and processing the data (`kinetics`, `rgb_stacking`, `robotap`).

**Directory Structure**
Ensure your directory structure matches the following layout after extraction and processing:

```text
Track4World/
└── evaluation/
    ├── 2d_track/                # 2D Tracking (TAP-Vid)
    │   ├── robotap/
    │   ├── tapvid_kinetics/
    │   └── tapvid_rgb_stacking/
    ├── flow/                    # Optical and Scene Flow 
    │   ├── blinkvision/
    │   ├── kitti/
    │   ├── kubric_long/
    │   └── kubric_short/
    ├── point_cloud/             # Point Cloud and Video Depth
    │   ├── Bonn/
    │   ├── GMUKitchens/
    │   ├── KITTI/
    │   ├── Kubric-3D/
    │   ├── Monkaa/
    │   ├── Scannet/
    │   ├── Sintel/
    │   └── Tum/
    └── track/                   # 3D Tracking
        ├── adt_mini/
        ├── ds_mini/
        ├── po_mini/
        └── pstudio_mini/
```

---

### 1. Optical and Scene Flow Estimation

Run the following commands to evaluate flow estimation on different datasets:

**BlinkVision & KITTI:**

```bash
python evaluation/flow/eval.py --dataset kitti
python evaluation/flow/eval.py --dataset blinkvision
```

**Kubric:**
Evaluate on different sequence lengths:

```bash
python evaluation/flow/eval.py --dataset kubric_short
python evaluation/flow/eval.py --dataset kubric_long
```

---

### 2. Point Cloud and Video Depth Estimation

We provide separate scripts for evaluating point cloud reconstruction and video depth estimation.

#### Point Cloud Evaluation

```bash
python evaluation/point_cloud/eval_pointcloud.py \
  --output evaluation/point_cloud/output/point/Sintel \
  --num_tokens 1200 \
  --gt-dataset-type Sintel
```

* **Supported Datasets (`--gt-dataset-type`):**
  `Tum`, `Sintel`, `Scannet`, `Monkaa`, `Kubric-3D`, `KITTI`, `GMUKitchens`

#### Video Depth Evaluation

```bash
python evaluation/point_cloud/eval_videodepth.py \
  --output evaluation/point_cloud/output/depth/Sintel \
  --num_tokens 1200 \
  --gt-dataset-type Sintel
```

* **Supported Datasets (`--gt-dataset-type`):**
  `Bonn`, `Sintel`, `Scannet`, `Monkaa`, `Kubric-3D`, `KITTI`, `GMUKitchens`

---

### 3. 3D Tracking Estimation

Evaluate 3D tracking performance by specifying the dataset and the number of frames.

```bash
python evaluation/track/eval.py \
  --dataset adt \
  --num_frames 16 \
  --world_eval
```

**Arguments:**

* `--dataset`: Choose from `['adt', 'ds', 'po', 'pstudio']`
* `--num_frames`: Choose from `[16, 50]`
* `--world_eval`: Evaluate results in the world coordinate system

---

### 4. 2D Tracking Estimation

Evaluate 2D tracking performance on TAP-Vid benchmarks (Kinetics, RGB-Stacking, RoboTAP).

```bash
python evaluation/2d_track/eval.py
```
