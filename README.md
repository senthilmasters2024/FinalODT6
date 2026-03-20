# RGB-D Object Detection & Segmentation for Autonomous Mobile Robotics

> **Comparative Evaluation of DETR and SegFormer for Real-Time Water Bottle Detection using Intel RealSense D435**
>
> ODT-6 — Frankfurt University of Applied Sciences | Course: Autonomous & Intelligent Systems

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Hardware Requirements](#3-hardware-requirements)
4. [Installation](#4-installation)
5. [Dataset Preparation](#5-dataset-preparation)
6. [Training the Models](#6-training-the-models)
7. [Running the Detectors](#7-running-the-detectors)
8. [ROS 2 Integration](#8-ros-2-integration)
9. [Recording Detection Sessions](#9-recording-detection-sessions)
10. [Understanding the Reports](#10-understanding-the-reports)
11. [Model Performance Summary](#11-model-performance-summary)
12. [Configuration Reference](#12-configuration-reference)
13. [Troubleshooting](#13-troubleshooting)
14. [References](#14-references)

---

## 1. Project Overview

This project implements and compares two transformer-based vision architectures — **DETR** (Detection Transformer) and **SegFormer** — for detecting everyday objects (primarily water bottles) using an **Intel RealSense D435** RGB-D camera. It was developed for the **ROSWITHA** autonomous mobile robot platform to support assistive robotics in indoor hospital-like environments.

### What the system does

- Captures synchronized **RGB + Depth** frames from the RealSense D435
- Runs real-time inference using one of three detection pipelines:
  - **Pipeline 1 — DETR (Configurable):** Pretrained DETR on COCO; target classes controlled via a JSON config file
  - **Pipeline 2 — DETR (All classes):** Pretrained DETR detecting all 80 COCO classes simultaneously
  - **Pipeline 3 — DETR Cascade (Custom-trained):** Two fine-tuned models in a cascade — M1 detects bottles, M2 rejects hard-negative false positives
  - **Pipeline 4 — SegFormer:** Custom-trained semantic segmentation model; converts pixel masks to bounding boxes
- Estimates **object depth** (distance in metres) for each detection
- Exports per-session **performance reports** (CSV + JSON + plots)
- Optionally publishes results to **ROS 2 topics** for robotic integration

### Why two model families?

| Aspect | DETR | SegFormer |
|---|---|---|
| Output | Bounding boxes | Pixel-level segmentation masks → boxes |
| Strength | Global context via self-attention, no NMS needed | Dense scene understanding, fast inference |
| Best for | Task-specific object detection | Cluttered scenes, partial occlusion |
| Inference speed | ~6–11 FPS (CPU/GPU) | ~25 FPS (GPU) |

The SegFormer-B2 model achieved the best overall F1-score and the fastest inference in evaluation, while the Cascade DETR achieved the highest per-detection precision.

---

## 2. Repository Structure

```
FinalODT6-main/
│
├── DETR/                               # DETR pipeline scripts
│   ├── realsensedetectionwithDETR.py   # Pipeline 1: pretrained DETR, class-filtered
│   ├── realsensedetectionwithDETR_all.py  # Pipeline 2: pretrained DETR, all classes
│   ├── realsenseensemble.py            # Pipeline 3: cascade (M1 + M2)
│   ├── fine_tune.py                    # Train M1 — bottle detector
│   ├── trainhardnegatives.py           # Train M2 — hard-negative rejector
│   ├── detector_config.json            # Runtime config for Pipeline 1
│   ├── water_bottle_model/             # Saved M1 model weights (HuggingFace format)
│   │   └── huggingface_model/
│   ├── water_bottle_model_hard_negatives/  # Saved M2 model weights
│   │   └── huggingface_model/
│   └── report_YYYYMMDD_HHMMSS/         # Auto-generated session reports
│       ├── frames.csv                  # Per-frame metrics
│       ├── session_summary.json        # Session-level summary
│       ├── fps_and_inference_time.png
│       ├── confidence_distribution.png
│       ├── detections_per_class.png
│       └── avg_confidence_per_class.png
│
├── SegFormer/                          # SegFormer pipeline scripts
│   ├── SegFormer.py                    # Pipeline 4: real-time SegFormer inference
│   ├── TrainSegFormerV2.py             # Train SegFormer-B2 on custom dataset
│   ├── water_bottle_segformer_model_v2/  # Saved SegFormer weights
│   │   ├── huggingface_model/          # Latest checkpoint
│   │   └── huggingface_model_best/     # Best checkpoint (highest bottle IoU)
│   └── segformer_report_YYYYMMDD_HHMMSS/  # Auto-generated session reports
│       ├── frames.csv
│       ├── session_summary.json
│       ├── fps_and_inference_time.png
│       ├── confidence_distribution.png
│       ├── detections_per_frame.png
│       ├── depth_distribution.png
│       └── precision_recall_f1.png
│
├── clean_all3_hardnegatives_only.py    # Dataset preprocessing utility
├── detection_recorder.py               # ROS 2 detection recorder node
├── requirements_detr.txt               # Python dependencies
└── README.md                           # This file
```

> **Note on `my_dataset/`:** The training dataset is **not included** in this repository (see [Dataset Preparation](#6-dataset-preparation)). You must create this folder manually before training.

---

## 4. Hardware Requirements

| Component | Requirement |
|---|---|
| RGB-D Camera | **Intel RealSense D435** |
| USB | USB 3.0 (blue port) — required for full bandwidth |
| GPU | NVIDIA GPU strongly recommended (CUDA ≥ 11.7) |
| RAM | ≥ 8 GB (16 GB recommended for training) |
| OS | Ubuntu 20.04 / 22.04 (also tested on Windows 10) |
| ROS 2 | Humble Hawksbill (optional, for robotic integration) |

### Verify your RealSense camera

Before running any scripts, open **Intel RealSense Viewer** and confirm the D435 streams are working. If the camera is not detected, check the USB 3.0 connection and update the firmware from within the Viewer.

---

## 5. Installation

### Step 1 — Clone the repository

```bash
git clone <your-repo-url>
cd FinalODT6-main
```

### Step 2 — Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements_detr.txt
```

This installs:

| Package | Purpose |
|---|---|
| `pyrealsense2` | RealSense D435 camera SDK |
| `torch` + `torchvision` | Deep learning framework |
| `transformers` | HuggingFace DETR and SegFormer models |
| `opencv-python` | Video capture and visualization |
| `Pillow` | Image loading and preprocessing |
| `numpy` | Numerical operations |

For SegFormer training you also need:

```bash
pip install scikit-learn tqdm matplotlib
```

### Step 4 — Verify GPU availability (optional but recommended)

```python
import torch
print(torch.cuda.is_available())  # Should print True if CUDA is set up
print(torch.cuda.get_device_name(0))
```

---

## 6. Dataset Preparation

The custom dataset is **not included** in this repository. You need to prepare it yourself following the structure below.

### Required folder layout

```
my_dataset/
├── rgb/                          # Positive images — bottles present
│   ├── img_0001.jpg
│   ├── img_0002.jpg
│   └── ...
├── All-1.json                    # COCO annotations for positive images
│                                 # (bounding boxes + polygon segmentations)
├── TrainingDatasetNoBottle/
│   └── rgb/                      # Negative images — NO bottles
│       ├── neg_0001.jpg
│       └── ...
├── All-3.json                    # Raw COCO annotations for negative images
└── All-3_clean_bbox_only.json    # Cleaned version — generated by script below
```

### Dataset statistics used in this project

| Split | Images | Description |
|---|---|---|
| Positive (bottles) | 672 | Bottles under varying lighting, angles, clutter |
| Negative (no bottles) | 77 | Scenes with visually similar objects (blue objects, containers) |

### Annotating your own images

1. Capture RGB images using the RealSense D435 (or any camera)
2. Upload to **[Datatorch](https://datatorch.io)** or any COCO-compatible annotation tool
3. Label with two classes: `water_bottle` and `no_Object`
4. Export as **COCO JSON format**
5. Place exported files at `my_dataset/All-1.json` (positives) and `my_dataset/All-3.json` (negatives)

### Clean the negative annotation file

Before training, run the preprocessing utility to strip polygon annotations from the negative set and remap category IDs:

```bash
python clean_all3_hardnegatives_only.py
```

This reads `my_dataset/All-3.json` and writes `my_dataset/All-3_clean_bbox_only.json`. You must run this **once** before training the hard-negative model or SegFormer.

**What it does:**
- Keeps only annotations with `category_id == 2` (the `no_Object` class)
- Removes annotations with zero-area bounding boxes
- Strips polygon segmentation fields (not needed for DETR training)
- Remaps category IDs to `0` and renames the class to `no-object`

---

## 7. Training the Models

There are three training scripts. Run them in the order shown below.

### 7.1 — Train M1: Bottle Detector (DETR)

**Script:** `DETR/fine_tune.py`

Fine-tunes `facebook/detr-resnet-50` on your positive bottle images.

**Before running, edit the paths at the top of the file:**

```python
IMAGES_DIR       = "./my_dataset/rgb"
ANNOTATIONS_FILE = "./my_dataset/All-1.json"
SAVE_DIR         = "./water_bottle_model"

EPOCHS       = 50
BATCH_SIZE   = 2
LEARNING_RATE = 1e-5
SAVE_EVERY   = 10   # save checkpoint every N epochs
```

**Run:**

```bash
cd DETR
python fine_tune.py
```

**Output:** Checkpoints saved to `DETR/water_bottle_model/huggingface_model/` in HuggingFace format (loadable with `from_pretrained()`).

**Training tips:**
- On a GPU, 50 epochs over ~650 images takes roughly 1–2 hours
- Lower `BATCH_SIZE` to `1` if you run out of VRAM
- Increase `EPOCHS` to 100 if validation loss is still decreasing at epoch 50

---

### 7.2 — Train M2: Hard-Negative Rejector (DETR)

**Script:** `DETR/trainhardnegatives.py`

Trains a second DETR model on the negative (no-bottle) images. This model learns what false-positive regions look like so they can be suppressed in the cascade pipeline.

**Before running, edit the paths:**

```python
IMAGES_DIR       = "./my_dataset/TrainingDatasetNoBottle/rgb"
ANNOTATIONS_FILE = "./my_dataset/All-3_clean_bbox_only.json"
SAVE_DIR         = "./water_bottle_model_hard_negatives"

EPOCHS     = 30
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
```

**Run:**

```bash
cd DETR
python trainhardnegatives.py
```

**Output:** Checkpoints saved to `DETR/water_bottle_model_hard_negatives/huggingface_model/`.

> **Important:** Run `clean_all3_hardnegatives_only.py` from the root directory first, or this script will fail to find valid annotations.

---

### 7.3 — Train SegFormer-B2 (Segmentation Model)

**Script:** `SegFormer/TrainSegFormerV2.py`

Fine-tunes `nvidia/segformer-b2-finetuned-ade-512-512` for binary bottle segmentation using both positive and negative examples.

**Before running, verify these paths at the top of the file:**

```python
POSITIVE_IMAGES_DIR  = "../my_dataset/rgb"
POSITIVE_ANNOTATIONS = "../my_dataset/All-1.json"
NEGATIVE_IMAGES_DIR  = "../my_dataset/TrainingDatasetNoBottle/rgb"
NEGATIVE_ANNOTATIONS = "../my_dataset/All-3_clean_bbox_only.json"
SAVE_DIR = "./water_bottle_segformer_model_v2"
```

**Key training hyperparameters:**

```python
EPOCHS      = 80
BATCH_SIZE  = 4
LR          = 6e-5
IMAGE_SIZE  = (512, 512)
TRAIN_SPLIT = 0.85     # 85% train, 15% validation
```

**Run:**

```bash
cd SegFormer
python TrainSegFormerV2.py
```

**Output:** Two model directories are saved:
- `water_bottle_segformer_model_v2/huggingface_model/` — latest epoch checkpoint
- `water_bottle_segformer_model_v2/huggingface_model_best/` — best checkpoint by bottle-class IoU ← **use this for inference**

**What the training script does:**
- Loads polygon annotations from COCO JSON and converts them to binary masks
- Combines positive and negative samples in a single DataLoader
- Applies data augmentation: horizontal flip, brightness/contrast jitter, random crop
- Uses **weighted cross-entropy loss** to handle the class imbalance between bottle pixels and background
- Saves the best model based on bottle-class IoU (not mean IoU)
- Monitors validation metrics after every epoch

**Training tips:**
- On a GPU, 80 epochs over ~750 images takes roughly 2–4 hours
- Reduce `BATCH_SIZE` to `2` if you run out of VRAM
- The `huggingface_model_best/` checkpoint is typically significantly better than the final epoch — always use it for inference

---

## 8. Running the Detectors

All detection scripts stream live from the D435 and display an annotated window. Press **`q`** to quit and save the session report.

### Pipeline 1 — Pretrained DETR with class filter

Runs the pretrained `facebook/detr-resnet-50` model and limits detections to classes defined in `detector_config.json`.

```bash
cd DETR
python realsensedetectionwithDETR.py
```

To change the target class, edit `DETR/detector_config.json`:

```json
{
  "detect_only": ["bottle"],
  "confidence_threshold": 0.7,
  "inference_every_n_frames": 3,
  "model": "facebook/detr-resnet-50"
}
```

- Set `"detect_only"` to `null` to detect all 80 COCO classes
- Adjust `"confidence_threshold"` between `0.5` (more detections) and `0.9` (higher precision)
- Increase `"inference_every_n_frames"` to `5` or `10` on slow hardware

### Pipeline 2 — Pretrained DETR, all classes

```bash
cd DETR
python realsensedetectionwithDETR_all.py
```

Detects all 80 COCO classes simultaneously. Useful for general scene understanding.

### Pipeline 3 — Cascade DETR (custom-trained, recommended for bottles)

Requires that M1 and M2 training have been completed first.

```bash
cd DETR
python realsenseensemble.py
```

**How the cascade works:**

```
Frame input
    │
    ▼
M1 (bottle detector)  →  candidate detections
    │
    ▼
M2 (hard-negative rejector)  →  flags false-positive regions
    │
    ▼
IoU overlap check:  if M1 detection overlaps M2 flag → REJECT
    │
    ▼
Final detections  (high precision, reduced false positives)
```

The model paths are hardcoded at the top of `realsenseensemble.py`. If you changed the `SAVE_DIR` in training, update these lines accordingly:

```python
M1_MODEL_PATH = "./water_bottle_model/huggingface_model"
M2_MODEL_PATH = "./water_bottle_model_hard_negatives/huggingface_model"
```

### Pipeline 4 — SegFormer (custom-trained, best F1)

Requires that SegFormer training has been completed first.

```bash
cd SegFormer
python SegFormer.py
```

The script loads `water_bottle_segformer_model_v2/huggingface_model_best/` by default. To use the latest checkpoint instead, edit the model path near the top of `SegFormer.py`.

**ROS 2 mode (optional):** If ROS 2 Humble is installed and sourced, SegFormer.py will automatically detect it and start publishing to `/segformer/detections`. See [Section 9](#9-ros-2-integration) for details.

---

## 9. ROS 2 Integration

All four detection scripts include an optional ROS 2 publishing layer. It activates automatically if `rclpy` is importable (i.e., ROS 2 is installed and sourced).

### Topics published by each pipeline

| Pipeline | Topic prefix | Detections | Depths | Annotated image |
|---|---|---|---|---|
| DETR (configurable) | `/detr` | `/detr/detections` | `/detr/detection_depths` | `/detr/annotated_image` |
| DETR (all classes) | `/detr_all` | `/detr_all/detections` | `/detr_all/detection_depths` | `/detr_all/annotated_image` |
| SegFormer | `/segformer` | `/segformer/detections` | `/segformer/detection_depths` | `/segformer/annotated_image` |

### Message types

| Topic | ROS 2 type |
|---|---|
| `/*/detections` | `vision_msgs/Detection2DArray` |
| `/*/detection_depths` | `std_msgs/Float32MultiArray` |
| `/*/annotated_image` | `sensor_msgs/Image` (BGR8) |

### Monitoring in real time

```bash
# List active topics
ros2 topic list

# Echo detection results
ros2 topic echo /segformer/detections

# View annotated image stream
ros2 run image_tools showimage --ros-args -r image:=/segformer/annotated_image
```

### Required ROS 2 packages

```bash
sudo apt install ros-humble-vision-msgs python3-cv-bridge
```

---

## 10. Recording Detection Sessions

`detection_recorder.py` is a standalone ROS 2 node that subscribes to any detection topic and saves all data to a timestamped JSON file — useful for logging, replay, and offline analysis.

### Usage

```bash
# Record from SegFormer (default: DETR)
python3 detection_recorder.py --source segformer

# Record from cascade DETR (not yet a separate pipeline name; use detr)
python3 detection_recorder.py --source detr

# Record from all three simultaneously
python3 detection_recorder.py --source all

# Also save annotated image frames to disk
python3 detection_recorder.py --source segformer --images

# Custom output folder name
python3 detection_recorder.py --source segformer --output my_hospital_session
```

Press **Ctrl+C** to stop — the JSON file is written automatically on exit.

### Output format

Each session produces a folder (e.g., `ros_recording_YYYYMMDD_HHMMSS/`) containing:

```
detections.json
├── source: "segformer"
├── session_start: "2026-03-13T14:49:01"
├── frames: [
│     { "timestamp": 0.03,
│       "detections": [
│           { "label": "water_bottle",
│             "confidence": 0.82,
│             "bbox": [x1, y1, x2, y2],
│             "depth_m": 1.21 }
│       ]
│     }, ...
│   ]
```

---

## 11. Understanding the Reports

Each detection script automatically generates a report folder when you quit (press `q`). Reports are saved inside the same directory as the script.

### DETR report contents

| File | Description |
|---|---|
| `frames.csv` | Per-frame: timestamp, inference_ms, FPS, M1 candidates, M2 rejections (cascade only), final detections, avg confidence |
| `session_summary.json` | Total frames, session duration, average FPS, total detections, overall avg confidence |
| `fps_and_inference_time.png` | Dual-axis plot: FPS and inference time over the session |
| `confidence_distribution.png` | Histogram of all detection confidence scores |
| `detections_per_class.png` | Bar chart of detection counts per class |
| `avg_confidence_per_class.png` | Average confidence score per class |

### SegFormer report contents

| File | Description |
|---|---|
| `frames.csv` | Per-frame: timestamp, inference_ms, FPS, detection count, avg confidence, avg depth |
| `session_summary.json` | Session summary including avg depth and avg object area |
| `fps_and_inference_time.png` | FPS and inference time over the session |
| `confidence_distribution.png` | Confidence histogram |
| `detections_per_frame.png` | Detection count over time |
| `depth_distribution.png` | Distribution of detected object depths (metres) |
| `precision_recall_f1.png` | Precision / recall / F1 if ground-truth evaluation was run |

---

## 12. Model Performance Summary

Results from evaluation sessions using the Intel RealSense D435 in an indoor lab environment.

| Model | Avg FPS | Avg Inference (ms) | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Pretrained DETR (filtered) | 11.48 | 86.93 | High | Low | — |
| Cascade DETR (custom M1+M2) | 6.72 | 147.87 | **0.987** | 0.570 | 0.722 |
| SegFormer-B2 (custom) | **25.87** | **38.5** | — | — | **Best** |

**Key findings:**

- The **Cascade DETR** achieves the highest per-detection precision (98.7%) but misses a portion of true detections (recall 57%). It rejected 43.9% of M1 candidates via M2, reducing false positives significantly.
- **SegFormer-B2** achieved the best overall F1-score with the fastest inference speed (~26 FPS), making it the most suitable pipeline for real-time deployment.
- Color bias (blue objects producing false positives) was the main challenge for DETR. The hard-negative training stage and cascade architecture substantially mitigate this.
- All custom-trained models significantly outperform the generic pretrained DETR for this specific task.

---

## 13. Configuration Reference

### `DETR/detector_config.json`

Controls Pipeline 1 (`realsensedetectionwithDETR.py`) at runtime — no code changes needed.

```json
{
  "detect_only": ["bottle"],
  "confidence_threshold": 0.7,
  "inference_every_n_frames": 3,
  "model": "facebook/detr-resnet-50"
}
```

| Field | Type | Description |
|---|---|---|
| `detect_only` | `list` or `null` | Class names to detect. Set to `null` for all 80 COCO classes. |
| `confidence_threshold` | `float` | Minimum confidence to show a detection. Range: `0.0`–`1.0`. |
| `inference_every_n_frames` | `int` | Run inference every N frames. `1` = every frame. Higher = faster display, lower detection rate. |
| `model` | `string` | HuggingFace model identifier. Change to `"facebook/detr-resnet-101"` for higher accuracy. |

### Camera resolution options

All DETR scripts try three resolutions in sequence during initialization:

| Resolution | FPS | Notes |
|---|---|---|
| 640 × 480 | 30 | Default fallback — fastest |
| 848 × 480 | 30 | Wide-angle alternative |
| 1280 × 720 | 30 | Best quality |

SegFormer uses a fixed 640 × 480 to ensure consistent input dimensions.

---

## 14. Troubleshooting

### Camera not detected

```
RuntimeError: No frames received from RealSense
```

- Ensure the D435 is plugged into a **USB 3.0 port** (blue)
- Open **RealSense Viewer** and verify streams are live
- Update D435 firmware via RealSense Viewer → Devices → Update Firmware

### Model download fails on first run

DETR and SegFormer base models are downloaded from HuggingFace (~160–230 MB) on the first run.

- Check internet connectivity
- If behind a proxy, set `HTTPS_PROXY` environment variable
- Models are cached at `~/.cache/huggingface/` after first download

### CUDA out of memory

```
RuntimeError: CUDA out of memory
```

- Reduce `BATCH_SIZE` to `1` in training scripts
- Switch camera resolution to 640 × 480
- Close other GPU-heavy applications
- The scripts automatically fall back to CPU if CUDA is unavailable

### Cascade DETR rejects everything

If M2 rejects almost all M1 detections, the M2 confidence threshold may be too low. In `realsenseensemble.py`, look for the M2 threshold and increase it slightly:

```python
M2_CONFIDENCE_THRESHOLD = 0.5  # raise to 0.6 or 0.7
```

### SegFormer not detecting bottles

- Make sure you are loading `huggingface_model_best/` not `huggingface_model/`
- Verify the model was trained for at least 40–50 epochs with stable bottle IoU
- Check that `clean_all3_hardnegatives_only.py` was run before training

### ROS 2 topics not appearing

- Source ROS 2: `source /opt/ros/humble/setup.bash`
- Verify the detection script printed `ROS 2 publishing active` on startup
- Install missing packages: `sudo apt install ros-humble-vision-msgs python3-cv-bridge`

---

## 15. References

1. Ren, S. et al. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." *NeurIPS*, 2015.
2. Redmon, J. et al. "You Only Look Once: Unified, Real-Time Object Detection." *CVPR*, 2016.
3. Dosovitskiy, A. et al. "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale." *ICLR*, 2021.
4. Carion, N. et al. "End-to-End Object Detection with Transformers (DETR)." *ECCV*, 2020. [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
5. Xie, E. et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." *NeurIPS*, 2021.
6. Gupta, S. et al. "Learning Rich Features from RGB-D Images for Object Detection and Segmentation." *ECCV*, 2014.
7. Dai, A. et al. "ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes." *CVPR*, 2017.
8. Quigley, M. et al. "ROS: An Open-Source Robot Operating System." *ICRA Workshop*, 2009.

**Tools & frameworks:**
- [Intel RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [ROS 2 Humble](https://docs.ros.org/en/humble/)
- [Datatorch Annotation Platform](https://datatorch.io)
- [COCO Dataset Format](https://cocodataset.org/#format-data)

---

## Authors

**Team ODT6** — Frankfurt University of Applied Sciences, Information Technology

- Senthil Arumugam Ramasamy
- Amrita Elizabeth
- Hibist Markos Gute

---

*This project was developed for research and educational purposes as part of the Autonomous & Intelligent Systems coursework.*
