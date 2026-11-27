# KITTI Semantic Segmentation with UNet

A PyTorch implementation of semantic segmentation on the KITTI dataset using a UNet architecture. This project provides end-to-end training and inference capabilities for autonomous driving scene understanding.

## Features

 - UNet Architecture - Custom with skip connections
 - KITTI Dataset Utils
 - Data Augmentation
 - Efficient Training

## Results

 - Side by side segmented visualization
 - Overlay Visualization with transparency
 - Training Progress Video

### Segmentation Inference Results

#### Side by side visualization
<img src="results/vis_000079_10_result.png" alt="Segmentation Result 1" width="800"/>
<img src="results/data_comparison.gif" alt="Segmentation Result 2" width="800"/>

#### Overlay Results
<img src="results/000079_10_overlay.png" alt="Segmentation Result 3" width="800"/>
<img src="results/data_overlay.gif" alt="Segmentation Result 4" width="800"/>

#### Training Progress
<img src="results/training_progress.gif" alt="Segmentation Result 5" width="800"/>

## Installation

### Requirements

```bash
pip install torch torchvision
pip install albumentations
pip install matplotlib
pip install numpy
pip install Pillow
pip install tqdm
```

### Dataset Setup

  1. Download the KITTI semantic segmentation dataset
  2. Extract it into `data/` in the repo root, like this:
     ```text
     data/
      ├── data_semantics/
      │   ├── training/
      │   │   ├── image_2/        # RGB images
      │   │   └── semantic/       # Ground truth masks
      │   └── testing/
      │       └── image_2/        # Test images
      ├── devkit_semantics/       # KITTI devkit with label definitions
      └── raw/
          └── 2011_09_26/2011_09_26_drive_0018_sync/   # raw drive, only needed for video_inference.py
     ```

`data/` and `outputs/` are git-ignored. Checkpoints, plots and inference results from my first training run are in `outputs/v1/`. A new `python train.py` run writes to `outputs/train/`, so it doesn't overwrite them.

## Usage

### Training

```bash
python train.py
```

### Inference

```bash
python inference.py
```

## Acknowledgments

  - KITTI dataset creators for providing the benchmark
  - PyTorch team for the deep learning framework
  - Albumentations library for data augmentation tools
