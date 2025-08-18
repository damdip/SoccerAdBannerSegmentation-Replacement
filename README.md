# SoccerAdBannerSegmentation-Replacement
Deep Learning project about ad banner detection and replacement in soccer matches videos.

Code Base from: https://debuggercafe.com/dinov2-for-semantic-segmentation


# DINOv2 for Semantic Segmentation

This project implements semantic segmentation using Facebook's DINOv2 vision transformer as the backbone. It's designed to perform pixel-wise classification of images, where each pixel is assigned to a specific class.

## Project Structure

```
├── config.py           # Configuration settings and class definitions
├── datasets.py         # Dataset loading and transformations
├── engine.py          # Training and validation loops
├── model.py           # DINOv2 model architecture
├── train.py           # Main training script
├── utils.py           # Utility functions
├── infer_image.py     # Script for inference on single images
├── infer_video.py     # Script for inference on video files
└── requirements.txt    # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Place your data in the following structure:
```
input/
└── dataset/
    ├── train_images/
    ├── train_masks/
    ├── valid_images/
    └── valid_masks/
```

## Training

To train the model, use the following command:

```bash
python train.py --lr 0.001 --batch 20 --imgsz 1914 1074 --epochs 10 --scheduler --scheduler-epochs 8
```

Parameters:
- `--lr`: Learning rate (default: 0.001)
- `--batch`: Batch size (default: 20)
- `--imgsz`: Input image dimensions (width height)
- `--epochs`: Number of training epochs
- `--scheduler`: Enable learning rate scheduler
- `--scheduler-epochs`: Epoch to apply learning rate reduction

## Inference

For single image inference:
```bash
python infer_image.py
```

For video inference:
```bash
python infer_video.py
```

## Model Architecture

The model uses DINOv2 as a backbone feature extractor, followed by a segmentation head for pixel-wise classification. The architecture is optimized for processing images with dimensions that are multiples of 14 (DINOv2's patch size).

## Outputs

Training outputs are saved in the `outputs/` directory:
- Model checkpoints (best model based on loss and IoU)
- Training plots (loss, accuracy, mIoU)
- Validation predictions
- Inference results

## Requirements

Main dependencies:
- torch
- torchvision
- albumentations
- opencv-python
- tqdm
- matplotlib

## Notes

- The model expects input dimensions to be multiples of 14 (DINOv2's patch size) minus padding (4 pixel for each side)
- xFormers optimization is optional but can improve training performance if available



Generated with Claude Sonnet :)
