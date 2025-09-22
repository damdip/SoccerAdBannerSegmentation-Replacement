# Soccer Ad Banner Segmentation & Replacement

This project implements semantic segmentation for detecting advertising banners in soccer footage using state-of-the-art Vision Transformer models (DINOv2 and DINOv3). The system can accurately identify TV monitor/banner areas in soccer images and videos, enabling automated advertisement replacement.

## 🎯 Project Overview

The project focuses on **binary semantic segmentation** to distinguish between:
- **Background**: General soccer field, players, and other elements
- **TV Monitor/Banner**: Advertising displays and banners around the field

This enables applications like:
- Automated advertisement replacement in sports broadcasts
- Content analysis for sports marketing
- Real-time banner detection and tracking

## 📁 Project Structure

```
SoccerAdBannerSegmentation-Replacement/
├── README                          # This comprehensive guide
├── Dinov2/                        # DINOv2 implementation (legacy)
│   ├── model.py                   # DINOv2 segmentation model
│   ├── train.py                   # Training script
│   ├── infer_image.py            # Image inference
│   ├── config.py                 # Configuration settings
│   ├── datasets.py               # Data loading utilities
│   ├── engine.py                 # Training/validation loops
│   ├── utils.py                  # Helper functions
│   ├── requirements.txt          # Dependencies
│   ├── README.md                 # DINOv2 specific documentation
│   └── inference_results_image/   # Sample results
├── Dinov2-few-shot/              # Few-shot learning experiments
│   └── start.txt                 # Placeholder for future work
└── Dinov3/                       # DINOv3 implementation (recommended)
    ├── model.py                  # DINOv3 segmentation model
    ├── train.py                  # Training script with enhanced features
    ├── infer_image.py           # Image inference with improved performance
    ├── config.py                # Configuration settings
    ├── datasets.py              # Data loading with augmentations
    ├── engine.py                # Training/validation loops
    ├── utils.py                 # Utility functions
    ├── requirements.txt         # Updated dependencies
    ├── README.md                # DINOv3 specific documentation
    ├── input/                   # Input data directory
    │   └── inference_images/    # Test images
    └── outputs/                 # Training and inference 
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ GPU memory for training

### Installation


 **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   For CUDA 11.8 (optimal performance):
   ```bash
   pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt --no-deps
   ```

### Dataset Setup

Organize your data in the following structure:

```
input/
└── dataset/
    ├── train_images/          # Training images
    ├── train_masks/           # Training binary masks (0=background, 255=banner)
    ├── valid_images/          # Validation images
    └── valid_masks/           # Validation binary masks
```

**Mask Format:**
- Background pixels: `(0, 0, 0)` - Black
- Banner/TV monitor pixels: `(255, 255, 255)` - White

## 🏋️ Training

### DINOv3 Training (Recommended)

```bash
cd Dinov3

# Quick training (small images)
python train.py --lr 0.001 --batch 4 --imgsz 224 224 --epochs 10

# High-quality training (larger images)
python train.py --lr 0.001 --batch 2 --imgsz 384 384 --epochs 15 --scheduler --scheduler-epochs 12

# Production training (full resolution)
python train.py --lr 0.0005 --batch 1 --imgsz 512 512 --epochs 20 --scheduler --scheduler-epochs 15
```

### DINOv2 Training (Legacy)

```bash
cd Dinov2

# Standard training
python train.py --lr 0.001 --batch 20 --imgsz 1914 1074 --epochs 10 --scheduler --scheduler-epochs 8
```

### Training Parameters

- `--lr`: Learning rate (0.0001-0.001)
- `--batch`: Batch size (adjust based on GPU memory)
- `--imgsz`: Input dimensions [width height] (must be multiples of 16 for DINOv3)
- `--epochs`: Number of training epochs
- `--scheduler`: Enable learning rate scheduling
- `--scheduler-epochs`: Epoch to reduce learning rate

## 🔮 Inference

### Single Image Inference

```bash
# DINOv3
cd Dinov3
python infer_image.py --imgsz 224 224 --model outputs/best_model_iou.pth --input input/inference_images

# DINOv2
cd Dinov2
python infer_image.py --model outputs/best_model.pth --input path/to/images
```

### Batch Inference

Place multiple images in the input directory and run the inference script. Results will be saved in `outputs/inference_results_image/` with overlay visualizations.

## 🏗️ Model Architecture

### DINOv3 (Current)

- **Backbone**: Meta's DINOv3 Vision Transformer
- **Available sizes**: 
  - Small: ViT-S/16 (21M params, 384 hidden dims)
  - Base: ViT-B/16 (86M params, 768 hidden dims)
  - Large: ViT-L/16 (300M params, 1024 hidden dims)
- **Patch size**: 16x16 pixels
- **Features**: Register tokens for enhanced feature quality
- **Input**: Flexible resolution (multiples of 16)

### DINOv2 (Legacy)

- **Backbone**: Facebook's DINOv2 Vision Transformer
- **Patch size**: 14x14 pixels
- **Input**: Fixed or flexible resolution
- **Performance**: Good baseline performance

### Segmentation Head

Both models use a simple yet effective segmentation head:
1. Linear projection from transformer features
2. Bilinear upsampling to original resolution
3. Binary classification (background vs banner)

## 📊 Performance Monitoring

Training outputs comprehensive metrics:

- **Loss plots**: Training and validation loss curves
- **Accuracy**: Pixel-wise classification accuracy
- **mIoU**: Mean Intersection over Union
- **Model checkpoints**: Best models saved by loss and IoU

All plots and checkpoints are saved in the `outputs/` directory.

## 🔧 Configuration

Modify `config.py` to adjust:

```python
ALL_CLASSES = ['background', 'tvmonitor']  # Class names
LABEL_COLORS_LIST = [(0, 0, 0), (255, 255, 255)]  # Mask colors
VIS_LABEL_MAP = [(0, 0, 0), (0, 255, 0)]  # Visualization colors
```

## 🎨 Results

The system produces:

1. **Segmentation masks**: Binary masks identifying banner regions
2. **Overlay visualizations**: Original image with colored overlay
3. **Performance metrics**: Quantitative evaluation results

Sample results are available in the `inference_results_image/` directories showing successful banner detection in various soccer scenarios.

## 🔄 Migration Guide (DINOv2 → DINOv3)

Key improvements in DINOv3:

- **Better dense features** for segmentation tasks
- **Simplified API** using Hugging Face Transformers
- **Register tokens** for enhanced feature quality
- **Improved scaling** for different input resolutions
- **Better memory efficiency**

To migrate existing models, retrain with DINOv3 using similar hyperparameters but adjusted batch sizes.

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or image dimensions
2. **Model download fails**: Check internet connection and Hugging Face access
3. **Poor segmentation quality**: Increase training epochs or use larger model
4. **Input dimension errors**: Ensure dimensions are multiples of 16 (DINOv3)

### Performance Tips

- Use GPU for training (CPU inference is possible but slow)
- Start with smaller models and scale up based on results
- Monitor GPU memory usage and adjust batch size accordingly
- Use mixed precision training for memory efficiency


**Note**: This project is designed for research and educational purposes. For production use, ensure proper testing and validation on your specific data.
