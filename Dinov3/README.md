# DINOv3 for Semantic Segmentation

This project implements semantic segmentation using Meta's DINOv3 vision transformer as the backbone. It's designed to perform pixel-wise classification of images, where each pixel is assigned to a specific class.

## 🔄 Migration from DINOv2 to DINOv3

This project has been migrated from DINOv2 to DINOv3, bringing several improvements:

- **Better Performance**: DINOv3 offers superior dense features for segmentation tasks
- **Simplified API**: Using Hugging Face Transformers for easier model loading
- **Register Tokens**: Enhanced feature quality with learnable register tokens
- **Multiple Architectures**: Support for both ViT and ConvNeXt backbones
- **Better Scaling**: Improved handling of different input resolutions

### Key Changes in Migration:
- Replaced `torch.hub.load` with Hugging Face `AutoModel` 
- Updated feature extraction to handle DINOv3's output format
- Adapted to new patch token structure (CLS + register tokens + patch tokens)
- Optimized for patch size 16 (instead of 14 in DINOv2)

## Project Structure

```
├── config.py               # Configuration settings and class definitions
├── datasets.py             # Dataset loading and transformations  
├── engine.py               # Training and validation loops
├── model.py                # DINOv3 model architecture 
├── train.py                # Main training script
├── utils.py                # Utility functions
├── infer_image.py          # Script for inference on single images
├── infer_video.py          # Script for inference on video files
├── test_dinov3_migration.py # Test script to verify migration
└── requirements.txt         # Project dependencies
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
# For standard training
python train.py --lr 0.001 --batch 4 --imgsz 224 224 --epochs 10

# For larger images (multiples of 16)
python train.py --lr 0.001 --batch 2 --imgsz 384 384 --epochs 10 --scheduler --scheduler-epochs 8
```

Parameters:
- `--lr`: Learning rate (default: 0.0001)
- `--batch`: Batch size (default: 4)
- `--imgsz`: Input image dimensions (width height) - should be multiples of 16
- `--epochs`: Number of training epochs
- `--scheduler`: Enable learning rate scheduler
- `--scheduler-epochs`: Epoch to apply learning rate reduction

## Inference

For single image inference:
```bash
python infer_image.py --imgsz 224 224 --model outputs/best_model_iou.pth
```

## Model Architecture

The model uses DINOv3 as a backbone feature extractor, followed by a segmentation head for pixel-wise classification. 

### DINOv3 Backbone Options:
- **small**: ViT-S/16 (21M parameters, 384 hidden dimensions)
- **base**: ViT-B/16 (86M parameters, 768 hidden dimensions)  
- **large**: ViT-L/16 (300M parameters, 1024 hidden dimensions)

You can change the backbone in `model.py` by modifying `BACKBONE_SIZE`.

### Architecture Details:
- Patch size: 16 (vs 14 in DINOv2)
- Register tokens: 4 learnable tokens for better feature quality
- Input resolution: Flexible, should be multiples of 16
- Output resolution: Input resolution / 16

## Outputs

Training outputs are saved in the `outputs/` directory:
- Model checkpoints (best model based on loss and IoU)
- Training plots (loss, accuracy, mIoU)
- Validation predictions  
- Inference results

## Requirements

Main dependencies:
- torch >= 2.7.1
- torchvision
- transformers >= 4.43.0 (for DINOv3 support)
- safetensors >= 0.4.5
- albumentations
- opencv-python
- tqdm
- matplotlib

## Performance Tips

1. **Batch Size**: Start with smaller batch sizes (2-4) for larger models
2. **Input Size**: Use multiples of 16 (224, 256, 320, 384, etc.)
3. **Memory**: DINOv3 requires more memory than DINOv2
4. **Training**: Consider using gradient checkpointing for memory efficiency

## Notes

- The model expects input dimensions to be multiples of 16 (DINOv3's patch size)
- DINOv3 models require agreement to license terms for access
- For the first run, models will be downloaded from Hugging Face Hub
- CUDA is recommended but the model works on CPU for inference
