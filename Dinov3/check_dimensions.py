#!/usr/bin/env python3
"""
Script to check and analyze dataset dimensions for DINOv3 compatibility.
"""

import os
from PIL import Image
from utils import validate_dinov3_dimensions, calculate_optimal_dimensions

def check_dataset_dimensions(images_dir, masks_dir):
    """
    Check all images and masks in the dataset for DINOv3 dimension compatibility.
    """
    print("🔍 Checking dataset dimensions for DINOv3 compatibility...\n")
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith('.png')]
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    dimensions_summary = {}
    incompatible_files = []
    
    # Check first 10 files as sample
    sample_files = image_files[:10] if len(image_files) >= 10 else image_files
    
    for img_file in sample_files:
        img_path = os.path.join(images_dir, img_file)
        
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                
                # Check if dimensions are compatible
                is_valid = validate_dinov3_dimensions(width, height)
                
                dimension_key = f"{width}x{height}"
                if dimension_key not in dimensions_summary:
                    dimensions_summary[dimension_key] = {
                        'count': 0,
                        'valid': is_valid,
                        'optimal': calculate_optimal_dimensions(width, height) if not is_valid else (width, height)
                    }
                dimensions_summary[dimension_key]['count'] += 1
                
                if not is_valid:
                    incompatible_files.append(img_file)
                    
        except Exception as e:
            print(f"❌ Error reading {img_file}: {e}")
    
    # Print summary
    print("\n📊 Dimension Summary:")
    print("=" * 60)
    for dim, info in dimensions_summary.items():
        status = "✅ Compatible" if info['valid'] else "❌ Incompatible"
        print(f"{dim}: {info['count']} files - {status}")
        
        if not info['valid']:
            optimal = info['optimal']
            print(f"   Optimal: {optimal[0]}x{optimal[1]}")
    
    if incompatible_files:
        print(f"\n⚠️ Found {len(incompatible_files)} incompatible files (showing first 5):")
        for file in incompatible_files[:5]:
            print(f"   - {file}")
    else:
        print("\n✅ All checked files are compatible with DINOv3!")
    
    return dimensions_summary

def suggest_resize_parameters():
    """
    Suggest optimal resize parameters for the dataset.
    """
    print("\n💡 Resize Recommendations:")
    print("=" * 60)
    
    common_sizes = [(512, 512), (384, 384), (256, 256)]
    
    for width, height in common_sizes:
        is_valid = validate_dinov3_dimensions(width, height)
        status = "✅ Compatible" if is_valid else "❌ Incompatible"
        print(f"{width}x{height}: {status}")
        
        if not is_valid:
            optimal = calculate_optimal_dimensions(width, height)
            print(f"   Optimal: {optimal[0]}x{optimal[1]}")

if __name__ == "__main__":
    # Check if dataset directories exist
    images_dir = "Dataset_Cleaned/Tagged Images"
    masks_dir = "Dataset_Cleaned/Masks"
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        exit(1)
        
    if not os.path.exists(masks_dir):
        print(f"❌ Masks directory not found: {masks_dir}")
        exit(1)
    
    # Run dimension check
    dimensions_summary = check_dataset_dimensions(images_dir, masks_dir)
    
    # Provide suggestions
    suggest_resize_parameters()
    
    print("\n📋 Next Steps:")
    print("1. If images are incompatible, update resize parameters in config.py")
    print("2. Use utils.calculate_optimal_dimensions() to find best sizes")
    print("3. Test training with compatible dimensions")
