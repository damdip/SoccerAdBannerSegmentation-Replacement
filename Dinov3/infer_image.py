from model import DINOv3Segmentation, hidden_size
from config import ALL_CLASSES
from utils import (
    draw_segmentation_map, 
    image_overlay,
    get_segment_labels,
    safe_torch_load,
    print_dinov3_dimension_info,
    calculate_dinov3_dimensions
)

import argparse
import cv2
import os
import glob
import torch.nn as nn
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    help='path to the input image directory',
    default='input/inference_data/images'
)
parser.add_argument(
    '--device',
    default='cuda:0',
    help='compute device, cpu or cuda'
)
parser.add_argument(
    '--imgsz', 
    default=None,
    type=int,
    nargs='+',
    help='width, height'
)
parser.add_argument(
    '--model',
    default='outputs/model_iou'
)
parser.add_argument(
    '--no-display',
    action='store_true',
    help='skip displaying images and just save results'
)
args = parser.parse_args()

out_dir = 'outputs/inference_results_image'
os.makedirs(out_dir, exist_ok=True)

model = DINOv3Segmentation()
model.decode_head.conv_seg = nn.Conv2d(hidden_size, len(ALL_CLASSES), kernel_size=(1, 1), stride=(1, 1))

# Load checkpoint with CPU mapping if CUDA is not available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_location = device if torch.cuda.is_available() else 'cpu'

# Load checkpoint safely
ckpt = safe_torch_load(args.model, map_location=map_location)
model.load_state_dict(ckpt['model_state_dict'])
_ = model.to(device).eval()

image_paths = glob.glob(os.path.join(args.input, '*'))
for image_path in image_paths:
    image = cv2.imread(image_path)
    
    # Handle image resizing with DINOv3 dimension requirements
    if args.imgsz is not None:
        requested_width, requested_height = args.imgsz[0], args.imgsz[1]
        
        # Show dimension analysis
        print(f"\n🖼️ Processing: {os.path.basename(image_path)}")
        print_dinov3_dimension_info(requested_width, requested_height)
        
        # Calculate optimal dimensions
        optimal_width, optimal_height = calculate_dinov3_dimensions(requested_width, requested_height)
        
        if (optimal_width, optimal_height) != (requested_width, requested_height):
            print(f"📏 Adjusting dimensions: {requested_width}x{requested_height} → {optimal_width}x{optimal_height}")
            image = cv2.resize(image, (optimal_width, optimal_height))
        else:
            print(f"✅ Dimensions are already optimal: {requested_width}x{requested_height}")
            image = cv2.resize(image, (requested_width, requested_height))
    else:
        # If no size specified, show info for original dimensions
        h, w = image.shape[:2]
        print(f"\n🖼️ Processing: {os.path.basename(image_path)} (original {w}x{h})")
        print_dinov3_dimension_info(w, h)
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get labels with actual image size
    current_size = [image.shape[1], image.shape[0]]  # [width, height]
    labels = get_segment_labels(image, model, device, current_size)
    
    # Get segmentation map.
    seg_map = draw_segmentation_map(labels.cpu())

    outputs = image_overlay(image, seg_map)
    
    if not args.no_display:
        cv2.imshow('Image', outputs)
        cv2.waitKey(0)
    
    # Save path.
    image_name = image_path.split(os.path.sep)[-1]
    save_path = os.path.join(
        out_dir, '_'+image_name
    )
    cv2.imwrite(save_path, outputs)
    print(f"✅ Processed and saved: {save_path}")