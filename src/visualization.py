import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_training_curves(metrics_csv: str, output_dir: str):
    """Generates loss and mIoU curves from the metrics CSV."""
    if not os.path.exists(metrics_csv):
        return
        
    df = pd.read_csv(metrics_csv)
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'train_loss.png'))
    plt.close()
    
    # mIoU plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_iou'], label='Train mIoU')
    plt.plot(df['epoch'], df['val_iou'], label='Val mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('Training and Validation mIoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'miou_curve.png'))
    plt.close()

def generate_overlay(image_path: str, mask: np.ndarray, output_path: str, alpha: float = 0.5):
    """Blends an 8-class multiclass mask onto the original image and saves it."""
    import rasterio
    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3]).transpose(1, 2, 0)
        
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    overlay = image.copy()
    color_mask = np.zeros_like(image)
    
    # 8-Class Color Mapping (RGB)
    colors = {
        0: [255, 0, 0],     # Building (Red)
        1: [0, 255, 0],     # Tree (Lime)
        2: [0, 0, 255],     # Parking (Blue)
        3: [255, 255, 0],   # Shrub (Yellow)
        4: [0, 255, 255],   # Turf (Cyan)
        5: [255, 0, 255],   # Road (Magenta)
        6: [192, 192, 192], # Sidewalk (Silver)
        7: [0, 128, 128]    # Water (Teal)
    }
    
    for class_id, color in colors.items():
        color_mask[mask == class_id] = color
        
    cv2.addWeighted(color_mask, alpha, overlay, 1 - alpha, 0, overlay)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, overlay_bgr)

def generate_difference_map(pred_mask: np.ndarray, gt_mask: np.ndarray, output_path: str):
    """Generates an error map: TP=Green, FP=Red, FN=Blue."""
    h, w = pred_mask.shape
    diff_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    # True Positives (Green)
    diff_map[(pred_mask == 1) & (gt_mask == 1)] = [0, 255, 0]
    # False Positives (Red)
    diff_map[(pred_mask == 1) & (gt_mask == 0)] = [255, 0, 0]
    # False Negatives (Blue)
    diff_map[(pred_mask == 0) & (gt_mask == 1)] = [0, 0, 255]
    
    diff_map_bgr = cv2.cvtColor(diff_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, diff_map_bgr)
