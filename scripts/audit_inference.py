import os
import sys
import glob
import cv2
import numpy as np
import rasterio
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Ensure src is in path
sys.path.append(os.getcwd())

from src.data_ingestion import load_and_prepare_labels, discover_classes, find_geojson_files, find_image_files, match_images_to_labels
from src.geo_utils import get_affine_transform, get_bbox_transform
from src.rasterize import create_mask_from_gdf
from src.visualization import generate_overlay
from src.alignment import evaluate_alignment

def decode_colored_mask(colored_mask, colors):
    """Decodes a colored mask back to integer class labels using nearest neighbor color matching."""
    # colored_mask is RGB
    decoded = np.zeros(colored_mask.shape[:2], dtype=np.uint8)
    # Using simple exact matching because we generated these colors
    for class_id, color in colors.items():
        # color is [R, G, B]
        mask = (colored_mask == color).all(axis=-1)
        decoded[mask] = class_id
    return decoded

def calculate_iou(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=np.arange(num_classes))
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    
    # Avoid division by zero
    iou = np.divide(intersection, union.astype(np.float32), out=np.zeros_like(intersection, dtype=float), where=union != 0)
    
    return iou, ground_truth_set

def main():
    images = find_image_files('aerial_imagery_pack')
    labels = find_geojson_files('feature_layers')
    matched = match_images_to_labels(images, labels)
    
    class_mapping = discover_classes(labels)
    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    num_classes = len(class_mapping)
    
    colors = {0: [255,0,0], 1: [0,255,0], 2: [0,0,255], 3: [255,255,0], 4: [0,255,255], 5: [255,0,255], 6: [192,192,192], 7: [0,128,128]}
    
    os.makedirs('results/audit_real', exist_ok=True)
    
    all_ious = []
    all_gt_pixels = []
    
    for img_path, lbl_path in matched:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(f"Auditing {basename}...")
        
        # Load image & transform
        with rasterio.open(img_path) as src:
            img_np = src.read([1,2,3]).transpose(1, 2, 0)
            transform = get_affine_transform(img_path)
            
        # Load ground truth GeoJSON
        gdf = load_and_prepare_labels(lbl_path)
        
        # Handle alignment gate like LOOCV
        report = evaluate_alignment(img_path, lbl_path)
        if report['coordinate_system'] == "Pixel-space":
            transform = rasterio.transform.Affine.identity()
        elif report['confidence'] == 50:
            transform = get_bbox_transform(gdf.total_bounds, img_np.shape[1], img_np.shape[0])
            
        gt_mask = create_mask_from_gdf(gdf, transform, img_np.shape[:2], class_mapping=class_mapping)
        
        # Save Ground Truth Overlay for comparison
        gt_overlay_path = f"results/audit_real/{basename}_gt_overlay.png"
        generate_overlay(img_path, gt_mask, gt_overlay_path)
        
        # Save GT Mask colored
        gt_mask_colored = np.zeros_like(img_np)
        for class_id, color in colors.items():
            gt_mask_colored[gt_mask == class_id] = color
        cv2.imwrite(f"results/audit_real/{basename}_gt_mask.png", cv2.cvtColor(gt_mask_colored, cv2.COLOR_RGB2BGR))
        
        # Load Predicted Mask (colored)
        pred_colored_path = f"results/inference/{basename}_mask.png"
        if not os.path.exists(pred_colored_path):
            print(f"Warning: {pred_colored_path} not found. Skipping metrics.")
            continue
            
        # Read in BGR and convert to RGB to match colors dict
        pred_colored = cv2.imread(pred_colored_path)
        pred_colored = cv2.cvtColor(pred_colored, cv2.COLOR_BGR2RGB)
        
        pred_mask = decode_colored_mask(pred_colored, colors)
        
        # Calculate IoU
        iou, gt_pixels = calculate_iou(gt_mask, pred_mask, num_classes)
        all_ious.append(iou)
        all_gt_pixels.append(gt_pixels)
        
        print(f"  - mIoU: {np.nanmean(iou[gt_pixels > 0]):.4f}")
        
    # Aggregate over all images
    mean_ious = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    
    for iou, gt_pixels in zip(all_ious, all_gt_pixels):
        for c in range(num_classes):
            if gt_pixels[c] > 0:
                mean_ious[c] += iou[c]
                class_counts[c] += 1
                
    final_class_ious = np.divide(mean_ious, class_counts, out=np.zeros_like(mean_ious), where=class_counts!=0)
    
    print("\n--- FINAL AUDIT METRICS ---")
    for c in range(num_classes):
        print(f"{inv_class_mapping[c]}: {final_class_ious[c]:.4f}")
        
    valid_classes = class_counts > 0
    mIoU = np.mean(final_class_ious[valid_classes])
    print(f"\nOverall mIoU: {mIoU:.4f}")
    
    # Save to CSV
    metrics = {"Class": [inv_class_mapping[c] for c in range(num_classes)], "IoU": final_class_ious, "Present_In_Images": class_counts}
    df = pd.DataFrame(metrics)
    df.to_csv("results/audit_real/true_class_statistics.csv", index=False)
    print("Saved true metrics to results/audit_real/true_class_statistics.csv")

if __name__ == '__main__':
    main()
