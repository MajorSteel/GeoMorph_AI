import os
import numpy as np
import argparse
import logging
import cv2
import rasterio
import torch

from src.config import PipelineConfig
from src.logging_utils import get_logger
from src.augmentation import get_validation_augmentation
from src.predictor import predict_image
from src.postprocess import postprocess_mask
from src.visualization import generate_overlay
from src.model_registry import get_model
from src.vectorize import mask_to_polygons
from src.gis_export import export_predictions
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Geospatial Semantic Segmentation Inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output_dir", default="results/inference", help="Directory to save outputs")
    parser.add_argument("--weights", default="results/training/fold_0/best.pt", help="Path to model weights")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    logger = get_logger()
    cfg = PipelineConfig.from_yaml(args.config)
    
    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.image))[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    logger.info(f"Using device: {device}")
    
    with rasterio.open(args.image) as src:
        image_np = src.read([1, 2, 3]).transpose(1, 2, 0)
        transform = src.transform
        
    model = get_model(
        architecture=cfg.model.architecture,
        encoder=cfg.model.encoder,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.classes
    ).to(device)
    
    if os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    else:
        logger.warning(f"Checkpoint not found at {args.weights}. Running with random weights.")
        
    transform_val = get_validation_augmentation()
    prob_map = predict_image(model, image_np, transform_val, cfg.data.tile_size, cfg.data.inference_overlap, device)
    
    mask = postprocess_mask(prob_map, threshold=cfg.inference.threshold, min_area=cfg.inference.min_area)
    
    overlay_path = os.path.join(args.output_dir, f"{basename}_overlay.png")
    generate_overlay(args.image, mask, overlay_path)
    
    mask_colored = np.zeros_like(image_np)
    colors = {0: [255,0,0], 1: [0,255,0], 2: [0,0,255], 3: [255,255,0], 4: [0,255,255], 5: [255,0,255], 6: [192,192,192], 7: [0,128,128]}
    class_names = {0: 'Building', 1: 'Tree', 2: 'Parking', 3: 'Shrub', 4: 'Turf', 5: 'Road', 6: 'Sidewalk', 7: 'Water'}
    
    for class_id, color in colors.items():
        mask_colored[mask == class_id] = color
        
    cv2.imwrite(os.path.join(args.output_dir, f"{basename}_mask.png"), cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR))
    
    # --- GIS Vectorization ---
    logger.info("Starting vectorization to GIS formats...")
    all_gdfs = []
    
    for class_id, class_name in class_names.items():
        # Create binary mask for the current class
        binary_mask = (mask == class_id).astype(np.uint8)
        
        # Skip if class is not present in prediction
        if not np.any(binary_mask):
            continue
            
        class_gdf = mask_to_polygons(binary_mask, transform, simplify_tolerance=cfg.inference.simplify_tolerance)
        
        if not class_gdf.empty:
            class_gdf['class_id'] = class_id
            class_gdf['class_name'] = class_name
            all_gdfs.append(class_gdf)
            
    if all_gdfs:
        combined_gdf = pd.concat(all_gdfs, ignore_index=True)
        export_predictions(combined_gdf, args.output_dir, basename)
    else:
        logger.warning("No polygons generated across any class. Skipping GIS export.")
        
    logger.info(f"Inference completed successfully. Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
