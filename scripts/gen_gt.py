import os
import rasterio
import numpy as np
from src.data_ingestion import load_and_prepare_labels
from src.rasterize import create_mask_from_gdf
from src.visualization import generate_overlay
from src.config import PipelineConfig

cfg = PipelineConfig.from_yaml('configs/baseline.yaml')
image_path = 'aerial_imagery_pack/aerial_imagery_pack/1.jpg'
geojson_path = 'aerial_imagery_pack/aerial_imagery_pack/1.geojson'

# Get image shape and transform
with rasterio.open(image_path) as src:
    shape = src.shape
    transform = src.transform

# Generate integer mask from GeoJSON
gdf = load_and_prepare_labels(geojson_path)
mask = create_mask_from_gdf(gdf, transform, shape, class_mapping=cfg.data.class_mapping)

# Use generate_overlay to create a colored visualization
generate_overlay(image_path, mask, 'assets/1_ground_truth.png')
print("Generated assets/1_ground_truth.png")
