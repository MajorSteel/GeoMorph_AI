import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import logging

logger = logging.getLogger("geospatial_pipeline.rasterize")

def create_mask_from_gdf(gdf: gpd.GeoDataFrame, transform: rasterio.transform.Affine, shape: tuple, class_mapping: dict = None) -> np.ndarray:
    """
    Rasterizes a GeoDataFrame into a numpy array mask.
    
    Args:
        gdf: GeoDataFrame containing polygons.
        transform: Affine transform from geographic to pixel coordinates.
        shape: (height, width) of the output mask.
        class_mapping: dict mapping feature values to integers. If None, produces a binary mask (1 for feature, 0 for bg).
        
    Returns:
        np.ndarray of shape (height, width) with integer class labels.
    """
    if len(gdf) == 0:
        logger.warning("Empty GeoDataFrame provided. Returning all-zero mask.")
        return np.zeros(shape, dtype=np.uint8)

    # Convert geometries to shapes format for rasterio: ((geom, value), ...)
    shapes = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        # Map the string class name to its integer ID for 8-class segmentation
        if class_mapping and 'class' in row:
            class_name = row['class']
            value = class_mapping.get(class_name, 0)
        else:
            value = 1 # Fallback to binary if mapping is missing
            
        shapes.append((geom, value))
        
    mask = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=False, # Center of pixel rule
        dtype=np.uint8
    )
    
    return mask
