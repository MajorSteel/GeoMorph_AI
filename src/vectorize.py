import cv2
import numpy as np
import rasterio
from shapely.geometry import Polygon
import geopandas as gpd
import logging

from src.geometry_validator import validate_and_repair_geometries

logger = logging.getLogger("geospatial_pipeline.vectorize")

def mask_to_polygons(mask: np.ndarray, transform: rasterio.transform.Affine, simplify_tolerance: float = 2.0) -> gpd.GeoDataFrame:
    """
    Converts a binary mask into a GeoDataFrame of polygons in geographic coordinates.
    
    Args:
        mask: (H, W) binary numpy array
        transform: Affine transform from pixel to geographic coordinates
        simplify_tolerance: Douglas-Peucker simplification tolerance (in geographic units, ~pixels mapped to crs)
    """
    # Find contours
    # RETR_EXTERNAL extracts only the outer boundaries (no holes for now, simplifies topology)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # A valid polygon needs at least 4 points (triangle + closed)
        if contour.shape[0] >= 3:
            # OpenCV contours are (N, 1, 2) where 2 is (x, y)
            pts = contour.squeeze(1)
            
            # Convert pixel coordinates to geographic coordinates
            # transform * (x, y) = (lon, lat)
            geo_pts = [transform * (pt[0], pt[1]) for pt in pts]
            
            try:
                poly = Polygon(geo_pts)
                if poly.is_valid or not poly.is_valid: # We'll repair invalid ones later
                    polygons.append(poly)
            except Exception as e:
                pass # Skip degenerate polygons
                
    if not polygons:
        logger.warning("No valid polygons extracted from mask.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
    crs = "EPSG:4326" if not transform.is_identity else None
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    
    # Validate and repair
    gdf = validate_and_repair_geometries(gdf)
    
    # Simplify
    if simplify_tolerance > 0 and len(gdf) > 0:
        # transform.a is the pixel width in geographic units
        actual_tolerance = simplify_tolerance * abs(transform.a)
        gdf.geometry = gdf.geometry.simplify(actual_tolerance, preserve_topology=True)
        # Re-validate after simplify
        gdf = validate_and_repair_geometries(gdf)
        
    logger.info(f"Vectorized into {len(gdf)} valid polygons.")
    return gdf
