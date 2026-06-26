import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
import logging

logger = logging.getLogger("geospatial_pipeline.geo_utils")

def get_affine_transform(image_path: str) -> Affine:
    """
    Extracts the affine transform from the image metadata.
    rasterio automatically checks for embedded GeoTIFF tags or sidecar files (.jgw, .tfw).
    Returns Identity transform if no georeferencing is found.
    """
    with rasterio.open(image_path) as src:
        transform = src.transform
            
    return transform

def get_bbox_transform(bbox: list, width: int, height: int) -> Affine:
    """
    Computes an affine transform by mapping the spatial bounding box exactly 
    to the image width and height.
    bbox format: [minx, miny, maxx, maxy]
    """
    minx, miny, maxx, maxy = bbox
    res_x = (maxx - minx) / width
    res_y = (maxy - miny) / height
    return Affine(res_x, 0.0, minx, 0.0, -res_y, maxy)

def verify_or_reproject_crs(gdf, target_epsg: int = 4326):
    """
    Ensures a GeoDataFrame is in the target EPSG CRS.
    If it has no CRS, sets it to target.
    If it has a different CRS, reprojects it.
    """
    target_crs = CRS.from_epsg(target_epsg)
    
    if gdf.crs is None:
        bounds = gdf.total_bounds
        # Check if it looks like pixel space (values > 180 or 90)
        if max(abs(bounds[0]), abs(bounds[2])) > 180 or max(abs(bounds[1]), abs(bounds[3])) > 90:
            logger.warning("GeoDataFrame has no CRS, but coordinates exceed 180/90. Assuming pixel space. Not setting EPSG.")
            return gdf
            
        logger.warning(f"GeoDataFrame has no CRS. Assuming EPSG:{target_epsg}")
        gdf.set_crs(target_crs, inplace=True)
    elif gdf.crs != target_crs:
        logger.info(f"Reprojecting GeoDataFrame from {gdf.crs} to EPSG:{target_epsg}")
        gdf = gdf.to_crs(target_crs)
        
    return gdf
