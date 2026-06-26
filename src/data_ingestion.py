import os
import glob
import logging
import geopandas as gpd
from typing import List, Tuple, Dict
from src.geo_utils import verify_or_reproject_crs
from src.geometry_validator import validate_and_repair_geometries

logger = logging.getLogger("geospatial_pipeline.data_ingestion")

def find_image_files(image_dir: str) -> List[str]:
    """Finds all standard image files in the directory."""
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
    return sorted(list(set(files)))

def find_geojson_files(label_dir: str) -> List[str]:
    """Finds all geojson files in the directory."""
    files = glob.glob(os.path.join(label_dir, '**', '*.geojson'), recursive=True)
    return sorted(files)

def match_images_to_labels(image_files: List[str], label_files: List[str]) -> List[Tuple[str, str]]:
    """
    Naively matches images to labels by base filename.
    e.g. '1.jpg' matches '1.geojson'.
    Returns a list of tuples: (image_path, label_path).
    """
    matched = []
    for img in image_files:
        basename = os.path.splitext(os.path.basename(img))[0]
        # Find matching label
        for lbl in label_files:
            lbl_basename = os.path.splitext(os.path.basename(lbl))[0]
            if basename == lbl_basename:
                matched.append((img, lbl))
                break
    return matched

def discover_classes(label_files: List[str]) -> Dict[str, int]:
    """
    Inspects GeoJSON files to see if there is a consistent class property.
    If not, defaults to binary segmentation.
    Returns a mapping of class_name to integer index (0 is background).
    """
    all_properties = set()
    for lbl in label_files:
        gdf = gpd.read_file(lbl)
        for col in gdf.columns:
            if col != 'geometry':
                all_properties.add(col)
                
    logger.info(f"Discovered GeoJSON properties: {all_properties}")
    
    # SegFormer 8-Class Mapping
    class_mapping = {
        "Building": 0,
        "Tree": 1,
        "Parking": 2,
        "Shrub": 3,
        "Turf": 4,
        "Road": 5,
        "Sidewalk": 6,
        "Water": 7
    }
    logger.info("Using 8-class semantic segmentation mapping.")
    return class_mapping

def load_and_prepare_labels(label_path: str) -> gpd.GeoDataFrame:
    """Loads GeoJSON, verifies CRS, repairs geometries."""
    gdf = gpd.read_file(label_path)
    gdf = verify_or_reproject_crs(gdf, target_epsg=4326)
    gdf = validate_and_repair_geometries(gdf)
    return gdf
