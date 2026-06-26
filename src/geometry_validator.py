import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
import logging

logger = logging.getLogger("geospatial_pipeline.geometry_validator")

def validate_and_repair_geometries(gdf: gpd.GeoDataFrame, min_area: float = 0.0) -> gpd.GeoDataFrame:
    """
    Cleans and validates vector geometries:
    1. Removes null geometries.
    2. Repairs invalid geometries (e.g., self-intersections) using make_valid().
    3. Buffers by 0 as a secondary repair step.
    4. Ensures only Polygon and MultiPolygon geometries are retained.
    5. Filters out polygons below min_area.
    """
    initial_count = len(gdf)
    
    # 1. Remove null/empty
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    
    repaired_geoms = []
    for idx, geom in gdf.geometry.items():
        # 2 & 3. Repair
        if not geom.is_valid:
            geom = make_valid(geom)
            
        # Buffer by 0 can resolve some lingering topological issues
        geom = geom.buffer(0)
        repaired_geoms.append(geom)
        
    gdf.geometry = repaired_geoms
    
    # 4. Keep only Polygons / MultiPolygons
    valid_types = ['Polygon', 'MultiPolygon']
    gdf = gdf[gdf.geometry.geom_type.isin(valid_types)].copy()
    
    # 5. Filter by area (if projected, min_area should be in square meters/degrees)
    if min_area > 0:
        gdf = gdf[gdf.geometry.area >= min_area].copy()
        
    final_count = len(gdf)
    if initial_count != final_count:
        logger.info(f"Geometry repair dropped {initial_count - final_count} invalid features.")
        
    return gdf
