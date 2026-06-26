import os
import json
import geopandas as gpd
import logging

logger = logging.getLogger("geospatial_pipeline.gis_export")

def export_predictions(gdf: gpd.GeoDataFrame, output_dir: str, basename: str):
    """
    Exports a GeoDataFrame to standard GIS formats inside output_dir.
    Generates:
    - GeoJSON
    - Shapefile (with .shp, .shx, .dbf, .prj)
    - validation.json
    """
    if len(gdf) == 0:
        logger.warning(f"GeoDataFrame is empty. Nothing to export to {output_dir}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. GeoJSON
    geojson_path = os.path.join(output_dir, f"{basename}.geojson")
    gdf.to_file(geojson_path, driver="GeoJSON")
    logger.info(f"Exported {len(gdf)} features to {geojson_path}")
    
    # 2. Shapefile (geopandas handles sidecars automatically when driver="ESRI Shapefile")
    shp_path = os.path.join(output_dir, f"{basename}.shp")
    # ESRI Shapefile does not support datetime fields easily or long column names, 
    # but we only have 'class' and 'confidence'
    gdf.to_file(shp_path, driver="ESRI Shapefile")
    logger.info(f"Exported Shapefile components to {output_dir}")
    
    # 3. Validation JSON
    validation = {
        "crs": str(gdf.crs),
        "geometry_validity": "all_valid" if gdf.is_valid.all() else "contains_invalid",
        "polygon_count": len(gdf),
        "bounding_box": list(gdf.total_bounds),
        "average_area": gdf.geometry.area.mean(),
        "invalid_polygons": int((~gdf.is_valid).sum()),
        "coordinate_ranges": {
            "min_x": float(gdf.total_bounds[0]),
            "min_y": float(gdf.total_bounds[1]),
            "max_x": float(gdf.total_bounds[2]),
            "max_y": float(gdf.total_bounds[3])
        }
    }
    
    val_path = os.path.join(output_dir, "validation.json")
    with open(val_path, 'w') as f:
        json.dump(validation, f, indent=4)
        
    logger.info(f"Exported GIS validation metrics to {val_path}")
