import os
import json
import logging
import rasterio
import geopandas as gpd

logger = logging.getLogger("geospatial_pipeline.alignment")

def evaluate_alignment(image_path: str, label_path: str) -> dict:
    """
    Implements the Smart Alignment Gate decision tree and confidence scoring.
    Returns a dictionary report.
    """
    with rasterio.open(image_path) as src:
        transform = src.transform
        img_width = src.width
        img_height = src.height
        is_identity = transform.is_identity
        
    gdf = gpd.read_file(label_path)
    bounds = gdf.total_bounds # [minx, miny, maxx, maxy]
    
    # Check for World Files
    base, _ = os.path.splitext(image_path)
    has_world_file = False
    for ext in ['.jgw', '.tfw', '.pgw', '.wld']:
        if os.path.exists(base + ext):
            has_world_file = True
            break
            
    # Decision Tree for Coordinate System
    coord_system = "Unknown"
    if gdf.crs is not None:
        if gdf.crs.is_geographic:
            coord_system = "Geographic"
        elif gdf.crs.is_projected:
            coord_system = "Projected"
        else:
            coord_system = "Unknown"
    
    # Heuristic fallback if CRS is missing or unknown
    if coord_system == "Unknown":
        if bounds[0] >= 0 and bounds[1] >= 0 and bounds[2] <= img_width and bounds[3] <= img_height:
            coord_system = "Pixel-space"
        elif max(abs(bounds[0]), abs(bounds[2])) <= 180 and max(abs(bounds[1]), abs(bounds[3])) <= 90:
            coord_system = "Geographic"
        else:
            coord_system = "Projected (Heuristic)"
            
    # Calculate Confidence Score
    confidence = 0
    reason = ""
    status = "FAILED"
    recommendation = ""
    
    if coord_system in ["Geographic", "Projected", "Projected (Heuristic)"]:
        if is_identity:
            confidence = 50
            reason = "Missing affine transform. Fabricating from GeoJSON bounding box (Per Design Document)."
            recommendation = "Use fabricated transform for training, but verify alignment."
            status = "WARNING"
        else:
            if image_path.endswith(('.tif', '.tiff')) and not has_world_file:
                confidence = 100
                reason = "Valid GeoTIFF affine transform found"
                status = "SUCCESS"
            elif has_world_file:
                confidence = 95
                reason = "Valid World file transform found"
                status = "SUCCESS"
            else:
                confidence = 90
                reason = "Valid RPC or other metadata transform found"
                status = "SUCCESS"
    elif coord_system == "Pixel-space":
        confidence = 85
        reason = "Verified pixel coordinates"
        status = "SUCCESS"
        recommendation = "No georeferencing needed."
        
    report = {
        "image": os.path.basename(image_path),
        "status": status,
        "reason": reason,
        "annotation_crs": str(gdf.crs) if gdf.crs else "None",
        "image_georeferencing": "Absent" if is_identity else "Present",
        "coordinate_system": coord_system,
        "confidence": confidence,
        "recommendation": recommendation,
        "image_metadata": f"{img_width}x{img_height} (bands: {src.count})",
        "transform_matrix": "Identity" if is_identity else str(transform).replace('\n', ' '),
        "world_file_presence": "Present" if has_world_file else "Absent",
        "coordinate_ranges": f"X: [{bounds[0]:.4f}, {bounds[2]:.4f}], Y: [{bounds[1]:.4f}, {bounds[3]:.4f}]"
    }
    
    return report

def generate_reports(alignment_reports: list, out_dir: str):
    """Generates ALIGNMENT_REPORT.json and PREPROCESSING_REPORT.md"""
    os.makedirs(out_dir, exist_ok=True)
    
    # JSON Report
    json_path = os.path.join(out_dir, "ALIGNMENT_REPORT.json")
    with open(json_path, 'w') as f:
        json.dump(alignment_reports, f, indent=2)
        
    # MD Report
    md_path = os.path.join(out_dir, "PREPROCESSING_REPORT.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Preprocessing Report\n\n")
        f.write("| Image | Image Readable | GeoJSON Readable | CRS Detected | Coordinate System | Affine Transform | Rasterization | Reason |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        
        for r in alignment_reports:
            img_read = "✅"
            geo_read = "✅"
            crs = r['annotation_crs']
            coord = r['coordinate_system']
            affine = "❌ Missing" if r['image_georeferencing'] == "Absent" else "✅ Present"
            rast = "Skipped" if r['status'] == "FAILED" else "Ready"
            reason = r['reason']
            
            f.write(f"| {r['image']} | {img_read} | {geo_read} | {crs} | {coord} | {affine} | {rast} | {reason} |\n")
            
    # Dataset Audit Report
    audit_path = os.path.join(out_dir, "DATASET_AUDIT.md")
    with open(audit_path, 'w', encoding='utf-8') as f:
        f.write("# Dataset Audit\n\n")
        f.write("| Image | Metadata | CRS | Transform | World File | Coordinate Ranges | Classification | Confidence | Final Decision |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in alignment_reports:
            f.write(f"| {r['image']} | {r['image_metadata']} | {r['annotation_crs']} | {r['transform_matrix']} | {r['world_file_presence']} | {r['coordinate_ranges']} | {r['coordinate_system']} | {r['confidence']}% | {r['status']} |\n")
            
    logger.info(f"Generated alignment reports in {out_dir}")
