# Troubleshooting Guide

### 1. `rasterio NotGeoreferencedWarning`
**Issue**: You see a warning that the dataset has no geotransform, gcps, or rpcs.
**Solution**: This is a harmless warning. If your `.jpg` image does not have embedded geospatial coordinates, the pipeline will fallback to an Identity Matrix, meaning it will treat the image as a standard Cartesian grid.

### 2. `CUDA OutOfMemoryError`
**Issue**: Your GPU runs out of VRAM during training or inference.
**Solution**: 
- In `configs/baseline.yaml`, lower the `batch_size` from `8` to `4` or `2`.
- Enable `mixed_precision: true` to utilize AMP (Automatic Mixed Precision).

### 3. `Albumentations ValueError`
**Issue**: Error regarding bounding box geometries or invalid arguments.
**Solution**: This usually means your `.geojson` contains malformed polygons (e.g., Bowtie polygons). The pipeline's `geometry_validator.py` attempts to repair these, but severely broken geometries must be fixed manually in QGIS.
