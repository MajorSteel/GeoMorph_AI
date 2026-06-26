# Rasterization Engine

To train a semantic segmentation model, vector geometries (GeoJSONs) must be translated into integer grid matrices (Masks). GeoMorph AI performs this dynamically in memory.

## Process Flow
1. **Data Ingestion**: `geopandas` loads the `.geojson` and forces it into the `EPSG:4326` geographic coordinate system.
2. **Affine Translation**: `rasterio` calculates the exact mathematical affine transform of the background image to map geographic coordinates to pixel space `[x, y]`.
3. **Shape Rasterization**: `rasterio.features.rasterize` burns the geometries into a zero-matrix.

## Class Priority (Z-Indexing)
When polygons overlap (e.g., a `Tree` canopy hanging over a `Building`), the engine determines pixel ownership based on class hierarchy or a predefined Z-Index configuration, ensuring no pixels are assigned multiple classes.

## Dynamic Augmentation
Because rasterization occurs dynamically *during* the PyTorch DataLoader yield, we can instantly apply spatial augmentations. Both the raw image and the newly generated mask undergo `ShiftScaleRotate`, `GridDistortion`, and `RandomCrop` simultaneously.
