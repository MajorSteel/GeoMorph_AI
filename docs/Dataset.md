# Dataset Structure & Data Ingestion

GeoMorph AI requires a paired dataset of raw geospatial imagery and manual vector annotations (GeoJSON) to train the SegFormer model. The pipeline features a dynamic rasterization engine that converts polygon annotations into integer segmentation masks on the fly.

## Directory Structure
To train the model, your dataset should be organized in a flat or nested directory structure where images and their corresponding labels share the same base filename.

```text
aerial_imagery_pack/
├── 1.jpg
├── 1.geojson
├── 2.tiff
├── 2.geojson
├── 3.png
└── 3.geojson
```

## Supported Formats
- **Imagery**: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff` (High-resolution imagery is seamlessly tiled during training).
- **Labels**: `.geojson` (Must contain polygon or multipolygon geometries).

## Class Mapping & Attributes
The `GeoJSON` files must contain properties that define the class of each polygon. By default, GeoMorph AI expects a `'class'` attribute mapping to one of the 8 core feature classes:

| Class ID | Feature Name | Description |
| :---: | :--- | :--- |
| `0` | **Building** | Commercial and residential structures |
| `1` | **Tree** | Canopies and individual trees |
| `2` | **Parking** | Paved parking lots |
| `3` | **Shrub** | Low-lying vegetation and bushes |
| `4` | **Turf** | Maintained grass and lawns |
| `5` | **Road** | Asphalt and concrete roadways |
| `6` | **Sidewalk** | Pedestrian walkways |
| `7` | **Water** | Pools, lakes, and rivers |

> **Note**: If your dataset uses a different naming convention, you can modify the `class_mapping` dictionary inside `src/data_ingestion.py`.

## Dynamic Rasterization
Unlike legacy pipelines that require pre-processing shapefiles into PNG masks, GeoMorph AI handles rasterization dynamically. 

When the dataset loader reads a tile:
1. It extracts the bounding box of the tile from the raw imagery.
2. It queries the `.geojson` file for polygons intersecting that bounding box.
3. It rasterizes those polygons into a strict `(512, 512)` integer mask matrix.
4. It applies aggressive `albumentations` transformations (GridDistortion, ColorJitter, RandomCrop) to both the image tile and the rasterized mask simultaneously.