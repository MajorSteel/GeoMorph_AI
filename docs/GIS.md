# GIS Vectorization Intelligence

GeoMorph AI acts as a complete bridge between Deep Learning and Geographic Information Systems (GIS). It doesn't just stop at generating a PNG mask; it converts predictions back into real-world geographic data.

## Pixel-to-Polygon Translation
1. **Contour Extraction**: The raw prediction matrix is parsed using OpenCV (`cv2.findContours`) to extract the exact boundaries of every predicted class (Trees, Roads, Buildings).
2. **Douglas-Peucker Simplification**: Raw pixel boundaries are highly jagged. We apply the Douglas-Peucker algorithm (`shapely.simplify`) to drastically reduce vertex counts, turning jagged pixel steps into smooth, realistic vector lines.
3. **Georeferencing**: Using the original image's Affine Transform matrix, the `[x, y]` pixel coordinates are inverted back into true geographic Longitude/Latitude coordinates.

## Export Formats
The pipeline automatically outputs two production-ready formats:
- **GeoJSON**: Ideal for web mapping, Mapbox, and lightweight storage.
- **ESRI Shapefile (.shp)**: The industry standard format ready to be dragged and dropped into **QGIS** or **ArcGIS Pro**.
