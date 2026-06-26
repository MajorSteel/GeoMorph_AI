# Inference Pipeline

Running inference on massive, gigapixel satellite imagery presents a severe memory challenge. GeoMorph AI solves this using an **Overlapping Sliding-Window Inference** mechanism combined with **Cosine-Bell Distance Weighting**.

## Sliding-Window Strategy
When a large `.tiff` is passed to `geomorph predict`, the pipeline does not attempt to load the entire image into GPU memory. 
Instead, it chunks the image into `512x512` tiles with a configurable overlap (e.g., 25%). 

## The Seam Problem & Tile Stitching
A common issue with chunked inference is that objects on the edge of a tile are poorly segmented, resulting in visible "grid seams" when stitched back together. 
GeoMorph AI solves this via the `TileStitcher`. Predictions near the center of a tile are given a higher confidence weight than predictions near the edge, using a **Cosine-Bell** mathematical distribution. When overlapping tiles are merged, their probabilities are seamlessly blended.

## Running Inference
```bash
geomorph predict --image sample.tiff --weights best.pt --output_dir results/
```
**Outputs Generated**:
1. `_mask.png`: The raw semantic mask.
2. `_overlay.png`: A colored visualization superimposed on the original image.
3. `.geojson`: The vectorized GIS geometry.
