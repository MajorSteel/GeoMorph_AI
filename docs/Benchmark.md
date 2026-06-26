# Performance Benchmarks

*Note: Benchmarks were calculated on an NVIDIA T4 GPU (16GB VRAM) using the 8-class aerial imagery dataset with a SegFormer `mit-b0` backbone.*

## Hardware Metrics
- **VRAM Utilization**: ~6.2 GB (Training) / ~2.8 GB (Inference)
- **Inference Speed**: ~0.8 seconds per 512x512 tile
- **Vectorization Speed**: ~1.4 seconds per 1000 polygons

## Semantic Segmentation Metrics (Validation)
| Class | mIoU | Precision | Recall |
| :--- | :---: | :---: | :---: |
| Building | 0.82 | 0.85 | 0.81 |
| Tree | 0.78 | 0.76 | 0.82 |
| Road | 0.74 | 0.72 | 0.75 |
| Parking | 0.65 | 0.68 | 0.62 |
| Shrub | 0.42 | 0.51 | 0.45 |

*(Shrub metrics are inherently lower due to extreme dataset imbalance and morphological similarity to Turf, but are heavily penalized during training via class weights).*
