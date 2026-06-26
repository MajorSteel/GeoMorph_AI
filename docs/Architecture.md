# Architecture & Model Design

GeoMorph AI departs from traditional Convolutional Neural Networks (CNNs) like U-Net and DeepLabV3 by utilizing a purely attention-based **SegFormer** architecture. Specifically, the pipeline is built on top of the HuggingFace `nvidia/mit-b0` backbone.

## The SegFormer Advantage
SegFormer features a hierarchical Transformer encoder that does not rely on positional encoding. This provides two massive advantages for geospatial imagery:
1. **Resolution Agnosticism**: It inherently adapts to the massive, varied resolutions of aerial imagery without hardcoded crop limitations.
2. **Global Receptive Field**: Unlike CNNs that are limited by kernel sizes (e.g., 3x3), SegFormer computes self-attention across the entire tile, allowing it to understand the context of a river relative to a distant road, rather than just local pixel textures.

## Model Flow
```mermaid
graph TD
    A[Input Image Tile 512x512] --> B[Mix Transformer Encoder]
    B --> C[Overlapped Patch Merging]
    C --> D[Multi-Level Feature Maps 1/4, 1/8, 1/16, 1/32]
    D --> E[All-MLP Decoder]
    E --> F[8-Class Segmentation Mask]
```

## The All-MLP Decoder
By utilizing an extremely lightweight All-MLP decoder, GeoMorph AI achieves rapid inference speeds (sub-second per tile) while maintaining state-of-the-art mIoU metrics, making it highly feasible for edge deployment.
