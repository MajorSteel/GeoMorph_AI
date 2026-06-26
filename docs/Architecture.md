# Architecture & Model Design

GeoMorph AI departs from traditional Convolutional Neural Networks (CNNs) like U-Net and DeepLabV3 by utilizing a purely attention-based **SegFormer** architecture. Specifically, the pipeline is built on top of the HuggingFace `nvidia/mit-b0` backbone.

## The SegFormer Advantage
SegFormer features a hierarchical Transformer encoder that does not rely on positional encoding. This provides two massive advantages for geospatial imagery:
1. **Resolution Agnosticism**: It inherently adapts to the massive, varied resolutions of aerial imagery without hardcoded crop limitations.
2. **Global Receptive Field**: Unlike CNNs that are limited by kernel sizes (e.g., 3x3), SegFormer computes self-attention across the entire tile, allowing it to understand the context of a river relative to a distant road, rather than just local pixel textures.

## Model Flow
```mermaid
graph TD
    %% Styling
    classDef dataLayer fill:#1e1e2f,stroke:#4a4a75,stroke-width:2px,color:#fff
    classDef processingLayer fill:#2b2b40,stroke:#626296,stroke-width:2px,color:#fff
    classDef neuralLayer fill:#3a1c40,stroke:#8a4096,stroke-width:2px,color:#fff
    classDef outputLayer fill:#1c3a30,stroke:#409672,stroke-width:2px,color:#fff
    
    subgraph Data Ingestion
        A1[(Raw Multi-Gigapixel Imagery)]:::dataLayer
        A2[(Manual GIS Annotations .geojson)]:::dataLayer
        A1 --> B1[Dynamic Tile Generator 512x512]:::processingLayer
        A2 --> B2[Affine Projection Engine]:::processingLayer
    end

    subgraph Preprocessing & Rasterization
        B2 --> C1[Rasterio Z-Index Geometry Burner]:::processingLayer
        C1 --> C2[Boolean Feature Masks]:::processingLayer
        B1 --> C3{Albumentations Augmenter}:::processingLayer
        C2 --> C3
        C3 -->|GridDistortion, ColorJitter| D1[PyTorch DataLoader]:::processingLayer
    end

    subgraph SegFormer Transformer Backbone
        D1 --> E1[Overlapped Patch Merging Layer]:::neuralLayer
        E1 --> E2(Transformer Block 1: 1/4 Scale):::neuralLayer
        E2 --> E3(Transformer Block 2: 1/8 Scale):::neuralLayer
        E3 --> E4(Transformer Block 3: 1/16 Scale):::neuralLayer
        E4 --> E5(Transformer Block 4: 1/32 Scale):::neuralLayer
        
        E2 --> F1[Mix-FFN & Efficient Self-Attention]:::neuralLayer
        E3 --> F1
        E4 --> F1
        E5 --> F1
        
        F1 --> G1[All-MLP Decoder Head]:::neuralLayer
        G1 --> G2[Softmax Probability Tensor]:::neuralLayer
    end

    subgraph Inference & Post-Processing
        G2 --> H1[Cosine-Bell Distance Weighting]:::processingLayer
        H1 --> H2[Overlapping Tile Stitcher]:::processingLayer
        H2 --> H3[Confidence Thresholding > 0.5]:::processingLayer
        H3 --> H4[Argmax Class Mask]:::processingLayer
    end

    subgraph GIS Vectorization Intelligence
        H4 --> I1[OpenCV Contour Extraction]:::outputLayer
        I1 --> I2[Douglas-Peucker Geometry Simplification]:::outputLayer
        I2 --> I3[Inverse Affine Georeferencing EPSG:4326]:::outputLayer
    end

    subgraph Production Export
        I3 --> J1([ESRI Shapefile .shp]):::outputLayer
        I3 --> J2([Mapbox GeoJSON .geojson]):::outputLayer
        H4 --> J3([Color-Coded Visual Overlay .png]):::outputLayer
    end
```

## The All-MLP Decoder
By utilizing an extremely lightweight All-MLP decoder, GeoMorph AI achieves rapid inference speeds (sub-second per tile) while maintaining state-of-the-art mIoU metrics, making it highly feasible for edge deployment.
