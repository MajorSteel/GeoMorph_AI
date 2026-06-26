# Training the SegFormer Pipeline

GeoMorph AI utilizes a sophisticated training protocol designed to maximize generalization on small to medium geospatial datasets. The core of this protocol relies on **Leave-One-Out Cross-Validation (LOOCV)** and a **Two-Phase Transfer Learning** strategy.

## The LOOCV Protocol
Because high-quality annotated geospatial data is often scarce, we use LOOCV to ensure the model generalizes perfectly.
If you have 3 images in your dataset:
- **Fold 1**: Trains on Image 2 and 3. Validates on Image 1.
- **Fold 2**: Trains on Image 1 and 3. Validates on Image 2.
- **Fold 3**: Trains on Image 1 and 2. Validates on Image 3.

The pipeline automatically handles fold generation, metric tracking, and aggregates the best weights.

## Two-Phase Training
GeoMorph AI loads a HuggingFace `nvidia/mit-b0` SegFormer backbone. To prevent catastrophic forgetting of the pre-trained ImageNet weights, training is split into two phases:

1. **Phase 1 (Frozen Encoder)**: 
   - The Transformer encoder is frozen.
   - Only the MLP decode head is trained for 1 epoch at a high learning rate.
   - This rapidly adapts the classifier to the 8-class geospatial schema.
2. **Phase 2 (Full Tuning)**:
   - The entire network is unfrozen.
   - Trained for up to 40 epochs with a lower learning rate.
   - Utilizes **Early Stopping** (patience of 5 epochs) to prevent overfitting.

## Class-Weighted Loss Function
Geospatial data is inherently imbalanced (e.g., Turf covers 50% of the image, while Shrubs cover 2%). To combat this, we utilize a custom **Weighted Cross-Entropy + Dice Loss**:
- `Shrub`: 5.0x multiplier
- `Water`: 0.2x multiplier (Down-weighted to prevent false positives in shadows)
- `Road`: 2.0x multiplier

## Running the Training Pipeline

To initiate the automated LOOCV training loop, use the GeoMorph CLI:

```bash
geomorph train --config configs/baseline.yaml
```

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 8GB of VRAM (e.g., RTX 3060, Tesla T4, A100).
- **RAM**: 16GB System RAM for raster caching.

### Output Artifacts
During training, the pipeline will generate the following artifacts in the `results/training/` directory:
- `fold_N/best.pt`: The model weights achieving the highest Validation mIoU.
- `fold_N/last.pt`: The final model weights before early stopping.
- `training.log`: A comprehensive step-by-step metric logger.