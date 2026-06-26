<div align="center">
  <h1>🌍 GeoMorph AI</h1>
  <p><strong>End-to-End Geospatial Semantic Segmentation & GIS Vectorization</strong></p>
</div>

## 📌 Overview
**GeoMorph AI** is a state-of-the-art computer vision and geospatial pipeline designed to process high-resolution aerial and satellite imagery. Utilizing a HuggingFace **SegFormer (nvidia/mit-b0)** backbone, the pipeline dynamically extracts **8 core land-cover classes** and automatically translates raw pixel predictions into production-ready **GIS Vector Polygons** (`.shp`, `.geojson`).

---

## 🚀 Key Features
* **Transformer-Based Architecture:** Replaces legacy U-Nets with a lightweight, highly accurate SegFormer.
* **Class-Weighted CE+Dice Loss:** Intelligently balances extreme data disparity, ensuring minority classes (e.g., Shrubs, Water, Roads) are segmented accurately against dominant classes (e.g., Turf, Buildings).
* **Sliding-Window Inference:** Seamlessly handles massive geospatial rasters without memory overflow using Gaussian-weighted tile stitching.
* **Automated GIS Vectorization:** Bridges the gap between Deep Learning and GIS by dynamically routing prediction masks into Douglas-Peucker simplified vector polygons.
* **Robust Evaluation:** Built-in Leave-One-Out Cross-Validation (LOOCV) harness and true performance auditing (mIoU).

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/omega-pipeline.git
cd omega-pipeline

# Install the dependencies
pip install -r requirements.txt
```
> **Note:** We recommend running this project in a Google Colab instance with a T4 GPU for optimal training speeds.

---

## 📂 Project Structure

```text
├── configs/
│   └── baseline.yaml          # Core pipeline configurations (Loss, Weights, Augmentations)
├── scripts/
│   ├── omega_pipeline.py      # The master execution protocol
│   ├── run_loocv.py           # The training loop (Cross-Validation)
│   └── audit_inference.py     # True performance metric calculations
├── src/
│   ├── augmentation.py        # Advanced augmentations (GridDistortion, ColorJitter)
│   ├── dataset.py             # Geospatial raster-to-tile data ingestion
│   ├── gis_export.py          # Exports predictions to .shp and .geojson
│   ├── losses.py              # Custom class-weighted ce_dice loss
│   ├── model_registry.py      # SegFormer model loader
│   ├── postprocess.py         # Confidence thresholding and mask cleaning
│   ├── predictor.py           # Sliding-window inference logic
│   ├── stitcher.py            # Cosine-Bell distance weighting tile stitcher
│   └── vectorize.py           # Pixel-to-Polygon geometric translation
├── inference.py               # Standalone inference entry point
└── requirements.txt
```

---

## 🧠 Usage

### 1. The Omega Execution Protocol (Automated)
The easiest way to run the entire pipeline from start to finish is via the master script. This will sequentially run the 3-Fold LOOCV training, run inference on the testing set, generate the GIS files, and calculate the true audit metrics.

```bash
python scripts/omega_pipeline.py
```

### 2. Standalone Training
If you want to tweak parameters (found in `configs/baseline.yaml`) and train the model independently:

```bash
python -m scripts.run_loocv --config configs/baseline.yaml
```

### 3. Standalone Inference
Once you have your `best.pt` model weights, you can run inference on any geospatial image. The pipeline will automatically generate a visual color-overlay, a `.geojson` file, and an ESRI Shapefile (`.shp`).

```bash
python inference.py \
    --image path/to/your/image.tiff \
    --config configs/baseline.yaml \
    --weights results/training/fold_0/best.pt \
    --output_dir results/my_outputs
```

---

## 📊 Classes
By default, the pipeline is configured to extract the following 8 classes from your GeoJSON ground truths:
0. Building
1. Tree
2. Parking
3. Shrub
4. Turf
5. Road
6. Sidewalk
7. Water

---

## 🤝 Contributing
Pull requests are welcome! If you'd like to improve the tile-stitching logic, add new transformer backbones, or expand the GIS capabilities, please fork the repository and open a PR.

## 📝 License
This project is licensed under the MIT License.
