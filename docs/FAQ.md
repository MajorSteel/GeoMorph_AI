# Frequently Asked Questions

**Q: Can I train on custom classes?**
A: Yes! Simply update the `class_mapping` dictionary inside `src/data_ingestion.py` and ensure your `.geojson` files contain those exact string properties. The model will automatically adjust its architecture to support the new class count.

**Q: Does it support multi-spectral (e.g., 10-band Sentinel-2) imagery?**
A: Out of the box, the SegFormer is configured for 3-channel (RGB) imagery. To support multi-spectral imagery, you must modify the `in_channels` of the model registry and bypass standard image normalizers.

**Q: Why does inference take a long time on massive TIFFs?**
A: Massive TIFFs are processed using sliding windows. A `10,000 x 10,000` image requires hundreds of tile passes. Use a GPU to accelerate the model inference.
