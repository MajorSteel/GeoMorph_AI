# Deployment & Integration

GeoMorph AI is designed to be completely hardware-agnostic and highly portable. 

## 1. Web Deployment (HuggingFace Spaces / Gradio)
The repository includes a `demo_app.py` built on Gradio. This allows you to instantly spin up a local Web UI or deploy the repository directly to a HuggingFace Space.
```bash
python demo_app.py
```

## 2. Docker Containerization
To deploy on AWS EC2 or Kubernetes, we recommend wrapping the inference script in a Docker container.
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
WORKDIR /app
COPY . /app
RUN pip install -e .
CMD ["geomorph", "predict", "--image", "data/test.tif"]
```

## 3. API Integration (FastAPI)
For enterprise integration, you can wrap `scripts.cli.predict` inside a FastAPI endpoint to serve real-time geospatial inference requests via REST API.
