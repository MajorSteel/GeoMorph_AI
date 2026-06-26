import argparse
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description="GeoMorph AI - Geospatial Intelligence Framework")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model using LOOCV")
    train_parser.add_argument("--config", default="configs/baseline.yaml", help="Path to config file")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference on an image")
    predict_parser.add_argument("--image", required=True, help="Path to image")
    predict_parser.add_argument("--weights", required=True, help="Path to weights")
    predict_parser.add_argument("--config", default="configs/baseline.yaml", help="Path to config file")
    predict_parser.add_argument("--output_dir", default="results/inference", help="Output directory")

    # Vectorize command
    vectorize_parser = subparsers.add_parser("vectorize", help="Vectorize a prediction mask into GIS polygons")
    vectorize_parser.add_argument("--mask", required=True, help="Path to mask file")
    vectorize_parser.add_argument("--output", required=True, help="Path to output .shp or .geojson")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarking on the model")
    benchmark_parser.add_argument("--config", default="configs/baseline.yaml", help="Path to config")

    args = parser.parse_args()

    if args.command == "train":
        print(f"Running GeoMorph Training with config {args.config}...")
        subprocess.run([sys.executable, "-m", "scripts.run_loocv", "--config", args.config])
    
    elif args.command == "predict":
        print(f"Running GeoMorph Prediction on {args.image}...")
        subprocess.run([sys.executable, "inference.py", "--image", args.image, "--weights", args.weights, "--config", args.config, "--output_dir", args.output_dir])
    
    elif args.command == "vectorize":
        print(f"Running GeoMorph Vectorization on {args.mask}...")
        print("Vectorization standalone not fully implemented yet, delegating to pipeline.")
        # placeholder for vectorize call
    
    elif args.command == "benchmark":
        print("Running GeoMorph Benchmark...")
        subprocess.run([sys.executable, "-m", "scripts.benchmark", "--config", args.config])
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
