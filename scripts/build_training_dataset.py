import os
import json
import glob
import shutil

# Confidence thresholds specific to each class's known fallback performance
THRESHOLDS = {
    "Building": 0.75,
    "Tree": 0.80,
    "Water": 0.70,
    "Road": 0.60,
    "Parking": 0.65,
    "Sidewalk": 0.55,
    "Shrub": 0.60
}

INPUT_GEOJSON = "feature_layers/feature_layers/GeoJSON"
INPUT_PSEUDO = "feature_layers/PseudoLabels"
OUTPUT_DIR = "feature_layers/TrainingDataset"

def build_dataset():
    print("========================================")
    print(" BUILDING FINAL TRAINING DATASET ")
    print("========================================\n")
    
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    merged_data = {}
    
    # 1. Load immutable ground-truth (Turf)
    print("Loading Ground-Truth (Turf)...")
    gt_files = glob.glob(os.path.join(INPUT_GEOJSON, "*.geojson"))
    for f in gt_files:
        basename = os.path.basename(f)
        with open(f, 'r') as file:
            data = json.load(file)
            
        # Add class property to Turf since the original didn't have it
        for feature in data.get("features", []):
            feature["properties"]["class"] = "Turf"
            feature["properties"]["source"] = "Human"
            feature["properties"]["confidence"] = 1.0
            
        merged_data[basename] = data["features"]
        print(f"  -> Added {len(data['features'])} verified Turf polygons to {basename}")

    # 2. Safely merge Approved Pseudo-Labels
    print("\nProcessing Reviewed Pseudo-Labels...")
    classes = [d for d in os.listdir(INPUT_PSEUDO) if os.path.isdir(os.path.join(INPUT_PSEUDO, d))]
    
    for cls in classes:
        threshold = THRESHOLDS.get(cls, 0.6)
        files = glob.glob(os.path.join(INPUT_PSEUDO, cls, "*.geojson"))
        
        approved_count = 0
        dropped_count = 0
        
        for f in files:
            basename = os.path.basename(f)
            if basename not in merged_data:
                merged_data[basename] = []
                
            with open(f, 'r') as file:
                data = json.load(file)
                
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                
                # Rigid Validation Gates
                if props.get("review_status") != "approved":
                    dropped_count += 1
                    continue
                if props.get("confidence", 0) < threshold:
                    dropped_count += 1
                    continue
                
                merged_data[basename].append(feature)
                approved_count += 1
                
        print(f"  -> {cls}: Merged {approved_count} polygons. Dropped {dropped_count} (unreviewed/low-confidence).")
        
    # 3. Export to immutable TrainingDataset
    print("\nExporting to TrainingDataset/...")
    for basename, features in merged_data.items():
        out_path = os.path.join(OUTPUT_DIR, basename)
        feature_collection = {
            "type": "FeatureCollection",
            "name": "Compiled_Training_Data",
            "features": features
        }
        with open(out_path, 'w') as f:
            json.dump(feature_collection, f, indent=2)
        print(f"  -> Saved {out_path} ({len(features)} total features)")
        
    print("\nCOMPILATION COMPLETE! The pipeline is ready for retraining.")

if __name__ == "__main__":
    build_dataset()
