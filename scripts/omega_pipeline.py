import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - OMEGA - %(levelname)s - %(message)s')
logger = logging.getLogger("omega")

def run_command(cmd, desc):
    logger.info(f"Running Phase: {desc}")
    logger.info(f"Command: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        logger.error(f"Phase FAILED: {desc}")
        sys.exit(1)
    logger.info(f"Phase SUCCESS: {desc}\n")

def main():
    logger.info("Initializing TRUE OMEGA EXECUTION PROTOCOL...")
    
    # 1. Training & Validation
    run_command("python -m scripts.run_loocv --config configs/baseline.yaml", "Phase 1: Training & LOOCV")
    
    # 2. Inference & Generalization
    images = ["1.jpg", "2.tiff", "3.jpg"]
    os.makedirs("results/inference", exist_ok=True)
    for img in images:
        path = f"aerial_imagery_pack/aerial_imagery_pack/{img}"
        run_command(f"python inference.py --image {path} --config configs/baseline.yaml --weights results/training/fold_0/best.pt", f"Phase 2: Inference on {img}")
        
    # 3. True Auditing
    run_command("python scripts/audit_inference.py", "Phase 3: True Performance Audit")
    
    logger.info("OMEGA EXECUTION COMPLETE. All artifacts and true metrics generated successfully.")

if __name__ == "__main__":
    main()
