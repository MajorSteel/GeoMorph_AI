import os
import json
import platform
import subprocess
import time
import torch
import psutil

def get_git_commit():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
        return commit
    except Exception:
        return "Not a git repository"

def get_system_metadata():
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "os": platform.system() + " " + platform.release(),
        "python_version": platform.python_version(),
        "cpu_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "pytorch_version": torch.__version__,
        "git_commit": get_git_commit()
    }

def save_training_metadata(config, best_iou: float, output_path: str):
    """Saves comprehensive training metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    metadata = {
        "system": get_system_metadata(),
        "training_results": {
            "best_validation_miou": best_iou,
            "status": "completed"
        },
        "hyperparameters": {
            "architecture": config.model.architecture,
            "encoder": config.model.encoder,
            "epochs": config.training.epochs,
            "loss": config.training.loss,
            "optimizer": "AdamW",
            "encoder_lr": config.training.encoder_lr,
            "decoder_lr": config.training.decoder_lr,
            "tile_size": config.data.tile_size,
            "batch_size": config.data.batch_size
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    return metadata
