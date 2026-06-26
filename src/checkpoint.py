import os
import torch
import logging

logger = logging.getLogger("geospatial_pipeline.checkpoint")

def save_checkpoint(model, optimizer, scheduler, epoch: int, miou: float, config, output_dir: str, is_best: bool = False):
    """
    Saves model checkpoint and all required metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Model Weights
    last_path = os.path.join(output_dir, 'last.pt')
    torch.save(model.state_dict(), last_path)
    
    if is_best:
        best_path = os.path.join(output_dir, 'best.pt')
        torch.save(model.state_dict(), best_path)
        logger.info(f"Saved new best model (mIoU: {miou:.4f}) to {best_path}")
        
    # Save Optimizer and Scheduler
    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer_state.pt'))
    if scheduler:
        torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler_state.pt'))
        
    # Save Training Metadata
    import json
    metadata = {
        "encoder": config.model.encoder,
        "epochs": epoch,
        "validation_score": float(miou),
        "timestamp": __import__("datetime").datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, 'training_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    # Save Config
    import yaml
    try:
        from omegaconf import OmegaConf
        yaml_str = OmegaConf.to_yaml(config)
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            f.write(yaml_str)
    except:
        pass
