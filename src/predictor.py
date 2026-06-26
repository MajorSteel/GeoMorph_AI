import torch
import numpy as np
import logging
from typing import Tuple
import torch.nn.functional as F
from src.tiling import get_tile_coordinates, extract_tile
from src.stitcher import TileStitcher

logger = logging.getLogger("geospatial_pipeline.predictor")

@torch.no_grad()
def predict_image(
    model: torch.nn.Module, 
    image: np.ndarray, 
    transform, 
    tile_size: int = 512, 
    overlap: int = 128,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> np.ndarray:
    """
    Runs sliding window inference over a full image.
    """
    model.eval()
    model.to(device)
    
    height, width = image.shape[:2]
    num_classes = 8
    
    stitcher = TileStitcher((height, width), num_classes=num_classes)
    coords_list = get_tile_coordinates((height, width), tile_size, overlap)
    
    total_tiles = len(coords_list)
    logger.info(f"Running inference on {total_tiles} tiles...")
    
    for i, coords in enumerate(coords_list):
        if i % 50 == 0:
            logger.info(f"Processing tile {i}/{total_tiles}")
            
        tile_np = extract_tile(image, coords)
        
        augmented = transform(image=tile_np)
        tile_tensor = augmented['image'].unsqueeze(0).to(device) # (1, C, H, W)
        
        with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            if hasattr(model, 'segformer'):
                outputs = model(tile_tensor)
                logits = outputs.logits
                logits = F.interpolate(logits, size=tile_tensor.shape[-2:], mode="bilinear", align_corners=False)
            else:
                logits = model(tile_tensor)
                
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy() # (8, H, W)
                
        stitcher.add_prediction(probs, coords)
        
    logger.info("Inference complete. Stitching probabilities...")
    return stitcher.get_final_probabilities()
