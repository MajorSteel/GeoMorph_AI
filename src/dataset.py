import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Callable
import logging

logger = logging.getLogger("geospatial_pipeline.dataset")

class AerialSegmentationDataset(Dataset):
    def __init__(
        self, 
        images: List[np.ndarray], 
        masks: List[np.ndarray], 
        coordinates: List[List[Tuple[int, int, int, int]]],
        transform: Callable = None,
        bg_suppression_threshold: float = 1.0
    ):
        """
        images: List of full images
        masks: List of full masks
        coordinates: List of coordinate lists. coordinates[i] has tiles for images[i]
        bg_suppression_threshold: Skip tiles where background (0) > threshold
        """
        self.images = images
        self.masks = masks
        self.transform = transform
        
        # Flatten the coordinates and filter out mostly-background tiles
        self.samples = []
        skipped = 0
        
        for img_idx, coords_list in enumerate(coordinates):
            for coords in coords_list:
                ymin, ymax, xmin, xmax = coords
                # Check background ratio in mask
                mask_tile = self.masks[img_idx][ymin:ymax, xmin:xmax]
                bg_ratio = np.sum(mask_tile == 0) / mask_tile.size
                
                if bg_ratio < bg_suppression_threshold:
                    self.samples.append((img_idx, coords))
                else:
                    skipped += 1
                    
        logger.info(f"Dataset created with {len(self.samples)} valid tiles. Skipped {skipped} background tiles.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, (ymin, ymax, xmin, xmax) = self.samples[idx]
        
        image = self.images[img_idx][ymin:ymax, xmin:xmax]
        mask = self.masks[img_idx][ymin:ymax, xmin:xmax]
        
        # Apply padding if on the edge
        # We ensure all outputs are exactly tile_size handled externally or implicitly via crop
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Ensure mask is long tensor
        mask = mask.long()
        
        return image, mask
