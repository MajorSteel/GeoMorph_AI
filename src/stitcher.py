import numpy as np
import scipy.ndimage

def get_distance_weights(tile_size: int) -> np.ndarray:
    """
    Creates a distance-to-center weighting matrix for a square tile.
    Center pixels have weight ~1, edge pixels approach 0.
    """
    center = tile_size // 2
    y, x = np.ogrid[:tile_size, :tile_size]
    
    # Distance from center
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Max distance is at the corners
    max_dist = np.sqrt(2 * (center**2))
    
    # Invert and normalize (cosine taper is better but linear is fine for baseline)
    # Using cosine bell
    weights = np.cos((dist_from_center / max_dist) * (np.pi / 2))
    # Ensure no exact zeros at the edge to prevent division by zero
    weights = np.clip(weights, 1e-4, 1.0)
    
    return weights

class TileStitcher:
    def __init__(self, image_shape: tuple, num_classes: int = 1):
        """
        image_shape: (height, width)
        num_classes: Number of prediction classes
        """
        self.height, self.width = image_shape[:2]
        self.num_classes = num_classes
        
        # Accumulators
        if num_classes == 1:
            self.prob_map = np.zeros((self.height, self.width), dtype=np.float32)
        else:
            self.prob_map = np.zeros((num_classes, self.height, self.width), dtype=np.float32)
            
        self.weight_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Cache weights to avoid recomputing for same tile sizes
        self.weight_cache = {}

    def add_prediction(self, prob_tile: np.ndarray, coords: tuple):
        """
        prob_tile: (height, width) or (num_classes, height, width)
        coords: (ymin, ymax, xmin, xmax)
        """
        ymin, ymax, xmin, xmax = coords
        tile_h = ymax - ymin
        tile_w = xmax - xmin
        
        # In case the prob_tile is larger (due to padding during inference), crop it
        if prob_tile.ndim == 2:
            prob_tile = prob_tile[:tile_h, :tile_w]
        else:
            prob_tile = prob_tile[:, :tile_h, :tile_w]
            
        # Get weight matrix
        tile_size = max(prob_tile.shape[-2:])
        if tile_size not in self.weight_cache:
            self.weight_cache[tile_size] = get_distance_weights(tile_size)
            
        w_tile = self.weight_cache[tile_size][:tile_h, :tile_w]
        
        if self.num_classes == 1:
            self.prob_map[ymin:ymax, xmin:xmax] += (prob_tile * w_tile)
        else:
            # Broadcast weight across classes
            self.prob_map[:, ymin:ymax, xmin:xmax] += (prob_tile * w_tile)
            
        self.weight_map[ymin:ymax, xmin:xmax] += w_tile

    def get_final_probabilities(self) -> np.ndarray:
        """
        Returns the merged probability map.
        """
        # Avoid division by zero
        safe_weight = np.where(self.weight_map == 0, 1.0, self.weight_map)
        
        if self.num_classes == 1:
            final_probs = self.prob_map / safe_weight
            final_probs = np.clip(final_probs, 0.0, 1.0)
        else:
            final_probs = self.prob_map / safe_weight
            final_probs = np.clip(final_probs, 0.0, 1.0)
            
        return final_probs
