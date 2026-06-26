import math
from typing import List, Tuple
import numpy as np

def get_tile_coordinates(image_shape: Tuple[int, int], tile_size: int, overlap: int = 0) -> List[Tuple[int, int, int, int]]:
    """
    Generates a list of coordinates for tiling an image.
    
    Args:
        image_shape: (height, width)
        tile_size: Size of the square tile
        overlap: Number of overlapping pixels
        
    Returns:
        List of (ymin, ymax, xmin, xmax)
    """
    height, width = image_shape[:2]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("Overlap must be smaller than tile size")

    y_coords = list(range(0, height - tile_size + 1, stride))
    if y_coords == [] or y_coords[-1] + tile_size < height:
        y_coords.append(max(0, height - tile_size))

    x_coords = list(range(0, width - tile_size + 1, stride))
    if x_coords == [] or x_coords[-1] + tile_size < width:
        x_coords.append(max(0, width - tile_size))

    coords = []
    for y in y_coords:
        for x in x_coords:
            coords.append((y, min(height, y + tile_size), x, min(width, x + tile_size)))
            
    return coords

def extract_tile(image: np.ndarray, coords: Tuple[int, int, int, int], pad_size: int = 0) -> np.ndarray:
    """
    Extracts a tile from an image using provided coordinates, with padding if it exceeds boundaries.
    """
    ymin, ymax, xmin, xmax = coords
    tile = image[ymin:ymax, xmin:xmax]
    
    # If the extracted tile is smaller than expected (e.g., at the edge), pad it
    h, w = tile.shape[:2]
    expected_h = ymax - ymin
    expected_w = xmax - xmin
    
    if h < expected_h or w < expected_w:
        # Pad with reflect or constant
        pad_width = []
        pad_width.append((0, expected_h - h)) # y padding
        pad_width.append((0, expected_w - w)) # x padding
        if tile.ndim == 3:
            pad_width.append((0, 0)) # channel padding
            
        tile = np.pad(tile, pad_width, mode='reflect')
        
    return tile
