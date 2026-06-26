import numpy as np

def postprocess_mask(prob_map: np.ndarray, threshold: float = 0.5, min_area: int = 32) -> np.ndarray:
    """
    Converts (8, H, W) probability map to an (H, W) integer mask via argmax.
    Pixels where the max probability is below threshold are assigned to 255 (Unclassified).
    """
    mask = np.argmax(prob_map, axis=0).astype(np.uint8)
    max_probs = np.max(prob_map, axis=0)
    mask[max_probs < threshold] = 255
    return mask
