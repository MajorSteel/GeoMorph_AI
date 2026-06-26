import torch

def compute_iou(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Computes Intersection over Union for multiclass segmentation.
    Uses argmax to select the predicted class, and computes macro-IoU across present classes.
    """
    with torch.no_grad():
        # y_pred: (N, C, H, W) -> (N, H, W)
        pred_labels = torch.argmax(y_pred, dim=1)
        
        # Ensure y_true has no channel dimension
        if y_true.ndim == 4 and y_true.size(1) == 1:
            y_true = y_true.squeeze(1)
        
        y_true = y_true.long()
        
        iou_list = []
        classes = torch.unique(y_true)
        
        for cls in classes:
            pred_inds = pred_labels == cls
            target_inds = y_true == cls
            
            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()
            
            if union > 0:
                iou_list.append(intersection / union)
                
        if not iou_list:
            return 1.0 # Both empty
            
        return float(torch.tensor(iou_list).mean())
