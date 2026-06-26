import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def get_loss_fn(loss_name: str, **kwargs):
    """
    Returns the configured loss function.
    Supports combined Focal + Dice as default.
    """
    loss_name = loss_name.lower()
    
    # Extract optional class weights
    class_weights = kwargs.get('class_weights', None)
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    if loss_name == "focal":
        return smp.losses.FocalLoss(mode=smp.losses.MULTICLASS_MODE, alpha=kwargs.get('alpha', 0.25), gamma=kwargs.get('gamma', 2.0))
        
    elif loss_name == "dice":
        return smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE)
        
    elif loss_name == "focal_dice":
        focal = smp.losses.FocalLoss(mode=smp.losses.MULTICLASS_MODE, alpha=kwargs.get('alpha', 0.25), gamma=kwargs.get('gamma', 2.0))
        dice = smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE)
        
        class CombinedFocalDiceLoss(nn.Module):
            def __init__(self, focal, dice):
                super().__init__()
                self.focal = focal
                self.dice = dice
                
            def forward(self, y_pred, y_true):
                y_true = y_true.long()
                if y_true.ndim == 4 and y_true.size(1) == 1:
                    y_true = y_true.squeeze(1)
                return self.focal(y_pred, y_true) + self.dice(y_pred, y_true)
                
        return CombinedFocalDiceLoss(focal, dice)
        
    elif loss_name == "ce_dice":
        ce = nn.CrossEntropyLoss(weight=class_weights)
        dice = smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE)
        
        class CombinedCEDiceLoss(nn.Module):
            def __init__(self, ce, dice):
                super().__init__()
                self.ce = ce
                self.dice = dice
                
            def forward(self, y_pred, y_true):
                y_true = y_true.long()
                if y_true.ndim == 4 and y_true.size(1) == 1:
                    y_true = y_true.squeeze(1)
                
                # CrossEntropy expects y_true to not be one-hot, which it isn't (shape: N, H, W)
                ce_loss = self.ce(y_pred, y_true)
                dice_loss = self.dice(y_pred, y_true)
                return ce_loss + dice_loss
                
        return CombinedCEDiceLoss(ce, dice)
        
    else:
        # Default fallback
        return nn.BCEWithLogitsLoss()
