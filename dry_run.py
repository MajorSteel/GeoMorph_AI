import torch
from src.model_registry import get_model
from src.losses import get_loss_fn
from src.metrics import compute_iou
import os

print("Testing SegFormer Architecture Setup...")
try:
    # 1. Instantiate Segformer for 8 classes
    model = get_model(architecture="segformer", encoder="nvidia/mit-b0", encoder_weights="imagenet", in_channels=3, classes=8)
    print("[OK] Model instantiated successfully.")

    # 2. Dummy Data (Batch Size 2, 3 Channels, 512x512)
    dummy_images = torch.randn(2, 3, 512, 512)
    # Dummy Targets (Batch Size 2, 512x512) - values 0-7
    dummy_targets = torch.randint(0, 8, (2, 512, 512)).long()
    
    # 3. Forward Pass
    outputs = model(dummy_images)
    print(f"[OK] Forward pass successful. Output type: {type(outputs)}")
    
    # 4. Upsampling logic (replicating train.py)
    if hasattr(outputs, "logits"):
        logits = outputs.logits
        print(f"[OK] Logits extracted. Original shape: {logits.shape}")
        logits = torch.nn.functional.interpolate(logits, size=dummy_targets.shape[-2:], mode="bilinear", align_corners=False)
        print(f"[OK] Logits upsampled. New shape: {logits.shape}")
    else:
        logits = outputs
        
    # 5. Loss Calculation (MULTICLASS_MODE)
    criterion = get_loss_fn("focal_dice")
    loss = criterion(logits, dummy_targets)
    print(f"[OK] Loss computed successfully: {loss.item()}")
    
    # 6. Backward Pass
    loss.backward()
    print(f"[OK] Backward pass (gradient flow) successful.")
    
    # 7. Metrics
    iou = compute_iou(logits, dummy_targets)
    print(f"[OK] Macro-IoU computed successfully: {iou}")
    
    print("\nALL SEGFORMER TESTS PASSED!")
    
except Exception as e:
    print(f"\n[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()
