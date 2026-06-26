import segmentation_models_pytorch as smp
import logging
from transformers import SegformerForSemanticSegmentation, SegformerConfig

logger = logging.getLogger("geospatial_pipeline.model_registry")

def get_model(architecture: str, encoder: str, encoder_weights: str, in_channels: int, classes: int):
    """
    Returns a segmentation model based on configuration.
    Currently focuses on the robust SMP baseline to secure the pipeline.
    """
    architecture = architecture.lower()
    
    if architecture == "unet":
        logger.info(f"Creating U-Net with {encoder} encoder")
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights if encoder_weights != "none" else None,
            in_channels=in_channels,
            classes=classes
        )
    elif architecture == "unetplusplus":
        logger.info(f"Creating U-Net++ with {encoder} encoder")
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights if encoder_weights != "none" else None,
            in_channels=in_channels,
            classes=classes
        )
    elif architecture == "deeplabv3plus":
        logger.info(f"Creating DeepLabV3+ with {encoder} encoder")
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights if encoder_weights != "none" else None,
            in_channels=in_channels,
            classes=classes
        )
    elif architecture == "segformer":
        logger.info(f"Creating HuggingFace SegFormer with {encoder} encoder for {classes} classes")
        # Load the pre-trained segformer config, update num_labels, and create model
        config = SegformerConfig.from_pretrained(encoder)
        config.num_labels = classes
        model = SegformerForSemanticSegmentation.from_pretrained(
            encoder, 
            config=config, 
            ignore_mismatched_sizes=True
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
        
    return model
