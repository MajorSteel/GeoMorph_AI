import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ProjectConfig:
    name: str = "geospatial-feature-detection"
    seed: int = 42

@dataclass
class DataConfig:
    image_dir: str = "aerial_imagery_pack"
    label_dir: str = "feature_layers"
    num_workers: int = 4
    tile_size: int = 512
    inference_overlap: int = 128
    batch_size: int = 4
    bg_suppression_threshold: float = 0.95

@dataclass
class ModelConfig:
    architecture: str = "segformer"
    encoder: str = "nvidia/mit-b0"
    encoder_weights: str = "imagenet"
    in_channels: int = 3
    classes: int = 8

@dataclass
class TrainingConfig:
    epochs: int = 40
    phase1_epochs: int = 10
    encoder_lr: float = 1e-5
    decoder_lr: float = 1e-4
    weight_decay: float = 1e-2
    patience: int = 7
    mixed_precision: bool = True
    loss: str = "focal_dice"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    class_weights: Optional[list] = None

@dataclass
class InferenceConfig:
    threshold: float = 0.5
    min_area: int = 32
    simplify_tolerance: float = 2.0

@dataclass
class PipelineConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r") as f:
            cfg_dict = yaml.safe_load(f)
            
        return cls(
            project=ProjectConfig(**cfg_dict.get("project", {})),
            data=DataConfig(**cfg_dict.get("data", {})),
            model=ModelConfig(**cfg_dict.get("model", {})),
            training=TrainingConfig(**cfg_dict.get("training", {})),
            inference=InferenceConfig(**cfg_dict.get("inference", {}))
        )
