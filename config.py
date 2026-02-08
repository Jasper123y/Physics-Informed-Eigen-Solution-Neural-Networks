"""
Minimal configuration classes - keeps training parameters in training.py
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Data paths and basic properties."""
    input_folder: str = './data_training/Input'
    k_squared_file: str = ''
    all_mode_folders: list = field(default_factory=list)
    image_shape: tuple = (128, 128)
    peak_value: float = 4.0e-10
    real_width: float = 500e-6
    test_size: float = 0.9
    random_state: int = 42
    
    @property
    def pixel_size(self):
        return self.real_width / self.image_shape[0]


@dataclass
class ModelConfig:
    """Model architecture settings."""
    input_shape: tuple = (128, 128, 1)
    filters: tuple = (32, 64, 128, 256)
    kernel_size: tuple = (5, 5)
    output_activation: str = 'tanh'


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 230
    batch_size: int = 32
    initial_lr: float = 1e-5
    lr_schedule: dict = field(default_factory=dict)
    loss_schedule: dict = field(default_factory=dict)


@dataclass
class WandbConfig:
    """Weights & Biases settings."""
    project: str = "vibration-mode-prediction"
    entity: str = ""
    name: str = ""
    enabled: bool = True


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
