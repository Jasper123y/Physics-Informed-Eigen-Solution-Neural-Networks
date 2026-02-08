"""
Vibration Mode Prediction Package

A U-Net based approach with physics-informed loss for predicting 
2D vibration modes from structural geometry.
"""

from .data_loader import load_geometry_images, load_mode_shapes, load_all_mode_shapes, prepare_dataset
from .model import create_unet_model
from .losses import CustomHelmholtzLoss, symmetric_loss_2d, helmholtz_loss, orthogonality_loss
from .training import train_model, PlotLosses, WandbCallback
from .visualization import display_images, display_comparison, log_comparison_to_wandb

__version__ = "1.0.0"
__author__ = "Jiapeng xU"

__all__ = [
    "load_geometry_images",
    "load_mode_shapes",
    "load_all_mode_shapes",
    "prepare_dataset",
    "create_unet_model",
    "CustomHelmholtzLoss",
    "symmetric_loss_2d",
    "helmholtz_loss",
    "orthogonality_loss",
    "train_model",
    "PlotLosses",
    "WandbCallback",
    "display_images",
    "display_comparison",
    "log_comparison_to_wandb",
]
