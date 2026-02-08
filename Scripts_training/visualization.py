"""
Visualization Module

Functions for displaying and logging vibration mode predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def display_images(
    images: np.ndarray,
    title: str = "Image",
    is_binary: bool = True,
    num_images: int = 9,
    figsize: Tuple[int, int] = (10, 10)
):
    """
    Display a grid of images.

    Parameters
    ----------
    images : np.ndarray
        Array of images to display.
    title : str, optional
        Title prefix for images (default: "Image").
    is_binary : bool, optional
        Whether to use grayscale colormap (default: True).
    num_images : int, optional
        Number of images to display (default: 9).
    figsize : tuple, optional
        Figure size (default: (10, 10)).
    """
    n = min(num_images, len(images))
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    plt.figure(figsize=figsize)
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        img = images[i].squeeze() if images[i].ndim > 2 else images[i]
        plt.imshow(img, cmap='gray' if is_binary else 'viridis')
        plt.title(f"{title} {i+1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def display_comparison(
    input_images: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    num_samples: int = 3,
    vmin: float = -1,
    vmax: float = 1,
    figsize: Tuple[int, int] = (12, 4)
):
    """
    Display input, prediction, and ground truth side by side.

    Parameters
    ----------
    input_images : np.ndarray
        Input geometry images.
    predictions : np.ndarray
        Model predictions.
    ground_truth : np.ndarray
        Ground truth mode shapes.
    num_samples : int, optional
        Number of samples to display (default: 3).
    vmin : float, optional
        Minimum value for colormap (default: -1).
    vmax : float, optional
        Maximum value for colormap (default: 1).
    figsize : tuple, optional
        Figure size per row (default: (12, 4)).
    """
    n = min(num_samples, len(input_images))
    
    for i in range(n):
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Input
        inp = input_images[i].squeeze()
        axes[0].imshow(inp, cmap='gray')
        axes[0].set_title(f"Input {i+1}")
        axes[0].axis("off")
        
        # Prediction
        pred = predictions[i].squeeze()
        im1 = axes[1].imshow(pred, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Prediction {i+1}")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Ground truth
        truth = ground_truth[i].squeeze()
        im2 = axes[2].imshow(truth, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[2].set_title(f"Ground Truth {i+1}")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        plt.show()


def log_comparison_to_wandb(
    input_images: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    num_samples: int = 6,
    vmin: float = -1,
    vmax: float = 1
):
    """
    Log comparison images to Weights & Biases.

    Parameters
    ----------
    input_images : np.ndarray
        Input geometry images.
    predictions : np.ndarray
        Model predictions.
    ground_truth : np.ndarray
        Ground truth mode shapes.
    num_samples : int, optional
        Number of samples to log (default: 6).
    vmin : float, optional
        Minimum value for colormap (default: -1).
    vmax : float, optional
        Maximum value for colormap (default: 1).
    """
    if not WANDB_AVAILABLE:
        print("wandb not available, skipping logging")
        return
    
    n = min(num_samples, len(input_images))
    
    for i in range(n):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Input
        inp = input_images[i].squeeze()
        axes[0].imshow(inp, cmap='gray')
        axes[0].set_title(f"Input", fontsize=12, fontweight='bold')
        axes[0].axis("off")
        
        # Prediction
        pred = predictions[i].squeeze()
        im1 = axes[1].imshow(pred, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Prediction", fontsize=12, fontweight='bold')
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Ground truth
        truth = ground_truth[i].squeeze()
        im2 = axes[2].imshow(truth, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[2].set_title(f"Ground Truth", fontsize=12, fontweight='bold')
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        
        wandb.log({f"comparison_sample_{i+1}": wandb.Image(fig)})
        plt.close(fig)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Compute prediction metrics for mode shapes.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth mode shapes.
    y_pred : np.ndarray
        Predicted mode shapes.

    Returns
    -------
    dict
        Dictionary containing MSE, MAE, and correlation metrics.
    """
    # Handle sign ambiguity - compare both orientations
    mse_pos = np.mean((y_true - y_pred) ** 2)
    mse_neg = np.mean((y_true + y_pred) ** 2)
    
    # Use the better orientation
    if mse_neg < mse_pos:
        y_pred_aligned = -y_pred
        mse = mse_neg
    else:
        y_pred_aligned = y_pred
        mse = mse_pos
    
    mae = np.mean(np.abs(y_true - y_pred_aligned))
    
    # Compute correlation per sample
    correlations = []
    for i in range(len(y_true)):
        true_flat = y_true[i].flatten()
        pred_flat = y_pred_aligned[i].flatten()
        if np.std(true_flat) > 0 and np.std(pred_flat) > 0:
            corr = np.corrcoef(true_flat, pred_flat)[0, 1]
            correlations.append(corr)
    
    mean_correlation = np.mean(correlations) if correlations else 0.0
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'mean_correlation': mean_correlation
    }


def plot_training_history(
    history: dict,
    figsize: Tuple[int, int] = (12, 4)
):
    """
    Plot training history.

    Parameters
    ----------
    history : dict
        Training history dictionary.
    figsize : tuple, optional
        Figure size (default: (12, 4)).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=10)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate plot
    if 'learning_rate' in history:
        axes[1].plot(history['learning_rate'], linewidth=2, color='green')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_yscale('log')
        axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
