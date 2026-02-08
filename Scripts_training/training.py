"""
Training Utilities Module

Custom training loop, callbacks, learning rate schedulers, and wandb integration.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable, List, Dict, Any
from IPython.display import clear_output

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class PlotLosses:
    """
    Real-time loss plotting during training.

    Attributes
    ----------
    epochs : list
        List of epoch numbers.
    losses : list
        List of training losses.
    val_losses : list
        List of validation losses.
    """

    def __init__(self):
        self.epochs: List[int] = []
        self.losses: List[float] = []
        self.val_losses: List[float] = []

    def update(self, epoch: int, loss: float, val_loss: float):
        """Update history and refresh plot."""
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.val_losses.append(val_loss)
        self.plot()

    def plot(self):
        """Display loss curves."""
        clear_output(wait=True)
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.losses, label="Training Loss", linewidth=2)
        plt.plot(self.epochs, self.val_losses, label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.yscale('log')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def get_history(self) -> dict:
        """Return training history as dictionary."""
        return {
            'epochs': self.epochs,
            'loss': self.losses,
            'val_loss': self.val_losses
        }


class WandbCallback:
    """
    Callback for logging metrics to Weights & Biases.

    Parameters
    ----------
    log_gradients : bool, optional
        Whether to log gradient statistics (default: False).
    """

    def __init__(self, log_gradients: bool = False):
        self.log_gradients = log_gradients

    def on_epoch_end(
        self,
        epoch: int,
        loss: float,
        val_loss: float,
        learning_rate: float,
        extra_metrics: Optional[Dict[str, Any]] = None
    ):
        """Log metrics at end of epoch."""
        if not WANDB_AVAILABLE:
            return
        
        metrics = {
            'epoch': epoch,
            'loss': loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate
        }
        
        if extra_metrics:
            metrics.update(extra_metrics)
        
        wandb.log(metrics)


def create_lr_schedule(
    schedule_config: Dict[tuple, float]
) -> Callable[[int], float]:
    """
    Create a learning rate schedule from configuration.

    Parameters
    ----------
    schedule_config : dict
        Dictionary mapping (start_epoch, end_epoch) to learning rate.

    Returns
    -------
    callable
        Learning rate schedule function.

    Examples
    --------
    >>> schedule = create_lr_schedule({
    ...     (0, 20): 1e-5,
    ...     (20, 40): 5e-5,
    ...     (40, 100): 1e-4,
    ... })
    >>> lr = schedule(30)  # Returns 5e-5
    """
    def schedule(epoch: int) -> float:
        for (start, end), lr in schedule_config.items():
            if start <= epoch < end:
                return lr
        # Return last defined LR as fallback
        return list(schedule_config.values())[-1]
    return schedule


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: Optional[tf.data.Dataset],
    loss_fn,
    training_config,
    use_wandb: bool = False,
    other_modes_datasets: Optional[Dict[str, List[tf.data.Dataset]]] = None
) -> Dict[str, Any]:
    """
    Custom training loop with physics-informed loss.

    Parameters
    ----------
    model : tf.keras.Model
        The U-Net model to train.
    train_dataset : tf.data.Dataset
        Training dataset yielding (input, k_squared, output) tuples.
    val_dataset : tf.data.Dataset or None
        Validation dataset. If None, validation is skipped.
    loss_fn : CustomHelmholtzLoss
        Physics-informed loss function.
    training_config : TrainingConfig
        Training configuration object.
    use_wandb : bool, optional
        Whether to log to Weights & Biases (default: False).
    other_modes_datasets : dict, optional
        Dictionary with 'train' and 'val' keys, each containing list of
        datasets for other eigenmodes (for orthogonality loss).

    Returns
    -------
    dict
        Training history containing losses and metrics.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=training_config.initial_lr)
    
    # Create LR schedule
    lr_schedule = create_lr_schedule(training_config.lr_schedule)
    
    # Initialize logging
    plot_losses = None if use_wandb else PlotLosses()
    wandb_callback = WandbCallback() if use_wandb else None
    
    history = {
        'loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    for epoch in range(training_config.epochs):
        # Update epoch for loss function
        loss_fn.set_current_epoch(epoch)
        
        # Update learning rate
        new_lr = lr_schedule(epoch)
        optimizer.learning_rate.assign(new_lr)
        
        # Training loop
        total_loss = 0.0
        num_batches = 0
        
        for batch_data in train_dataset:
            # Unpack batch data (with or without indices)
            if len(batch_data) == 4:
                img_batch, k_squared_batch, target_batch, indices_batch = batch_data
            else:
                img_batch, k_squared_batch, target_batch = batch_data
                indices_batch = None
            
            # Get other modes for this batch if available (using indices)
            if other_modes_datasets and 'train' in other_modes_datasets and indices_batch is not None:
                indices = indices_batch.numpy()
                other_modes_batch = [
                    tf.gather(mode, indices) for mode in other_modes_datasets['train']
                ]
                loss_fn.set_other_modes(other_modes_batch)
            
            with tf.GradientTape() as tape:
                predictions = model([img_batch, k_squared_batch], training=True)
                loss_fn.set_k_squared(k_squared_batch)
                loss = loss_fn(target_batch, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            total_loss += loss.numpy()
            num_batches += 1
            
            if not use_wandb and num_batches % 10 == 0:
                print(f"\rEpoch {epoch+1}, Batch {num_batches}, Loss: {loss.numpy():.6f}", end="")
        
        epoch_loss = total_loss / num_batches

        # Validation loop (skip if no validation dataset)
        epoch_val_loss = None
        if val_dataset is not None:
            total_val_loss = 0.0
            num_val_batches = 0
            
            for batch_data in val_dataset:
                # Unpack batch data (with or without indices)
                if len(batch_data) == 4:
                    img_batch, k_squared_batch, target_batch, indices_batch = batch_data
                else:
                    img_batch, k_squared_batch, target_batch = batch_data
                    indices_batch = None
                
                # Get other modes for this batch if available (using indices)
                if other_modes_datasets and 'val' in other_modes_datasets and indices_batch is not None:
                    indices = indices_batch.numpy()
                    other_modes_batch = [
                        tf.gather(mode, indices) for mode in other_modes_datasets['val']
                    ]
                    loss_fn.set_other_modes(other_modes_batch)
                
                predictions = model([img_batch, k_squared_batch], training=False)
                loss_fn.set_k_squared(k_squared_batch)
                val_loss = loss_fn(target_batch, predictions)
                
                total_val_loss += val_loss.numpy()
                num_val_batches += 1
            
            epoch_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else None

        # Record history
        history['loss'].append(epoch_loss)
        history['val_loss'].append(epoch_val_loss)
        history['learning_rate'].append(new_lr)

        # Update visualization/logging
        if use_wandb and wandb_callback:
            wandb_callback.on_epoch_end(
                epoch + 1, epoch_loss, epoch_val_loss, new_lr
            )
        elif plot_losses:
            plot_losses.update(epoch + 1, epoch_loss, epoch_val_loss)

        val_loss_str = f"{epoch_val_loss:.6f}" if epoch_val_loss is not None else "N/A"
        print(f"\nEpoch {epoch + 1}/{training_config.epochs}, "
              f"Loss: {epoch_loss:.6f}, Val Loss: {val_loss_str}, LR: {new_lr:.2e}")

    return history
