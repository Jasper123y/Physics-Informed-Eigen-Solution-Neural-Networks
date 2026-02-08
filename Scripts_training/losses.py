"""
Physics-Informed Loss Functions Module

Custom loss functions for vibration mode prediction including
Helmholtz equation constraints.
"""

import tensorflow as tf
from typing import Optional, List


def compute_laplacian(x: tf.Tensor, pixel_size: float) -> tf.Tensor:
    """
    Compute the Laplacian of a 2D field using finite differences.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor with shape (batch, height, width, channels).
    pixel_size : float
        Physical size of each pixel in meters.

    Returns
    -------
    tf.Tensor
        Laplacian of the input field.
    """
    laplacian_kernel = tf.constant([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=tf.float32) / (pixel_size ** 2)
    laplacian_kernel = laplacian_kernel[:, :, tf.newaxis, tf.newaxis]
    return tf.nn.conv2d(x, laplacian_kernel, strides=[1, 1, 1, 1], padding='SAME')


def symmetric_loss_2d(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Symmetric loss that handles sign ambiguity in mode shapes.

    Mode shapes can be equally valid with opposite signs. This loss
    computes MSE for both orientations and returns the minimum.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth mode shapes.
    y_pred : tf.Tensor
        Predicted mode shapes.

    Returns
    -------
    tf.Tensor
        Symmetric loss value.
    """
    loss_positive = tf.keras.losses.mean_squared_error(y_true, y_pred)
    loss_negative = tf.keras.losses.mean_squared_error(-y_true, y_pred)
    # Penalize predictions skewing too positive/negative
    balance_penalty = tf.abs(tf.reduce_mean(y_pred))
    symmetric_loss = tf.minimum(loss_positive, loss_negative) + balance_penalty
    return symmetric_loss


def helmholtz_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    k_squared: tf.Tensor,
    pixel_size: float,
    peak_value: float = 4.0e-10,
    threshold_ratio: float = 0.05
) -> tf.Tensor:
    """
    Helmholtz equation residual loss.

    Computes the residual of the Helmholtz equation: ∇²u + k²u = 0
    Only considers regions with significant displacement.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth mode shapes (normalized).
    y_pred : tf.Tensor
        Predicted mode shapes (normalized).
    k_squared : tf.Tensor
        Wave number squared values.
    pixel_size : float
        Physical pixel size in meters.
    peak_value : float, optional
        Peak displacement for denormalization (default: 4.0e-10).
    threshold_ratio : float, optional
        Threshold for significant displacement (default: 0.05).

    Returns
    -------
    tf.Tensor
        Helmholtz residual loss.
    """
    # Denormalize to physical displacement
    y_pred_real = y_pred * peak_value
    y_true_real = y_true * peak_value
    
    # Find significant displacement regions
    max_displacement = tf.reduce_max(tf.abs(y_true_real), axis=[1, 2], keepdims=True)
    threshold = threshold_ratio * max_displacement
    significant_mask = tf.abs(y_true_real) > threshold
    
    # Apply mask
    significant_pred = y_pred_real * tf.cast(significant_mask, y_pred_real.dtype)
    
    # Compute Helmholtz residue: ∇²u + k²u = 0
    laplacian = compute_laplacian(tf.abs(significant_pred), pixel_size)
    helmholtz_residue = laplacian + k_squared * tf.abs(significant_pred)
    
    return tf.reduce_mean(tf.square(helmholtz_residue))


def orthogonality_loss(
    y_pred: tf.Tensor,
    other_modes: List[tf.Tensor]
) -> tf.Tensor:
    """
    Orthogonality loss to enforce predicted mode is orthogonal to other eigenmodes.

    For eigenmodes, the inner product between different modes should be zero.
    This loss penalizes non-zero inner products.

    Parameters
    ----------
    y_pred : tf.Tensor
        Predicted mode shape.
    other_modes : list of tf.Tensor
        List of other eigenmode shapes to enforce orthogonality against.

    Returns
    -------
    tf.Tensor
        Orthogonality loss value.
    """
    total_loss = tf.constant(0.0)
    for mode in other_modes:
        dot_product = tf.reduce_sum(y_pred * mode, axis=[1, 2, 3], keepdims=True)
        total_loss += tf.reduce_mean(tf.abs(dot_product))
    return total_loss


def k_squared_deviation_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    pixel_size: float,
    peak_value: float = 4.0e-10,
    threshold_ratio: float = 0.05
) -> tf.Tensor:
    """
    Loss based on standard deviation of computed k² values.

    For a true eigenmode, k² should be constant across the domain.
    This loss penalizes spatial variation in the computed k².

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth mode shapes (normalized).
    y_pred : tf.Tensor
        Predicted mode shapes (normalized).
    pixel_size : float
        Physical pixel size in meters.
    peak_value : float, optional
        Peak displacement for denormalization (default: 4.0e-10).
    threshold_ratio : float, optional
        Threshold for significant displacement (default: 0.05).

    Returns
    -------
    tf.Tensor
        K-squared deviation loss.
    """
    # Denormalize
    y_pred_real = y_pred * peak_value
    y_true_real = y_true * peak_value
    
    # Find significant displacement regions
    max_displacement = tf.reduce_max(tf.abs(y_true_real), axis=[1, 2], keepdims=True)
    threshold = threshold_ratio * max_displacement
    significant_mask = tf.abs(y_true_real) > threshold
    
    # Apply mask
    significant_pred = y_pred_real * tf.cast(significant_mask, y_pred_real.dtype)
    
    # Compute theoretical k²: k² = -∇²u / u
    epsilon = 1e-12
    laplacian = compute_laplacian(tf.abs(significant_pred), pixel_size)
    theoretical_k_squared = -laplacian / (tf.abs(significant_pred) + epsilon)
    
    return tf.math.reduce_std(theoretical_k_squared)


class CustomHelmholtzLoss(tf.keras.losses.Loss):
    """
    Custom physics-informed loss combining MSE and Helmholtz constraints.

    This loss function adapts its behavior based on the training epoch,
    gradually introducing physics-based terms.

    Parameters
    ----------
    pixel_size : float
        Physical size of each pixel in meters.
    peak_value : float, optional
        Peak displacement for normalization (default: 4.0e-10).
    name : str, optional
        Name of the loss function.

    Attributes
    ----------
    k_squared : tf.Tensor
        Current batch's k-squared values (must be set before each batch).
    current_epoch : int
        Current training epoch (affects loss weighting).

    Examples
    --------
    >>> loss_fn = CustomHelmholtzLoss(pixel_size=3.125e-6)
    >>> loss_fn.set_k_squared(k_batch)
    >>> loss_fn.set_current_epoch(50)
    >>> loss = loss_fn(y_true, y_pred)
    """

    def __init__(
        self,
        pixel_size: float,
        peak_value: float = 4.0e-10,
        name: str = "custom_helmholtz_loss"
    ):
        super().__init__(name=name)
        self.pixel_size = pixel_size
        self.peak_value = peak_value
        self.k_squared: Optional[tf.Tensor] = None
        self.other_modes: Optional[List[tf.Tensor]] = None
        self.current_epoch: int = 0
        
        # Loss weights (can be customized)
        self.loss_schedule = {
            (0, 50): {'mse': 1.0, 'symmetric': 0, 'helmholtz': 0, 'k_dev': 0, 'orthogonality': 0},
            (50, 100): {'mse': 1.0, 'symmetric': 0, 'helmholtz': 0, 'k_dev': 0, 'orthogonality': 0},
            (100, 150): {'mse': 0, 'symmetric': 1.0, 'helmholtz': 5e-3, 'k_dev': 2e-15, 'orthogonality': 3e-5},
            (150, 200): {'mse': 0, 'symmetric': 1.0, 'helmholtz': 5e-3, 'k_dev': 2e-15, 'orthogonality': 3e-5},
            (200, float('inf')): {'mse': 0, 'symmetric': 1.0, 'helmholtz': 0, 'k_dev': 0, 'orthogonality': 0},
        }

    def set_k_squared(self, k_squared: tf.Tensor):
        """Set the k-squared values for the current batch."""
        self.k_squared = k_squared

    def set_other_modes(self, other_modes: List[tf.Tensor]):
        """Set the other eigenmodes for orthogonality loss."""
        self.other_modes = other_modes

    def set_current_epoch(self, epoch: int):
        """Update the current epoch."""
        self.current_epoch = epoch
    
    def set_loss_schedule(self, schedule: dict):
        """
        Set a custom loss schedule.
        
        Parameters
        ----------
        schedule : dict
            Dictionary mapping (start_epoch, end_epoch) tuples to weight dicts.
        """
        self.loss_schedule = schedule

    def _get_weights(self) -> dict:
        """Get loss weights for current epoch."""
        for (start, end), weights in self.loss_schedule.items():
            if start <= self.current_epoch < end:
                return weights
        return {'mse': 1.0, 'symmetric': 0, 'helmholtz': 0, 'k_dev': 0}

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the combined loss.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth mode shapes.
        y_pred : tf.Tensor
            Predicted mode shapes.

        Returns
        -------
        tf.Tensor
            Combined loss value.
        """
        if self.k_squared is None:
            raise ValueError("k_squared not set. Call set_k_squared() first.")

        weights = self._get_weights()
        total_loss = tf.constant(0.0)

        if weights['mse'] > 0:
            mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
            total_loss += weights['mse'] * tf.reduce_mean(mse_loss)

        if weights['symmetric'] > 0:
            sym_loss = symmetric_loss_2d(y_true, y_pred)
            total_loss += weights['symmetric'] * tf.reduce_mean(sym_loss)

        if weights['helmholtz'] > 0:
            helm_loss = helmholtz_loss(
                y_true, y_pred, self.k_squared,
                self.pixel_size, self.peak_value
            )
            total_loss += weights['helmholtz'] * helm_loss

        if weights['k_dev'] > 0:
            k_dev_loss = k_squared_deviation_loss(
                y_true, y_pred, self.pixel_size, self.peak_value
            )
            total_loss += weights['k_dev'] * k_dev_loss

        if weights.get('orthogonality', 0) > 0 and self.other_modes is not None:
            ortho_loss = orthogonality_loss(y_pred, self.other_modes)
            total_loss += weights['orthogonality'] * ortho_loss

        return total_loss
