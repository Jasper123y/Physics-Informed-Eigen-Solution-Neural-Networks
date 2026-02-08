"""
Data Loading Module

Functions for loading geometry images, vibration mode shapes, and k-squared values.
"""

import numpy as np
import os
import tensorflow as tf
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split


def load_geometry_images(
    folder_path: str,
    image_shape: Tuple[int, int] = (128, 128),
    add_channel_dim: bool = True
) -> np.ndarray:
    """
    Load geometry images from text files.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing .txt data files.
    image_shape : tuple of int, optional
        Shape to reshape the data into (default: (128, 128)).
    add_channel_dim : bool, optional
        Whether to add a channel dimension for CNN input (default: True).

    Returns
    -------
    np.ndarray
        Array of geometry images.
    """
    data_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    data_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    
    reshaped_data_list = []
    for data_file in data_files:
        data_path = os.path.join(folder_path, data_file)
        data = np.loadtxt(data_path, delimiter=',', usecols=[2], skiprows=1)
        reshaped_data = data.reshape(image_shape)
        reshaped_data_list.append(reshaped_data)

    images = np.array(reshaped_data_list)
    
    if add_channel_dim:
        images = images.reshape(images.shape[0], *image_shape, 1)
    
    return images


def load_mode_shapes(
    folder_path: str,
    image_shape: Tuple[int, int] = (128, 128),
    transpose: bool = True,
    add_channel_dim: bool = True
) -> np.ndarray:
    """
    Load vibration mode shape images from text files.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing mode shape .txt files.
    image_shape : tuple of int, optional
        Shape to reshape the data into (default: (128, 128)).
    transpose : bool, optional
        Whether to transpose the data (default: True).
    add_channel_dim : bool, optional
        Whether to add a channel dimension (default: True).

    Returns
    -------
    np.ndarray
        Array of mode shape images.
    """
    data_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    # Sort by second-to-last element for truth files
    data_files.sort(key=lambda f: int(f.split('_')[-2]))
    
    reshaped_data_list = []
    for data_file in data_files:
        data_path = os.path.join(folder_path, data_file)
        data = np.loadtxt(data_path, delimiter=',', usecols=[2], skiprows=1)
        reshaped_data = data.reshape(image_shape)
        if transpose:
            reshaped_data = reshaped_data.T
        reshaped_data_list.append(reshaped_data)

    images = np.array(reshaped_data_list)
    
    if add_channel_dim:
        images = images.reshape(images.shape[0], *image_shape, 1)
    
    return images


def load_all_mode_shapes(
    folder_paths: List[str],
    image_shape: Tuple[int, int] = (128, 128),
    transpose: bool = True,
    add_channel_dim: bool = True
) -> List[np.ndarray]:
    """
    Load all 6 eigenmode shapes from multiple folders.

    Parameters
    ----------
    folder_paths : list of str
        List of paths to folders containing mode shape files for each mode.
    image_shape : tuple of int, optional
        Shape to reshape the data into (default: (128, 128)).
    transpose : bool, optional
        Whether to transpose the data (default: True).
    add_channel_dim : bool, optional
        Whether to add a channel dimension (default: True).

    Returns
    -------
    list of np.ndarray
        List of arrays, one for each eigenmode.

    Examples
    --------
    >>> folders = [f'/path/to/Truthfile_{i}th' for i in range(1, 7)]
    >>> all_modes = load_all_mode_shapes(folders)
    >>> target_mode = all_modes[1]  # 2nd mode for training
    >>> other_modes = [all_modes[i] for i in [0, 2, 3, 4, 5]]  # Others for orthogonality
    """
    all_modes = []
    for folder_path in folder_paths:
        modes = load_mode_shapes(
            folder_path, image_shape, transpose, add_channel_dim
        )
        all_modes.append(modes)
    return all_modes


def load_k_squared(file_path: str, column: int = 2) -> np.ndarray:
    """
    Load k-squared values from a file.

    Parameters
    ----------
    file_path : str
        Path to the k-squared values file.
    column : int, optional
        Column index to read (default: 2).

    Returns
    -------
    np.ndarray
        Array of k-squared values with shape (N, 1).
    """
    k_squared = np.genfromtxt(file_path, usecols=(column), delimiter=',', skip_header=1)
    return k_squared.reshape(-1, 1)


def normalize_data(
    input_images: np.ndarray,
    output_images: np.ndarray,
    peak_value: float = 4.0e-10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize input and output images.

    Parameters
    ----------
    input_images : np.ndarray
        Raw input geometry images.
    output_images : np.ndarray
        Raw output mode shape images.
    peak_value : float, optional
        Peak displacement value for normalization (default: 4.0e-10).

    Returns
    -------
    tuple of np.ndarray
        (normalized_input, normalized_output)
    """
    normalized_input = input_images.astype(np.float32)
    normalized_output = output_images.astype(np.float32) / peak_value
    return normalized_input, normalized_output


def expand_k_squared(
    k_squared: np.ndarray,
    target_shape: Tuple[int, ...]
) -> tf.Tensor:
    """
    Expand k_squared to match the shape of images for broadcasting.

    Parameters
    ----------
    k_squared : np.ndarray
        K-squared values with shape (N, 1).
    target_shape : tuple
        Target shape to broadcast to.

    Returns
    -------
    tf.Tensor
        Expanded k_squared tensor.
    """
    k_squared = tf.cast(k_squared, tf.float32)
    k_squared_expanded = tf.expand_dims(tf.expand_dims(k_squared, -1), -1)
    return tf.broadcast_to(k_squared_expanded, target_shape)


def prepare_dataset(
    input_path: str,
    output_path: str,
    k_squared_path: str,
    peak_value: float = 4.0e-10,
    test_size: float = 0.3,
    val_split: float = 0.5,
    random_state: int = 42
) -> dict:
    """
    Load and prepare complete dataset with train/val/test splits.

    Parameters
    ----------
    input_path : str
        Path to input geometry images folder.
    output_path : str
        Path to output mode shapes folder.
    k_squared_path : str
        Path to k-squared values file.
    peak_value : float, optional
        Peak displacement for normalization (default: 4.0e-10).
    test_size : float, optional
        Fraction for test+val split (default: 0.3).
    val_split : float, optional
        Fraction of test_size for validation (default: 0.5).
    random_state : int, optional
        Random seed (default: 42).

    Returns
    -------
    dict
        Dictionary containing all dataset splits.
    """
    # Load raw data
    input_images = load_geometry_images(input_path)
    output_images = load_mode_shapes(output_path)
    k_squared = load_k_squared(k_squared_path)
    
    # Normalize
    input_norm, output_norm = normalize_data(input_images, output_images, peak_value)
    
    # First split: train vs test+val
    (input_train, input_temp, 
     output_train, output_temp,
     k_train, k_temp) = train_test_split(
        input_norm, output_norm, k_squared,
        test_size=test_size, random_state=random_state
    )
    
    # Second split: val vs test
    (input_val, input_test,
     output_val, output_test,
     k_val, k_test) = train_test_split(
        input_temp, output_temp, k_temp,
        test_size=val_split, random_state=random_state
    )
    
    # Expand k_squared for training and validation
    k_train_expanded = expand_k_squared(k_train, tf.shape(input_train))
    k_val_expanded = expand_k_squared(k_val, tf.shape(input_val))
    
    return {
        'train': {
            'input': input_train,
            'output': output_train,
            'k_squared': k_train,
            'k_squared_expanded': k_train_expanded
        },
        'val': {
            'input': input_val,
            'output': output_val,
            'k_squared': k_val,
            'k_squared_expanded': k_val_expanded
        },
        'test': {
            'input': input_test,
            'output': output_test,
            'k_squared': k_test
        }
    }


def create_tf_dataset(
    inputs: np.ndarray,
    k_squared: np.ndarray,
    outputs: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    include_indices: bool = False
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from numpy arrays.

    Parameters
    ----------
    inputs : np.ndarray
        Input images.
    k_squared : np.ndarray
        K-squared values (expanded).
    outputs : np.ndarray
        Output mode shapes.
    batch_size : int, optional
        Batch size (default: 32).
    shuffle : bool, optional
        Whether to shuffle (default: True).
    include_indices : bool, optional
        Whether to include sample indices (default: False).
        Useful for orthogonality loss to align with other modes.

    Returns
    -------
    tf.data.Dataset
        TensorFlow dataset.
    """
    if include_indices:
        indices = np.arange(len(inputs), dtype=np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, k_squared, outputs, indices))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((inputs, k_squared, outputs))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(inputs))
    return dataset.batch(batch_size)
