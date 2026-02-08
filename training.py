"""
Vibration Mode Prediction using U-Net with Physics-Informed Loss

This script demonstrates the complete workflow for training a U-Net model
with attention gates to predict 2D vibration modes from structural geometry.

Supports two training modes:
- Physics: Uses physics-informed loss (MSE + Helmholtz + Symmetry + K-deviation + Orthogonality)
- Non-physics: Uses MSE loss only

Author: Jiapeng XU
Date: December 2025
"""

import numpy as np
import tensorflow as tf
import wandb
import os

# Import from local package
from Scripts_training import (
    load_geometry_images,
    load_mode_shapes,
    load_all_mode_shapes,
    create_unet_model,
    CustomHelmholtzLoss,
    train_model,
    display_images,
    display_comparison,
    log_comparison_to_wandb,
)
from Scripts_training.data_loader import (
    load_k_squared, normalize_data, expand_k_squared, create_tf_dataset
)
from Scripts_training.visualization import compute_metrics, plot_training_history
from Scripts_training.config import Config, DataConfig, ModelConfig, TrainingConfig, WandbConfig
from sklearn.model_selection import train_test_split


def train_single_mode(target_mode, input_images, all_modes, k_squared_base_path, config_template, training_type='physics'):
    """Train a model for a single target mode.
    
    Args:
        target_mode: The mode number to train (1-6)
        input_images: Input geometry images
        all_modes: List of all 6 mode shapes
        k_squared_base_path: Path to k-squared files
        config_template: Configuration template
        training_type: 'physics' for physics-informed training, 'nonphysics' for MSE-only training
    """
    
    print(f"\n{'='*80}")
    print(f"TRAINING MODE {target_mode} ({training_type.upper()} MODE)")
    print(f"{'='*80}")
    
    # Build k_squared file path for this mode
    suffix = "st" if target_mode == 1 else "nd" if target_mode == 2 else "rd" if target_mode == 3 else "th"
    k_squared_file = f'{k_squared_base_path}/{target_mode}{suffix}.txt'
    
    # Update config for this mode
    # Determine wandb run name based on training type
    wandb_name = f"run_mode{target_mode}_V0_{training_type}"
    
    config = Config(
        data=DataConfig(
            input_folder=config_template.data.input_folder,
            k_squared_file=k_squared_file,
            all_mode_folders=config_template.data.all_mode_folders,
            image_shape=config_template.data.image_shape,
            peak_value=config_template.data.peak_value,
            real_width=config_template.data.real_width,
            test_size=config_template.data.test_size,
            random_state=config_template.data.random_state,
        ),
        model=config_template.model,
        training=config_template.training,
        wandb=WandbConfig(
            project=config_template.wandb.project,
            entity=config_template.wandb.entity,
            name=wandb_name,
            enabled=config_template.wandb.enabled,
        ),
    )
    
    # The target mode is the one we're predicting
    output_images = all_modes[target_mode - 1]  # 0-indexed
    
    # Other modes for orthogonality loss (all except target)
    other_mode_images = [all_modes[i] for i in range(6) if i != target_mode - 1]
    
    k_squared = load_k_squared(config.data.k_squared_file)
    
    print(f"Input images shape: {input_images.shape}")
    print(f"Output images shape: {output_images.shape}")
    print(f"Other modes: {len(other_mode_images)} arrays of shape {other_mode_images[0].shape}")
    print(f"K-squared shape: {k_squared.shape}")

    # Normalize data
    input_norm, output_norm = normalize_data(
        input_images, output_images, config.data.peak_value
    )
    
    # Normalize other modes
    other_modes_norm = []
    for mode in other_mode_images:
        _, mode_norm = normalize_data(input_images, mode, config.data.peak_value)
        other_modes_norm.append(mode_norm)

    # =========================================================================
    # Train-Validation Split 
    # =========================================================================
    n_samples = input_norm.shape[0]
    indices = np.arange(n_samples)
    
    train_idx, val_idx = train_test_split(
        indices,
        test_size=config.data.test_size,  # This is the validation ratio (e.g., 0.1)
        random_state=config.data.random_state
    )
    
    # Split main data using indices
    input_train, input_val = input_norm[train_idx], input_norm[val_idx]
    output_train, output_val = output_norm[train_idx], output_norm[val_idx]
    k_train, k_val = k_squared[train_idx], k_squared[val_idx]
    
    # Split other modes using same indices (for orthogonality loss)
    other_modes_train = [mode[train_idx] for mode in other_modes_norm]
    other_modes_val = [mode[val_idx] for mode in other_modes_norm]
    
    # Expand k_squared for broadcasting
    k_train_expanded = expand_k_squared(k_train, tf.shape(input_train))
    k_val_expanded = expand_k_squared(k_val, tf.shape(input_val))
    
    print(f"\nTraining set size: {input_train.shape[0]}")
    print(f"Validation set size: {input_val.shape[0]}")

    # Create TF datasets
    train_dataset = create_tf_dataset(
        input_train, k_train_expanded, output_train,
        batch_size=config.training.batch_size, shuffle=True, include_indices=True
    )
    
    val_dataset = create_tf_dataset(
        input_val, k_val_expanded, output_val,
        batch_size=config.training.batch_size, shuffle=False, include_indices=True
    )
    
    # Prepare other modes datasets for orthogonality loss
    other_modes_datasets = {
        'train': other_modes_train,  # List of 5 arrays, each (n_train, H, W, 1)
        'val': other_modes_val,      # List of 5 arrays, each (n_val, H, W, 1)
    }

    # =========================================================================
    # Initialize Wandb Run
    # =========================================================================
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.name,
            config={
                "architecture": "U-Net with Attention",
                "target_mode": target_mode,
                "input_shape": config.model.input_shape,
                "filters": config.model.filters,
                "epochs": config.training.epochs,
                "batch_size": config.training.batch_size,
                "initial_lr": config.training.initial_lr,
                "peak_value": config.data.peak_value,
                "pixel_size": config.data.pixel_size,
                "train_size": input_train.shape[0],
                "val_size": input_val.shape[0],
            }
        )

    # =========================================================================
    # Model Creation
    # =========================================================================
    print("\nCreating U-Net model...")
    model = create_unet_model(
        input_shape=config.model.input_shape,
        filters=config.model.filters,
        kernel_size=config.model.kernel_size,
        output_activation=config.model.output_activation
    )
    model.summary()

    # =========================================================================
    # Loss Function Setup
    # =========================================================================
    loss_fn = CustomHelmholtzLoss(
        pixel_size=config.data.pixel_size,
        peak_value=config.data.peak_value
    )
    loss_fn.set_loss_schedule(config.training.loss_schedule)

    # =========================================================================
    # Model Training
    # =========================================================================
    print("\nStarting training...")
    print(f"Training mode {target_mode} with orthogonality loss against {len(other_modes_train)} other modes")
    
    history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn=loss_fn,
        training_config=config.training,
        use_wandb=config.wandb.enabled,
        other_modes_datasets=other_modes_datasets
    )

    # =========================================================================
    # Evaluation (on validation set)
    # =========================================================================
    print("\nEvaluating model on validation set...")
    
    # Prepare validation k_squared (already expanded: k_val_expanded)
    # Get predictions
    predictions = model.predict([input_val, k_val_expanded])
    
    # Compute metrics
    metrics = compute_metrics(output_val, predictions)
    print("\nPrediction Metrics (Validation Set):")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  Mean Correlation: {metrics['mean_correlation']:.4f}")
    
    # Log metrics to wandb
    if config.wandb.enabled:
        wandb.log({
            "val/mse": metrics['mse'],
            "val/mae": metrics['mae'],
            "val/rmse": metrics['rmse'],
            "val/mean_correlation": metrics['mean_correlation'],
        })

    # =========================================================================
    # Visualization
    # =========================================================================
    if config.wandb.enabled:
        log_comparison_to_wandb(input_val, predictions, output_val, num_samples=6)

    # =========================================================================
    # Save Model
    # =========================================================================
    # Create output folder based on training type
    if training_type == 'physics':
        output_folder = 'models_trained/Physics_models/'
    else:
        output_folder = 'models_trained/Nonphysics_models/'
    os.makedirs(output_folder, exist_ok=True)
    
    model_path = os.path.join(output_folder, f'vibration_{target_mode}_mode_model.h5')
    model.save(model_path)
    print(f"\nModel saved to '{model_path}'")
    
    if config.wandb.enabled:
        wandb.save(model_path)
        wandb.finish()
        print("Wandb run completed.")
    
    # Clear model from memory
    tf.keras.backend.clear_session()
    
    return model_path, metrics


def main(runner_config=None, training_type='physics'):
    """Main training and evaluation pipeline for all 6 modes.
    
    Args:
        runner_config: Optional config from runner.py. If provided, some settings
                      may be overridden (currently unused, reserved for future use).
        training_type: Type of training - 'physics' (physics-informed loss) or 
                      'nonphysics' (MSE-only loss). Default: 'physics'
    """
    
    print(f"\n{'='*80}")
    print(f"TRAINING TYPE: {training_type.upper()}")
    if training_type == 'physics':
        print("Using physics-informed loss (MSE + Helmholtz + Symmetry + K-deviation + Orthogonality)")
    else:
        print("Using non-physics loss (MSE only)")
    print(f"{'='*80}\n")
    
    # =========================================================================
    # Weights & Biases Initialization
    # =========================================================================
    wandb.login()
    
    # =========================================================================
    # Configuration Template (minimal - detailed params defined below)
    # =========================================================================
    
    # Define loss schedules based on training type
    if training_type == 'physics':
        # Physics-informed loss schedule
        loss_schedule = {
            (0, 100): {'mse': 1.0, 'symmetric': 0, 'helmholtz': 0, 'k_dev': 0, 'orthogonality': 0},
            (100, 200): {'mse': 0, 'symmetric': 1.0, 'helmholtz': 5e-3, 'k_dev': 2e-15, 'orthogonality': 3e-5},
            (200, float('inf')): {'mse': 0, 'symmetric': 1.0, 'helmholtz': 0, 'k_dev': 0, 'orthogonality': 0},
        }
    else:
        # Non-physics loss schedule (MSE only)
        loss_schedule = {
            (0, float('inf')): {'mse': 1.0, 'symmetric': 0, 'helmholtz': 0, 'k_dev': 0, 'orthogonality': 0},
        }
    
    config_template = Config(
        data=DataConfig(
            input_folder='./data_training/Input',
            all_mode_folders=[
                './data_training/Truth/1st',
                './data_training/Truth/2nd',
                './data_training/Truth/3rd',
                './data_training/Truth/4th',
                './data_training/Truth/5th',
                './data_training/Truth/6th',
            ],
            image_shape=(128, 128),
            peak_value=4.0e-10,
            real_width=500e-6,
            test_size=0.2,
            random_state=42,
        ),
        model=ModelConfig(
            input_shape=(128, 128, 1),
            filters=(32, 64, 128, 256),
            kernel_size=(5, 5),
            output_activation='tanh',
        ),
        training=TrainingConfig(
            epochs=230,
            batch_size=32,
            initial_lr=1e-5,
            lr_schedule={
                (0, 200): 5e-5,
                (200, 250): 1e-5,
                (250, 1000): 1e-5,
            },
            loss_schedule=loss_schedule,
        ),
        wandb=WandbConfig(
            project="vibration-mode-prediction",
            entity="xu7639",
            enabled=True,
        ),
    )

    # =========================================================================
    # Data Loading (once for all modes)
    # =========================================================================
    print("Loading data...")
    
    # DEBUG FLAG: Set to None to load all data, or set a number to limit samples
    MAX_SAMPLES = None  # Change to None for full training
    
    if MAX_SAMPLES is not None:
        print(f"\n*** DEBUG MODE: Loading only {MAX_SAMPLES} samples ***\n")
    
    input_images = load_geometry_images(
        config_template.data.input_folder,
        image_shape=config_template.data.image_shape,
        max_samples=MAX_SAMPLES
    )
    
    # Load all 6 eigenmodes
    all_modes = load_all_mode_shapes(
        config_template.data.all_mode_folders,
        image_shape=config_template.data.image_shape,
        max_samples=MAX_SAMPLES
    )
    
    print(f"Loaded {len(all_modes)} eigenmodes")
    print(f"Input images shape: {input_images.shape}")

    # =========================================================================
    # Visualize Sample Data
    # =========================================================================
    # print("\nVisualizing sample input images...")
    # input_norm_preview, _ = normalize_data(input_images, all_modes[0], config_template.data.peak_value)
    # display_images(input_norm_preview, "Input", is_binary=True)

    # =========================================================================
    # Train All 6 Modes Sequentially
    # =========================================================================
    k_squared_base_path = './data_training/Kfile'
    
    results = {}
    for target_mode in range(1, 7):
        model_path, metrics = train_single_mode(
            target_mode=target_mode,
            input_images=input_images,
            all_modes=all_modes,
            k_squared_base_path=k_squared_base_path,
            config_template=config_template,
            training_type=training_type
        )
        results[target_mode] = {
            'model_path': model_path,
            'metrics': metrics
        }
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print(f"TRAINING COMPLETE - SUMMARY ({training_type.upper()} MODE)")
    print("="*80)
    for mode, result in results.items():
        print(f"\nMode {mode}:")
        print(f"  Model: {result['model_path']}")
        print(f"  MSE: {result['metrics']['mse']:.6f}")
        print(f"  Correlation: {result['metrics']['mean_correlation']:.4f}")


if __name__ == "__main__":
    main()


