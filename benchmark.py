"""
Benchmark script to plot all 6 modes of one input in one figure.
Loads data from the pre-saved Benchmark_Test_Data folder.
Plots 10 samples, each showing all 6 modes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import configuration
from config import get_config

# Get configuration instance
_cfg = get_config()

# ============== CONFIGURATION (from config.py) ==============
MODELS_FOLDER = _cfg.model.physics_models_folder  # Default to physics models
STORAGE_MAIN_FOLDER = _cfg.data.benchmark_folder
INDIVIDUAL_PEAK_VALUE = _cfg.data.individual_peak_value
N_SAMPLES = _cfg.benchmark.n_samples
 
# ============== HELPER FUNCTIONS ==============
def load_test_data_simple(main_folder='data_benchmark/'):

    def load_folder_data_sequential(folder_path, transpose=False):
        """Load files in sequential order (0001.txt, 0002.txt, etc.)."""
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
        data_list = []
        for f in files:
            filepath = os.path.join(folder_path, f)
            data = np.loadtxt(filepath, delimiter=',', usecols=[2], skiprows=1)
            reshaped = data.reshape(128, 128)
            if transpose:
                reshaped = reshaped.T
            data_list.append(reshaped)
        return np.array(data_list)
    
    # Load input data
    input_test_test = load_folder_data_sequential(
        os.path.join(main_folder, 'Input'), transpose=True
    )
    input_test_test = input_test_test.reshape(-1, 128, 128, 1).astype(np.float32)
    
    # Load output data for each mode
    outputs = []
    for mode_idx in range(1, 7):
        output_data = load_folder_data_sequential(
            os.path.join(main_folder, f'Truth/outputs_mode{mode_idx}')
        )
        output_data = output_data.reshape(-1, 128, 128, 1).astype(np.float32) / INDIVIDUAL_PEAK_VALUE
        outputs.append(output_data)
    
    # # Load k_squared
    # k_squared = np.loadtxt(os.path.join(main_folder, 'k_squared_test.txt'), delimiter=',').reshape(-1, 1)
    # Use zeros for k_squared with the correct shape (matching number of samples)
    num_samples = input_test_test.shape[0]
    k_squared = np.zeros((num_samples, 1), dtype=np.float32)
    return input_test_test, outputs, k_squared


def load_all_models(models_folder):
    """
    Load all .h5 models from a folder, sorted by the number in the second-to-last
    underscore-separated term (e.g., regression_3phys_1_orth6modes.h5 -> 1).
    """
    # Find all .h5 files in the folder
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5')]
    
    if not model_files:
        raise FileNotFoundError(f"No .h5 model files found in {models_folder}")
    
    def extract_mode_number(filename):
        """Extract the number from the second-to-last underscore term."""
        # Remove .h5 extension and split by underscore
        name_without_ext = filename.replace('.h5', '')
        parts = name_without_ext.split('_')
        # Get the second-to-last part (index -2)
        if len(parts) >= 2:
            return int(parts[-2])
        return 0
    
    # Sort files by the extracted mode number
    model_files_sorted = sorted(model_files, key=extract_mode_number)
    
    print(f"Found {len(model_files_sorted)} model files in '{models_folder}':")
    for f in model_files_sorted:
        mode_num = extract_mode_number(f)
        print(f"  - {f} (mode {mode_num})")
    
    # Load models in sorted order
    models = []
    for model_file in model_files_sorted:
        model_path = os.path.join(models_folder, model_file)
        model = load_model(model_path)
        models.append(model)
        print(f"Loaded model: {model_path}")
    
    return models


def display_all_modes_per_sample(input_images, predictions_list, truths_list, n_samples=10):
    """
    Plot all 6 modes for each sample in one figure.
    Each figure has: 1 row for input + 6 rows for modes (Prediction | Truth | Residual)
    
    predictions_list: list of 6 prediction arrays (one per mode)
    truths_list: list of 6 truth arrays (one per mode)
    """
    n_modes = len(predictions_list)
    
    for sample_idx in range(n_samples):

        # Extract only the last folder name from the path
        folder_name = os.path.basename(os.path.normpath(MODELS_FOLDER))
        # Create figure: 7 rows (1 input + 6 modes), 4 columns
        fig, axes = plt.subplots(n_modes + 1, 4, figsize=(14, 3 * (n_modes + 1)))
        fig.suptitle(f'Sample {sample_idx + 1} - All 6 Modes', fontsize=26, fontweight='bold')
        
        # Row 0: Input image (span across or show in first column)
        axes[0, 0].imshow(input_images[sample_idx], cmap='gray')
        axes[0, 0].set_title("Input Shape", fontsize=26)
        axes[0, 0].axis("off")
        
        # Hide other columns in row 0
        axes[0, 1].axis("off")
        axes[0, 2].axis("off")
        axes[0, 3].axis("off")
        
        # Add column headers in row 0
        axes[0, 1].text(0.5, 0.5, f'Prediction\n({folder_name})', ha='center', va='center', fontsize=26, fontweight='bold')
        axes[0, 2].text(0.5, 0.5, "Truth", ha='center', va='center', fontsize=26, fontweight='bold')
        axes[0, 3].text(0.5, 0.5, "Residual", ha='center', va='center', fontsize=26, fontweight='bold')
        
        # Rows 1-6: Each mode
        for mode_idx in range(n_modes):
            row = mode_idx + 1
            
            pred = -predictions_list[mode_idx][sample_idx]
            truth = -truths_list[mode_idx][sample_idx]
            
            # Symmetric normalization [-1, +1]
            pred_norm = pred / (np.max(np.abs(pred)) + 1e-8)
            truth_norm = truth / (np.max(np.abs(truth)) + 1e-8)
            
            # Phase selection
            loss_pos = tf.reduce_mean(tf.keras.losses.MSE(truth_norm, pred_norm))
            loss_neg = tf.reduce_mean(tf.keras.losses.MSE(truth_norm, -pred_norm))
            
            if loss_pos <= loss_neg:
                phase = +1
                residual_loss = loss_pos
            else:
                phase = -1
                residual_loss = loss_neg
            
            pred_plot = phase * pred
            residual = truth_norm - phase * pred_norm
            
            # Mode label
            axes[row, 0].text(0.5, 0.5, f"Mode {mode_idx + 1}", ha='center', va='center', 
                            fontsize=26, fontweight='bold')
            axes[row, 0].axis("off")
            
            # Prediction
            axes[row, 1].imshow(pred_plot, vmin=-1, vmax=1)
            axes[row, 1].axis("off")
            
            # Truth
            axes[row, 2].imshow(truth, vmin=-1, vmax=1)
            axes[row, 2].axis("off")
            
            # Residual
            axes[row, 3].imshow(residual, vmin=-1, vmax=1)
            axes[row, 3].set_title(f"MSE: {residual_loss.numpy():.4e}", fontsize=18)
            axes[row, 3].axis("off")
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.3)  # Narrow column gaps
        plt.savefig(f'benchmark_sample_{sample_idx + 1}_{folder_name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close figure to free memory
        print(f"Saved: benchmark_sample_{sample_idx + 1}_{folder_name}.png")


if __name__ == "__main__":
    # When run directly, use the runner for proper validation
    print("For proper validation, use: python runner.py --mode benchmark")
    print("Running benchmark directly...\n")
    
    # Refresh config values in case they were updated
    MODELS_FOLDER = _cfg.model.physics_models_folder
    STORAGE_MAIN_FOLDER = _cfg.data.benchmark_folder
    N_SAMPLES = _cfg.benchmark.n_samples
    
    # Step 1: Load stored data using simplified function
    print("=== Loading stored test data ===")
    input_test_test, output_truths, k_squared_test_test = load_test_data_simple(STORAGE_MAIN_FOLDER)
    print(f"Loaded {len(input_test_test)} test samples")
    
    # Step 2: Load all models
    print("\n=== Loading models ===")
    models = load_all_models(MODELS_FOLDER)
    
    # Step 3: Make predictions for all modes
    print("\n=== Running predictions ===")
    predictions = []
    for i, model in enumerate(models):
        pred = model.predict([
            input_test_test, k_squared_test_test,
            output_truths[0], output_truths[1], output_truths[2],
            output_truths[3], output_truths[4]
        ])
        predictions.append(pred)
        print(f"Completed prediction for mode {i+1}")
    
    # Step 4: Display all modes per sample
    print(f"\n=== Plotting all 6 modes for {N_SAMPLES} samples ===")
    display_all_modes_per_sample(
        input_test_test, predictions, output_truths, n_samples=N_SAMPLES
    )
    
    print(f"\nBenchmark complete! {N_SAMPLES} figures saved as 'benchmark_sample_X.png'")
