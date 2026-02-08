#!/usr/bin/env python3
"""
Main runner script for PINN-ECM project.

Usage:
    python runner.py --mode benchmark                            # Run benchmark mode
    python runner.py --mode training --training-type physics     # Run physics training
    python runner.py --mode training --training-type nonphysics  # Run non-physics training
    python runner.py --check                                     # Validate environment (all modes)
    python runner.py --mode training --check                     # Validate training environment only
"""

import argparse
import sys
import os
from config import Config, EnvironmentValidator, get_config


def run_benchmark_for_model_folder(cfg: Config, models_folder: str, model_type: str):
    """Run benchmark for a specific models folder."""
    print(f"\n{'=' * 50}")
    print(f"BENCHMARKING {model_type.upper()} MODELS")
    print(f"Models folder: {models_folder}")
    print(f"{'=' * 50}")
    
    if not os.path.exists(models_folder):
        print(f"⚠ Skipping {model_type} models - folder not found: {models_folder}")
        return False
    
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5')]
    if not model_files:
        print(f"⚠ Skipping {model_type} models - no .h5 files found")
        return False
    
    # Import here to avoid loading TF before validation
    from benchmark import (
        load_test_data_simple,
        load_all_models,
        display_all_modes_per_sample
    )
    
    # Load data
    print("\n=== Loading test data ===")
    input_test, output_truths, k_squared = load_test_data_simple(
        cfg.data.benchmark_folder
    )
    print(f"Loaded {len(input_test)} test samples")
    
    # Load models from specific folder
    print(f"\n=== Loading {model_type} models ===")
    models = load_all_models(models_folder)
    
    # Make predictions
    print("\n=== Running predictions ===")
    predictions = []
    for i, model in enumerate(models):
        pred = model.predict([
            input_test, k_squared,
            output_truths[0], output_truths[1], output_truths[2],
            output_truths[3], output_truths[4]
        ])
        predictions.append(pred)
        print(f"Completed prediction for mode {i + 1}")
    
    # Temporarily update the models folder for display function
    import benchmark
    original_folder = benchmark.MODELS_FOLDER
    benchmark.MODELS_FOLDER = models_folder
    
    # Display results
    print(f"\n=== Plotting all 6 modes for {cfg.benchmark.n_samples} samples ===")
    display_all_modes_per_sample(
        input_test, predictions, output_truths, n_samples=cfg.benchmark.n_samples
    )
    
    # Restore original folder
    benchmark.MODELS_FOLDER = original_folder
    
    print(f"\n✓ {model_type} benchmark complete!")
    return True


def run_benchmark(cfg: Config):
    """Run benchmark mode for both Physics and Non-Physics models."""
    print("\n" + "=" * 50)
    print("RUNNING BENCHMARK MODE")
    print("=" * 50)
    
    results = {}
    
    # Benchmark Physics models
    results['physics'] = run_benchmark_for_model_folder(
        cfg, cfg.model.physics_models_folder, "Physics"
    )
    
    # Benchmark Non-Physics models
    results['nonphysics'] = run_benchmark_for_model_folder(
        cfg, cfg.model.nonphysics_models_folder, "Non-Physics"
    )
    
    # Summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    for model_type, success in results.items():
        status = "✓ Completed" if success else "✗ Skipped"
        print(f"  {model_type.capitalize()} models: {status}")
    
    if not any(results.values()):
        print("\n⚠ No models were benchmarked!")
        return False
    
    return True


def run_training(cfg: Config, training_type: str = 'physics'):
    """Run training mode.
    
    Args:
        cfg: Configuration object
        training_type: Type of training - 'physics' or 'nonphysics'
    """
    print("\n" + "=" * 50)
    print(f"RUNNING {training_type.upper()} TRAINING MODE")
    print("=" * 50)
    
    # Check if training script exists
    try:
        import training
        training.main(training_type=training_type)
    except ImportError:
        print("\n⚠ Training module not found.")
        print("Please create 'training.py' with a main() function.")
        print("\nExpected training data structure:")
        print(f"  {cfg.data.training_folder}/")
        print("    ├── inputs/")
        print("    │   ├── 0001.txt")
        print("    │   └── ...")
        print("    └── outputs_mode1/ ... outputs_mode6/")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PINN LEM GPU - Benchmark and Training Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py --mode benchmark # Run benchmark mode
  python runner.py --mode training  # Run physics training (default)
  python runner.py --mode training --training-type physics    # Run physics training
  python runner.py --mode training --training-type nonphysics # Run non-physics training (MSE only)
  python runner.py --check          # Validate all environments
  python runner.py --mode benchmark --check   # Validate benchmark environment only
  python runner.py --mode benchmark --samples 5  # Benchmark with 5 samples
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['benchmark', 'training'],
        default=None,
        help='Execution mode: benchmark or training'
    )
    
    parser.add_argument(
        '--training-type', '-t',
        choices=['physics', 'nonphysics'],
        default='physics',
        help='Training type: physics (with physics-informed loss) or nonphysics. Default: physics'
    )
    
    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Only check environment without running'
    )
    
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=None,
        help='Number of samples to benchmark (default: 10)'
    )
    
    parser.add_argument(
        '--physics-models',
        type=str,
        default=None,
        help='Path to physics models folder'
    )
    
    parser.add_argument(
        '--nonphysics-models',
        type=str,
        default=None,
        help='Path to non-physics models folder'
    )
    
    parser.add_argument(
        '--data-folder',
        type=str,
        default=None,
        help='Path to benchmark data folder'
    )
    
    args = parser.parse_args()
    
    # Handle --check without --mode (validates both modes)
    if args.check and args.mode is None:
        cfg = get_config()
        print("=" * 50)
        print("PINN LEM GPU - Environment Check")
        print("=" * 50)
        
        validator = EnvironmentValidator(cfg)
        
        print("\n--- Benchmark Environment ---")
        benchmark_valid = validator.validate_for_benchmark()
        validator.print_status()
        
        print("\n--- Training Environment ---")
        training_valid = validator.validate_for_training()
        validator.print_status()
        
        if benchmark_valid and training_valid:
            print("\n✓ All environment checks passed.")
            sys.exit(0)
        else:
            print("\n⚠ Some environment checks failed.")
            sys.exit(1)
    
    # If not check-only mode, --mode is required
    if args.mode is None:
        parser.error("the following arguments are required: --mode/-m")
    
    # Get configuration
    cfg = get_config()
    
    # Apply command-line overrides
    if args.samples is not None:
        cfg.benchmark.n_samples = args.samples
    if args.physics_models is not None:
        cfg.model.physics_models_folder = args.physics_models
    if args.nonphysics_models is not None:
        cfg.model.nonphysics_models_folder = args.nonphysics_models
    if args.data_folder is not None:
        cfg.data.benchmark_folder = args.data_folder
    
    cfg.mode = args.mode
    
    # Print header
    print("=" * 50)
    print("PINN LEM GPU - Runner")
    print("=" * 50)
    print(f"Mode: {cfg.mode}")
    if cfg.mode == 'benchmark':
        print(f"Physics models: {cfg.model.physics_models_folder}")
        print(f"Non-Physics models: {cfg.model.nonphysics_models_folder}")
        print(f"Benchmark data: {cfg.data.benchmark_folder}")
    else:
        print(f"Training data: {cfg.data.training_folder}")
        print(f"Training type: {args.training_type}")
    
    # Validate environment
    validator = EnvironmentValidator(cfg)
    
    if args.mode == 'benchmark':
        if not validator.validate_for_benchmark():
            validator.print_status()
            print("\n✗ Benchmark validation failed!")
            sys.exit(1)
        validator.print_status()
    
    elif args.mode == 'training':
        if not validator.validate_for_training():
            validator.print_status()
            print("\n✗ Training validation failed!")
            sys.exit(1)
        validator.print_status()
    
    # Check-only mode
    if args.check:
        print("\n✓ Environment check complete.")
        sys.exit(0)
    
    # Run the appropriate mode
    if cfg.mode == 'benchmark':
        run_benchmark(cfg)
    elif cfg.mode == 'training':
        run_training(cfg, training_type=args.training_type)


if __name__ == "__main__":
    main()
