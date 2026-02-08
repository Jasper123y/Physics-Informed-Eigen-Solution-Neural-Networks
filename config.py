"""
Minimal configuration for PINN LEM GPU project.
Only contains paths and basic settings - detailed training parameters stay in training.py
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Data paths and basic properties."""
    benchmark_folder: str = 'data_benchmark/'
    training_folder: str = 'data_training/'
    input_shape: tuple = (128, 128, 1)
    num_modes: int = 6
    individual_peak_value: float = 4.0e-10  # Peak value for individual normalization


@dataclass
class BenchmarkConfig:
    """Benchmark settings."""
    n_samples: int = 10  # Number of samples to benchmark


@dataclass
class ModelConfig:
    """Model paths."""
    physics_models_folder: str = 'models_trained/Physics_models/'
    nonphysics_models_folder: str = 'models_trained/Nonphysics_models/'
    model_extension: str = '.h5'


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    mode: str = 'auto'  # 'auto', 'benchmark', 'training'


# Global configuration instance
config = Config()


class EnvironmentValidator:
    """Validates environment, data, and model availability."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def check_python_dependencies(self) -> bool:
        """Check if required Python packages are installed."""
        required = ['numpy', 'tensorflow', 'matplotlib']
        missing = []
        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        
        if missing:
            self.errors.append(f"Missing packages: {', '.join(missing)}")
            return False
        return True
    
    def check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                self.warnings.append("No GPU detected. Training will use CPU.")
                return False
            print(f"✓ Found {len(gpus)} GPU(s)")
            return True
        except Exception as e:
            self.warnings.append(f"Could not check GPU: {e}")
            return False
    
    def check_benchmark_data(self) -> bool:
        """Check if benchmark data exists."""
        folder = self.cfg.data.benchmark_folder
        if not os.path.exists(folder):
            self.errors.append(f"Benchmark folder not found: {folder}")
            return False
        
        input_folder = os.path.join(folder, 'Input')
        if not os.path.exists(input_folder):
            self.errors.append(f"Input folder not found: {input_folder}")
            return False
        
        input_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
        if input_files:
            print(f"✓ Benchmark data found: {len(input_files)} samples")
            return True
        return False
    
    def check_training_data(self) -> bool:
        """Check if training data exists."""
        folder = self.cfg.data.training_folder
        if not os.path.exists(folder):
            self.warnings.append(f"Training folder not found: {folder}")
            return False
        
        input_folder = os.path.join(folder, 'Input')
        if os.path.exists(input_folder):
            input_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
            if input_files:
                print(f"✓ Training data found: {len(input_files)} samples")
                return True
        return False
    
    def check_models(self, folder: str) -> bool:
        """Check if trained models exist."""
        if not os.path.exists(folder):
            return False
        
        model_files = [f for f in os.listdir(folder) if f.endswith(self.cfg.model.model_extension)]
        if model_files:
            print(f"✓ Found {len(model_files)} model(s) in {folder}")
            return True
        return False
    
    def validate_for_benchmark(self) -> bool:
        """Validate environment for benchmark mode."""
        print("\n=== Validating for Benchmark Mode ===")
        
        if not self.check_python_dependencies():
            return False
        
        self.check_gpu_availability()
        
        data_ok = self.check_benchmark_data()
        models_ok = (self.check_models(self.cfg.model.physics_models_folder) or 
                     self.check_models(self.cfg.model.nonphysics_models_folder))
        
        return data_ok and models_ok
    
    def validate_for_training(self) -> bool:
        """Validate environment for training mode."""
        print("\n=== Validating for Training Mode ===")
        
        if not self.check_python_dependencies():
            return False
        
        self.check_gpu_availability()
        
        return self.check_training_data()
    
    def determine_mode(self) -> Optional[str]:
        """Auto-detect which mode can be run."""
        self.check_python_dependencies()
        self.check_gpu_availability()
        
        benchmark_ok = self.check_benchmark_data()
        models_ok = (self.check_models(self.cfg.model.physics_models_folder) or 
                     self.check_models(self.cfg.model.nonphysics_models_folder))
        training_ok = self.check_training_data()
        
        if benchmark_ok and models_ok:
            return 'benchmark'
        elif training_ok:
            return 'training'
        return None
    
    def print_status(self):
        """Print warnings and errors."""
        if self.warnings:
            print("\n⚠ WARNINGS:")
            for w in self.warnings:
                print(f"  - {w}")
        if self.errors:
            print("\n✗ ERRORS:")
            for e in self.errors:
                print(f"  - {e}")


def get_config() -> Config:
    """Get the global configuration instance."""
    return config
