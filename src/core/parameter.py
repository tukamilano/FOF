"""
Hyperparameter and Default Value Management Module

This module centrally manages hyperparameters and default values used
throughout the project, improving configuration consistency and maintainability.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


class DeviceType(Enum):
    """Device selection enumeration"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class DataFilterType(Enum):
    """Data filtering enumeration"""
    ALL = "all"
    SUCCESSFUL_ONLY = "successful_only"


@dataclass
class ModelParameters:
    """Transformer model hyperparameters"""
    # Basic settings
    vocab_size: int = 0  # Set dynamically
    pad_id: int = 0  # Set dynamically
    
    # Hierarchical classification settings
    use_hierarchical_classification: bool = True  # Whether to use hierarchical classification
    num_main_classes: int = 0  # Number of main tactic classes (set dynamically)
    num_arg1_classes: int = 0  # Number of first argument classes (set dynamically)
    num_arg2_classes: int = 0  # Number of second argument classes (set dynamically)
    
    # Token settings
    add_tactic_tokens: bool = True  # Whether to add tactic tokens
    num_tactic_tokens: int = 50  # Number of tactic tokens
    
    # Sequence length settings
    max_seq_len: int = 256
    
    # Transformer architecture
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    


@dataclass
class TrainingParameters:
    """Training-related parameters"""
    # Basic training settings
    learning_rate: float = 3e-4
    batch_size: int = 32
    num_epochs: int = 10
    
    # Loss function weight settings
    arg1_loss_weight: float = 0.8  # Weight for Arg1 loss
    arg2_loss_weight: float = 0.8  # Weight for Arg2 loss
    
    # Data collection settings
    collect_data: bool = False
    work_file: str = "temp_work.json"
    dataset_file: str = "training_data.json"
    filter_type: DataFilterType = DataFilterType.ALL
    
    # Evaluation settings
    warmup_iterations: int = 2
    max_interactions: int = 5


@dataclass
class GenerationParameters:
    """Formula generation and evaluation parameters"""
    # Basic generation settings
    count: int = 10
    difficulty: float = 0.7
    seed: int = 7
    max_len: int = 50
    
    # Variable settings
    variables: List[str] = None
    allow_const: bool = False
    
    # Proof search settings
    max_steps: int = 30
    max_depth: int = 8  # For auto_classical
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = ["a", "b", "c"]


@dataclass
class SystemParameters:
    """System and device-related parameters"""
    device: DeviceType = DeviceType.AUTO
    root_dir: str = ""  # Set dynamically
    token_py_path: str = ""  # Set dynamically
    pyprover_dir: str = ""  # Set dynamically


@dataclass
class SpecialTokens:
    """Special token definitions"""
    PAD = "[PAD]"
    CLS = "[CLS]"
    SEP = "[SEP]"
    UNK = "[UNK]"
    EOS = "[EOS]"


@dataclass
class HierarchicalLabels:
    """Label management for hierarchical classification"""
    main_tactics: List[str] = None  # List of main tactics
    arg1_values: List[str] = None   # List of first argument values
    arg2_values: List[str] = None   # List of second argument values
    num_tactic_tokens: int = 50     # Number of tactic tokens
    
    # Argument requirement mask per tactic (arg1_required, arg2_required)
    TACTIC_ARG_MASK: Dict[str, tuple] = None
    
    def __post_init__(self):
        if self.main_tactics is None:
            # Default main tactics (existing 9 + new tactic placeholders)
            existing_tactics = [
                "assumption", "intro", "split", "left", "right", "add_dn", 
                "apply", "destruct", "specialize"
            ]
            # Placeholders for new tactics (initially unused)
            new_tactic_placeholders = [f"TACTIC_{i}" for i in range(self.num_tactic_tokens)]
            self.main_tactics = existing_tactics + new_tactic_placeholders
        if self.arg1_values is None:
            # Default first argument (numeric)
            self.arg1_values = [str(i) for i in range(10)]  # 0-9 (10 values)
        if self.arg2_values is None:
            # Default second argument (numeric)
            self.arg2_values = [str(i) for i in range(10)]  # 0-9 (10 values)
        if self.TACTIC_ARG_MASK is None:
            # Argument requirement mask per tactic (arg1_required, arg2_required)
            self.TACTIC_ARG_MASK = {
                "intro": (False, False),
                "apply": (True, False),
                "specialize": (True, True),
                "split": (False, False),
                "left": (False, False),
                "right": (False, False),
                "destruct": (True, False),
                "assumption": (False, False),
                "add_dn": (False, False),
            }
            # All new tactics set to require no arguments
            for i in range(self.num_tactic_tokens):
                self.TACTIC_ARG_MASK[f"TACTIC_{i}"] = (False, False)
    
    def get_main_to_id(self) -> Dict[str, int]:
        """Mapping from main tactic string to ID"""
        return {tactic: i for i, tactic in enumerate(self.main_tactics)}
    
    def get_arg1_to_id(self) -> Dict[str, int]:
        """Mapping from first argument string to ID"""
        return {arg: i for i, arg in enumerate(self.arg1_values)}
    
    def get_arg2_to_id(self) -> Dict[str, int]:
        """Mapping from second argument string to ID"""
        return {arg: i for i, arg in enumerate(self.arg2_values)}
    
    def get_id_to_main(self) -> List[str]:
        """Mapping from ID to main tactic string"""
        return self.main_tactics.copy()
    
    def get_id_to_arg1(self) -> List[str]:
        """Mapping from ID to first argument string"""
        return self.arg1_values.copy()
    
    def get_id_to_arg2(self) -> List[str]:
        """Mapping from ID to second argument string"""
        return self.arg2_values.copy()


class ParameterManager:
    """Parameter management class"""
    
    def __init__(self):
        self.model = ModelParameters()
        self.training = TrainingParameters()
        self.generation = GenerationParameters()
        self.system = SystemParameters()
        self.special_tokens = SpecialTokens()
        self.hierarchical_labels = HierarchicalLabels()
        
        # Synchronize parameters
        self._sync_parameters()
    
    def _sync_parameters(self) -> None:
        """Synchronize parameters"""
        # Sync num_tactic_tokens from ModelParameters to HierarchicalLabels
        self.hierarchical_labels.num_tactic_tokens = self.model.num_tactic_tokens
        # Regenerate main_tactics
        existing_tactics = [
            "assumption", "intro", "split", "left", "right", "add_dn", 
            "apply", "destruct", "specialize"
        ]
        new_tactic_placeholders = [f"TACTIC_{i}" for i in range(self.model.num_tactic_tokens)]
        self.hierarchical_labels.main_tactics = existing_tactics + new_tactic_placeholders
    
    def update_model_params(self, **kwargs) -> None:
        """Update model parameters"""
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
            else:
                raise ValueError(f"Unknown model parameter: {key}")
        
        # Synchronize after parameter update
        self._sync_parameters()
    
    def update_training_params(self, **kwargs) -> None:
        """Update training parameters"""
        for key, value in kwargs.items():
            if hasattr(self.training, key):
                # Convert string to DataFilterType for filter_type
                if key == "filter_type" and isinstance(value, str):
                    try:
                        value = DataFilterType(value)
                    except ValueError:
                        # Keep as string if invalid value
                        pass
                setattr(self.training, key, value)
            else:
                raise ValueError(f"Unknown training parameter: {key}")
    
    def update_generation_params(self, **kwargs) -> None:
        """Update generation parameters"""
        for key, value in kwargs.items():
            if hasattr(self.generation, key):
                setattr(self.generation, key, value)
            else:
                raise ValueError(f"Unknown generation parameter: {key}")
    
    def update_system_params(self, **kwargs) -> None:
        """Update system parameters"""
        for key, value in kwargs.items():
            if hasattr(self.system, key):
                # Convert string to DeviceType for device
                if key == "device" and isinstance(value, str):
                    try:
                        value = DeviceType(value)
                    except ValueError:
                        # Keep as string if invalid value
                        pass
                setattr(self.system, key, value)
            else:
                raise ValueError(f"Unknown system parameter: {key}")
    
    def get_device(self) -> str:
        """Get device string"""
        if isinstance(self.system.device, DeviceType):
            if self.system.device == DeviceType.AUTO:
                try:
                    import torch
                    return "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    return "cpu"
            return self.system.device.value
        else:
            # Return as-is if string
            if self.system.device == "auto":
                try:
                    import torch
                    return "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    return "cpu"
            return self.system.device
    
    def get_filter_flags(self) -> Dict[str, bool]:
        """Get data filtering flags"""
        if isinstance(self.training.filter_type, DataFilterType):
            return {
                "filter_successful_only": self.training.filter_type == DataFilterType.SUCCESSFUL_ONLY
            }
        else:
            # Compare as string
            return {
                "filter_successful_only": self.training.filter_type == "successful_only"
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Get all parameters in dictionary format"""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "generation": self.generation.__dict__,
            "system": self.system.__dict__,
            "special_tokens": self.special_tokens.__dict__
        }


# Default parameter manager instance
default_params = ParameterManager()


# Convenience functions
def get_model_params() -> ModelParameters:
    """Get model parameters"""
    return default_params.model


def get_training_params() -> TrainingParameters:
    """Get training parameters"""
    return default_params.training


def get_generation_params() -> GenerationParameters:
    """Get generation parameters"""
    return default_params.generation


def get_system_params() -> SystemParameters:
    """Get system parameters"""
    return default_params.system


def get_special_tokens() -> SpecialTokens:
    """Get special tokens"""
    return default_params.special_tokens


def get_hierarchical_labels() -> HierarchicalLabels:
    """Get hierarchical classification labels"""
    return default_params.hierarchical_labels


def update_parameters(**kwargs) -> None:
    """Batch update parameters"""
    for category, params in kwargs.items():
        if category == "model":
            default_params.update_model_params(**params)
        elif category == "training":
            default_params.update_training_params(**params)
        elif category == "generation":
            default_params.update_generation_params(**params)
        elif category == "system":
            default_params.update_system_params(**params)
        else:
            raise ValueError(f"Unknown parameter category: {category}")


def update_model_params(**kwargs) -> None:
    """Update model parameters (global function)"""
    default_params.update_model_params(**kwargs)


# Commonly used configuration presets
PRESETS = {
    "fast_training": {
        "model": {"d_model": 64, "nhead": 2, "num_layers": 1, "dim_feedforward": 128},
        "training": {"batch_size": 16, "learning_rate": 1e-3},
        "generation": {"count": 5, "max_steps": 3}
    },
    "high_quality": {
        "model": {"d_model": 256, "nhead": 8, "num_layers": 4, "dim_feedforward": 512},
        "training": {"batch_size": 64, "learning_rate": 1e-4},
        "generation": {"count": 50, "max_steps": 10}
    },
    "debug": {
        "model": {"d_model": 32, "nhead": 1, "num_layers": 1, "dim_feedforward": 64},
        "training": {"batch_size": 4, "learning_rate": 1e-2},
        "generation": {"count": 2, "max_steps": 2}
    }
}


def apply_preset(preset_name: str) -> None:
    """Apply preset configuration"""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    
    update_parameters(**PRESETS[preset_name])


# Usage examples and documentation
if __name__ == "__main__":
    # Basic usage
    print("=== Parameter Management System Usage Examples ===")
    
    # Display default parameters
    print("\n1. Default Parameters:")
    print(f"Model d_model: {get_model_params().d_model}")
    print(f"Training batch_size: {get_training_params().batch_size}")
    print(f"Generation count: {get_generation_params().count}")
    
    # Update parameters
    print("\n2. Updating Parameters:")
    update_parameters(
        model={"d_model": 256, "nhead": 8},
        training={"batch_size": 64},
        generation={"count": 100}
    )
    print(f"After update d_model: {get_model_params().d_model}")
    print(f"After update batch_size: {get_training_params().batch_size}")
    
    # Apply preset
    print("\n3. Applying Preset:")
    apply_preset("fast_training")
    print(f"After fast_training d_model: {get_model_params().d_model}")
    print(f"After fast_training batch_size: {get_training_params().batch_size}")
    
    # Get device
    print(f"\n4. Device: {default_params.get_device()}")
    
    # Display all parameters
    print("\n5. All Parameters:")
    all_params = default_params.to_dict()
    for category, params in all_params.items():
        print(f"{category}: {params}")