"""
Configuration management for brute force training
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union
from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    
    # Model configuration
    trainer_type: str = "qwen2-vl"
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    output_dir: str = "./finetuned_model"
    device: str = "cuda"
    
    # Dataset configuration
    dataset_name: str = ""
    image_column: str = "image"
    text_column: str = "text"
    input_column: str = "input"
    output_column: str = "output"
    user_text: str = "Convert this image to text"
    
    # Training parameters
    max_steps: int = 10000
    eval_steps: int = 500
    num_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    train_batch_size: int = 1
    val_batch_size: int = 1
    
    # Data selection
    train_select_start: int = 0
    train_select_end: int = 1000
    val_select_start: int = 0
    val_select_end: int = 1000
    train_field: str = "train"
    val_field: str = "validation"
    
    # Model-specific parameters
    max_image_size: int = 500
    min_pixel: int = 256
    max_pixel: int = 384
    image_factor: int = 28
    max_length: int = 2048
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'TrainingConfig':
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'TrainingConfig':
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self, json_path: Union[str, Path]) -> None:
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save config to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)


# Predefined configurations for common use cases
CONFIGS = {
    "qwen2_vl_medieval": TrainingConfig(
        trainer_type="qwen2-vl",
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        dataset_name="CATMuS/medieval",
        image_column="im",
        text_column="text",
        user_text="Transcribe this medieval manuscript line",
        max_steps=10000,
        eval_steps=500,
        train_select_end=5000,
        val_select_start=5000,
        val_select_end=5500
    ),
    
    "qwen25_vl_basic": TrainingConfig(
        trainer_type="qwen25-vl",
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        max_steps=8000,
        eval_steps=1000,
        train_batch_size=2
    ),
    
    "lfm2_vl_small": TrainingConfig(
        trainer_type="lfm2-vl",
        model_name="LiquidAI/LFM2-VL-450M",
        max_steps=5000,
        eval_steps=500,
        train_batch_size=1,
        learning_rate=1e-5
    ),
    
    "qwen3_text": TrainingConfig(
        trainer_type="qwen3",
        model_name="Qwen/Qwen3-4B-Thinking-2507",
        max_steps=8000,
        eval_steps=1000,
        train_batch_size=4,
        val_batch_size=4,
        max_length=2048
    )
}


def get_config(config_name: str) -> TrainingConfig:
    """Get a predefined configuration by name."""
    if config_name not in CONFIGS:
        available = list(CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    return CONFIGS[config_name]


def list_configs() -> list:
    """List all available predefined configurations."""
    return list(CONFIGS.keys())
