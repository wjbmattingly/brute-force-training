"""
Brute Force Training - A no-thrills Python package for finetuning Vision-Language Models (VLMs)

This package provides simple, unoptimized training utilities for various VLM architectures including:
- Qwen2-VL
- Qwen2.5-VL  
- LFM2-VL
- Qwen3 (text-only)

Features:
- HuggingFace datasets integration
- Simple training loops with validation
- Configurable data filtering and preprocessing
- Model checkpointing
"""

__version__ = "0.1.0"
__author__ = "wjbmattingly"

from .trainers import (
    Qwen2VLTrainer,
    Qwen25VLTrainer,
    LFM2VLTrainer,
    Qwen3Trainer
)

from .datasets import (
    VisionLanguageDataset,
    TextOnlyDataset
)

from .utils import (
    ensure_pil_image,
    find_assistant_content_sublist_indexes
)

from .config import TrainingConfig, get_config, list_configs

__all__ = [
    "Qwen2VLTrainer",
    "Qwen25VLTrainer", 
    "LFM2VLTrainer",
    "Qwen3Trainer",
    "VisionLanguageDataset",
    "TextOnlyDataset",
    "ensure_pil_image",
    "find_assistant_content_sublist_indexes",
    "TrainingConfig",
    "get_config",
    "list_configs"
]
