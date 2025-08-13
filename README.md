---
base_model:
- LiquidAI/LFM2-VL-1.6B
---

<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wjbmattingly/brute-force-training/main/assets/bft-dark.png">
    <img alt="bft" src="https://raw.githubusercontent.com/wjbmattingly/brute-force-training/main/assets/bft-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
No thrills. Un-Optimized. Training.
</h3>


---

# Brute Force Training

A no-thrills, unoptimized Python package for finetuning Vision-Language Models (VLMs). This package provides simple training utilities for various VLM architectures with HuggingFace datasets integration.

## Supported Models

- **Qwen2-VL**: Vision-language models from the Qwen2-VL series
- **Qwen2.5-VL**: Enhanced vision-language models with improved capabilities  
- **LFM2-VL**: Liquid AI's vision-language models
- **Qwen3**: Text-only models from the Qwen3 series

## Features

- 🚀 Simple, unoptimized training loops - perfect for research and experimentation
- 📊 HuggingFace datasets integration out of the box
- 🔧 Configurable data filtering and preprocessing
- 💾 Automatic model checkpointing during training
- 🎯 Built-in validation loops
- 📸 Automatic image preprocessing and resizing
- 🏗️ Modular architecture with base classes for easy extension
- 📈 **Comprehensive documentation generation** - README.md for each checkpoint
- 🎨 **Training visualizations** - Loss curves and evaluation charts
- 📋 **HuggingFace model cards** - Automatic metadata generation
- 🔍 **Pre/post training evaluation** - Compare model performance
- 📊 **Training metrics tracking** - Detailed training history

## Installation

### From PyPI (when published)
```bash
pip install brute-force-training
```

### From Source
```bash
git clone https://github.com/wjbmattingly/brute-force-training.git
cd brute-force-training
pip install -e .
```

### Requirements
- Python 3.8+
- PyTorch 1.11.0+
- transformers 4.37.0+
- datasets 2.14.0+

## Quick Start

### Vision-Language Model Training (Qwen2-VL)

```python
from brute_force_training import Qwen2VLTrainer

# Initialize trainer
trainer = Qwen2VLTrainer(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    output_dir="./my_finetuned_model"
)

# Train the model
trainer.train_and_validate(
    dataset_name="your_dataset_name",
    image_column="image",
    text_column="text", 
    user_text="Describe this image",
    max_steps=1000,
    train_batch_size=2,
    learning_rate=1e-5,
    validate_before=True,    # Pre-training evaluation
    generate_docs=True       # Generate documentation
)
```

### Text-Only Model Training (Qwen3)

```python
from brute_force_training import Qwen3Trainer

# Initialize trainer
trainer = Qwen3Trainer(
    model_name="Qwen/Qwen3-4B-Thinking-2507",
    output_dir="./my_finetuned_qwen3"
)

# Train the model
trainer.train_and_validate(
    dataset_name="your_text_dataset",
    input_column="input",
    output_column="output",
    system_prompt="You are a helpful assistant.",  # ✨ System prompt support
    max_steps=1000,
    train_batch_size=4,
    learning_rate=1e-5
)
```

### System Prompts for Text Models

Just like vision models have `user_text`, text models now support `system_prompt`:

```python
# Math tutoring model
trainer.train_and_validate(
    dataset_name="math_problems",
    system_prompt="You are a mathematics tutor. Provide step-by-step solutions."
)

# Code assistant model  
trainer.train_and_validate(
    dataset_name="code_questions",
    system_prompt="You are a coding assistant. Write clean, efficient code."
)

# Creative writing model
trainer.train_and_validate(
    dataset_name="writing_prompts", 
    system_prompt="You are a creative writer. Write engaging stories."
)

# No system prompt (original behavior)
trainer.train_and_validate(
    dataset_name="general_qa",
    system_prompt=None  # Or just omit this parameter
)
```

## Documentation & Visualization Features

### Automatic Documentation Generation

Every checkpoint now includes comprehensive documentation:

```python
trainer.train_and_validate(
    dataset_name="your_dataset",
    # ... other parameters ...
    validate_before=True,    # Run evaluation before training starts
    generate_docs=True       # Generate docs and visualizations
)
```

Each saved checkpoint will contain:
- **README.md** - Detailed model card with training info
- **training_curves.png** - Loss and learning rate visualizations  
- **evaluation_comparison.png** - Before/after training performance
- **training_metrics.json** - Complete training history
- **model_card_metadata.json** - HuggingFace metadata

### Pre/Post Training Evaluation

Compare your model's performance before and after training:

```python
# This will automatically run if validate_before=True
# Shows output like:
# 🔍 Running pre-training evaluation...
# 📊 Pre-training - Loss: 2.456789, Perplexity: 11.67
# 
# [training happens]
#
# 🔍 Running post-training evaluation...  
# 📊 Post-training - Loss: 1.234567, Perplexity: 3.44
# 🎯 Loss improvement: +49.75% (from 2.456789 to 1.234567)
```

### Training Visualizations

Automatic generation of:
- **Loss curves** showing training and validation loss over time
- **Learning rate schedules** 
- **Evaluation comparisons** with before/after metrics
- **Training progress** with step-by-step metrics

## Advanced Usage

### Custom Data Filtering

```python
def my_filter_function(example):
    # Only include examples with text length between 50-1000 characters
    return 50 <= len(example['text']) <= 1000

trainer = Qwen2VLTrainer(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    output_dir="./filtered_model"
)

# Override the default filtering
trainer.filter_dataset = lambda dataset: dataset.filter(my_filter_function)

trainer.train_and_validate(
    dataset_name="your_dataset",
    image_column="image",
    text_column="text"
)
```

### Training Configuration

```python
trainer.train_and_validate(
    dataset_name="CATMuS/medieval",
    image_column="im",
    text_column="text",
    user_text="Transcribe this medieval manuscript line",
    
    # Training parameters
    max_steps=10000,
    eval_steps=500,
    num_accumulation_steps=4,
    learning_rate=1e-5,
    
    # Data selection
    train_select_start=0,
    train_select_end=5000,
    val_select_start=5000,
    val_select_end=6000,
    
    # Batch sizes
    train_batch_size=2,
    val_batch_size=2,
    
    # Image preprocessing
    max_image_size=500
)
```

## Model-Specific Examples

### LFM2-VL Training

```python
from brute_force_training import LFM2VLTrainer

trainer = LFM2VLTrainer(
    model_name="LiquidAI/LFM2-VL-450M",
    output_dir="./lfm2_finetuned"
)

trainer.train_and_validate(
    dataset_name="your_dataset",
    image_column="image",
    text_column="caption",
    user_text="What is in this image?",
    max_steps=5000,
    train_batch_size=1,  # LFM2-VL typically needs smaller batch sizes
    learning_rate=1e-5
)
```

### Qwen2.5-VL Training

```python
from brute_force_training import Qwen25VLTrainer

trainer = Qwen25VLTrainer(
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    output_dir="./qwen25_finetuned",
    min_pixel=256,
    max_pixel=384,
    image_factor=28
)

trainer.train_and_validate(
    dataset_name="your_dataset",
    image_column="image", 
    text_column="text",
    max_steps=8000,
    eval_steps=1000
)
```

## Dataset Format

### Vision-Language Datasets
Your HuggingFace dataset should have:
- An image column (PIL Images or base64 strings)
- A text column (string descriptions/captions)

### Text-Only Datasets  
Your HuggingFace dataset should have:
- An input column (input text)
- An output column (target text)

## Project Structure

```
brute_force_training/
├── __init__.py
├── datasets/
│   ├── __init__.py
│   ├── vision_language.py    # VisionLanguageDataset class
│   └── text_only.py         # TextOnlyDataset class
├── trainers/
│   ├── __init__.py
│   ├── base.py              # BaseTrainer abstract class
│   ├── qwen2_vl.py          # Qwen2VLTrainer
│   ├── qwen25_vl.py         # Qwen25VLTrainer
│   ├── lfm2_vl.py           # LFM2VLTrainer
│   └── qwen3.py             # Qwen3Trainer
└── utils/
    ├── __init__.py
    ├── image_utils.py       # Image preprocessing utilities
    └── tokenization.py     # Tokenization utilities
```

## Contributing

This is a research-focused package intended for experimentation. Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

The original training scripts were adapted from [zhangfaen/finetune-Qwen2-VL](https://github.com/zhangfaen/finetune-Qwen2-VL). We are deeply grateful for their foundational work.

## Limitations

This package is intentionally "brute force" and unoptimized. It's designed for:
- Research and experimentation
- Quick prototyping
- Educational purposes

For production use cases, consider more optimized training frameworks.

## Support

For questions, issues, or feature requests, please open an issue on GitHub.
