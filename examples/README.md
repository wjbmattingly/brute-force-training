# Examples

This directory contains example scripts and configuration files for using the brute-force-training package.

## Files

### Training Examples
- `train_qwen2_vl.py` - Example training script for Qwen2-VL models
- `train_qwen3.py` - Example training script for Qwen3 text models

### Configuration Examples
- `config_example.json` - Example JSON configuration file
- `use_config.py` - Examples of using configuration files and predefined configs

## Usage

### Run a Simple Training Example
```bash
cd examples
python train_qwen2_vl.py
```

### Use Configuration Files
```bash
cd examples
python use_config.py
```

### Command Line Interface
```bash
# Train with CLI arguments
brute-force-train --trainer qwen2-vl --model-name Qwen/Qwen2-VL-2B-Instruct \
                  --output-dir ./my_model --dataset-name CATMuS/medieval \
                  --image-column im --text-column text

# Train with config file
brute-force-train --config config_example.json
```

## Customization

You can modify these examples for your specific use case:

1. Change the `dataset_name` to your HuggingFace dataset
2. Adjust `image_column` and `text_column` to match your dataset schema
3. Modify training parameters like `max_steps`, `learning_rate`, etc.
4. Update data selection ranges with `train_select_start/end` and `val_select_start/end`
