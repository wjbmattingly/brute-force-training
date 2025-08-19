"""
Command-line interface for brute-force-training
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from .trainers import Qwen2VLTrainer, Qwen25VLTrainer, LFM2VLTrainer, Qwen3Trainer


TRAINER_CLASSES = {
    "qwen2-vl": Qwen2VLTrainer,
    "qwen25-vl": Qwen25VLTrainer,
    "lfm2-vl": LFM2VLTrainer,
    "qwen3": Qwen3Trainer,
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_trainer(trainer_type: str, **kwargs):
    """Create trainer instance based on type."""
    if trainer_type not in TRAINER_CLASSES:
        raise ValueError(f"Unknown trainer type: {trainer_type}. Available: {list(TRAINER_CLASSES.keys())}")
    
    trainer_class = TRAINER_CLASSES[trainer_type]
    return trainer_class(**kwargs)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Brute Force Training - Finetune Vision-Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Qwen2-VL on a vision-language dataset
  brute-force-train --trainer qwen2-vl --model-name Qwen/Qwen2-VL-2B-Instruct \\
                    --output-dir ./my_model --dataset-name my_dataset \\
                    --image-column image --text-column caption
  
  # Train Qwen3 on a text dataset
  brute-force-train --trainer qwen3 --model-name Qwen/Qwen3-4B-Thinking-2507 \\
                    --output-dir ./my_text_model --dataset-name my_text_dataset \\
                    --input-column input --output-column output
  
  # Use configuration file
  brute-force-train --config config.json
        """
    )
    
    # Main arguments
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--trainer", 
        type=str, 
        choices=list(TRAINER_CLASSES.keys()),
        help="Type of trainer to use"
    )
    
    # Model arguments
    parser.add_argument("--model-name", type=str, help="HuggingFace model name")
    parser.add_argument("--output-dir", type=str, help="Output directory for trained model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    # Dataset arguments
    parser.add_argument("--dataset-name", type=str, help="HuggingFace dataset name")
    parser.add_argument("--image-column", type=str, help="Column name for images (vision models)")
    parser.add_argument("--text-column", type=str, help="Column name for text/captions")
    parser.add_argument("--input-column", type=str, default="input", help="Input column for text models")
    parser.add_argument("--output-column", type=str, default="output", help="Output column for text models")
    parser.add_argument("--user-text", type=str, default="Convert this image to text", 
                       help="User prompt text for vision models")
    
    # Training arguments
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Steps between evaluations")
    parser.add_argument("--train-batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=1, help="Validation batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num-accumulation-steps", type=int, default=2, 
                       help="Gradient accumulation steps")
    
    # Data selection arguments
    parser.add_argument("--train-select-start", type=int, default=0, help="Training data start index")
    parser.add_argument("--train-select-end", type=int, default=1000, help="Training data end index")
    parser.add_argument("--val-select-start", type=int, default=0, help="Validation data start index")
    parser.add_argument("--val-select-end", type=int, default=1000, help="Validation data end index")
    parser.add_argument("--train-field", type=str, default="train", help="Dataset field for training")
    parser.add_argument("--val-field", type=str, default="validation", help="Dataset field for validation")
    
    # Model-specific arguments
    parser.add_argument("--max-image-size", type=int, default=500, help="Maximum image size")
    parser.add_argument("--min-pixel", type=int, default=256, help="Minimum pixel size (Qwen models)")
    parser.add_argument("--max-pixel", type=int, default=384, help="Maximum pixel size (Qwen models)")
    parser.add_argument("--image-factor", type=int, default=28, help="Image factor (Qwen models)")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length (text models)")
    
    # Display/debugging arguments
    parser.add_argument("--show-predictions", action="store_true", help="Show model predictions and ground truth during evaluation")
    parser.add_argument("--show-diff", action="store_true", help="Show colored diff between predictions and ground truth during evaluation")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    if args.config:
        config = load_config(args.config)
        
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                config[key.replace('_', '-')] = value
                
        # Convert back to underscore format for function calls
        config = {k.replace('-', '_'): v for k, v in config.items()}
    else:
        config = {k: v for k, v in vars(args).items() if v is not None}
    
    # Validate required arguments
    required_args = ['trainer', 'model_name', 'output_dir', 'dataset_name']
    missing_args = [arg for arg in required_args if arg not in config]
    if missing_args:
        parser.error(f"Missing required arguments: {', '.join(missing_args)}")
    
    # Extract trainer configuration
    trainer_type = config.pop('trainer')
    trainer_args = {
        'model_name': config.pop('model_name'),
        'output_dir': config.pop('output_dir'),
        'device': config.pop('device', 'cuda'),
        'show_predictions': config.pop('show_predictions', False),
        'show_diff': config.pop('show_diff', False)
    }
    
    # Add model-specific arguments
    if trainer_type in ['qwen2-vl', 'qwen25-vl']:
        for arg in ['min_pixel', 'max_pixel', 'image_factor']:
            if arg in config:
                trainer_args[arg] = config.pop(arg)
    elif trainer_type == 'qwen3':
        if 'max_length' in config:
            trainer_args['max_length'] = config.pop('max_length')
    
    # Create trainer
    try:
        trainer = create_trainer(trainer_type, **trainer_args)
        print(f"Created {trainer_type} trainer with model: {trainer_args['model_name']}")
        
        # Start training
        print("Starting training...")
        trainer.train_and_validate(**config)
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
