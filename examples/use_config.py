"""
Example: Using configuration files for training
"""

from brute_force_training.config import TrainingConfig, get_config
from brute_force_training.trainers import create_trainer

def train_with_json_config():
    """Train using a JSON configuration file."""
    # Load config from JSON
    config = TrainingConfig.from_json("config_example.json")
    
    # Create trainer based on config
    trainer_args = {
        'model_name': config.model_name,
        'output_dir': config.output_dir,
        'device': config.device
    }
    
    # Add model-specific arguments
    if config.trainer_type in ['qwen2-vl', 'qwen25-vl']:
        trainer_args.update({
            'min_pixel': config.min_pixel,
            'max_pixel': config.max_pixel,
            'image_factor': config.image_factor
        })
    elif config.trainer_type == 'qwen3':
        trainer_args['max_length'] = config.max_length
    
    # Get trainer class and create instance
    if config.trainer_type == "qwen2-vl":
        from brute_force_training import Qwen2VLTrainer
        trainer = Qwen2VLTrainer(**trainer_args)
    elif config.trainer_type == "qwen25-vl":
        from brute_force_training import Qwen25VLTrainer
        trainer = Qwen25VLTrainer(**trainer_args)
    elif config.trainer_type == "lfm2-vl":
        from brute_force_training import LFM2VLTrainer
        trainer = LFM2VLTrainer(**trainer_args)
    elif config.trainer_type == "qwen3":
        from brute_force_training import Qwen3Trainer
        trainer = Qwen3Trainer(**trainer_args)
    else:
        raise ValueError(f"Unknown trainer type: {config.trainer_type}")
    
    # Extract training arguments
    training_args = {
        'dataset_name': config.dataset_name,
        'max_steps': config.max_steps,
        'eval_steps': config.eval_steps,
        'num_accumulation_steps': config.num_accumulation_steps,
        'learning_rate': config.learning_rate,
        'train_batch_size': config.train_batch_size,
        'val_batch_size': config.val_batch_size,
        'train_select_start': config.train_select_start,
        'train_select_end': config.train_select_end,
        'val_select_start': config.val_select_start,
        'val_select_end': config.val_select_end,
        'train_field': config.train_field,
        'val_field': config.val_field
    }
    
    # Add model-specific training arguments
    if config.trainer_type in ['qwen2-vl', 'qwen25-vl', 'lfm2-vl']:
        training_args.update({
            'image_column': config.image_column,
            'text_column': config.text_column,
            'user_text': config.user_text,
            'max_image_size': config.max_image_size
        })
    elif config.trainer_type == 'qwen3':
        training_args.update({
            'input_column': config.input_column,
            'output_column': config.output_column
        })
    
    # Start training
    trainer.train_and_validate(**training_args)

def train_with_predefined_config():
    """Train using a predefined configuration."""
    # Get a predefined config
    config = get_config("qwen2_vl_medieval")
    
    # Modify as needed
    config.output_dir = "./my_medieval_model"
    config.max_steps = 5000
    
    # Convert to dict and train (similar to above)
    print(f"Using predefined config for {config.trainer_type}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Max steps: {config.max_steps}")

def create_and_save_config():
    """Create a new configuration and save it."""
    config = TrainingConfig(
        trainer_type="lfm2-vl",
        model_name="LiquidAI/LFM2-VL-450M",
        output_dir="./lfm2_custom",
        dataset_name="your_dataset",
        image_column="image",
        text_column="caption",
        max_steps=3000,
        train_batch_size=1
    )
    
    # Save to JSON
    config.to_json("my_lfm2_config.json")
    print("Configuration saved to my_lfm2_config.json")

if __name__ == "__main__":
    # Example 1: Create and save a config
    create_and_save_config()
    
    # Example 2: Use predefined config
    train_with_predefined_config()
    
    # Example 3: Train with JSON config (uncomment to run)
    # train_with_json_config()
