"""
Example: Training Qwen3 on a text-to-text dataset
"""

from brute_force_training import Qwen3Trainer

def main():
    # Initialize the trainer
    trainer = Qwen3Trainer(
        model_name="Qwen/Qwen3-4B-Thinking-2507",
        output_dir="./qwen3_finetuned",
        device="cuda",
        max_length=2048
    )
    
    # Train the model
    trainer.train_and_validate(
        dataset_name="your_text_dataset",  # Replace with your dataset
        input_column="input",
        output_column="output",
        
        # System prompt (optional) - like user_text for VLMs
        system_prompt="You are a helpful assistant. Please provide clear and accurate responses.",
        
        # Training parameters
        max_steps=8000,
        eval_steps=1000,
        num_accumulation_steps=2,
        learning_rate=1e-5,
        
        # Data selection
        train_select_start=0,
        train_select_end=5000,
        val_select_start=5000,
        val_select_end=6000,
        
        # Batch sizes (can be larger for text-only)
        train_batch_size=4,
        val_batch_size=4
    )

if __name__ == "__main__":
    main()
