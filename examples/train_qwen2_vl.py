"""
Example: Training Qwen2-VL on a vision-language dataset
"""

from brute_force_training import Qwen2VLTrainer

def main():
    # Initialize the trainer
    trainer = Qwen2VLTrainer(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        output_dir="./qwen2_vl_finetuned",
        device="cuda",
        min_pixel=256,
        max_pixel=384,
        image_factor=28
    )
    
    # Train the model
    trainer.train_and_validate(
        dataset_name="CATMuS/medieval",  # Replace with your dataset
        image_column="im",
        text_column="text",
        user_text="Transcribe this medieval manuscript line",
        
        # Training parameters
        max_steps=5000,
        eval_steps=500,
        num_accumulation_steps=2,
        learning_rate=1e-5,
        
        # Data selection
        train_select_start=0,
        train_select_end=2000,
        val_select_start=2000,
        val_select_end=2500,
        
        # Batch sizes
        train_batch_size=1,
        val_batch_size=1,
        
        # Image preprocessing
        max_image_size=500
    )

if __name__ == "__main__":
    main()
