"""
Example: Training with comprehensive documentation and evaluation
"""

from brute_force_training import LFM2VLTrainer

def main():
    # Initialize the trainer
    trainer = LFM2VLTrainer(
        model_name="LiquidAI/LFM2-VL-450M",
        output_dir="./lfm2_vl_documented",
        device="cuda"
    )
    
    # Train with full documentation and evaluation
    trainer.train_and_validate(
        dataset_name="CATMuS/medieval",  # Replace with your dataset
        image_column="im",
        text_column="text",
        user_text="Transcribe this medieval manuscript",
        
        # Training parameters
        max_steps=2000,
        eval_steps=200,
        num_accumulation_steps=2,
        learning_rate=1e-5,
        
        # Data selection
        train_select_start=0,
        train_select_end=800,
        val_select_start=800,
        val_select_end=1000,
        
        # Batch sizes
        train_batch_size=1,
        val_batch_size=1,
        
        # New features
        validate_before=True,    # Run evaluation before training
        generate_docs=True,      # Generate documentation and visualizations
        
        # Image preprocessing
        max_image_size=500
    )
    
    print("🎉 Training completed with full documentation!")
    print("📁 Check the output directory for:")
    print("   - Model checkpoints")
    print("   - README.md files with training details")
    print("   - Training curve visualizations")
    print("   - Evaluation comparisons")
    print("   - HuggingFace model card metadata")

if __name__ == "__main__":
    main()
