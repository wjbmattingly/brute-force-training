"""
Fixed example for training on CATMuS/medieval dataset
"""

from brute_force_training import LFM2VLTrainer

def main():
    trainer = LFM2VLTrainer(
        model_name="LiquidAI/LFM2-VL-450M",
        output_dir="./lfm2-vl-finetuned-450M-test",
    )

    trainer.train_and_validate(
        dataset_name="CATMuS/medieval",
        image_column="im",
        text_column="text",
        user_text="Transcribe this medieval manuscript line",

        # Training parameters
        eval_steps=100,
        max_steps=1000,
        num_accumulation_steps=4,
        learning_rate=1e-5,
        
        # FIXED: Data selection within the actual dataset size
        # CATMuS/medieval has ~152K examples in the train split
        train_select_start=0,
        train_select_end=80000,      # Use first 80K for training
        val_select_start=80000,      # Use next 10K for validation  
        val_select_end=90000,
        
        # Both use the same "train" field since there's no separate validation split
        train_field="train",
        val_field="train",           # Same as train_field

        # Batch sizes
        train_batch_size=10,
        val_batch_size=10,

        validate_before=True,        # Pre-training evaluation
        generate_docs=True           # Full documentation
    )

if __name__ == "__main__":
    main()




