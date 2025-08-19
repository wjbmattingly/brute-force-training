"""
Base trainer class with common functionality
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from datasets import load_dataset
from ..utils.documentation import ModelDocumenter
from ..utils.evaluation import ModelEvaluator
import difflib


class BaseTrainer(ABC):
    """
    Base trainer class providing common training functionality.
    
    This abstract base class defines the interface and common functionality
    for all trainer implementations. Subclasses should implement the 
    abstract methods to provide model-specific behavior.
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        device: str = "cuda",
        show_predictions: bool = False,
        show_diff: bool = False
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device
        self.show_predictions = show_predictions
        self.show_diff = show_diff
        self.model = None
        self.tokenizer_or_processor = None
        self.documenter = ModelDocumenter(model_name, output_dir)
        self.current_step = 0
        
    def _display_training_prediction(self, inputs, labels, step: int, sample_idx: int = 0) -> None:
        """Display prediction for a training sample if flags are enabled."""
        if not (self.show_predictions or self.show_diff):
            return
            
        try:
            self.model.eval()  # Temporarily switch to eval mode
            with torch.no_grad():
                # Get single item from batch for prediction display
                single_inputs = {k: v[sample_idx:sample_idx+1] for k, v in inputs.items() if k != 'labels'}
                single_labels = labels[sample_idx:sample_idx+1]
                
                # Extract target text from labels
                target_tokens = single_labels[0][single_labels[0] != -100]
                if len(target_tokens) == 0:
                    return
                    
                target_text = self.tokenizer_or_processor.decode(target_tokens, skip_special_tokens=True).strip()
                
                # Generate prediction with more generous settings for better text quality
                generation_kwargs = {
                    'max_new_tokens': min(100, max(50, len(target_tokens))),  # More generous token limit
                    'do_sample': False,  # Greedy sampling for consistency
                    'pad_token_id': getattr(self.model.config, 'eos_token_id', getattr(self.model.config, 'pad_token_id', 0)),
                    'eos_token_id': getattr(self.model.config, 'eos_token_id', getattr(self.model.config, 'pad_token_id', 0)),
                    'use_cache': False,
                    'repetition_penalty': 1.1,  # Slight penalty to avoid repetition
                    'temperature': 1.0,
                    'top_p': 1.0,
                }
                
                final_kwargs = {**single_inputs, **generation_kwargs}
                generated_ids = self.model.generate(**final_kwargs)
                
                # Extract only the generated part
                input_length = single_inputs['input_ids'].size(1)
                generated_tokens = generated_ids[0][input_length:]
                
                # More robust decoding that handles special characters better
                try:
                    # Try different decoding strategies
                    generated_text = self.tokenizer_or_processor.decode(generated_tokens, skip_special_tokens=True).strip()
                    if not generated_text:  # If empty, try without skipping special tokens
                        generated_text = self.tokenizer_or_processor.decode(generated_tokens, skip_special_tokens=False).strip()
                except Exception as decode_error:
                    generated_text = f"[Decode Error: {str(decode_error)}]"
                
                # Display predictions and/or diffs
                if self.show_predictions:
                    print(f"\n📝 Training Step {step} Sample:")
                    print(f"🎯 Ground Truth: {repr(target_text)}")  # Use repr to show special characters
                    print(f"🤖 Prediction:   {repr(generated_text)}")  # Use repr to show special characters
                    print(f"📏 GT Length: {len(target_text)}, Pred Length: {len(generated_text)}")
                    
                    # Quick quality metrics for this sample
                    if target_text and generated_text:
                        import Levenshtein
                        char_dist = Levenshtein.distance(generated_text, target_text)
                        char_acc = 1.0 - (char_dist / max(len(target_text), len(generated_text), 1))
                        print(f"📊 Sample CER: {char_dist}/{max(len(target_text), len(generated_text))} = {1-char_acc:.3f}, Char Acc: {char_acc:.3f}")
                
                if self.show_diff:
                    self._display_training_diff(generated_text, target_text, step)
                    
        except Exception as e:
            # Don't interrupt training if prediction display fails
            print(f"  ⚠️ Training prediction display failed at step {step}: {e}")
        finally:
            self.model.train()  # Switch back to training mode
    
    def _display_training_diff(self, predicted: str, ground_truth: str, step: int) -> None:
        """Display a colored diff for training samples."""
        print(f"\n📊 Training Step {step} Diff:")
        print("=" * 50)
        
        # Generate unified diff
        diff = list(difflib.unified_diff(
            ground_truth.splitlines(keepends=True),
            predicted.splitlines(keepends=True),
            fromfile='Ground Truth',
            tofile='Prediction',
            lineterm=''
        ))
        
        if len(diff) > 2:  # Only show if there are actual differences
            print("🔍 Diff:")
            for line in diff:
                if line.startswith('+++') or line.startswith('---'):
                    print(f"  {line.strip()}")
                elif line.startswith('+'):
                    print(f"  \033[92m{line.rstrip()}\033[0m")  # Green for additions
                elif line.startswith('-'):
                    print(f"  \033[91m{line.rstrip()}\033[0m")  # Red for deletions
                elif line.startswith('@@'):
                    print(f"  \033[94m{line.rstrip()}\033[0m")  # Blue for line numbers
                else:
                    print(f"  {line.rstrip()}")
        else:
            print("✅ Texts are identical!")
        print("=" * 50)
        
    @abstractmethod
    def load_model_and_processor(self) -> None:
        """Load the model and tokenizer/processor."""
        pass
        
    @abstractmethod
    def create_dataset(self, dataset: Any, **kwargs) -> Any:
        """Create a dataset instance from HuggingFace dataset."""
        pass
        
    @abstractmethod
    def create_collate_fn(self) -> callable:
        """Create the collate function for data loading."""
        pass
        
    def filter_dataset(self, dataset: Any, filter_fn: Optional[callable] = None) -> Any:
        """Filter dataset using provided function. No default filtering applied."""
        if filter_fn is None:
            return dataset
        return dataset.filter(filter_fn)
        
    def create_data_loaders(
        self,
        train_dataset: Any,
        val_dataset: Any,
        train_batch_size: int = 1,
        val_batch_size: int = 1
    ) -> tuple:
        """Create training and validation data loaders."""
        collate_fn = self.create_collate_fn()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            collate_fn=collate_fn,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader
        
    def validate(self, val_loader: DataLoader) -> Optional[float]:
        """
        Validate the model on the validation dataset.
        
        Args:
            val_loader: DataLoader for the validation dataset.
            
        Returns:
            The average validation loss, or None if validation set is empty.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model_and_processor() first.")
            
        self.model.eval()
        
        # Check if validation loader is empty
        if len(val_loader) == 0:
            print("Warning: Validation loader is empty. Skipping validation.")
            self.model.train()
            return None
        
        total_val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Handle both standard and enhanced batch formats
                if len(batch) == 3:
                    inputs, labels, target_texts = batch
                else:
                    inputs, labels = batch
                    target_texts = None
                
                outputs = self.model(**inputs, labels=labels)
                
                # Use hybrid loss if available, otherwise standard loss
                if hasattr(self, 'compute_hybrid_loss') and hasattr(self, 'use_error_rate_loss') and self.use_error_rate_loss:
                    loss, _ = self.compute_hybrid_loss(outputs, labels, target_texts)
                else:
                    loss = outputs.loss
                    
                total_val_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else None
        self.model.train()
        return avg_val_loss
        
    def save_model(self, step: int, is_final: bool = False) -> None:
        """Save the model, tokenizer/processor, and documentation."""
        if self.model is None or self.tokenizer_or_processor is None:
            raise RuntimeError("Model or processor not loaded.")
            
        if is_final:
            save_dir = os.path.join(self.output_dir, "final")
        else:
            save_dir = os.path.join(self.output_dir, f"model_step_{step}")
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_dir)
        self.tokenizer_or_processor.save_pretrained(save_dir)
        
        # Save model card metadata for HuggingFace
        model_card_metadata = self.documenter.create_model_card_metadata()
        with open(os.path.join(save_dir, 'model_card_metadata.json'), 'w') as f:
            json.dump(model_card_metadata, f, indent=2)
        
        # Save comprehensive documentation
        self.documenter.save_documentation(save_dir, is_final)
        
        print(f"✅ Model {'and documentation ' if is_final else ''}saved to {save_dir}")
        
    def train_and_validate(
        self,
        dataset_name: str,
        num_accumulation_steps: int = 2,
        eval_steps: int = 500,
        max_steps: int = 10000,
        train_select_start: int = 0,
        train_select_end: int = 1000,
        val_select_start: int = 0,
        val_select_end: int = 1000,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        train_field: str = "train",
        val_field: str = "validation",
        learning_rate: float = 1e-5,
        validate_before: bool = True,
        generate_docs: bool = True,
        validation_text_metrics: bool = True,  # Whether to include text metrics during training validation
        validation_samples: int = 100,  # Number of samples to use for validation during training
        **kwargs
    ) -> None:
        """
        Main training and validation loop.
        
        Args:
            dataset_name: Name of the HuggingFace dataset to use.
            num_accumulation_steps: Number of steps for gradient accumulation.
            eval_steps: Number of steps between evaluations.
            max_steps: Maximum number of training steps.
            train_select_start: Starting index for selecting training data.
            train_select_end: Ending index for selecting training data.
            val_select_start: Starting index for selecting validation data.
            val_select_end: Ending index for selecting validation data.
            train_batch_size: Batch size for training.
            val_batch_size: Batch size for validation.
            train_field: Field name for training data in the dataset.
            val_field: Field name for validation data in the dataset.
            learning_rate: Learning rate for the optimizer.
            validate_before: Whether to run evaluation before training starts.
            generate_docs: Whether to generate documentation and visualizations.
            **kwargs: Additional arguments passed to create_dataset.
        """
        # Load model and processor
        self.load_model_and_processor()
        
        # Load and prepare dataset
        dataset = load_dataset(dataset_name)
        
        # Apply filtering
        filtered_train_dataset = self.filter_dataset(dataset[train_field])
        filtered_val_dataset = self.filter_dataset(dataset[val_field])
        
        print(f"Number of training examples after filtering: {len(filtered_train_dataset)}")
        print(f"Number of validation examples after filtering: {len(filtered_val_dataset)}")
        
        # Handle case where train and validation come from the same split
        if train_field == val_field:
            print("📝 Using same dataset split for training and validation")
            # Use the training dataset for both, but split it properly
            total_examples = len(filtered_train_dataset)
            
            # Calculate actual selection ranges
            actual_train_end = min(train_select_end, total_examples)
            actual_val_start = min(val_select_start, total_examples)
            actual_val_end = min(val_select_end, total_examples)
            
            print(f"Total examples: {total_examples}")
            print(f"Training selection: {train_select_start}:{actual_train_end}")
            print(f"Validation selection: {actual_val_start}:{actual_val_end}")
            
            # Ensure validation range is valid
            if actual_val_start >= total_examples or actual_val_start >= actual_val_end:
                print(f"⚠️ Warning: Invalid validation range ({actual_val_start}:{actual_val_end}) for dataset size {total_examples}")
                print("🔧 Auto-adjusting: Using last 10% of training data for validation")
                
                # Auto-adjust: use last 10% of available data for validation
                val_size = max(100, int(actual_train_end * 0.1))  # At least 100 examples or 10%
                actual_val_start = max(0, actual_train_end - val_size)
                actual_val_end = actual_train_end
                actual_train_end = actual_val_start  # Adjust training end to not overlap
                
                print(f"📊 Adjusted - Training: {train_select_start}:{actual_train_end}, Validation: {actual_val_start}:{actual_val_end}")
            
            # Select subsets from the same filtered dataset
            shuffled_dataset = filtered_train_dataset.shuffle(seed=42)
            train_dataset = shuffled_dataset.select(range(train_select_start, actual_train_end))
            val_dataset = shuffled_dataset.select(range(actual_val_start, actual_val_end))
            
        else:
            # Different splits for training and validation
            print("📝 Using separate dataset splits for training and validation")
            
            # Calculate actual selection ranges
            actual_train_end = min(train_select_end, len(filtered_train_dataset))
            actual_val_end = min(val_select_end, len(filtered_val_dataset))
            
            print(f"Training selection: {train_select_start}:{actual_train_end}")
            print(f"Validation selection: {val_select_start}:{actual_val_end}")
            
            # Select subsets
            train_dataset = filtered_train_dataset.shuffle(seed=42).select(
                range(train_select_start, actual_train_end)
            )
            
            # Handle validation dataset selection
            if val_select_start >= len(filtered_val_dataset):
                print(f"⚠️ Warning: Validation start index ({val_select_start}) >= dataset size ({len(filtered_val_dataset)})")
                print("Creating empty validation set - validation will be skipped")
                val_dataset = filtered_val_dataset.select([])  # Empty selection
            else:
                val_dataset = filtered_val_dataset.shuffle(seed=42).select(
                    range(val_select_start, actual_val_end)
                )
        
        print(f"Final training examples selected: {len(train_dataset)}")
        print(f"Final validation examples selected: {len(val_dataset)}")
        
        # Create dataset instances
        train_dataset = self.create_dataset(train_dataset, **kwargs)
        val_dataset = self.create_dataset(val_dataset, **kwargs)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            train_dataset, val_dataset, train_batch_size, val_batch_size
        )
        
        # Store training configuration in documenter
        if generate_docs:
            training_config = {
                'dataset_name': dataset_name,
                'model_name': self.model_name,
                'max_steps': max_steps,
                'eval_steps': eval_steps,
                'num_accumulation_steps': num_accumulation_steps,
                'learning_rate': learning_rate,
                'train_batch_size': train_batch_size,
                'val_batch_size': val_batch_size,
                'train_select_start': train_select_start,
                'train_select_end': train_select_end,
                'val_select_start': val_select_start,
                'val_select_end': val_select_end,
                'train_field': train_field,
                'val_field': val_field,
                **kwargs
            }
            self.documenter.set_training_config(training_config)
        
        # Pre-training evaluation
        if validate_before and len(val_loader) > 0:
            print("🔍 Running pre-training evaluation...")
            evaluator = ModelEvaluator(self.model, val_loader, self.tokenizer_or_processor, trainer=self)
            pre_training_results = evaluator.evaluate_model(num_samples=min(50, len(val_loader)), include_text_metrics=True)
            
            # Print comprehensive pre-training results
            print(f"📊 Pre-training Results:")
            print(f"   Loss: {pre_training_results['loss']:.6f}")
            print(f"   Perplexity: {pre_training_results['perplexity']:.2f}")
            if pre_training_results.get('avg_char_accuracy'):
                print(f"   Character Accuracy: {pre_training_results['avg_char_accuracy']*100:.1f}%")
            if pre_training_results.get('avg_word_accuracy'):
                print(f"   Word Accuracy: {pre_training_results['avg_word_accuracy']*100:.1f}%")
            
            if generate_docs:
                self.documenter.set_pre_training_eval(pre_training_results)
        
        # Setup training
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        global_step = 0
        progress_bar = tqdm(total=max_steps, desc="Training")
        
        # Training loop
        while global_step < max_steps:
            for batch in train_loader:
                global_step += 1
                self.current_step = global_step
                
                # Handle both standard and enhanced batch formats
                if len(batch) == 3:
                    inputs, labels, target_texts = batch
                else:
                    inputs, labels = batch
                    target_texts = None
                
                outputs = self.model(**inputs, labels=labels)
                
                # Use hybrid loss if available, otherwise standard loss
                if hasattr(self, 'compute_hybrid_loss') and hasattr(self, 'use_error_rate_loss') and self.use_error_rate_loss:
                    loss, loss_components = self.compute_hybrid_loss(outputs, labels, target_texts)
                    loss = loss / num_accumulation_steps
                    
                    # Log detailed loss components if documentation is enabled
                    if generate_docs and global_step % 100 == 0:  # Log every 100 steps
                        print(f"Step {global_step} Loss Components: CE={loss_components['ce_loss']:.4f}, CER={loss_components['cer_loss']:.4f}, WER={loss_components['wer_loss']:.4f}")
                else:
                    loss = outputs.loss / num_accumulation_steps
                
                # Display training predictions periodically if requested
                if (self.show_predictions or self.show_diff) and global_step % 100 == 0:
                    self._display_training_prediction(inputs, labels, global_step)
                
                loss.backward()
                
                # Log training metrics
                if generate_docs:
                    current_lr = optimizer.param_groups[0]['lr']
                    self.documenter.log_training_step(
                        step=global_step, 
                        loss=loss.item() * num_accumulation_steps,
                        learning_rate=current_lr
                    )
                
                if global_step % num_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item() * num_accumulation_steps})

                # Perform evaluation and save model
                if global_step % eval_steps == 0 or global_step == max_steps:
                    if len(val_loader) > 0:
                        print(f"\n🔍 Running validation at step {global_step}...")
                        evaluator = ModelEvaluator(self.model, val_loader, self.tokenizer_or_processor, trainer=self)
                        # Use configurable sample size during training for efficiency
                        eval_samples = min(validation_samples, len(val_loader))
                        eval_results = evaluator.evaluate_model(num_samples=eval_samples, include_text_metrics=validation_text_metrics)
                        
                        print(f"📈 Step {global_step} Validation Results:")
                        print(f"   Loss: {eval_results['loss']:.6f}")
                        print(f"   Perplexity: {eval_results['perplexity']:.2f}")
                        if eval_results.get('avg_char_accuracy') is not None:
                            print(f"   Character Accuracy: {eval_results['avg_char_accuracy']*100:.1f}%")
                        if eval_results.get('avg_word_accuracy') is not None:
                            print(f"   Word Accuracy: {eval_results['avg_word_accuracy']*100:.1f}%")
                        if eval_results.get('text_samples_evaluated', 0) > 0:
                            print(f"   Text samples evaluated: {eval_results['text_samples_evaluated']}/{eval_samples}")
                        
                        # Log comprehensive evaluation metrics
                        if generate_docs:
                            checkpoint_type = "final" if global_step == max_steps else "checkpoint"
                            self.documenter.log_evaluation_checkpoint(global_step, eval_results, checkpoint_type)
                    else:
                        print(f"\n⚠️ Step {global_step}: Validation skipped (empty validation set)")

                    self.save_model(global_step)
                    self.model.train()  # Set back to training mode

                if global_step >= max_steps:
                    # Final comprehensive evaluation is already done above in the eval step
                    # Just save the final model
                    self.save_model(global_step, is_final=True)
                    
                    # Show improvement summary if we have both pre and post training data
                    if generate_docs and self.documenter.pre_training_eval and len(self.documenter.evaluation_history) > 1:
                        pre_eval = self.documenter.evaluation_history[0]  # First (pre-training)
                        final_eval = self.documenter.evaluation_history[-1]  # Last (final)
                        
                        print(f"\n🎯 Training Summary:")
                        pre_loss = pre_eval.get('loss')
                        final_loss = final_eval.get('loss')
                        if pre_loss and final_loss:
                            loss_improvement = ((pre_loss - final_loss) / pre_loss * 100)
                            print(f"   Loss improvement: {loss_improvement:+.2f}% (from {pre_loss:.6f} to {final_loss:.6f})")
                        
                        if pre_eval.get('avg_char_accuracy') and final_eval.get('avg_char_accuracy'):
                            char_improvement = (final_eval['avg_char_accuracy'] - pre_eval['avg_char_accuracy']) * 100
                            print(f"   Character accuracy improvement: {char_improvement:+.1f}% (from {pre_eval['avg_char_accuracy']*100:.1f}% to {final_eval['avg_char_accuracy']*100:.1f}%)")
                        
                        if pre_eval.get('avg_word_accuracy') and final_eval.get('avg_word_accuracy'):
                            word_improvement = (final_eval['avg_word_accuracy'] - pre_eval['avg_word_accuracy']) * 100
                            print(f"   Word accuracy improvement: {word_improvement:+.1f}% (from {pre_eval['avg_word_accuracy']*100:.1f}% to {final_eval['avg_word_accuracy']*100:.1f}%)")
                    
                    break

            if global_step >= max_steps:
                break

        progress_bar.close()
        print("🎉 Training completed!")
