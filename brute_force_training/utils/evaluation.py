"""
Evaluation utilities for model assessment
"""

import torch
from typing import Dict, Any, List
from tqdm import tqdm
import Levenshtein
import difflib

class CharacterLevelMetrics:
    """Calculate character-level metrics for text comparison using proven methods."""
    
    @staticmethod
    def calculate_metrics(predicted: str, ground_truth: str) -> Dict[str, float]:
        """
        Calculate character-level metrics between predicted and ground truth text.
        
        Args:
            predicted: Predicted transcription
            ground_truth: Ground truth transcription
            
        Returns:
            Dictionary of metrics
        """
        # Clean strings
        pred_clean = predicted.strip()
        gt_clean = ground_truth.strip()
        
        # Character-level metrics
        char_accuracy = CharacterLevelMetrics._character_accuracy(pred_clean, gt_clean)
        edit_distance = Levenshtein.distance(pred_clean, gt_clean)
        normalized_edit_distance = edit_distance / max(len(pred_clean), len(gt_clean), 1)
        
        # Word-level metrics
        word_accuracy = CharacterLevelMetrics._word_accuracy(pred_clean, gt_clean)
        
        # Length metrics
        length_ratio = len(pred_clean) / max(len(gt_clean), 1)
        
        return {
            'character_accuracy': char_accuracy,
            'edit_distance': edit_distance,
            'normalized_edit_distance': normalized_edit_distance,
            'word_accuracy': word_accuracy,
            'length_ratio': length_ratio,
            'predicted_length': len(pred_clean),
            'ground_truth_length': len(gt_clean)
        }
    
    @staticmethod
    def _character_accuracy(predicted: str, ground_truth: str) -> float:
        """Calculate character-level accuracy using Levenshtein distance."""
        if not ground_truth:
            return 1.0 if not predicted else 0.0
        
        # Use edit distance for accuracy
        max_len = max(len(predicted), len(ground_truth))
        if max_len == 0:
            return 1.0
        
        edit_dist = Levenshtein.distance(predicted, ground_truth)
        return max(0.0, (max_len - edit_dist) / max_len)
    
    @staticmethod
    def _word_accuracy(predicted: str, ground_truth: str) -> float:
        """Calculate word-level accuracy using Jaccard similarity."""
        pred_words = set(predicted.lower().split())
        gt_words = set(ground_truth.lower().split())
        
        if not gt_words:
            return 1.0 if not pred_words else 0.0
        
        intersection = len(pred_words.intersection(gt_words))
        union = len(pred_words.union(gt_words))
        
        return intersection / union if union > 0 else 0.0


class ModelEvaluator:
    """Handles model evaluation before and after training."""
    
    def __init__(self, model, data_loader, processor=None, trainer=None):
        self.model = model
        self.data_loader = data_loader
        self.processor = processor or getattr(model, 'processor', None)
        self.trainer = trainer  # Keep reference to trainer for hybrid loss
        
    def _display_diff(self, predicted: str, ground_truth: str, sample_idx: int) -> None:
        """Display a colored diff between predicted and ground truth text."""
        print(f"\n📊 Sample {sample_idx} Comparison:")
        print("=" * 60)
        print(f"🎯 Ground Truth: {ground_truth}")
        print(f"🤖 Prediction:   {predicted}")
        
        # Generate unified diff
        diff = list(difflib.unified_diff(
            ground_truth.splitlines(keepends=True),
            predicted.splitlines(keepends=True),
            fromfile='Ground Truth',
            tofile='Prediction',
            lineterm=''
        ))
        
        if len(diff) > 2:  # Only show if there are actual differences
            print("\n🔍 Diff:")
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
            print("\n✅ Texts are identical!")
        print("=" * 60)
        
    def evaluate_model(self, num_samples: int = None, include_text_metrics: bool = True) -> Dict[str, Any]:
        """
        Evaluate model on the dataset with comprehensive metrics using proper text generation.
        
        Args:
            num_samples: Number of samples to evaluate (None for all)
            include_text_metrics: Whether to include character/word-level metrics
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        losses = []
        
        # Text metrics tracking
        text_metrics_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.data_loader, desc="Evaluating")):
                if num_samples and i >= num_samples:
                    break
                    
                # Handle both standard and enhanced batch formats
                if len(batch) == 3:
                    inputs, labels, target_texts = batch
                else:
                    inputs, labels = batch
                    target_texts = None
                
                # Calculate loss
                outputs = self.model(**inputs, labels=labels)
                
                # Use hybrid loss if available on the trainer, otherwise standard loss
                if (self.trainer and hasattr(self.trainer, 'compute_hybrid_loss') and 
                    hasattr(self.trainer, 'use_error_rate_loss') and self.trainer.use_error_rate_loss):
                    loss, _ = self.trainer.compute_hybrid_loss(outputs, labels, target_texts, inputs)
                else:
                    loss = outputs.loss
                    
                total_loss += loss.item()
                losses.append(loss.item())
                total_samples += inputs['input_ids'].size(0)
                
                # Generate text for quality metrics (if requested)
                if include_text_metrics:
                    try:
                        # Process each item in the batch
                        for batch_idx in range(inputs['input_ids'].size(0)):
                            # Get single item from batch
                            single_inputs = {k: v[batch_idx:batch_idx+1] for k, v in inputs.items() if k != 'labels'}
                            single_labels = labels[batch_idx:batch_idx+1]
                            
                            # Extract target text from labels
                            target_tokens = single_labels[0][single_labels[0] != -100]
                            if len(target_tokens) == 0:
                                continue
                                
                            target_text = self.processor.decode(target_tokens, skip_special_tokens=True).strip()
                            
                            # Generate prediction with more conservative settings
                            try:
                                # Validate input tensors before generation
                                if 'pixel_values' in single_inputs and single_inputs['pixel_values'].numel() == 0:
                                    print(f"  Empty pixel values in batch {i}, sample {batch_idx} - skipping text evaluation")
                                    continue
                                
                                # Additional validation for vision-language models
                                if 'pixel_values' in single_inputs:
                                    pixel_shape = single_inputs['pixel_values'].shape
                                    if len(pixel_shape) < 3 or pixel_shape[0] == 0:
                                        print(f"  Invalid pixel shape {pixel_shape} in batch {i}, sample {batch_idx} - skipping")
                                        continue
                                
                                # For models with image token constraints (e.g., LFM2-VL), use conservative generation
                                generation_kwargs = {
                                    'max_new_tokens': min(50, len(target_tokens)),  # Very conservative for image-text models
                                    'do_sample': False,  # Disable sampling completely
                                    'temperature': 1.0,  # Set explicitly even though do_sample=False
                                    'top_p': 1.0,  # Set explicitly to prevent issues
                                    'top_k': 50,  # Set explicitly to prevent issues
                                    'pad_token_id': getattr(self.model.config, 'eos_token_id', getattr(self.model.config, 'pad_token_id', 0)),
                                    'eos_token_id': getattr(self.model.config, 'eos_token_id', getattr(self.model.config, 'pad_token_id', 0)),
                                }
                                
                                # Merge single_inputs with generation_kwargs, avoiding duplicates
                                final_kwargs = {**single_inputs, **generation_kwargs}
                                
                                generated_ids = self.model.generate(**final_kwargs)
                                
                            except Exception as gen_error:
                                # More detailed error logging for debugging
                                error_msg = str(gen_error)
                                if "Image features and image tokens do not match" in error_msg:
                                    print(f"  Image token mismatch in batch {i}, sample {batch_idx} - skipping text evaluation for this sample")
                                elif "shape" in error_msg.lower():
                                    print(f"  Shape error in batch {i}, sample {batch_idx} - skipping text evaluation for this sample")
                                elif "size mismatch" in error_msg.lower():
                                    print(f"  Size mismatch in batch {i}, sample {batch_idx} - skipping text evaluation for this sample")
                                else:
                                    # For debugging: show first occurrence of unknown errors
                                    if not hasattr(self, '_generation_error_logged'):
                                        print(f"  Generation error details: {error_msg}")
                                        self._generation_error_logged = True
                                    print(f"  Generation failed for batch {i}, sample {batch_idx} - skipping text evaluation")
                                continue
                            
                            # Extract only the generated part (remove input)
                            input_length = single_inputs['input_ids'].size(1)
                            generated_tokens = generated_ids[0][input_length:]
                            generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True).strip()
                            
                            # Calculate metrics using your proven method
                            if target_text and generated_text:  # Only if both are non-empty
                                metrics = CharacterLevelMetrics.calculate_metrics(generated_text, target_text)
                                text_metrics_list.append(metrics)
                                
                                # Display predictions and/or diffs if requested by trainer
                                sample_idx = i * inputs['input_ids'].size(0) + batch_idx
                                if self.trainer and hasattr(self.trainer, 'show_predictions') and self.trainer.show_predictions:
                                    print(f"\n📝 Sample {sample_idx}:")
                                    print(f"🎯 Ground Truth: {target_text}")
                                    print(f"🤖 Prediction:   {generated_text}")
                                
                                if self.trainer and hasattr(self.trainer, 'show_diff') and self.trainer.show_diff:
                                    self._display_diff(generated_text, target_text, sample_idx)
                                
                    except Exception as e:
                        # Skip text metrics if generation fails, but continue with loss
                        print(f"Text generation failed for batch {i}: {e}")
                        continue
        
        # Calculate basic metrics
        avg_loss = total_loss / len(losses) if losses else float('inf')
        min_loss = min(losses) if losses else float('inf')
        max_loss = max(losses) if losses else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss != float('inf') else float('inf')
        
        eval_results = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'min_loss': min_loss,
            'max_loss': max_loss,
            'num_samples': total_samples,
            'num_batches': len(losses)
        }
        
        # Add text quality metrics if available
        if include_text_metrics and text_metrics_list:
            # Calculate averages using your proven approach
            avg_char_accuracy = sum(m['character_accuracy'] for m in text_metrics_list) / len(text_metrics_list)
            avg_word_accuracy = sum(m['word_accuracy'] for m in text_metrics_list) / len(text_metrics_list)
            avg_edit_distance = sum(m['edit_distance'] for m in text_metrics_list) / len(text_metrics_list)
            avg_normalized_edit_distance = sum(m['normalized_edit_distance'] for m in text_metrics_list) / len(text_metrics_list)
            avg_length_ratio = sum(m['length_ratio'] for m in text_metrics_list) / len(text_metrics_list)
            
            eval_results.update({
                'avg_char_accuracy': avg_char_accuracy,
                'avg_word_accuracy': avg_word_accuracy,
                'avg_edit_distance': avg_edit_distance,
                'avg_normalized_edit_distance': avg_normalized_edit_distance,
                'avg_length_ratio': avg_length_ratio,
                'text_samples_evaluated': len(text_metrics_list),
                'detailed_text_metrics': text_metrics_list  # Store for debugging
            })
            
            print(f"  Text evaluation: {len(text_metrics_list)} samples processed")
            print(f"  Avg character accuracy: {avg_char_accuracy:.3f}")
            print(f"  Avg word accuracy: {avg_word_accuracy:.3f}")
        elif include_text_metrics:
            # Text metrics were requested but none succeeded
            print(f"  ⚠️ Text evaluation failed - no successful generations out of {total_samples} samples")
            print(f"  Continuing with loss-only metrics...")
        
        self.model.train()  # Set back to training mode
        return eval_results
        
    def generate_sample_outputs(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Generate sample outputs from the model for qualitative evaluation.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of sample inputs and outputs
        """
        self.model.eval()
        samples = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.data_loader):
                if i >= num_samples:
                    break
                    
                inputs, labels = batch
                
                # Generate output
                with torch.no_grad():
                    # For generation, we might want to use generate() method
                    # This is a simplified version that just gets the model output
                    outputs = self.model(**inputs)
                    
                # Store sample info
                sample = {
                    'batch_index': i,
                    'input_ids': inputs['input_ids'][0].cpu().tolist(),  # First item in batch
                    'generated_logits': outputs.logits[0].cpu(),
                    'loss': self.model(**inputs, labels=labels).loss.item()
                }
                samples.append(sample)
        
        self.model.train()
        return samples
