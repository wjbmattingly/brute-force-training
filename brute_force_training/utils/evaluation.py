"""
Evaluation utilities for model assessment
"""

import torch
import re
from typing import Dict, Any, List
from tqdm import tqdm
import difflib


class ModelEvaluator:
    """Handles model evaluation before and after training."""
    
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        
    def evaluate_model(self, num_samples: int = None, include_text_metrics: bool = True) -> Dict[str, Any]:
        """
        Evaluate model on the dataset with comprehensive metrics.
        
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
        total_chars_generated = 0
        total_chars_target = 0
        total_words_generated = 0
        total_words_target = 0
        char_accuracy_scores = []
        word_accuracy_scores = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.data_loader, desc="Evaluating")):
                if num_samples and i >= num_samples:
                    break
                    
                inputs, labels = batch
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                losses.append(loss.item())
                total_samples += inputs['input_ids'].size(0)
                
                # Generate text for quality metrics (if requested)
                if include_text_metrics and hasattr(self.model, 'generate'):
                    try:
                        # Simple generation for comparison
                        generated_ids = self.model.generate(
                            **{k: v for k, v in inputs.items() if k != 'labels'},
                            max_new_tokens=50,
                            do_sample=False,
                            pad_token_id=self.model.config.eos_token_id
                        )
                        
                        # Get the processor/tokenizer for decoding
                        if hasattr(self.model, 'processor'):
                            processor = self.model.processor
                        else:
                            # Try to get tokenizer from the model
                            processor = getattr(self.model, 'tokenizer', None)
                        
                        if processor:
                            # Decode generated and target text
                            generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
                            target_text = processor.decode(labels[0][labels[0] != -100], skip_special_tokens=True)
                            
                            # Calculate character-level metrics
                            char_acc = self._calculate_character_accuracy(generated_text, target_text)
                            char_accuracy_scores.append(char_acc)
                            total_chars_generated += len(generated_text)
                            total_chars_target += len(target_text)
                            
                            # Calculate word-level metrics
                            word_acc = self._calculate_word_accuracy(generated_text, target_text)
                            word_accuracy_scores.append(word_acc)
                            total_words_generated += len(generated_text.split())
                            total_words_target += len(target_text.split())
                            
                    except Exception as e:
                        # Skip text metrics if generation fails
                        pass
        
        avg_loss = total_loss / len(losses) if losses else float('inf')
        
        # Calculate additional metrics
        min_loss = min(losses) if losses else float('inf')
        max_loss = max(losses) if losses else float('inf')
        
        # Calculate perplexity (for language models)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        eval_results = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'min_loss': min_loss,
            'max_loss': max_loss,
            'num_samples': total_samples,
            'num_batches': len(losses)
        }
        
        # Add text quality metrics if available
        if include_text_metrics and char_accuracy_scores:
            eval_results.update({
                'avg_char_accuracy': sum(char_accuracy_scores) / len(char_accuracy_scores),
                'avg_word_accuracy': sum(word_accuracy_scores) / len(word_accuracy_scores),
                'total_chars_generated': total_chars_generated,
                'total_chars_target': total_chars_target,
                'total_words_generated': total_words_generated,
                'total_words_target': total_words_target,
                'char_length_ratio': total_chars_generated / max(total_chars_target, 1),
                'word_length_ratio': total_words_generated / max(total_words_target, 1)
            })
        
        self.model.train()  # Set back to training mode
        return eval_results
    
    def _calculate_character_accuracy(self, generated: str, target: str) -> float:
        """Calculate character-level accuracy using edit distance."""
        if not target:
            return 1.0 if not generated else 0.0
        
        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, generated, target)
        return matcher.ratio()
    
    def _calculate_word_accuracy(self, generated: str, target: str) -> float:
        """Calculate word-level accuracy."""
        generated_words = generated.split()
        target_words = target.split()
        
        if not target_words:
            return 1.0 if not generated_words else 0.0
        
        # Use difflib for sequence matching at word level
        matcher = difflib.SequenceMatcher(None, generated_words, target_words)
        return matcher.ratio()
        
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
