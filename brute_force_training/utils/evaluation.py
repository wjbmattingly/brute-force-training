"""
Evaluation utilities for model assessment
"""

import torch
from typing import Dict, Any, List
from tqdm import tqdm


class ModelEvaluator:
    """Handles model evaluation before and after training."""
    
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        
    def evaluate_model(self, num_samples: int = None) -> Dict[str, Any]:
        """
        Evaluate model on the dataset.
        
        Args:
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        losses = []
        
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
