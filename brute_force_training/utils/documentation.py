"""
Documentation and visualization utilities for model training
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class ModelDocumenter:
    """Handles model documentation, metadata, and visualizations."""
    
    def __init__(self, base_model_name: str, output_dir: str):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.training_metrics = []
        self.validation_metrics = []
        self.training_config = {}
        self.pre_training_eval = None
        self.post_training_eval = None
        
    def log_training_step(self, step: int, loss: float, learning_rate: float = None):
        """Log training metrics for a step."""
        self.training_metrics.append({
            'step': step,
            'loss': loss,
            'learning_rate': learning_rate,
            'timestamp': datetime.now().isoformat()
        })
        
    def log_validation_step(self, step: int, val_loss: float):
        """Log validation metrics for a step."""
        self.validation_metrics.append({
            'step': step,
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat()
        })
        
    def set_training_config(self, config: Dict[str, Any]):
        """Store training configuration."""
        self.training_config = config.copy()
        
    def set_pre_training_eval(self, eval_results: Dict[str, Any]):
        """Store pre-training evaluation results."""
        self.pre_training_eval = eval_results.copy()
        
    def set_post_training_eval(self, eval_results: Dict[str, Any]):
        """Store post-training evaluation results."""
        self.post_training_eval = eval_results.copy()
        
    def create_visualizations(self, save_dir: str):
        """Create training visualizations."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Create loss curve
        if self.training_metrics:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Training loss
            steps = [m['step'] for m in self.training_metrics]
            losses = [m['loss'] for m in self.training_metrics]
            
            ax1.plot(steps, losses, 'b-', alpha=0.7, linewidth=2, label='Training Loss')
            
            # Add validation loss if available
            if self.validation_metrics:
                val_steps = [m['step'] for m in self.validation_metrics]
                val_losses = [m['val_loss'] for m in self.validation_metrics]
                ax1.plot(val_steps, val_losses, 'r-', alpha=0.7, linewidth=2, label='Validation Loss')
            
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Learning rate schedule (if available)
            lr_data = [m for m in self.training_metrics if m.get('learning_rate') is not None]
            if lr_data:
                lr_steps = [m['step'] for m in lr_data]
                lr_values = [m['learning_rate'] for m in lr_data]
                
                ax2.plot(lr_steps, lr_values, 'g-', alpha=0.7, linewidth=2)
                ax2.set_xlabel('Training Steps')
                ax2.set_ylabel('Learning Rate')
                ax2.set_title('Learning Rate Schedule')
                ax2.grid(True, alpha=0.3)
                ax2.set_yscale('log')
            else:
                ax2.text(0.5, 0.5, 'No learning rate data available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Learning Rate Schedule')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        # Create evaluation comparison chart
        if self.pre_training_eval and self.post_training_eval:
            self._create_evaluation_comparison(save_dir)
            
    def _create_evaluation_comparison(self, save_dir: str):
        """Create before/after evaluation comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract common metrics
        pre_metrics = self.pre_training_eval
        post_metrics = self.post_training_eval
        
        # Find common keys
        common_keys = set(pre_metrics.keys()) & set(post_metrics.keys())
        
        if 'loss' in common_keys:
            categories = ['Pre-training', 'Post-training']
            losses = [pre_metrics['loss'], post_metrics['loss']]
            
            bars = ax.bar(categories, losses, color=['#ff7f0e', '#2ca02c'], alpha=0.8)
            ax.set_ylabel('Loss')
            ax.set_title('Model Performance: Before vs After Training')
            
            # Add value labels on bars
            for bar, value in zip(bars, losses):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'evaluation_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
    def generate_model_card(self, save_dir: str, is_final: bool = False) -> str:
        """Generate a comprehensive model card README."""
        model_name = os.path.basename(save_dir)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate training progress
        total_steps = max([m['step'] for m in self.training_metrics]) if self.training_metrics else 0
        final_loss = self.training_metrics[-1]['loss'] if self.training_metrics else "N/A"
        final_val_loss = self.validation_metrics[-1]['val_loss'] if self.validation_metrics else "N/A"
        
        readme_content = f"""# {model_name}

## Model Description

This model is a fine-tuned version of **{self.base_model_name}** using the brute-force-training package.

- **Base Model**: {self.base_model_name}
- **Training Status**: {"✅ Complete" if is_final else "🔄 In Progress"}
- **Generated**: {current_time}
- **Training Steps**: {total_steps:,}

## Training Details

### Dataset
- **Dataset**: {self.training_config.get('dataset_name', 'N/A')}
- **Training Examples**: {self.training_config.get('train_select_end', 0) - self.training_config.get('train_select_start', 0):,}
- **Validation Examples**: {self.training_config.get('val_select_end', 0) - self.training_config.get('val_select_start', 0):,}

### Training Configuration
- **Max Steps**: {self.training_config.get('max_steps', 'N/A'):,}
- **Batch Size**: {self.training_config.get('train_batch_size', 'N/A')}
- **Learning Rate**: {self.training_config.get('learning_rate', 'N/A')}
- **Gradient Accumulation**: {self.training_config.get('num_accumulation_steps', 'N/A')} steps
- **Evaluation Frequency**: Every {self.training_config.get('eval_steps', 'N/A'):,} steps

### Current Performance
- **Training Loss**: {final_loss if isinstance(final_loss, str) else f"{final_loss:.6f}"}
- **Validation Loss**: {final_val_loss if isinstance(final_val_loss, str) else f"{final_val_loss:.6f}"}

"""

        # Add evaluation comparison if available
        if self.pre_training_eval and self.post_training_eval:
            readme_content += f"""### Model Performance Comparison

| Metric | Pre-training | Post-training | Improvement |
|--------|--------------|---------------|-------------|
"""
            for key in set(self.pre_training_eval.keys()) & set(self.post_training_eval.keys()):
                pre_val = self.pre_training_eval[key]
                post_val = self.post_training_eval[key]
                if isinstance(pre_val, (int, float)) and isinstance(post_val, (int, float)):
                    improvement = ((pre_val - post_val) / pre_val * 100) if pre_val != 0 else 0
                    readme_content += f"| {key} | {pre_val:.6f} | {post_val:.6f} | {improvement:+.2f}% |\n"

        # Add training history section
        if self.training_metrics:
            readme_content += f"""
## Training History

### Recent Training Steps
| Step | Training Loss | Validation Loss | Timestamp |
|------|---------------|-----------------|-----------|
"""
            # Show last 10 training steps
            recent_steps = self.training_metrics[-10:]
            val_dict = {v['step']: v['val_loss'] for v in self.validation_metrics}
            
            for step_info in recent_steps:
                step = step_info['step']
                loss = step_info['loss']
                val_loss = val_dict.get(step, 'N/A')
                timestamp = step_info['timestamp'][:16]  # Just date and time
                val_loss_str = f"{val_loss:.6f}" if isinstance(val_loss, (int, float)) else val_loss
                readme_content += f"| {step:,} | {loss:.6f} | {val_loss_str} | {timestamp} |\n"

        # Add visualizations section
        readme_content += f"""
## Visualizations

The following training visualizations are available:

- **Training Curves**: `training_curves.png` - Shows training and validation loss over time
"""
        if self.pre_training_eval and self.post_training_eval:
            readme_content += "- **Evaluation Comparison**: `evaluation_comparison.png` - Before vs after training performance\n"

        # Add usage section
        readme_content += f"""
## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
# For vision-language models, use appropriate imports

model = AutoModelForCausalLM.from_pretrained("./{model_name}")
tokenizer = AutoTokenizer.from_pretrained("./{model_name}")

# Your inference code here
```

## Training Configuration

```json
{json.dumps(self.training_config, indent=2)}
```

## Model Card Metadata

- **Base Model**: {self.base_model_name}
- **Training Framework**: brute-force-training
- **Training Type**: {"Instruction Following" if any("instruct" in self.base_model_name.lower() for _ in [1]) else "Fine-tuning"}
- **License**: Inherited from base model
- **Language**: Inherited from base model

---

*This model card was automatically generated by brute-force-training on {current_time}*
"""

        return readme_content
        
    def save_documentation(self, save_dir: str, is_final: bool = False):
        """Save complete documentation for a checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate and save README
        readme_content = self.generate_model_card(save_dir, is_final)
        with open(os.path.join(save_dir, 'README.md'), 'w') as f:
            f.write(readme_content)
            
        # Save training metrics
        metrics_data = {
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'training_config': self.training_config,
            'pre_training_eval': self.pre_training_eval,
            'post_training_eval': self.post_training_eval,
            'base_model': self.base_model_name,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(save_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics_data, f, indent=2)
            
        # Create visualizations
        self.create_visualizations(save_dir)
        
    def create_model_card_metadata(self) -> Dict[str, Any]:
        """Create HuggingFace model card metadata."""
        metadata = {
            'base_model': self.base_model_name,
            'training_framework': 'brute-force-training',
            'training_date': datetime.now().isoformat(),
            'training_steps': max([m['step'] for m in self.training_metrics]) if self.training_metrics else 0,
            'dataset': self.training_config.get('dataset_name', 'custom'),
            'training_config': self.training_config
        }
        
        # Add performance metrics if available
        if self.training_metrics:
            metadata['final_training_loss'] = self.training_metrics[-1]['loss']
        if self.validation_metrics:
            metadata['final_validation_loss'] = self.validation_metrics[-1]['val_loss']
            
        return metadata
