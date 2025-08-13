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
        self.evaluation_history = []  # Only evaluation checkpoints
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
        
    def log_evaluation_checkpoint(self, step: int, eval_results: Dict[str, Any], checkpoint_type: str = "checkpoint"):
        """Log comprehensive evaluation results for a checkpoint."""
        eval_entry = {
            'step': step,
            'checkpoint_type': checkpoint_type,  # 'pre_training', 'checkpoint', 'final'
            'timestamp': datetime.now().isoformat(),
            **eval_results  # Include all metrics from evaluation
        }
        self.evaluation_history.append(eval_entry)
        
    def set_training_config(self, config: Dict[str, Any]):
        """Store training configuration."""
        self.training_config = config.copy()
        
    def set_pre_training_eval(self, eval_results: Dict[str, Any]):
        """Store pre-training evaluation results."""
        self.pre_training_eval = eval_results.copy()
        # Also log as evaluation checkpoint
        self.log_evaluation_checkpoint(0, eval_results, "pre_training")
        
    def set_post_training_eval(self, eval_results: Dict[str, Any]):
        """Store post-training evaluation results."""
        self.post_training_eval = eval_results.copy()
        # Also log as evaluation checkpoint
        final_step = max([m['step'] for m in self.training_metrics]) if self.training_metrics else 0
        self.log_evaluation_checkpoint(final_step, eval_results, "final")
        
    def create_visualizations(self, save_dir: str):
        """Create comprehensive training visualizations."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Create main training curves
        if self.training_metrics and self.evaluation_history:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Training loss curve
            steps = [m['step'] for m in self.training_metrics]
            losses = [m['loss'] for m in self.training_metrics]
            ax1.plot(steps, losses, 'b-', alpha=0.6, linewidth=1, label='Training Loss')
            
            # Add evaluation checkpoints
            eval_steps = [e['step'] for e in self.evaluation_history if e.get('loss')]
            eval_losses = [e['loss'] for e in self.evaluation_history if e.get('loss')]
            if eval_steps:
                ax1.scatter(eval_steps, eval_losses, color='red', s=60, zorder=5, 
                           label='Evaluation Checkpoints', alpha=0.8)
            
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss with Evaluation Checkpoints')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Character accuracy over time (if available)
            char_acc_data = [e for e in self.evaluation_history if e.get('avg_char_accuracy')]
            if char_acc_data:
                char_steps = [e['step'] for e in char_acc_data]
                char_accs = [e['avg_char_accuracy'] * 100 for e in char_acc_data]  # Convert to percentage
                ax2.plot(char_steps, char_accs, 'g-o', linewidth=2, markersize=6, label='Character Accuracy')
                ax2.set_xlabel('Training Steps')
                ax2.set_ylabel('Character Accuracy (%)')
                ax2.set_title('Character-Level Accuracy Progress')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 100)
            else:
                ax2.text(0.5, 0.5, 'Character accuracy data not available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Character-Level Accuracy Progress')
            
            # 3. Word accuracy over time (if available)
            word_acc_data = [e for e in self.evaluation_history if e.get('avg_word_accuracy')]
            if word_acc_data:
                word_steps = [e['step'] for e in word_acc_data]
                word_accs = [e['avg_word_accuracy'] * 100 for e in word_acc_data]  # Convert to percentage
                ax3.plot(word_steps, word_accs, 'm-o', linewidth=2, markersize=6, label='Word Accuracy')
                ax3.set_xlabel('Training Steps')
                ax3.set_ylabel('Word Accuracy (%)')
                ax3.set_title('Word-Level Accuracy Progress')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(0, 100)
            else:
                ax3.text(0.5, 0.5, 'Word accuracy data not available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Word-Level Accuracy Progress')
            
            # 4. Perplexity over time
            perp_data = [e for e in self.evaluation_history if e.get('perplexity')]
            if perp_data:
                perp_steps = [e['step'] for e in perp_data]
                perplexities = [e['perplexity'] for e in perp_data]
                ax4.plot(perp_steps, perplexities, 'orange', marker='s', linewidth=2, markersize=6, label='Perplexity')
                ax4.set_xlabel('Training Steps')
                ax4.set_ylabel('Perplexity')
                ax4.set_title('Perplexity Progress')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_yscale('log')
            else:
                ax4.text(0.5, 0.5, 'Perplexity data not available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Perplexity Progress')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        # Create evaluation comparison chart
        if len(self.evaluation_history) >= 2:
            self._create_evaluation_comparison(save_dir)
            
    def _create_evaluation_comparison(self, save_dir: str):
        """Create comprehensive evaluation comparison across all checkpoints."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Get all evaluation points
        eval_data = self.evaluation_history
        
        if len(eval_data) >= 2:
            # 1. Loss comparison
            steps = [e['step'] for e in eval_data]
            losses = [e.get('loss', 0) for e in eval_data]
            
            colors = ['red' if e.get('checkpoint_type') == 'pre_training' else 
                     'green' if e.get('checkpoint_type') == 'final' else 'blue' 
                     for e in eval_data]
            
            bars1 = ax1.bar(range(len(steps)), losses, color=colors, alpha=0.7)
            ax1.set_xlabel('Checkpoint')
            ax1.set_ylabel('Loss')
            ax1.set_title('Loss Across All Checkpoints')
            ax1.set_xticks(range(len(steps)))
            ax1.set_xticklabels([f"Step {s}" if s > 0 else "Pre" for s in steps], rotation=45)
            
            # Add value labels
            for bar, value in zip(bars1, losses):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            # 2. Character accuracy comparison (if available)
            char_accs = [e.get('avg_char_accuracy', 0) * 100 for e in eval_data]
            if any(char_accs):
                bars2 = ax2.bar(range(len(steps)), char_accs, color=colors, alpha=0.7)
                ax2.set_xlabel('Checkpoint')
                ax2.set_ylabel('Character Accuracy (%)')
                ax2.set_title('Character Accuracy Across Checkpoints')
                ax2.set_xticks(range(len(steps)))
                ax2.set_xticklabels([f"Step {s}" if s > 0 else "Pre" for s in steps], rotation=45)
                ax2.set_ylim(0, 100)
                
                for bar, value in zip(bars2, char_accs):
                    if value > 0:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
            else:
                ax2.text(0.5, 0.5, 'Character accuracy not available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Character Accuracy Across Checkpoints')
            
            # 3. Word accuracy comparison (if available)
            word_accs = [e.get('avg_word_accuracy', 0) * 100 for e in eval_data]
            if any(word_accs):
                bars3 = ax3.bar(range(len(steps)), word_accs, color=colors, alpha=0.7)
                ax3.set_xlabel('Checkpoint')
                ax3.set_ylabel('Word Accuracy (%)')
                ax3.set_title('Word Accuracy Across Checkpoints')
                ax3.set_xticks(range(len(steps)))
                ax3.set_xticklabels([f"Step {s}" if s > 0 else "Pre" for s in steps], rotation=45)
                ax3.set_ylim(0, 100)
                
                for bar, value in zip(bars3, word_accs):
                    if value > 0:
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
            else:
                ax3.text(0.5, 0.5, 'Word accuracy not available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Word Accuracy Across Checkpoints')
            
            # 4. Perplexity comparison
            perplexities = [e.get('perplexity', 0) for e in eval_data]
            if any(perplexities):
                bars4 = ax4.bar(range(len(steps)), perplexities, color=colors, alpha=0.7)
                ax4.set_xlabel('Checkpoint')
                ax4.set_ylabel('Perplexity')
                ax4.set_title('Perplexity Across Checkpoints')
                ax4.set_xticks(range(len(steps)))
                ax4.set_xticklabels([f"Step {s}" if s > 0 else "Pre" for s in steps], rotation=45)
                ax4.set_yscale('log')
                
                for bar, value in zip(bars4, perplexities):
                    if value > 0:
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
            else:
                ax4.text(0.5, 0.5, 'Perplexity not available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Perplexity Across Checkpoints')
            
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
        
        # Get the latest evaluation results
        latest_eval = self.evaluation_history[-1] if self.evaluation_history else {}
        final_eval_loss = latest_eval.get('loss', 'N/A')
        
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
- **Evaluation Loss**: {final_eval_loss if isinstance(final_eval_loss, str) else f"{final_eval_loss:.6f}"}

"""

        # Add pre-training evaluation results
        if self.pre_training_eval:
            readme_content += f"""## Pre-Training Evaluation

**Initial Model Performance (before training):**
- **Loss**: {self.pre_training_eval.get('loss', 'N/A'):.6f if isinstance(self.pre_training_eval.get('loss'), (int, float)) else 'N/A'}
- **Perplexity**: {self.pre_training_eval.get('perplexity', 'N/A'):.2f if isinstance(self.pre_training_eval.get('perplexity'), (int, float)) else 'N/A'}
"""
            if self.pre_training_eval.get('avg_char_accuracy'):
                readme_content += f"- **Character Accuracy**: {self.pre_training_eval['avg_char_accuracy']*100:.1f}%\n"
            if self.pre_training_eval.get('avg_word_accuracy'):
                readme_content += f"- **Word Accuracy**: {self.pre_training_eval['avg_word_accuracy']*100:.1f}%\n"
            
            readme_content += "\n"

        # Add comprehensive evaluation history
        if self.evaluation_history:
            readme_content += f"""## Evaluation History

### All Checkpoint Evaluations
| Step | Checkpoint Type | Loss | Perplexity | Char Acc | Word Acc | Improvement vs Pre |
|------|----------------|------|------------|----------|----------|--------------------|
"""
            pre_loss = self.pre_training_eval.get('loss') if self.pre_training_eval else None
            
            for eval_data in self.evaluation_history:
                step = eval_data.get('step', 0)
                checkpoint_type = eval_data.get('checkpoint_type', 'checkpoint')
                loss = eval_data.get('loss', 'N/A')
                perplexity = eval_data.get('perplexity', 'N/A')
                char_acc = eval_data.get('avg_char_accuracy', 0) * 100 if eval_data.get('avg_char_accuracy') else 'N/A'
                word_acc = eval_data.get('avg_word_accuracy', 0) * 100 if eval_data.get('avg_word_accuracy') else 'N/A'
                
                # Calculate improvement vs pre-training
                if pre_loss and isinstance(loss, (int, float)) and isinstance(pre_loss, (int, float)):
                    improvement = ((pre_loss - loss) / pre_loss * 100)
                    improvement_str = f"{improvement:+.1f}%"
                else:
                    improvement_str = "N/A"
                
                # Format values
                loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
                perp_str = f"{perplexity:.2f}" if isinstance(perplexity, (int, float)) else str(perplexity)
                char_str = f"{char_acc:.1f}%" if isinstance(char_acc, (int, float)) else str(char_acc)
                word_str = f"{word_acc:.1f}%" if isinstance(word_acc, (int, float)) else str(word_acc)
                
                step_display = "Pre" if step == 0 and checkpoint_type == "pre_training" else f"{step:,}"
                
                readme_content += f"| {step_display} | {checkpoint_type} | {loss_str} | {perp_str} | {char_str} | {word_str} | {improvement_str} |\n"

        # Add training history section
        if self.training_metrics:
            readme_content += f"""
## Training Progress

### Recent Training Steps (Loss Only)
| Step | Training Loss | Timestamp |
|------|---------------|-----------|
"""
            # Show last 10 training steps
            recent_steps = self.training_metrics[-10:]
            
            for step_info in recent_steps:
                step = step_info['step']
                loss = step_info['loss']
                timestamp = step_info['timestamp'][:16]  # Just date and time
                readme_content += f"| {step:,} | {loss:.6f} | {timestamp} |\n"

        # Add visualizations section with embedded images
        readme_content += f"""
## Training Visualizations

### Training Progress and Evaluation Metrics
![Training Curves](training_curves.png)

*This chart shows the training loss progression, character accuracy, word accuracy, and perplexity over time. Red dots indicate evaluation checkpoints.*

### Evaluation Comparison Across All Checkpoints  
![Evaluation Comparison](evaluation_comparison.png)

*Comprehensive comparison of all evaluation metrics across training checkpoints. Red=Pre-training, Blue=Checkpoints, Green=Final.*

### Available Visualization Files:
- **`training_curves.png`** - 4-panel view: Training loss with eval points, Character accuracy, Word accuracy, Perplexity
- **`evaluation_comparison.png`** - 4-panel comparison: Loss, Character accuracy, Word accuracy, Perplexity across all checkpoints
"""

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
            'evaluation_history': self.evaluation_history,
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
        if self.evaluation_history:
            latest_eval = self.evaluation_history[-1]
            metadata['final_evaluation_loss'] = latest_eval.get('loss')
            if latest_eval.get('avg_char_accuracy'):
                metadata['final_char_accuracy'] = latest_eval['avg_char_accuracy']
            if latest_eval.get('avg_word_accuracy'):
                metadata['final_word_accuracy'] = latest_eval['avg_word_accuracy']
            
        return metadata
