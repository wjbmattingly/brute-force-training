"""
Trainer for LFM2-VL models
"""

import torch
from functools import partial
from transformers import AutoModelForImageTextToText, AutoProcessor
try:
    import Levenshtein
except ImportError:
    print("Warning: python-Levenshtein not installed. Error rate loss will not work.")
    Levenshtein = None

from .base import BaseTrainer
from ..datasets import VisionLanguageDataset
from ..utils.tokenization import find_assistant_content_sublist_indexes


class LFM2VLTrainer(BaseTrainer):
    """Trainer class for LFM2-VL models."""
    
    def __init__(
        self,
        model_name: str = "LiquidAI/LFM2-VL-450M",
        output_dir: str = "./lfm2_vl_finetuned",
        device: str = "cuda",
        use_error_rate_loss: bool = False,
        cer_weight: float = 0.3,
        wer_weight: float = 0.2,
        ce_weight: float = 0.5
    ):
        super().__init__(model_name, output_dir, device)
        self.use_error_rate_loss = use_error_rate_loss
        self.cer_weight = cer_weight
        self.wer_weight = wer_weight
        self.ce_weight = ce_weight
        
    def load_model_and_processor(self) -> None:
        """Load the LFM2-VL model and processor."""
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )
        self.tokenizer_or_processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            max_image_tokens=256,
            padding_side="right"
        )
        
    def create_dataset(
        self, 
        dataset,
        image_column: str,
        text_column: str,
        user_text: str = "Convert this image to text",
        max_image_size: int = 500,
        **kwargs
    ):
        """Create a VisionLanguageDataset instance."""
        return VisionLanguageDataset(
            dataset, image_column, text_column, user_text, max_image_size
        )
        
    def filter_dataset(self, dataset):
        """No default filtering applied - returns dataset unchanged."""
        return dataset
        
    def create_collate_fn(self) -> callable:
        """Create collate function for LFM2-VL."""
        return partial(self._collate_fn, processor=self.tokenizer_or_processor, device=self.device, use_error_rate_loss=self.use_error_rate_loss)
        
    def _collate_fn(self, batch, processor, device, use_error_rate_loss=False):
        """
        Collate function for processing batches of data for the LFM2-VL model.
        """
        messages = [item['messages'] for item in batch]
        
        # Use processor's apply_chat_template which handles both text and images
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=False
        )

        # Move the inputs to the specified device
        inputs = inputs.to(device)

        # Convert input IDs to a list of lists for easier processing
        input_ids_lists = inputs['input_ids'].tolist()
        labels_list = []
        target_texts = []  # For error rate loss calculation
        
        for i, ids_list in enumerate(input_ids_lists):
            # Initialize label IDs with -100 (ignored in loss calculation)
            label_ids = [-100] * len(ids_list)
            target_text = ""
            
            # Find the indexes of assistant content in the input IDs
            for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list, processor.tokenizer):
                # Set the label IDs for assistant content
                start_idx = begin_end_indexs[0]  # Use exact start of assistant content
                end_idx = begin_end_indexs[1]
                if start_idx <= end_idx:
                    label_ids[start_idx:end_idx+1] = ids_list[start_idx:end_idx+1]
                    
                    # Extract target text for error rate loss if needed
                    if use_error_rate_loss:
                        target_tokens = ids_list[start_idx:end_idx+1]
                        target_text = processor.tokenizer.decode(target_tokens, skip_special_tokens=True).strip()
            
            labels_list.append(label_ids)
            target_texts.append(target_text)

        # Convert the labels list to a tensor
        labels_ids = torch.tensor(labels_list, dtype=torch.int64)

        if use_error_rate_loss:
            return inputs, labels_ids, target_texts
        else:
            return inputs, labels_ids
    
    def calculate_error_rate_loss(self, logits, labels, target_texts):
        """Calculate character and word error rate losses."""
        if not self.use_error_rate_loss or Levenshtein is None:
            return 0.0, 0.0
            
        batch_size = logits.size(0)
        total_cer = 0.0
        total_wer = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            if i >= len(target_texts) or not target_texts[i]:
                continue
                
            target_text = target_texts[i].strip()
            if not target_text:
                continue
                
            # Get predicted tokens (greedy sampling)
            sample_logits = logits[i]
            predicted_ids = torch.argmax(sample_logits, dim=-1)
            
            # Extract only assistant content
            sample_labels = labels[i]
            assistant_mask = sample_labels != -100
            
            if not torch.any(assistant_mask):
                continue
                
            predicted_assistant_tokens = predicted_ids[assistant_mask]
            predicted_text = self.tokenizer_or_processor.tokenizer.decode(
                predicted_assistant_tokens, skip_special_tokens=True
            ).strip()
            
            if predicted_text:
                # Character Error Rate
                char_distance = Levenshtein.distance(predicted_text, target_text)
                max_char_len = max(len(predicted_text), len(target_text), 1)
                cer = char_distance / max_char_len
                
                # Word Error Rate
                pred_words = predicted_text.lower().split()
                target_words = target_text.lower().split()
                
                # Calculate word-level differences
                word_distance = 0
                max_words = max(len(pred_words), len(target_words))
                
                for j in range(max_words):
                    pred_word = pred_words[j] if j < len(pred_words) else ""
                    target_word = target_words[j] if j < len(target_words) else ""
                    if pred_word != target_word:
                        word_distance += 1
                
                wer = word_distance / max_words if max_words > 0 else 0.0
                
                total_cer += cer
                total_wer += wer
                valid_samples += 1
        
        if valid_samples == 0:
            return 0.0, 0.0
        
        avg_cer = total_cer / valid_samples
        avg_wer = total_wer / valid_samples
        
        return avg_cer, avg_wer
    
    def compute_hybrid_loss(self, outputs, labels, target_texts=None):
        """Compute hybrid loss combining cross-entropy with error rate losses."""
        ce_loss = outputs.loss
        
        if not self.use_error_rate_loss or target_texts is None:
            return ce_loss, {'ce_loss': ce_loss.item(), 'cer_loss': 0.0, 'wer_loss': 0.0, 'total_loss': ce_loss.item()}
        
        try:
            cer_loss, wer_loss = self.calculate_error_rate_loss(outputs.logits, labels, target_texts)
            
            # Convert to tensors if they aren't already
            if not isinstance(cer_loss, torch.Tensor):
                cer_loss = torch.tensor(cer_loss, device=ce_loss.device, dtype=ce_loss.dtype)
            if not isinstance(wer_loss, torch.Tensor):
                wer_loss = torch.tensor(wer_loss, device=ce_loss.device, dtype=ce_loss.dtype)
            
            # Combine losses
            total_loss = (self.ce_weight * ce_loss + 
                         self.cer_weight * cer_loss + 
                         self.wer_weight * wer_loss)
            
            loss_components = {
                'ce_loss': ce_loss.item(),
                'cer_loss': cer_loss.item() if hasattr(cer_loss, 'item') else cer_loss,
                'wer_loss': wer_loss.item() if hasattr(wer_loss, 'item') else wer_loss,
                'total_loss': total_loss.item()
            }
            
            return total_loss, loss_components
            
        except Exception as e:
            print(f"Error rate calculation failed, falling back to CE loss: {e}")
            return ce_loss, {'ce_loss': ce_loss.item(), 'cer_loss': 0.0, 'wer_loss': 0.0, 'total_loss': ce_loss.item()}
