"""
Trainer for Qwen2.5-VL models
"""

import torch
from functools import partial
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from .base import BaseTrainer
from ..datasets import VisionLanguageDataset
from ..utils.tokenization import find_assistant_content_sublist_indexes

# Try to import Levenshtein for error rate calculations
try:
    import Levenshtein
except ImportError:
    Levenshtein = None
    print("⚠️ Levenshtein not available - error rate loss will be disabled")


class Qwen25VLTrainer(BaseTrainer):
    """Trainer class for Qwen2.5-VL models."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        output_dir: str = "./qwen25_vl_finetuned",
        device: str = "cuda",
        min_pixel: int = 256,
        max_pixel: int = 384,
        image_factor: int = 28,
        show_predictions: bool = False,
        show_diff: bool = False,
        use_error_rate_loss: bool = False,
        cer_weight: float = 0.3,
        wer_weight: float = 0.2,
        ce_weight: float = 0.5,
        debug_cer_generation: bool = False
    ):
        super().__init__(model_name, output_dir, device, show_predictions, show_diff)
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.image_factor = image_factor
        
        # Error rate loss configuration
        self.use_error_rate_loss = use_error_rate_loss
        self.cer_weight = cer_weight
        self.wer_weight = wer_weight
        self.ce_weight = ce_weight
        self.debug_cer_generation = debug_cer_generation
        
    def load_model_and_processor(self) -> None:
        """Load the Qwen2.5-VL model and processor."""
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        self.tokenizer_or_processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=self.min_pixel * self.image_factor * self.image_factor,
            max_pixels=self.max_pixel * self.image_factor * self.image_factor,
            padding_side="right"
        )
        
        # Ensure clean generation config to avoid warnings
        if hasattr(self.model, 'generation_config'):
            # Remove any problematic generation parameters
            if hasattr(self.model.generation_config, 'temperature'):
                delattr(self.model.generation_config, 'temperature')
            # Set conservative defaults
            self.model.generation_config.do_sample = False
            self.model.generation_config.pad_token_id = getattr(self.model.config, 'eos_token_id', getattr(self.model.config, 'pad_token_id', 0))
            self.model.generation_config.eos_token_id = getattr(self.model.config, 'eos_token_id', getattr(self.model.config, 'pad_token_id', 0))
        
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
        
    def filter_dataset(self, dataset, filter_fn=None):
        """Filter dataset like the original working script - remove text that's too short or too long."""
        if filter_fn is not None:
            return dataset.filter(filter_fn)
            
        # Default filtering like the original working script
        def filter_text_length(example):
            # Try common text column names
            text_value = None
            for col in ['text', 'transcription', 'content', 'caption']:
                if col in example:
                    text_value = example[col]
                    break
            
            return text_value is not None and 30 < len(str(text_value)) <= 2700
        
        return dataset.filter(filter_text_length)
        
    def create_collate_fn(self) -> callable:
        """Create collate function for Qwen2.5-VL."""
        return partial(self._collate_fn, processor=self.tokenizer_or_processor, device=self.device, use_error_rate_loss=self.use_error_rate_loss)
        
    def _collate_fn(self, batch, processor, device, use_error_rate_loss=False):
        """
        Collate function for processing batches of data for the Qwen2.5-VL model.
        
        Simplified approach based on the original working script.
        """
        messages = [item['messages'] for item in batch]
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
        
        # Use the vision processing utility to extract image and video inputs (like original)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Process the text, images, and videos using the processor (like original)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
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
            # Find the indexes of assistant content in the input IDs
            for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
                # Set the label IDs for assistant content, skipping the first two tokens
                label_ids[begin_end_indexs[0]+2:begin_end_indexs[1]+1] = ids_list[begin_end_indexs[0]+2:begin_end_indexs[1]+1]
            labels_list.append(label_ids)
            
            # Extract target text for error rate loss calculation
            if use_error_rate_loss:
                # Extract the assistant response from the original message
                target_text = ""
                for message in messages[i]:
                    if message['role'] == 'assistant':
                        target_text = message['content']
                        break
                target_texts.append(target_text)

        # Convert the labels list to a tensor
        labels_ids = torch.tensor(labels_list, dtype=torch.int64)

        # Return with or without target texts based on error rate loss usage
        if use_error_rate_loss:
            return inputs, labels_ids, target_texts
        else:
            return inputs, labels_ids
    
    def calculate_error_rate_loss(self, logits, labels, target_texts, inputs=None):
        """
        Calculate character and word error rate losses using proper model generation.
        
        This method uses the same generation approach as training predictions
        for consistency between debug output and actual model predictions.
        """
        if not self.use_error_rate_loss or Levenshtein is None:
            return 0.0, 0.0
            
        batch_size = logits.size(0)
        total_cer = 0.0
        total_wer = 0.0
        valid_samples = 0
        
        # Switch to eval mode temporarily for generation
        was_training = self.model.training
        self.model.eval()
        
        with torch.no_grad():
            for i in range(batch_size):
                if i >= len(target_texts) or not target_texts[i]:
                    continue
                    
                target_text = target_texts[i].strip()
                if not target_text:
                    continue
                    
                # Use proper generation if enabled and inputs available, otherwise fast logits method
                if self.debug_cer_generation and inputs is not None:
                    try:
                        # Extract single sample from batch
                        single_inputs = {k: v[i:i+1] for k, v in inputs.items() if k != 'labels'}
                        single_labels = labels[i:i+1]
                        
                        # Extract target length for conservative generation
                        target_tokens = single_labels[0][single_labels[0] != -100]
                        if len(target_tokens) == 0:
                            continue
                        
                        # FASTER generation settings - reduce overhead
                        generation_kwargs = {
                            'max_new_tokens': min(len(target_tokens) + 5, 30),  # Much smaller limit
                            'do_sample': False,  # Greedy sampling
                            'pad_token_id': getattr(self.model.config, 'eos_token_id', getattr(self.model.config, 'pad_token_id', 0)),
                            'eos_token_id': getattr(self.model.config, 'eos_token_id', getattr(self.model.config, 'pad_token_id', 0)),
                            'use_cache': True,  # Enable cache for speed
                        }
                        
                        final_kwargs = {**single_inputs, **generation_kwargs}
                        generated_ids = self.model.generate(**final_kwargs)
                        
                        # Extract only the generated part
                        input_length = single_inputs['input_ids'].size(1)
                        generated_tokens = generated_ids[0][input_length:]
                        predicted_text = self.tokenizer_or_processor.decode(
                            generated_tokens, skip_special_tokens=True
                        ).strip()
                        
                    except Exception as e:
                        # Fall back to fast logits method
                        sample_logits = logits[i]
                        predicted_ids = torch.argmax(sample_logits, dim=-1)
                        
                        # Extract only assistant content
                        sample_labels = labels[i]
                        assistant_mask = sample_labels != -100
                        
                        if not torch.any(assistant_mask):
                            continue
                            
                        predicted_assistant_tokens = predicted_ids[assistant_mask]
                        predicted_text = self.tokenizer_or_processor.decode(
                            predicted_assistant_tokens, skip_special_tokens=True
                        ).strip()
                        
                        # Note the fallback in debug output
                        if hasattr(self, 'current_step') and self.current_step % 100 == 0 and valid_samples == 0:
                            print(f"   Note: Generation failed, using fast logits fallback: {e}")
                            
                else:
                    # FAST: Use original logits-based method (default behavior)
                    sample_logits = logits[i]
                    predicted_ids = torch.argmax(sample_logits, dim=-1)
                    
                    # Extract only assistant content
                    sample_labels = labels[i]
                    assistant_mask = sample_labels != -100
                    
                    if not torch.any(assistant_mask):
                        continue
                        
                    predicted_assistant_tokens = predicted_ids[assistant_mask]
                    predicted_text = self.tokenizer_or_processor.decode(
                        predicted_assistant_tokens, skip_special_tokens=True
                    ).strip()
                
                if predicted_text:
                    # Character Error Rate
                    char_distance = Levenshtein.distance(predicted_text, target_text)
                    max_char_len = max(len(predicted_text), len(target_text), 1)
                    cer = char_distance / max_char_len
                    
                    # Word Error Rate using proper edit distance
                    pred_words = predicted_text.lower().split()
                    target_words = target_text.lower().split()
                    
                    # Use Levenshtein distance on word sequences  
                    word_distance = Levenshtein.distance(pred_words, target_words)
                    max_word_len = max(len(pred_words), len(target_words), 1)
                    wer = word_distance / max_word_len
                    
                    # Debug output every 100 steps to understand what's happening
                    if hasattr(self, 'current_step') and self.current_step % 100 == 0 and valid_samples == 0:
                        method_desc = "proper generation" if self.debug_cer_generation else "fast greedy logits"
                        print(f"\n🔍 CER/WER Debug (Step {self.current_step}):")
                        print(f"   Target: {repr(target_text[:50])}...")
                        print(f"   Predicted ({method_desc}): {repr(predicted_text[:50])}...")
                        print(f"   CER: {cer:.3f}, WER: {wer:.3f}")
                        if self.debug_cer_generation:
                            print(f"   NOTE: Using same generation method as predictions (slower but consistent)")
                        else:
                            print(f"   NOTE: Using fast greedy logits (faster but may differ from actual predictions)")
                    
                    total_cer += cer
                    total_wer += wer
                    valid_samples += 1
        
        # Restore training mode
        if was_training:
            self.model.train()
        
        if valid_samples == 0:
            return 0.0, 0.0
        
        avg_cer = total_cer / valid_samples
        avg_wer = total_wer / valid_samples
        
        return avg_cer, avg_wer
    
    def compute_hybrid_loss(self, outputs, labels, target_texts=None, inputs=None):
        """Compute hybrid loss combining cross-entropy with error rate losses."""
        ce_loss = outputs.loss
        
        if not self.use_error_rate_loss or target_texts is None:
            return ce_loss, {'ce_loss': ce_loss.item(), 'cer_loss': 0.0, 'wer_loss': 0.0, 'total_loss': ce_loss.item()}
        
        try:
            cer_loss, wer_loss = self.calculate_error_rate_loss(outputs.logits, labels, target_texts, inputs)
            
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
