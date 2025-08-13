"""
Trainer for LFM2-VL models
"""

import torch
from functools import partial
from transformers import AutoModelForImageTextToText, AutoProcessor

from .base import BaseTrainer
from ..datasets import VisionLanguageDataset
from ..utils.tokenization import find_assistant_content_sublist_indexes


class LFM2VLTrainer(BaseTrainer):
    """Trainer class for LFM2-VL models."""
    
    def __init__(
        self,
        model_name: str = "LiquidAI/LFM2-VL-450M",
        output_dir: str = "./lfm2_vl_finetuned",
        device: str = "cuda"
    ):
        super().__init__(model_name, output_dir, device)
        
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
        """Filter dataset for appropriate text length."""
        def filter_fn(example):
            # This will be set when create_dataset is called with text_column
            # For now, use a generic approach
            text_fields = [field for field in example.keys() if 'text' in field.lower()]
            if not text_fields:
                return True
            text_field = text_fields[0]
            text_content = example.get(text_field, "")
            return text_content is not None and 30 < len(str(text_content)) <= 2700
        
        return dataset.filter(filter_fn)
        
    def create_collate_fn(self) -> callable:
        """Create collate function for LFM2-VL."""
        return partial(self._collate_fn, processor=self.tokenizer_or_processor, device=self.device)
        
    def _collate_fn(self, batch, processor, device):
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
        for ids_list in input_ids_lists:
            # Initialize label IDs with -100 (ignored in loss calculation)
            label_ids = [-100] * len(ids_list)
            # Find the indexes of assistant content in the input IDs
            for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list, processor.tokenizer):
                # Set the label IDs for assistant content, skipping some initial tokens
                start_idx = begin_end_indexs[0] + 1  # Skip assistant role token
                end_idx = begin_end_indexs[1]
                if start_idx < end_idx:
                    label_ids[start_idx:end_idx+1] = ids_list[start_idx:end_idx+1]
            labels_list.append(label_ids)

        # Convert the labels list to a tensor
        labels_ids = torch.tensor(labels_list, dtype=torch.int64)

        return inputs, labels_ids
