"""
Trainer for Qwen3 text-only models
"""

import torch
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseTrainer
from ..datasets import TextOnlyDataset
from ..utils.tokenization import find_assistant_content_sublist_indexes


class Qwen3Trainer(BaseTrainer):
    """Trainer class for Qwen3 text-only models."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Thinking-2507",
        output_dir: str = "./qwen3_finetuned",
        device: str = "cuda",
        max_length: int = 2048,
        show_predictions: bool = False,
        show_diff: bool = False
    ):
        super().__init__(model_name, output_dir, device, show_predictions, show_diff)
        self.max_length = max_length
        
    def load_model_and_processor(self) -> None:
        """Load the Qwen3 model and tokenizer."""
        self.tokenizer_or_processor = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        
    def create_dataset(
        self, 
        dataset,
        input_column: str = "input",
        output_column: str = "output",
        **kwargs
    ):
        """Create a TextOnlyDataset instance."""
        return TextOnlyDataset(dataset, input_column, output_column)
        
    def filter_dataset(self, dataset):
        """Filter dataset for appropriate text length."""
        def filter_fn(example):
            # Look for common input/output fields
            input_fields = [field for field in example.keys() if 'input' in field.lower()]
            output_fields = [field for field in example.keys() if 'output' in field.lower()]
            
            if not input_fields or not output_fields:
                return True
                
            input_text = example.get(input_fields[0], "")
            output_text = example.get(output_fields[0], "")
            
            return (input_text is not None and output_text is not None and 
                    10 < len(str(input_text)) <= 5000 and 
                    10 < len(str(output_text)) <= 5000)
        
        return dataset.filter(filter_fn)
        
    def create_collate_fn(self) -> callable:
        """Create collate function for Qwen3."""
        return partial(self._collate_fn, tokenizer=self.tokenizer_or_processor, device=self.device)
        
    def _collate_fn(self, batch, tokenizer, device):
        """
        Collate function for processing batches of data for the Qwen3 model.
        """
        messages = [item['messages'] for item in batch]
        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
        
        # Tokenize the texts
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
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
            for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
                # Set the label IDs for assistant content, skipping the first two tokens
                label_ids[begin_end_indexs[0]+2:begin_end_indexs[1]+1] = ids_list[begin_end_indexs[0]+2:begin_end_indexs[1]+1]
            labels_list.append(label_ids)

        # Convert the labels list to a tensor
        labels_ids = torch.tensor(labels_list, dtype=torch.int64)

        return inputs, labels_ids
