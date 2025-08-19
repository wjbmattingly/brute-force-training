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
        show_diff: bool = False
    ):
        super().__init__(model_name, output_dir, device, show_predictions, show_diff)
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.image_factor = image_factor
        
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
        """Create collate function for Qwen2.5-VL."""
        return partial(self._collate_fn, processor=self.tokenizer_or_processor, device=self.device)
        
    def _collate_fn(self, batch, processor, device):
        """
        Collate function for processing batches of data for the Qwen2.5-VL model.
        
        This uses the new qwen_vl_utils.process_vision_info for proper image/video processing.
        """
        messages = [item['messages'] for item in batch]
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
        
        # Use the new vision processing utility to extract image and video inputs
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Process the text, images, and videos using the processor
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
