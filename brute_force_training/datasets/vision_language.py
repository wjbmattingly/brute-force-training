"""
Vision-language dataset class for handling image-text pairs
"""

from torch.utils.data import Dataset
from typing import Dict, Any, Union
from ..utils.image_utils import ensure_pil_image


class VisionLanguageDataset(Dataset):
    """
    A custom Dataset class for handling HuggingFace datasets with image and text pairs.

    This class is designed to work with datasets that contain image-text pairs,
    specifically for use in vision-language models. It processes the data to create
    a format suitable for models like Qwen2-VL, structuring each item as a conversation
    with a user query (including an image) and an assistant response.

    Attributes:
        dataset: The HuggingFace dataset to be wrapped.
        image_column: The name of the column containing image data.
        text_column: The name of the column containing text data.
        user_text: The default user query text to pair with each image.
        max_image_size: Maximum size for image dimensions.
    """
    
    def __init__(
        self, 
        dataset: Any, 
        image_column: str, 
        text_column: str, 
        user_text: str = "Convert this image to text", 
        max_image_size: int = 500
    ):
        self.dataset = dataset
        self.image_column = image_column
        self.text_column = text_column
        self.user_text = user_text
        self.max_image_size = max_image_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        image = item[self.image_column]
        
        # Ensure the image is a PIL Image and meets size requirements
        image = ensure_pil_image(image, max_size=self.max_image_size)
        
        assistant_text = item[self.text_column]

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.user_text}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": str(assistant_text)}
                    ]
                }
            ]
        }
