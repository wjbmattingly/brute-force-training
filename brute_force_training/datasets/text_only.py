"""
Text-only dataset class for handling input-output text pairs
"""

from torch.utils.data import Dataset
from typing import Dict, Any


class TextOnlyDataset(Dataset):
    """
    A custom Dataset class for handling HuggingFace datasets with text input/output pairs.

    This class is designed to work with datasets that contain text input-output pairs,
    specifically for use in text-to-text models. It processes the data to create
    a format suitable for models like Qwen 3, structuring each item as a conversation
    with a user query and an assistant response.

    Attributes:
        dataset: The HuggingFace dataset to be wrapped.
        input_column: The name of the column containing input text data.
        output_column: The name of the column containing output text data.
    """
    
    def __init__(
        self, 
        dataset: Any, 
        input_column: str = "input", 
        output_column: str = "output"
    ):
        self.dataset = dataset
        self.input_column = input_column
        self.output_column = output_column

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        input_text = item[self.input_column]
        output_text = item[self.output_column]

        return {
            "messages": [
                {
                    "role": "user",
                    "content": str(input_text)
                },
                {
                    "role": "assistant",
                    "content": str(output_text)
                }
            ]
        }
