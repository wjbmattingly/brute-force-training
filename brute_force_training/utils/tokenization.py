"""
Tokenization utilities for training
"""

from typing import List, Tuple, Union, Optional
from transformers import PreTrainedTokenizerBase


def find_assistant_content_sublist_indexes(
    token_ids: List[int], 
    tokenizer: Optional[PreTrainedTokenizerBase] = None
) -> List[Tuple[int, int]]:
    """
    Find the start and end indexes of assistant content sublists within a given list.

    This function searches for assistant role tokens that indicate the beginning and end
    of assistant content in a tokenized list. It identifies pairs of start and end indexes
    for each occurrence of assistant content.

    Args:
        token_ids: A list of token IDs to search through.
        tokenizer: The tokenizer to get special tokens from (optional, for fallback).

    Returns:
        A list of (start_index, end_index) pairs indicating the positions
        of assistant content sublists within the input list.
    """
    start_indexes = []
    end_indexes = []
    
    if tokenizer is not None:
        # Try to use tokenizer-aware approach for more robust matching
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        
        # Look for assistant markers in the decoded text
        assistant_start = "<|im_start|>assistant"
        assistant_end = "<|im_end|>"
        
        # Find positions in the original token list by re-encoding substrings
        for i in range(len(token_ids)):
            # Look for sequences that might indicate assistant content start
            substring = tokenizer.decode(token_ids[max(0, i-5):i+10], skip_special_tokens=False)
            if "assistant" in substring.lower():
                start_indexes.append(i)
                # Look for end marker
                for j in range(i + 1, len(token_ids)):
                    end_substring = tokenizer.decode(token_ids[j:min(len(token_ids), j+5)], skip_special_tokens=False)
                    if any(end_marker in end_substring for end_marker in ["<|im_end|>", "</s>", tokenizer.eos_token or ""]):
                        end_indexes.append(j)
                        break
                if len(end_indexes) < len(start_indexes):
                    # If no end found, use end of sequence
                    end_indexes.append(len(token_ids) - 1)
        
        return list(zip(start_indexes[:len(end_indexes)], end_indexes))
    
    else:
        # Fallback to hardcoded token IDs (specific to Qwen models)
        # Iterate through the list to find starting points
        for i in range(len(token_ids) - 1):
            # Check if the current and next element form the start sequence
            if token_ids[i] == 151644 and token_ids[i + 1] == 77091:
                start_indexes.append(i)
                # Now look for the first 151645 after the start
                for j in range(i + 2, len(token_ids)):
                    if token_ids[j] == 151645:
                        end_indexes.append(j)
                        break  # Move to the next start after finding the end

        return list(zip(start_indexes, end_indexes))
