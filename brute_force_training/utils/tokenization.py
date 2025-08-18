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
    assistant_ranges = []
    
    if tokenizer is not None:
        # Decode the full sequence to understand the structure
        full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        
        # Model-specific token patterns
        model_name = getattr(tokenizer, 'name_or_path', '').lower()
        
        if 'lfm2' in model_name or 'liquid' in model_name:
            # LFM2-VL specific tokens: <|im_start|> = [6], assistant = [64015], <|im_end|> = [7]
            assistant_start_pattern = [6, 64015]  # <|im_start|>assistant
            end_token = 7  # <|im_end|>
        else:
            # Qwen models: [151644, 77091] for <|im_start|>assistant, [151645] for <|im_end|>
            assistant_start_pattern = [151644, 77091]
            end_token = 151645
        
        # Find assistant content ranges using token pattern matching
        i = 0
        while i < len(token_ids) - len(assistant_start_pattern):
            # Check if we found the assistant start pattern
            if token_ids[i:i+len(assistant_start_pattern)] == assistant_start_pattern:
                # Start of assistant content is after the pattern + any newline tokens
                start_idx = i + len(assistant_start_pattern)
                
                # Skip any immediate newline/whitespace tokens after assistant marker
                while start_idx < len(token_ids) and tokenizer.decode([token_ids[start_idx]]).strip() == "":
                    start_idx += 1
                
                # Find the end token
                end_idx = len(token_ids) - 1  # Default to end of sequence
                for j in range(start_idx, len(token_ids)):
                    if token_ids[j] == end_token:
                        end_idx = j - 1  # End before the end marker
                        break
                
                if start_idx <= end_idx:
                    assistant_ranges.append((start_idx, end_idx))
                
                i = end_idx + 1  # Move past this assistant section
            else:
                i += 1
    
    else:
        # Fallback: assume Qwen model tokens if no tokenizer provided
        assistant_start_pattern = [151644, 77091]  # <|im_start|>assistant for Qwen
        end_token = 151645  # <|im_end|> for Qwen
        
        i = 0
        while i < len(token_ids) - 1:
            if token_ids[i:i+2] == assistant_start_pattern:
                start_idx = i + 2
                # Find end token
                end_idx = len(token_ids) - 1
                for j in range(start_idx, len(token_ids)):
                    if token_ids[j] == end_token:
                        end_idx = j - 1
                        break
                
                if start_idx <= end_idx:
                    assistant_ranges.append((start_idx, end_idx))
                
                i = end_idx + 1
            else:
                i += 1
    
    return assistant_ranges
