"""
Base trainer class with common functionality
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from datasets import load_dataset


class BaseTrainer(ABC):
    """
    Base trainer class providing common training functionality.
    
    This abstract base class defines the interface and common functionality
    for all trainer implementations. Subclasses should implement the 
    abstract methods to provide model-specific behavior.
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device
        self.model = None
        self.tokenizer_or_processor = None
        
    @abstractmethod
    def load_model_and_processor(self) -> None:
        """Load the model and tokenizer/processor."""
        pass
        
    @abstractmethod
    def create_dataset(self, dataset: Any, **kwargs) -> Any:
        """Create a dataset instance from HuggingFace dataset."""
        pass
        
    @abstractmethod
    def create_collate_fn(self) -> callable:
        """Create the collate function for data loading."""
        pass
        
    def filter_dataset(self, dataset: Any, filter_fn: Optional[callable] = None) -> Any:
        """Filter dataset using provided function or default filtering."""
        if filter_fn is None:
            # Default filtering - can be overridden by subclasses
            return dataset
        return dataset.filter(filter_fn)
        
    def create_data_loaders(
        self,
        train_dataset: Any,
        val_dataset: Any,
        train_batch_size: int = 1,
        val_batch_size: int = 1
    ) -> tuple:
        """Create training and validation data loaders."""
        collate_fn = self.create_collate_fn()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            collate_fn=collate_fn,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader
        
    def validate(self, val_loader: DataLoader) -> Optional[float]:
        """
        Validate the model on the validation dataset.
        
        Args:
            val_loader: DataLoader for the validation dataset.
            
        Returns:
            The average validation loss, or None if validation set is empty.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model_and_processor() first.")
            
        self.model.eval()
        
        # Check if validation loader is empty
        if len(val_loader) == 0:
            print("Warning: Validation loader is empty. Skipping validation.")
            self.model.train()
            return None
        
        total_val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                inputs, labels = batch
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else None
        self.model.train()
        return avg_val_loss
        
    def save_model(self, step: int, is_final: bool = False) -> None:
        """Save the model and tokenizer/processor."""
        if self.model is None or self.tokenizer_or_processor is None:
            raise RuntimeError("Model or processor not loaded.")
            
        if is_final:
            save_dir = os.path.join(self.output_dir, "final")
        else:
            save_dir = os.path.join(self.output_dir, f"model_step_{step}")
            
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer_or_processor.save_pretrained(save_dir)
        
    def train_and_validate(
        self,
        dataset_name: str,
        num_accumulation_steps: int = 2,
        eval_steps: int = 500,
        max_steps: int = 10000,
        train_select_start: int = 0,
        train_select_end: int = 1000,
        val_select_start: int = 0,
        val_select_end: int = 1000,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        train_field: str = "train",
        val_field: str = "validation",
        learning_rate: float = 1e-5,
        **kwargs
    ) -> None:
        """
        Main training and validation loop.
        
        Args:
            dataset_name: Name of the HuggingFace dataset to use.
            num_accumulation_steps: Number of steps for gradient accumulation.
            eval_steps: Number of steps between evaluations.
            max_steps: Maximum number of training steps.
            train_select_start: Starting index for selecting training data.
            train_select_end: Ending index for selecting training data.
            val_select_start: Starting index for selecting validation data.
            val_select_end: Ending index for selecting validation data.
            train_batch_size: Batch size for training.
            val_batch_size: Batch size for validation.
            train_field: Field name for training data in the dataset.
            val_field: Field name for validation data in the dataset.
            learning_rate: Learning rate for the optimizer.
            **kwargs: Additional arguments passed to create_dataset.
        """
        # Load model and processor
        self.load_model_and_processor()
        
        # Load and prepare dataset
        dataset = load_dataset(dataset_name)
        
        # Apply filtering
        filtered_train_dataset = self.filter_dataset(dataset[train_field])
        filtered_val_dataset = self.filter_dataset(dataset[val_field])
        
        print(f"Number of training examples after filtering: {len(filtered_train_dataset)}")
        print(f"Number of validation examples after filtering: {len(filtered_val_dataset)}")
        
        # Calculate actual selection ranges
        actual_train_end = min(train_select_end, len(filtered_train_dataset))
        actual_val_end = min(val_select_end, len(filtered_val_dataset))
        
        print(f"Training selection: {train_select_start}:{actual_train_end}")
        print(f"Validation selection: {val_select_start}:{actual_val_end}")
        
        # Select subsets
        train_dataset = filtered_train_dataset.shuffle(seed=42).select(
            range(train_select_start, actual_train_end)
        )
        
        # Handle validation dataset selection
        if val_select_start >= len(filtered_val_dataset):
            print(f"Warning: Validation start index ({val_select_start}) >= dataset size ({len(filtered_val_dataset)})")
            print("Creating empty validation set - validation will be skipped")
            val_dataset = filtered_val_dataset.select([])  # Empty selection
        else:
            val_dataset = filtered_val_dataset.shuffle(seed=42).select(
                range(val_select_start, actual_val_end)
            )
        
        print(f"Final training examples selected: {len(train_dataset)}")
        print(f"Final validation examples selected: {len(val_dataset)}")
        
        # Create dataset instances
        train_dataset = self.create_dataset(train_dataset, **kwargs)
        val_dataset = self.create_dataset(val_dataset, **kwargs)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            train_dataset, val_dataset, train_batch_size, val_batch_size
        )
        
        # Setup training
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        global_step = 0
        progress_bar = tqdm(total=max_steps, desc="Training")
        
        # Training loop
        while global_step < max_steps:
            for batch in train_loader:
                global_step += 1
                inputs, labels = batch
                outputs = self.model(**inputs, labels=labels)
                
                loss = outputs.loss / num_accumulation_steps
                loss.backward()
                
                if global_step % num_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item() * num_accumulation_steps})

                # Perform evaluation and save model
                if global_step % eval_steps == 0 or global_step == max_steps:
                    avg_val_loss = self.validate(val_loader)
                    if avg_val_loss is not None:
                        print(f"\nStep {global_step}: Validation Loss = {avg_val_loss:.4f}")
                    else:
                        print(f"\nStep {global_step}: Validation skipped (empty validation set)")

                    self.save_model(global_step)
                    self.model.train()  # Set back to training mode

                if global_step >= max_steps:
                    self.save_model(global_step, is_final=True)
                    break

            if global_step >= max_steps:
                self.save_model(global_step, is_final=True)
                break

        progress_bar.close()
