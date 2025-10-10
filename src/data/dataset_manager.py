#!/usr/bin/env python3
"""
Dataset Manager for Historia Scribe

This module handles loading and preparing datasets from various sources
including Hugging Face datasets and local processed data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from PIL import Image
import torch


def load_iam_dataset(data_dir: Path) -> Dataset:
    """
    Load the IAM Handwriting Database.
    
    Args:
        data_dir: Directory containing IAM dataset
        
    Returns:
        Hugging Face Dataset object
    """
    try:
        # Try to load from Hugging Face datasets
        dataset = load_dataset("Teklia/IAM-line", cache_dir=str(data_dir))
        return dataset
    except Exception as e:
        print(f"Failed to load IAM dataset from Hugging Face: {e}")
        print("Please download the dataset manually and place it in data/raw/")
        raise


def load_local_dataset(data_dir: Path, split: str = "train") -> Dataset:
    """
    Load a dataset from local processed data directory.
    
    Args:
        data_dir: Directory containing processed dataset
        split: Dataset split (train, validation, test)
        
    Returns:
        Hugging Face Dataset object
    """
    split_dir = data_dir / split
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")
    
    # Load images and corresponding text files
    image_paths = list(split_dir.glob("*.png"))
    
    if not image_paths:
        raise FileNotFoundError(f"No images found in {split_dir}")
    
    data = {"image_path": [], "text": []}
    
    for image_path in image_paths:
        # Look for corresponding text file
        text_path = image_path.with_suffix(".txt")
        
        if text_path.exists():
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            # If no text file, use filename as placeholder
            text = ""
        
        data["image_path"].append(str(image_path))
        data["text"].append(text)
    
    return Dataset.from_dict(data)


def create_dataset_splits(
    dataset: Dataset, 
    train_split: float = 0.8, 
    val_split: float = 0.1, 
    test_split: float = 0.1
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Input dataset
        train_split: Proportion for training
        val_split: Proportion for validation
        test_split: Proportion for testing
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, total_size))
    
    return train_dataset, val_dataset, test_dataset


def prepare_dataset_for_training(
    dataset: Dataset, 
    processor, 
    max_length: int = 512
) -> Dataset:
    """
    Prepare dataset for TrOCR training.
    
    Args:
        dataset: Input dataset with 'image_path' and 'text' columns
        processor: TrOCR processor
        max_length: Maximum sequence length
        
    Returns:
        Prepared dataset with 'pixel_values' and 'labels'
    """
    def process_example(example):
        # Load image
        image = Image.open(example['image_path']).convert('RGB')
        
        # Process image and text
        encoding = processor(
            images=image, 
            text=example['text'],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': encoding['pixel_values'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }
    
    return dataset.map(
        process_example, 
        remove_columns=dataset.column_names,
        batched=False
    )


class DatasetManager:
    """Manager class for handling multiple datasets."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config['data_paths']['processed_data_dir'])
        self.datasets = {}
    
    def load_dataset(self, dataset_name: str, split: str = "train") -> Dataset:
        """
        Load a dataset by name.
        
        Args:
            dataset_name: Name of the dataset
            split: Dataset split
            
        Returns:
            Loaded dataset
        """
        if dataset_name == "IAM":
            dataset = load_iam_dataset(self.data_dir)
            # Convert to standard format
            if split in dataset:
                return dataset[split]
            else:
                return dataset['train']
        else:
            # Assume local dataset
            return load_local_dataset(self.data_dir / dataset_name, split)
    
    def get_training_datasets(self, dataset_name: str) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Get train, validation, and test datasets for training.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (train, val, test) datasets
        """
        dataset = self.load_dataset(dataset_name)
        
        train_split = self.config['dataset_config']['train_split']
        val_split = self.config['dataset_config']['val_split']
        test_split = self.config['dataset_config']['test_split']
        
        return create_dataset_splits(dataset, train_split, val_split, test_split)
    
    def save_dataset_info(self, dataset: Dataset, output_path: Path) -> None:
        """
        Save dataset information to file.
        
        Args:
            dataset: Dataset to save info for
            output_path: Path to save info file
        """
        info = {
            'num_samples': len(dataset),
            'column_names': dataset.column_names,
            'features': str(dataset.features)
        }
        
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)


def main():
    """Test the dataset manager."""
    # Example usage
    config = {
        'data_paths': {
            'processed_data_dir': 'data/processed'
        },
        'dataset_config': {
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        }
    }
    
    manager = DatasetManager(config)
    
    try:
        # Try to load IAM dataset
        dataset = manager.load_dataset("IAM")
        print(f"Loaded IAM dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Could not load IAM dataset: {e}")


if __name__ == "__main__":
    main()
