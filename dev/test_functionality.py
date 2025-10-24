#!/usr/bin/env python3
"""
Test Script for Historia Scribe

This script tests the core functionality of the Historia Scribe project
including configuration loading, dataset management, and model inference.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

import yaml
from data.dataset_manager import DatasetManager
from model.inference import ModelManager


def test_configuration():
    """Test configuration loading."""
    print("Testing configuration loading...")
    
    config_path = Path("configs/config.yml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("[OK] Configuration loaded successfully")
        return config
    else:
        print("[ERROR] Configuration file not found")
        return None


def test_dataset_manager(config):
    """Test dataset manager functionality."""
    print("\nTesting dataset manager...")
    
    try:
        manager = DatasetManager(config)
        print("[OK] DatasetManager initialized successfully")
        
        # Test loading IAM dataset (this will fail if dataset not downloaded)
        try:
            dataset = manager.load_dataset("IAM")
            print(f"[OK] IAM dataset loaded with {len(dataset)} samples")
        except Exception as e:
            print(f"[WARNING] IAM dataset not available: {e}")
            
    except Exception as e:
        print(f"[ERROR] DatasetManager failed: {e}")


def test_model_manager(config):
    """Test model manager functionality."""
    print("\nTesting model manager...")
    
    try:
        manager = ModelManager(config)
        print("[OK] ModelManager initialized successfully")
        
        models = manager.list_models()
        if models:
            print(f"[OK] Found {len(models)} models: {', '.join(models)}")
        else:
            print("[WARNING] No models found (this is expected before training)")
            
    except Exception as e:
        print(f"[ERROR] ModelManager failed: {e}")


def test_preprocessing():
    """Test preprocessing functionality."""
    print("\nTesting preprocessing...")
    
    try:
        from preprocess.image_pipeline import binarize_image, load_image
        import cv2
        import numpy as np
        
        # Create a dummy image for testing
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test binarization
        binary = binarize_image(dummy_image, method="otsu")
        print("[OK] Image binarization working")
        
    except Exception as e:
        print(f"[ERROR] Preprocessing test failed: {e}")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Historia Scribe - Functionality Test")
    print("=" * 50)
    
    # Test configuration
    config = test_configuration()
    
    if config:
        # Test dataset manager
        test_dataset_manager(config)
        
        # Test model manager
        test_model_manager(config)
        
        # Test preprocessing
        test_preprocessing()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Download datasets using: python src/data/download_data.py IAM")
    print("2. Train a model using: python src/model/train_model.py --dataset IAM")
    print("3. Run the GUI using: python src/app/main.py")


if __name__ == "__main__":
    main()
