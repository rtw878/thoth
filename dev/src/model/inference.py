#!/usr/bin/env python3
"""
Model Inference for Historia Scribe

This module handles loading fine-tuned models and performing
actual transcription of historical document images.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))


class TranscriptionModel:
    """Class for handling model loading and inference."""
    
    def __init__(self, model_path: Path, device: str = "auto"):
        """
        Initialize transcription model.
        
        Args:
            model_path: Path to the fine-tuned model directory
            device: Device to run inference on (auto, cuda, cpu)
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.processor = None
        
        self.load_model()
    
    def _setup_device(self, device: str) -> torch.device:
        """Set up the device for inference."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {device}")
        return torch.device(device)
    
    def load_model(self) -> None:
        """Load the fine-tuned model and processor."""
        try:
            # Load processor
            self.processor = TrOCRProcessor.from_pretrained(self.model_path)
            
            # Load base model
            base_model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-large-handwritten"
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Successfully loaded model from {self.model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path: Path) -> torch.Tensor:
        """
        Preprocess image for model inference.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            pixel_values = self.processor(
                images=image, 
                return_tensors="pt"
            ).pixel_values
            
            return pixel_values.to(self.device)
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def transcribe_image(self, image_path: Path, max_length: int = 512) -> str:
        """
        Transcribe a single image.
        
        Args:
            image_path: Path to the image file
            max_length: Maximum sequence length for generation
            
        Returns:
            Transcribed text
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess image
            pixel_values = self.preprocess_image(image_path)
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode generated text
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            return generated_text
            
        except Exception as e:
            print(f"Error transcribing image {image_path}: {e}")
            raise
    
    def transcribe_batch(self, image_paths: List[Path], max_length: int = 512) -> List[str]:
        """
        Transcribe multiple images in batch.
        
        Args:
            image_paths: List of paths to image files
            max_length: Maximum sequence length for generation
            
        Returns:
            List of transcribed texts
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess all images
            pixel_values_list = []
            for image_path in image_paths:
                pixel_values = self.preprocess_image(image_path)
                pixel_values_list.append(pixel_values)
            
            # Stack tensors
            pixel_values_batch = torch.cat(pixel_values_list, dim=0)
            
            # Generate transcriptions
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values_batch,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode generated texts
            generated_texts = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            
            return generated_texts
            
        except Exception as e:
            print(f"Error transcribing batch: {e}")
            raise


class ModelManager:
    """Manager class for handling multiple models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models_dir = Path(config['data_paths']['models_dir'])
        self.models = {}
        self.load_available_models()
    
    def load_available_models(self) -> None:
        """Load available models from models directory."""
        # Load models from configuration
        available_models = self.config.get('app_settings', {}).get('available_models', [])
        
        for model_info in available_models:
            model_name = model_info['name']
            model_path = Path(model_info['path'])
            
            if model_path.exists():
                try:
                    self.models[model_name] = TranscriptionModel(model_path)
                    print(f"Loaded model: {model_name}")
                except Exception as e:
                    print(f"Failed to load model {model_name}: {e}")
            else:
                print(f"Model path not found: {model_path}")
    
    def get_model(self, model_name: str) -> Optional[TranscriptionModel]:
        """
        Get a loaded model by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            TranscriptionModel instance or None if not found
        """
        return self.models.get(model_name)
    
    def list_models(self) -> List[str]:
        """List all available model names."""
        return list(self.models.keys())
    
    def transcribe_with_model(self, model_name: str, image_path: Path) -> str:
        """
        Transcribe an image using a specific model.
        
        Args:
            model_name: Name of the model to use
            image_path: Path to the image file
            
        Returns:
            Transcribed text
        """
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        return model.transcribe_image(image_path)


def main():
    """Test the inference module."""
    # Example usage
    config = {
        'data_paths': {
            'models_dir': 'models'
        },
        'app_settings': {
            'available_models': [
                {
                    'name': '18th Century English Cursive',
                    'path': 'models/english_cursive_lora'
                },
                {
                    'name': 'Early Modern German Fraktur', 
                    'path': 'models/german_fraktur_lora'
                }
            ]
        }
    }
    
    manager = ModelManager(config)
    print(f"Available models: {manager.list_models()}")


if __name__ == "__main__":
    main()
