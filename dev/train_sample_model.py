#!/usr/bin/env python3
"""
Sample Model Training for Historia Scribe

This script trains a small model with minimal data to test the training pipeline.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from datasets import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model


def create_sample_dataset():
    """Create a tiny sample dataset for testing."""
    # Create dummy data - in practice, you'd use real images and text
    sample_data = {
        "pixel_values": [torch.randn(3, 384, 384) for _ in range(10)],
        "labels": [[1, 2, 3, 4, 5] for _ in range(10)]  # Dummy token IDs
    }
    return Dataset.from_dict(sample_data)


def main():
    """Train a sample model."""
    print("Starting sample model training...")
    
    # Create output directory
    output_dir = Path("models/sample_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processor and model
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    # Apply LoRA for efficient training
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Create sample dataset
    dataset = create_sample_dataset()
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    
    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=5e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,  # Just one epoch for testing
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=5,
        eval_steps=5,
        save_steps=10,
        save_total_limit=1,
        eval_strategy="steps",  # Fixed parameter name
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=2,
        report_to=[],  # Disable external logging for testing
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        tokenizer=processor.tokenizer,
    )
    
    # Start training
    print("Training sample model (this will be very fast with dummy data)...")
    trainer.train()
    
    # Save the model
    trainer.save_model()
    processor.save_pretrained(output_dir)
    
    print(f"Sample model saved to {output_dir}")
    print("You can now select this model in the GUI!")


if __name__ == "__main__":
    main()
