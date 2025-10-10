#!/usr/bin/env python3
"""
Model Training Script for Historia Scribe

This script handles fine-tuning the TrOCR model using LoRA (Parameter-Efficient Fine-Tuning)
for historical handwriting recognition. Based on specifications in Sections 4.3 and 7.2.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import yaml
from datasets import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset_manager import DatasetManager, prepare_dataset_for_training


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset_from_config(config: Dict[str, Any], processor) -> tuple:
    """
    Load and prepare dataset based on configuration.
    
    Args:
        config: Configuration dictionary
        processor: TrOCR processor for preprocessing
        
    Returns:
        Tuple of (train_dataset, eval_dataset, test_dataset)
    """
    dataset_manager = DatasetManager(config)
    
    # Get dataset splits
    train_dataset, eval_dataset, test_dataset = dataset_manager.get_training_datasets(
        config.get('dataset_name', 'IAM')
    )
    
    # Prepare datasets for training
    train_dataset = prepare_dataset_for_training(
        train_dataset, 
        processor, 
        config["model_params"]["max_length"]
    )
    eval_dataset = prepare_dataset_for_training(
        eval_dataset, 
        processor, 
        config["model_params"]["max_length"]
    )
    
    return train_dataset, eval_dataset, test_dataset


def setup_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """
    Set up LoRA configuration for parameter-efficient fine-tuning.
    
    Args:
        config: Configuration dictionary with LoRA parameters
        
    Returns:
        LoRA configuration object
    """
    lora_params = config["model_params"]["lora"]
    
    return LoraConfig(
        r=lora_params["r"],
        lora_alpha=lora_params["lora_alpha"],
        lora_dropout=lora_params["lora_dropout"],
        target_modules=lora_params["target_modules"],
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )


def prepare_model_and_processor(config: Dict[str, Any]) -> tuple:
    """
    Load pre-trained model and processor, then apply LoRA.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, processor)
    """
    model_params = config["model_params"]
    
    # Load processor and model
    processor = TrOCRProcessor.from_pretrained(model_params["processor_name"])
    model = VisionEncoderDecoderModel.from_pretrained(model_params["base_model"])
    
    # Apply LoRA
    lora_config = setup_lora_config(config)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, processor


def compute_metrics(eval_pred, processor):
    """
    Compute evaluation metrics (CER, WER) for the model.
    
    Args:
        eval_pred: Evaluation predictions from trainer
        processor: TrOCR processor for decoding
        
    Returns:
        Dictionary of metrics
    """
    import evaluate
    
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels as we can't decode them
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    
    # Compute CER and WER
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    
    cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {
        "cer": cer,
        "wer": wer,
    }


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Historia Scribe model with LoRA"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yml"),
        help="Path to configuration file (default: configs/config.yml)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/trocr_finetuned"),
        help="Output directory for trained model (default: models/trocr_finetuned)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="IAM",
        help="Dataset to use for training (default: IAM)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['dataset_name'] = args.dataset
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare model and processor
    model, processor = prepare_model_and_processor(config)
    
    # Load dataset
    train_dataset, eval_dataset, test_dataset = load_dataset_from_config(config, processor)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Set up training arguments
    training_args = config["training_args"]
    seq2seq_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=training_args["learning_rate"],
        per_device_train_batch_size=training_args["per_device_train_batch_size"],
        per_device_eval_batch_size=training_args["per_device_eval_batch_size"],
        num_train_epochs=training_args["num_train_epochs"],
        warmup_steps=training_args["warmup_steps"],
        weight_decay=training_args["weight_decay"],
        logging_steps=training_args["logging_steps"],
        eval_steps=training_args["eval_steps"],
        save_steps=training_args["save_steps"],
        save_total_limit=training_args["save_total_limit"],
        evaluation_strategy="steps",
        predict_with_generate=True,
        generation_max_length=config["model_params"]["max_length"],
        generation_num_beams=config["model_params"]["beam_size"],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to=["tensorboard"],
    )
    
    # Create trainer with processor reference for metrics
    trainer = Seq2SeqTrainer(
        model=model,
        args=seq2seq_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, processor),
        tokenizer=processor.tokenizer,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    # Save training configuration
    with open(args.output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Training completed. Model saved to {args.output_dir}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
