#!/usr/bin/env python3
"""
Baseline LoRA Implementation for Parameter-Efficient Fine-Tuning Study

This script implements standard LoRA fine-tuning to establish baseline performance
for comparing against selective layer placement strategies.

Usage:
    python baseline_lora.py --config configs/base_config.yaml
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from data_utils import (
    load_alpaca_dataset,
    preprocess_dataset,
    get_data_collator
)
from metrics import (
    LoRAMetricsCallback,
    count_trainable_parameters,
    log_experiment_summary,
    compute_metrics_for_eval
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model_and_tokenizer(config: dict):
    """
    Load and configure the model and tokenizer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_config = config['model']
    lora_config_dict = config['lora']
    
    logger.info(f"Loading model: {model_config['name']}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['name'],
        cache_dir=model_config.get('cache_dir'),
        trust_remote_code=True
    )
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        cache_dir=model_config.get('cache_dir'),
        torch_dtype=torch.float16 if config['training'].get('fp16', False) else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Create LoRA configuration
    lora_config = LoraConfig(
        r=lora_config_dict['r'],
        lora_alpha=lora_config_dict['lora_alpha'],
        lora_dropout=lora_config_dict['lora_dropout'],
        bias=lora_config_dict['bias'],
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_config_dict['target_modules']
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Print model info
    param_info = count_trainable_parameters(model)
    logger.info(f"Model loaded with LoRA configuration:")
    logger.info(f"  Total parameters: {param_info['total_parameters']:,}")
    logger.info(f"  Trainable parameters: {param_info['trainable_parameters']:,}")
    logger.info(f"  Trainable percentage: {param_info['trainable_percentage']:.2f}%")
    
    return model, tokenizer


def setup_datasets(config: dict, tokenizer):
    """
    Load and preprocess datasets.
    
    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer to use for preprocessing
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    data_config = config['data']
    
    logger.info("Loading and preprocessing datasets...")
    
    # Load datasets
    datasets = load_alpaca_dataset(
        dataset_name=data_config['dataset_name'],
        train_split=data_config['train_split'],
        eval_split=data_config['eval_split'],
        cache_dir=config['model'].get('cache_dir')
    )
    
    # Preprocess datasets
    train_dataset = preprocess_dataset(
        datasets['train'],
        tokenizer,
        data_config['instruction_template'],
        data_config['max_length']
    )
    
    eval_dataset = preprocess_dataset(
        datasets['eval'],
        tokenizer,
        data_config['instruction_template'],
        data_config['max_length']
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def setup_training_arguments(config: dict) -> TrainingArguments:
    """
    Create training arguments from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TrainingArguments object
    """
    training_config = config['training']
    experiment_config = config['experiment']
    
    # Ensure output directory exists
    output_dir = experiment_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        warmup_ratio=training_config['warmup_ratio'],
        weight_decay=training_config['weight_decay'],
        
        # Memory optimization
        fp16=training_config.get('fp16', False),
        gradient_checkpointing=training_config.get('gradient_checkpointing', False),
        dataloader_pin_memory=training_config.get('dataloader_pin_memory', False),
        
        # Evaluation and logging
        evaluation_strategy=training_config['evaluation_strategy'],
        eval_steps=training_config['eval_steps'],
        logging_steps=training_config['logging_steps'],
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],
        logging_dir=experiment_config['logging_dir'],
        
        # Other settings
        remove_unused_columns=False,
        report_to=None,  # We'll handle logging manually
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Baseline LoRA Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed for reproducibility
    set_seed(config['experiment']['seed'])
    
    logger.info(f"Starting baseline LoRA experiment: {config['experiment']['name']}")
    logger.info(f"Configuration: {args.config}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Setup datasets
    train_dataset, eval_dataset = setup_datasets(config, tokenizer)
    
    # Setup training arguments
    training_args = setup_training_arguments(config)
    
    # Setup data collator
    data_collator = get_data_collator(tokenizer)
    
    # Setup custom callback for metrics tracking
    metrics_callback = LoRAMetricsCallback()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_eval,
        callbacks=[metrics_callback],
    )
    
    # Initialize Weights & Biases if requested
    if args.wandb:
        import wandb
        wandb.init(
            project=config['experiment'].get('wandb_project', 'lora-experiments'),
            name=config['experiment'].get('wandb_run_name', config['experiment']['name']),
            config=config
        )
    
    try:
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Run final evaluation
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        
        # Get efficiency metrics from callback
        training_time = metrics_callback.efficiency_metrics.get_training_time()
        peak_memory = metrics_callback.efficiency_metrics.peak_memory_mb
        
        # Create experiment summary
        summary = log_experiment_summary(
            model=model,
            config=config,
            training_time=training_time,
            peak_memory_mb=peak_memory,
            final_metrics=eval_results
        )
        
        # Save summary
        summary_path = os.path.join(config['experiment']['output_dir'], 'experiment_summary.yaml')
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"Experiment completed successfully!")
        logger.info(f"Final evaluation loss: {eval_results.get('eval_loss', 'N/A')}")
        logger.info(f"Final perplexity: {eval_results.get('eval_perplexity', 'N/A')}")
        logger.info(f"Training time: {training_time / 60:.2f} minutes")
        logger.info(f"Peak memory: {peak_memory:.1f} MB")
        logger.info(f"Experiment summary saved to: {summary_path}")
        
        # Save the final model
        model.save_pretrained(os.path.join(config['experiment']['output_dir'], 'final_model'))
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main() 