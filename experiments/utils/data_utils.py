"""
Data utilities for loading and preprocessing datasets for LoRA experiments.
"""

import logging
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_alpaca_dataset(
    dataset_name: str = "tatsu-lab/alpaca",
    train_split: str = "train[:1000]",
    eval_split: str = "train[1000:1200]",
    cache_dir: Optional[str] = None
) -> Dict[str, Dataset]:
    """
    Load the Alpaca instruction-following dataset.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        train_split: Training split specification
        eval_split: Evaluation split specification
        cache_dir: Directory to cache the dataset
        
    Returns:
        Dictionary with 'train' and 'eval' Dataset objects
    """
    logger.info(f"Loading dataset: {dataset_name}")
    logger.info(f"Train split: {train_split}")
    logger.info(f"Eval split: {eval_split}")
    
    # Load the dataset
    dataset = load_dataset(
        dataset_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Create train/eval splits
    if "train" in dataset:
        train_dataset = dataset["train"].select(range(*_parse_split_range(train_split)))
        eval_dataset = dataset["train"].select(range(*_parse_split_range(eval_split)))
    else:
        # If no train split exists, use the first available split
        split_name = list(dataset.keys())[0]
        train_dataset = dataset[split_name].select(range(*_parse_split_range(train_split)))
        eval_dataset = dataset[split_name].select(range(*_parse_split_range(eval_split)))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    return {
        "train": train_dataset,
        "eval": eval_dataset
    }


def _parse_split_range(split_str: str) -> tuple:
    """Parse split string like 'train[:1000]' to extract range."""
    if "[" in split_str and "]" in split_str:
        range_part = split_str.split("[")[1].split("]")[0]
        if ":" in range_part:
            start, end = range_part.split(":")
            start = int(start) if start else 0
            end = int(end) if end else None
            return (start, end)
        else:
            return (0, int(range_part))
    return (0, 1000)  # Default fallback


def format_alpaca_prompt(example: Dict[str, str], instruction_template: str) -> str:
    """
    Format an Alpaca example into the instruction-response format.
    
    Args:
        example: Dictionary with 'instruction', 'input', and 'output' keys
        instruction_template: Template string for formatting
        
    Returns:
        Formatted prompt string
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    # Combine instruction and input if input exists
    if input_text:
        full_instruction = f"{instruction}\n\nInput: {input_text}"
    else:
        full_instruction = instruction
    
    # Format using the template
    formatted = instruction_template.format(
        instruction=full_instruction,
        output=output
    )
    
    return formatted


def tokenize_function(
    examples: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    instruction_template: str,
    max_length: int = 512
) -> Dict[str, List[List[int]]]:
    """
    Tokenize examples for instruction following.
    
    Args:
        examples: Batch of examples from the dataset
        tokenizer: Tokenizer to use
        instruction_template: Template for formatting prompts
        max_length: Maximum sequence length
        
    Returns:
        Tokenized batch
    """
    # Format all examples in the batch
    formatted_texts = []
    for i in range(len(examples["instruction"])):
        example = {
            "instruction": examples["instruction"][i],
            "input": examples.get("input", [""] * len(examples["instruction"]))[i],
            "output": examples["output"][i]
        }
        formatted_text = format_alpaca_prompt(example, instruction_template)
        formatted_texts.append(formatted_text)
    
    # Tokenize
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        padding=False,  # We'll pad dynamically during training
        max_length=max_length,
        return_tensors=None
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    instruction_template: str,
    max_length: int = 512,
    remove_columns: Optional[List[str]] = None
) -> Dataset:
    """
    Preprocess a dataset for LoRA fine-tuning.
    
    Args:
        dataset: Dataset to preprocess
        tokenizer: Tokenizer to use
        instruction_template: Template for formatting prompts
        max_length: Maximum sequence length
        remove_columns: Columns to remove after tokenization
        
    Returns:
        Preprocessed dataset
    """
    if remove_columns is None:
        remove_columns = ["instruction", "input", "output"]
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(
            examples, tokenizer, instruction_template, max_length
        ),
        batched=True,
        remove_columns=remove_columns,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


def get_data_collator(tokenizer: PreTrainedTokenizer):
    """Get the appropriate data collator for causal language modeling."""
    from transformers import DataCollatorForLanguageModeling
    
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8  # For better performance on GPUs
    ) 