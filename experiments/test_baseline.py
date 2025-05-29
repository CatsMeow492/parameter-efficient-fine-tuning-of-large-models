#!/usr/bin/env python3
"""
Quick test script to validate baseline LoRA implementation.

This script runs a minimal version of the baseline to check if everything works
before running the full experiment.
"""

import os
import sys
import tempfile
import yaml

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from data_utils import load_alpaca_dataset, preprocess_dataset
from metrics import count_trainable_parameters

def test_data_loading():
    """Test data loading and preprocessing."""
    print("Testing data loading...")
    
    try:
        # Load a tiny subset
        datasets = load_alpaca_dataset(
            dataset_name="tatsu-lab/alpaca",
            train_split="train[:10]",
            eval_split="train[10:15]"
        )
        
        print(f"✅ Successfully loaded {len(datasets['train'])} train examples")
        print(f"✅ Successfully loaded {len(datasets['eval'])} eval examples")
        
        # Check data structure
        example = datasets['train'][0]
        print(f"✅ Example structure: {list(example.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False


def test_model_loading():
    """Test model and tokenizer loading with LoRA."""
    print("\nTesting model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Use a very small model for testing
        model_name = "microsoft/DialoGPT-small"
        
        print(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Create minimal LoRA config
        lora_config = LoraConfig(
            r=4,  # Very small rank for testing
            lora_alpha=8,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["c_attn", "c_proj", "c_fc"]
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Check parameters
        param_info = count_trainable_parameters(model)
        print(f"✅ Model loaded successfully")
        print(f"✅ Total parameters: {param_info['total_parameters']:,}")
        print(f"✅ Trainable parameters: {param_info['trainable_parameters']:,}")
        print(f"✅ Trainable percentage: {param_info['trainable_percentage']:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_tokenization():
    """Test data tokenization."""
    print("\nTesting tokenization...")
    
    try:
        from transformers import AutoTokenizer
        from data_utils import format_alpaca_prompt, tokenize_function
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test prompt formatting
        example = {
            "instruction": "What is the capital of France?",
            "input": "",
            "output": "The capital of France is Paris."
        }
        
        template = "### Instruction:\n{instruction}\n\n### Response:\n{output}"
        formatted = format_alpaca_prompt(example, template)
        print(f"✅ Formatted prompt: {formatted[:100]}...")
        
        # Test tokenization
        batch = {
            "instruction": [example["instruction"]],
            "input": [example["input"]],
            "output": [example["output"]]
        }
        
        tokenized = tokenize_function(batch, tokenizer, template, max_length=128)
        print(f"✅ Tokenized successfully, sequence length: {len(tokenized['input_ids'][0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), "configs", "base_config.yaml")
        
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['model', 'lora', 'training', 'data', 'experiment']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing config section: {section}")
                return False
        
        print(f"✅ Configuration loaded successfully")
        print(f"✅ Model: {config['model']['name']}")
        print(f"✅ LoRA rank: {config['lora']['r']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Running baseline implementation tests...\n")
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Data Loading", test_data_loading),
        ("Model Loading", test_model_loading),
        ("Tokenization", test_tokenization),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Ready to run baseline experiment.")
    else:
        print("❌ Some tests failed. Please fix issues before running experiments.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 