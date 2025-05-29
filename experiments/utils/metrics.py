"""
Metrics utilities for tracking performance and efficiency in LoRA experiments.
"""

import time
import psutil
import torch
import logging
from typing import Dict, Any, Optional
from transformers import TrainerCallback
import numpy as np

logger = logging.getLogger(__name__)


class EfficiencyMetrics:
    """Track efficiency metrics during training and evaluation."""
    
    def __init__(self):
        self.start_time = None
        self.peak_memory_mb = 0
        self.training_start_time = None
        
    def start_timing(self):
        """Start timing for the current operation."""
        self.start_time = time.time()
        
    def start_training_timing(self):
        """Start timing for the entire training process."""
        self.training_start_time = time.time()
        
    def get_elapsed_time(self) -> float:
        """Get elapsed time since start_timing() was called."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
        
    def get_training_time(self) -> float:
        """Get total training time."""
        if self.training_start_time is None:
            return 0.0
        return time.time() - self.training_start_time
        
    def update_peak_memory(self):
        """Update peak memory usage."""
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
            self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
            
    def get_system_memory_usage(self) -> float:
        """Get current system memory usage in MB."""
        return psutil.virtual_memory().used / 1024 / 1024
        
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
        
    def reset(self):
        """Reset all metrics."""
        self.start_time = None
        self.peak_memory_mb = 0
        self.training_start_time = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def count_trainable_parameters(model) -> Dict[str, int]:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": (trainable_params / total_params) * 100 if total_params > 0 else 0,
        "parameter_reduction": ((total_params - trainable_params) / total_params) * 100 if total_params > 0 else 0
    }


def analyze_model_structure(model) -> Dict[str, Any]:
    """
    Analyze the structure of a model to understand parameter distribution.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with structural analysis
    """
    layer_info = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            layer_info[name] = {
                "weight_shape": list(module.weight.shape),
                "weight_params": module.weight.numel(),
                "trainable": module.weight.requires_grad
            }
            
            if hasattr(module, 'bias') and module.bias is not None:
                layer_info[name]["bias_params"] = module.bias.numel()
                layer_info[name]["bias_trainable"] = module.bias.requires_grad
    
    return layer_info


class LoRAMetricsCallback(TrainerCallback):
    """Custom callback to track LoRA-specific metrics during training."""
    
    def __init__(self):
        self.efficiency_metrics = EfficiencyMetrics()
        self.epoch_times = []
        self.step_times = []
        
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Called at the beginning of training."""
        self.efficiency_metrics.start_training_timing()
        logger.info("Training started - beginning efficiency tracking")
        
        # Log model parameter information
        param_info = count_trainable_parameters(model)
        logger.info(f"Model parameters: {param_info}")
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch."""
        self.efficiency_metrics.start_timing()
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        epoch_time = self.efficiency_metrics.get_elapsed_time()
        self.epoch_times.append(epoch_time)
        logger.info(f"Epoch {state.epoch} completed in {epoch_time:.2f} seconds")
        
    def on_step_end(self, args, state, control, **kwargs):
        """Called after each training step."""
        # Update memory tracking
        self.efficiency_metrics.update_peak_memory()
        
        # Log memory usage periodically
        if state.global_step % args.logging_steps == 0:
            gpu_memory = self.efficiency_metrics.get_gpu_memory_usage()
            system_memory = self.efficiency_metrics.get_system_memory_usage()
            logger.info(f"Step {state.global_step}: GPU Memory: {gpu_memory:.1f}MB, "
                       f"System Memory: {system_memory:.1f}MB")
                       
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        total_time = self.efficiency_metrics.get_training_time()
        peak_memory = self.efficiency_metrics.peak_memory_mb
        
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Peak GPU memory usage: {peak_memory:.1f}MB")
        logger.info(f"Average epoch time: {np.mean(self.epoch_times):.2f} seconds")


def compute_metrics_for_eval(eval_pred) -> Dict[str, float]:
    """
    Compute metrics for evaluation.
    
    Args:
        eval_pred: EvalPrediction object from Trainer
        
    Returns:
        Dictionary of computed metrics
    """
    predictions, labels = eval_pred
    
    # For causal LM, we typically look at perplexity
    # predictions shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    
    # Flatten predictions and labels
    predictions = predictions.reshape(-1, predictions.shape[-1])
    labels = labels.reshape(-1)
    
    # Remove ignored tokens (typically -100)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    # Compute cross-entropy loss
    import torch.nn.functional as F
    loss = F.cross_entropy(torch.tensor(predictions), torch.tensor(labels))
    perplexity = torch.exp(loss).item()
    
    return {
        "perplexity": perplexity,
        "eval_loss": loss.item()
    }


def log_experiment_summary(
    model,
    config: Dict[str, Any],
    training_time: float,
    peak_memory_mb: float,
    final_metrics: Dict[str, float]
) -> Dict[str, Any]:
    """
    Create a comprehensive experiment summary.
    
    Args:
        model: Trained model
        config: Experiment configuration
        training_time: Total training time in seconds
        peak_memory_mb: Peak memory usage in MB
        final_metrics: Final evaluation metrics
        
    Returns:
        Complete experiment summary
    """
    param_info = count_trainable_parameters(model)
    
    summary = {
        "experiment_name": config.get("experiment", {}).get("name", "unknown"),
        "model_name": config.get("model", {}).get("name", "unknown"),
        "lora_config": config.get("lora", {}),
        "training_config": config.get("training", {}),
        "parameter_efficiency": param_info,
        "performance_metrics": final_metrics,
        "efficiency_metrics": {
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "peak_memory_mb": peak_memory_mb,
            "peak_memory_gb": peak_memory_mb / 1024
        },
        "efficiency_ratios": {
            "parameters_per_second": param_info["trainable_parameters"] / training_time if training_time > 0 else 0,
            "memory_per_parameter": peak_memory_mb / param_info["trainable_parameters"] if param_info["trainable_parameters"] > 0 else 0
        }
    }
    
    return summary 