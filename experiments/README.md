# Experiments Directory

This directory contains the implementation for Step 3 of the research roadmap: **Baseline LoRA Reproduction**.

## Quick Start

### 1. Install Dependencies
```bash
# From the project root
pip install -r requirements.txt
```

### 2. Test the Implementation
```bash
cd experiments
python test_baseline.py
```

### 3. Run Baseline Experiment
```bash
cd experiments
python baseline_lora.py --config configs/base_config.yaml
```

### 4. Optional: Enable Weights & Biases Logging
```bash
cd experiments
python baseline_lora.py --config configs/base_config.yaml --wandb
```

## File Structure

```
experiments/
├── baseline_lora.py          # Main baseline implementation
├── test_baseline.py          # Validation script
├── configs/
│   └── base_config.yaml      # Standard LoRA configuration
├── utils/
│   ├── data_utils.py         # Dataset loading and preprocessing
│   └── metrics.py            # Performance and efficiency tracking
└── notebooks/                # Analysis notebooks (to be added)
```

## Configuration

The `configs/base_config.yaml` file contains all experiment parameters:

- **Model**: Currently set to `microsoft/DialoGPT-medium` for initial testing
- **LoRA**: Standard settings (rank=16, alpha=32, dropout=0.1)
- **Training**: Small subset of Alpaca dataset for quick validation
- **Output**: Results saved to `./results/baseline_lora/`

## Outputs

After running the baseline experiment, you'll find:

- `./results/baseline_lora/experiment_summary.yaml` - Comprehensive metrics
- `./results/baseline_lora/final_model/` - Trained LoRA adapters
- `./results/baseline_lora/logs/` - Training logs

## Key Metrics Tracked

### Performance Metrics
- Validation loss and perplexity
- Training convergence curves
- Final model accuracy

### Efficiency Metrics  
- Number of trainable parameters
- Peak GPU memory usage (VRAM)
- Training time per epoch
- Total training time

## Next Steps

Once the baseline is validated:

1. **Scale up**: Switch to `meta-llama/Llama-2-7b-hf` for full experiments
2. **Implement variations**: Create configs for selective placement strategies
3. **Run comparisons**: Execute attention-only, feed-forward-only, and strategic subsets
4. **Analyze results**: Generate performance vs efficiency trade-off plots

## Troubleshooting

### Common Issues

**Memory Errors**: 
- Reduce batch size in config
- Enable gradient checkpointing
- Use smaller model for testing

**Dataset Loading Issues**:
- Check internet connection for HuggingFace Hub
- Verify dataset name in config
- Try smaller data splits first

**CUDA/GPU Issues**:
- Ensure PyTorch with CUDA support is installed
- Check GPU memory availability
- Fall back to CPU by setting `device_map=None`

### Getting Help

If you encounter issues:
1. Run `python test_baseline.py` to validate setup
2. Check logs in `./results/baseline_lora/logs/`
3. Verify configuration matches your hardware capabilities

## Research Context

This baseline implementation serves as the foundation for comparing selective LoRA placement strategies:

- **Research Question**: Can selective LoRA placement achieve comparable performance with fewer parameters?
- **Baseline**: Standard LoRA applied to all linear layers
- **Comparisons**: Attention-only, feed-forward-only, strategic layer subsets
- **Success Criteria**: Maintain >95% of baseline performance with reduced parameters 