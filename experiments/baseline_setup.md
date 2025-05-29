# Baseline LoRA Setup Plan

**Status:** Ready to begin (Step 3 of 10)  
**Objective:** Reproduce standard LoRA fine-tuning to establish baseline performance  

## Technical Stack

### Core Libraries
- **Model**: Hugging Face Transformers + PEFT
- **Base Model**: LLaMA-2-7B (or similar 7B model for Colab compatibility)
- **Dataset**: Alpaca instruction-tuning dataset (small subset for initial tests)
- **Logging**: Weights & Biases for experiment tracking
- **Compute**: Google Colab with GPU runtime

### Environment Setup
```bash
pip install transformers peft accelerate datasets wandb torch
```

## Experimental Configuration

### Standard LoRA Baseline
- **Target Modules**: All linear layers (attention + feed-forward)
- **LoRA Rank**: 16 (standard setting)
- **LoRA Alpha**: 32 (scaling factor)
- **Dropout**: 0.1
- **Training Steps**: 1000-2000 (enough for meaningful convergence)

### Dataset Configuration
- **Task**: Instruction following (Alpaca format)
- **Train/Eval Split**: 80/20
- **Sequence Length**: 512 tokens
- **Batch Size**: 4-8 (depending on GPU memory)

### Metrics to Track
1. **Performance Metrics**:
   - Validation perplexity
   - BLEU score on held-out instructions
   - Loss curves (training + validation)

2. **Efficiency Metrics**:
   - Number of trainable parameters
   - Peak GPU memory usage (VRAM)
   - Training time per epoch
   - Inference latency

## Implementation Plan

### Phase 1: Basic Setup (Week 3)
1. **Environment Configuration**
   - Set up Colab notebook with required libraries
   - Configure Weights & Biases logging
   - Load and preprocess Alpaca dataset

2. **Baseline Implementation**
   - Load LLaMA-2-7B with PEFT LoRA configuration
   - Implement standard training loop
   - Verify training runs and convergence

### Phase 2: Measurement & Validation (Week 4)
1. **Performance Validation**
   - Run full baseline training
   - Collect all efficiency metrics
   - Validate against published LoRA results

2. **Infrastructure for Variations**
   - Create modular code for different LoRA placements
   - Set up automated metric collection
   - Prepare experiment configuration system

## Success Criteria

### Baseline Performance Target
- **Convergence**: Training loss should decrease consistently
- **Validation**: Perplexity should improve compared to untuned model
- **Efficiency**: Should use significantly fewer parameters than full fine-tuning
- **Reproducibility**: Results should be consistent across runs

### Technical Validation
- [ ] Successfully fine-tune LLaMA-2-7B with standard LoRA
- [ ] Achieve reasonable convergence within compute budget
- [ ] Collect all planned efficiency metrics
- [ ] Establish baseline numbers for comparison

## Code Structure

```
experiments/
├── baseline_lora.py          # Main baseline implementation
├── configs/
│   ├── base_config.yaml      # Standard LoRA configuration
│   └── model_configs.yaml    # Model-specific settings
├── utils/
│   ├── data_utils.py         # Dataset loading and preprocessing
│   ├── metrics.py            # Evaluation metrics
│   └── logging_utils.py      # W&B integration
└── notebooks/
    └── baseline_analysis.ipynb # Results analysis and visualization
```

## Next Steps After Baseline

Once baseline is established:
1. **Implement placement variations** (attention-only, FF-only, strategic subsets)
2. **Run controlled comparisons** with identical training conditions
3. **Analyze trade-offs** between performance and efficiency
4. **Document findings** for paper draft

## Notes

- Start with smaller model/dataset if 7B model too large for Colab
- Monitor GPU memory usage closely
- Save checkpoints for comparison across placement strategies
- Document any unexpected findings or challenges 