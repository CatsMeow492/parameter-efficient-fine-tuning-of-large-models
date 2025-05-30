# Experimental Results Documentation

## Overview

This document tracks all experimental results for the **selective LoRA placement research project**. Each experiment tests different LoRA placement strategies to optimize the efficiency-performance trade-off.

## Research Hypothesis

> **"Feed-forward-only LoRA placement will achieve >95% performance with ~50% fewer parameters compared to standard LoRA placement"**

## Experimental Setup

- **Model**: microsoft/DialoGPT-medium (356M parameters)
- **Dataset**: tatsu-lab/alpaca (100 train, 20 eval samples)
- **LoRA Config**: r=16, alpha=32, dropout=0.1
- **Training**: 3 epochs, batch_size=1, lr=2e-4, cosine scheduler

## Results Summary

### Experiment 1: Baseline LoRA (Standard Placement)
**Date**: May 29, 2025  
**Status**: ✅ Complete  

**Configuration**:
- Target modules: `["c_attn", "c_proj", "c_fc"]` (all linear layers)
- Trainable parameters: 6,291,456 (1.74% of total)

**Results**:
- **Evaluation Loss**: 3.09
- **Perplexity**: 4,283
- **Training Time**: 90.5 seconds
- **Parameter Reduction**: 98.26%

### Experiment 2: Feed-Forward Only LoRA
**Date**: May 29, 2025  
**Status**: ✅ Complete  

**Configuration**:
- Target modules: `["c_fc"]` (feed-forward layers only)
- Trainable parameters: 1,966,080 (0.55% of total)

**Results**:
- **Evaluation Loss**: 4.25 (+37% vs baseline)
- **Perplexity**: 3,391 (-21% vs baseline)
- **Training Time**: 61.7 seconds (-32% vs baseline)
- **Parameter Reduction**: 99.45%

**Key Finding**: 
- ✅ **69% fewer trainable parameters** than baseline
- ✅ **32% faster training**
- ❌ **37% higher evaluation loss** (performance trade-off)
- ✅ **Better perplexity** (requires investigation)

### Experiment 3: Attention Only LoRA
**Date**: May 29, 2025  
**Status**: ✅ Complete  

**Configuration**:
- Target modules: `["c_attn", "c_proj"]` (attention layers only)
- Trainable parameters: 4,325,376 (1.20% of total)

**Results**:
- **Evaluation Loss**: 3.48 (+13% vs baseline)
- **Perplexity**: 2,272 (-47% vs baseline)
- **Training Time**: 77.2 seconds (-15% vs baseline)
- **Parameter Reduction**: 98.80%

**Key Finding**: 
- ✅ **31% fewer trainable parameters** than baseline
- ✅ **15% faster training**
- ✅ **Much better than feed-forward only** (3.48 vs 4.25 loss)
- ✅ **Best perplexity** of all three approaches

## Comparative Analysis

| Metric | Baseline | Feed-Forward Only | Attention Only | Best Result |
|--------|----------|-------------------|----------------|-------------|
| Eval Loss | **3.09** | 4.25 | 3.48 | 🏆 Baseline |
| Perplexity | 4,283 | 3,391 | **2,272** | 🏆 Attention-Only |
| Trainable Params | 6.3M | **1.97M** | 4.33M | 🏆 FF-Only |
| Training Time | 90.5s | **61.7s** | 77.2s | 🏆 FF-Only |
| Parameter Efficiency | 98.26% | **99.45%** | 98.80% | 🏆 FF-Only |

## Research Insights

### 1. Layer Importance Hierarchy
- **Attention layers are more critical** for performance than feed-forward layers
- **Feed-forward layers offer maximum parameter efficiency** with acceptable degradation
- **Full baseline remains best** for pure performance metrics

### 2. Performance vs Efficiency Trade-offs
- **Feed-forward only**: Maximum efficiency (99.45% reduction) with 37% performance hit
- **Attention only**: Balanced approach (98.80% reduction) with 13% performance hit
- **Baseline**: Best performance but least efficient (98.26% reduction)

### 3. Perplexity Contradiction Investigation Needed
- **Counterintuitive result**: Higher loss but better perplexity in some cases
- **Hypothesis**: Different mathematical formulations may explain this
- **Action**: Requires deeper investigation into metric calculation differences

### 4. Training Efficiency Insights
- **Feed-forward only**: 32% faster training (biggest speedup)
- **Attention only**: 15% faster training (modest speedup)
- Suggests **computational overhead differs significantly** between layer types

## Next Steps

1. ✅ **Complete attention-only experiment** for full ablation study
2. **🔄 CURRENT: Investigate perplexity contradiction** - why better despite higher loss?
3. **📊 NEXT: Create visualizations** comparing efficiency vs performance trade-offs
4. **📈 NEXT: Analyze layer importance** and computational overhead patterns
5. **🔬 FUTURE: Test hybrid approaches** (selective layer placement)
6. **📝 FUTURE: Begin methodology draft** for paper preparation
7. **🔍 FUTURE: Validate findings** on larger datasets and different models

## Methodology Notes

- All experiments use identical hyperparameters for fair comparison
- Small dataset used for rapid iteration (100 samples)
- Metrics tracked: loss, perplexity, training time, memory usage
- Results logged to W&B for detailed analysis

---

*Last Updated: May 29, 2025* 