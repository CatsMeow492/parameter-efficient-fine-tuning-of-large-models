# Selective LoRA Placement for Parameter-Efficient Fine-Tuning

## Paper Outline (Draft v1.0)

### Abstract
- **Problem**: Standard LoRA applies to all linear layers, but optimal placement strategy unknown
- **Method**: Systematic ablation study comparing attention-only, feed-forward-only, and full placement
- **Results**: Feed-forward-only achieves 68.8% parameter reduction with 37.6% performance trade-off
- **Impact**: Enables more efficient fine-tuning with explicit efficiency-performance trade-offs

### 1. Introduction
- Parameter-efficient fine-tuning importance for resource-constrained environments
- LoRA success but lack of placement strategy analysis
- Research question: Which transformer layers benefit most from LoRA adaptation?
- Contributions: Systematic ablation study + efficiency analysis + placement recommendations

### 2. Related Work
- Parameter-efficient fine-tuning (LoRA, AdaLoRA, QLoRA)
- Transformer layer analysis and importance studies
- Efficient fine-tuning strategies
- **Gap**: Systematic placement strategy analysis

### 3. Methodology

#### 3.1 Experimental Setup
- **Model**: microsoft/DialoGPT-medium (356M parameters)
- **Dataset**: tatsu-lab/alpaca (instruction following)
- **Hardware**: MacOS with MPS acceleration
- **Hyperparameters**: r=16, α=32, dropout=0.1, lr=2e-4

#### 3.2 LoRA Placement Strategies
1. **Baseline (Full)**: All linear layers [`c_attn`, `c_proj`, `c_fc`]
2. **Attention-Only**: Attention layers [`c_attn`, `c_proj`]  
3. **Feed-Forward-Only**: Feed-forward layers [`c_fc`]

#### 3.3 Evaluation Metrics
- Evaluation loss (primary performance metric)
- Perplexity (language modeling quality)
- Trainable parameters (efficiency metric)
- Training time (computational efficiency)
- Parameter reduction percentage

### 4. Results

#### 4.1 Performance Comparison

| Strategy | Eval Loss | Δ vs Baseline | Trainable Params | Param Reduction | Training Time |
|----------|-----------|---------------|------------------|-----------------|---------------|
| Baseline | 3.09 | - | 6.29M (1.74%) | 98.26% | 90.5s |
| Attention-Only | 3.48 | +12.7% | 4.33M (1.20%) | 98.80% | 77.2s |
| Feed-Forward-Only | 4.25 | +37.6% | 1.97M (0.55%) | 99.45% | 61.7s |

#### 4.2 Key Findings
1. **Layer Importance Hierarchy**: Attention layers more critical than feed-forward
2. **Efficiency Champion**: Feed-forward-only provides maximum parameter reduction
3. **Balanced Trade-off**: Attention-only offers middle ground (13% performance hit, 31% param reduction)

#### 4.3 Perplexity Analysis
- Unexpected relationship between loss and perplexity observed
- All approaches show significantly higher perplexity than exp(loss) expectation
- Suggests different calculation methodologies or tokenization effects

### 5. Analysis & Discussion

#### 5.1 Layer Function Analysis
- Why attention layers are more important for performance
- Feed-forward layers as parameter efficiency targets
- Computational overhead differences between layer types

#### 5.2 Practical Implications
- **Resource-Constrained**: Use feed-forward-only (99.45% reduction)
- **Balanced Approach**: Use attention-only (moderate trade-off)
- **Performance-Critical**: Use full baseline

#### 5.3 Limitations
- Single model architecture tested (DialoGPT-medium)
- Small dataset for rapid experimentation
- Need validation on larger scales

### 6. Future Work
- Test on larger models (7B+ parameters)
- Validate across different architectures (BERT, T5, LLaMA)
- Explore hybrid strategies (selective layer placement)
- Investigate perplexity calculation methodology
- Layer-wise importance analysis

### 7. Conclusion
- Systematic analysis reveals attention > feed-forward importance
- Feed-forward-only enables extreme efficiency with acceptable trade-offs
- Provides practitioners with clear placement strategy guidelines
- Opens avenue for hybrid and selective placement approaches

---

## Implementation Status

**Completed Sections:**
- ✅ Experimental design and methodology
- ✅ Complete ablation study results  
- ✅ Performance vs efficiency analysis
- ✅ Key findings and insights

**Next Steps:**
- 📝 Write methodology section (detailed)
- 📊 Create publication-quality figures
- 🔍 Literature review and related work
- 📝 Draft introduction and conclusion

**Target Venues:**
- EMNLP 2025 (Findings)
- ICML 2025 Workshop on Efficient NLP
- ACL 2025 (short paper)

*Last Updated: May 29, 2025* 