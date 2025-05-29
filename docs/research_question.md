# Research Micro-Question Definition

**Status:** ✅ Complete (Step 1 of 10)  
**Date:** Finalized based on literature review  

## Finalized Research Question

> **"Can selective LoRA placement in specific transformer layers (attention-only, feed-forward-only, or strategic layer subsets) achieve comparable performance to full-model LoRA placement while using fewer trainable parameters?"**

## Justification from Literature Review

This question directly addresses a **key gap identified across multiple papers**:

- **LoRA (Hu et al., 2021)**: "Which layers or sub-modules are most critical to adapt is not fully explored"
- **Houlsby et al. (2019)**: "Could we skip adapters in less critical layers?" and notes that "adaptation often focuses on higher layers"
- **AdaLoRA (Zhang et al., 2023)**: Demonstrates "not all layers are equal" for adaptation

## Experimental Design Framework

### Conditions to Test:
1. **Baseline**: Standard LoRA in all layers (attention + feed-forward)
2. **Attention-Only**: LoRA adapters only in attention blocks
3. **Feed-Forward-Only**: LoRA adapters only in feed-forward blocks  
4. **Strategic Subsets**: LoRA in specific layer ranges (e.g., upper 50%, lower 50%)

### Success Metrics:
- **Performance**: Task accuracy/perplexity (maintain >95% of baseline)
- **Efficiency**: Number of trainable parameters, VRAM usage, training time
- **Trade-off Analysis**: Performance per parameter ratio

### Experimental Controls:
- Same base model (LLaMA-2-7B or similar)
- Same dataset (Alpaca or GSM8K)
- Same total training budget
- Same random seeds for reproducibility

## Research Question Criteria ✅

- ✅ **Focused**: Single variable (LoRA placement strategy)
- ✅ **Testable**: Clear experimental conditions and metrics
- ✅ **Novel**: Gap identified in multiple key papers
- ✅ **Feasible**: Can be completed with Colab GPU in 12 weeks
- ✅ **Measurable**: Quantitative metrics for accuracy and efficiency

## Hypothesis

**Primary Hypothesis**: Feed-forward-only LoRA placement will achieve >95% of full LoRA performance while using ~50% fewer trainable parameters, as feed-forward layers contain more task-specific knowledge.

**Secondary Hypothesis**: Strategic layer placement (upper layers only) will outperform random layer selection, supporting the literature's observation that higher layers are more critical for adaptation.

## Next Steps

1. ✅ **Literature review complete** 
2. ✅ **Research question finalized**
3. **Move to Step 3**: Reproduce baseline LoRA implementation
4. **Implement experimental variations**
5. **Run controlled experiments**

## Expected Timeline
- Weeks 3-4: Baseline LoRA reproduction
- Weeks 5-8: Implement and test placement variations  
- Weeks 9-10: Analysis and visualization
- Weeks 11-12: Paper writing and submission prep

## Notes

- Keep the scope small for a first paper
- Focus on a single comparison/variable
- Ensure the question can be answered in ~12 weeks part-time 