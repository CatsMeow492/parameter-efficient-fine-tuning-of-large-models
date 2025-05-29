# Literature Review Notes

**Status:** ✅ Complete (Step 2 of 10)  
**Objective:** Read ~5 key papers/blogs on LoRA, QLoRA, adapters to identify gaps

## Key Papers Reviewed

### Core LoRA Papers

#### ✅ LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
**arXiv:** https://arxiv.org/abs/2106.09685  
- **Summary:** Introduces LoRA, a method that freezes all pre-trained weights and injects small low-rank trainable matrices into each Transformer layer. Reduces trainable parameters by up to ~10,000× in GPT-3 and cuts memory usage ~3×, yet matches or exceeds full fine-tuning performance. Achieves on-par results with no extra inference latency.
- **Notable Gaps:**
  - **Adaptive Rank Allocation:** LoRA uses fixed rank for all layers - optimal rank per layer/module remains open
  - **Adapter Placement:** Which layers or sub-modules are most critical to adapt is not fully explored
  - **Layer Criticality:** Understanding if only certain layers (attention vs feed-forward, higher vs lower) could be tuned
- **Relevance:** Demonstrates performance–efficiency sweet spot, sparks interest in where and how to insert adapters

#### ✅ QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)
**arXiv:** https://arxiv.org/abs/2305.14314  
- **Summary:** Builds on LoRA with quantization for extreme memory efficiency. Quantizes pre-trained model to 4-bit precision and fine-tunes via LoRA adapters, allowing 65B-parameter model finetuning on single 48GB GPU with no performance degradation. Finetuned 65B LLaMA to ~99.3% of ChatGPT performance in 24 hours on one GPU.
- **Key Techniques:** 4-bit NormalFloat data type, double quantization, paged optimizers
- **Notable Gaps:**
  - **Extreme Quantization:** Can we push to 3-bit or 2-bit without quality loss?
  - **Generality:** How well does this generalize to other architectures/domains?
  - **Adapter-Quantization Integration:** Quantizing adapters themselves, combining with other adapter types
- **Relevance:** Shows adapter design can be paired with compression for limited hardware

### Adapter Methods

#### ✅ Parameter-Efficient Transfer Learning for NLP (Houlsby et al., 2019)
**arXiv:** https://arxiv.org/abs/1902.00751  
- **Summary:** Introduced "adapters" for Transformers - small bottleneck neural layers inserted into each layer while original weights remain frozen. Achieved near SOTA with only ~3.6% extra parameters per task. BERT with adapters reached within 0.4% of full fine-tuning accuracy while training <4% parameters.
- **Notable Gaps:**
  - **Inference Overhead & Sparsity:** Could we skip adapters in less critical layers?
  - **Layer Importance:** Which layers are most important to adapt? (often focuses on higher layers)
  - **Scalability:** Better parameter sharing between adapters for related tasks
- **Relevance:** Originated modern adapter design, highlights placement and sizing as key levers

#### ✅ AdaLoRA: Adaptive Budget Allocation for PEFT (Zhang et al., 2023)
**arXiv:** https://arxiv.org/abs/2303.10512  
- **Summary:** Addresses LoRA's fixed-budget limitation by adaptively allocating parameter budget across weight matrices based on importance. Uses pseudo-SVD and prunes low singular values for less important layers. Achieves better performance than fixed-rank LoRA given same total parameters.
- **Notable Gaps:**
  - **Beyond Fixed Budgets:** Automated budget selection, dynamic growing/shrinking during training
  - **Importance Measures:** Alternative ways to identify which layers to prioritize
  - **Other Adapter Types:** Applying adaptive allocation beyond LoRA-style updates
- **Relevance:** Treats placement/design as learnable variables, reinforces that not all layers are equal

#### ✅ Prefix-Tuning: Optimizing Continuous Prompts for Generation (Li & Liang, 2021)
**arXiv:** https://arxiv.org/abs/2101.00190  
- **Summary:** Learns task-specific prefix vectors prepended at each layer's input while keeping model weights frozen. Tuning only 0.1% of parameters achieved performance comparable to full fine-tuning, often outperforming in low-data settings.
- **Notable Gaps:**
  - **Task Applicability:** Mixed success on non-generative tasks
  - **Prefix Design:** Optimal length and placement (all layers vs subset?)
  - **Combination with Adapters:** Hybrid prompt-based + weight-based approaches
- **Relevance:** Approaches efficiency from input space rather than weight space, expands design space

## Major Research Gaps Identified

### 🎯 **Optimal Adapter Placement & Allocation**
**Key Gap:** Determining where and how much to adapt within large models. Do we need adapters in every layer, or can we target specific layers (later layers, attention blocks only) for certain tasks?

### 🔄 **Combining and Composing Methods**
**Key Gap:** How to compose or select among PEFT techniques (adapters, LoRA, prefix-tuning). Hybrid approaches and unified frameworks for method selection.

### 📊 **Fine-Grained Efficiency–Performance Tradeoffs**
**Key Gap:** Quantifying relationship between adapter size and task performance. What's the "intrinsic task dimension" that sets floor on necessary parameters?

## Connections to Research Question

Based on this literature review, the most promising and feasible research direction appears to be **Optimal Adapter Placement** - specifically investigating whether selective layer adaptation can achieve comparable performance to full-model adaptation.

This gap is:
- ✅ **Novel:** Not directly answered in existing literature
- ✅ **Feasible:** Can be tested with limited compute resources
- ✅ **Measurable:** Clear metrics (accuracy, parameters, training time)
- ✅ **Focused:** Single variable (placement strategy)

## Recommended Research Question

> **"Can selective LoRA placement in specific transformer layers (attention-only, feed-forward-only, or strategic layer subsets) achieve comparable performance to full-model LoRA placement while using fewer trainable parameters?"**

This directly addresses the gap identified across multiple papers and provides a clear experimental framework.

## Research Tools
- **Connected Papers**: [Link to explore]
- **Elicit**: [Search results]
- **Google Scholar**: [Relevant searches]

## Identified Gaps
*To be filled as literature review progresses*

1. **Gap 1:** [Description]
2. **Gap 2:** [Description]
3. **Gap 3:** [Description]

## Next Steps
- Complete literature review
- Finalize research question based on identified gaps
- Move to Step 3: Baseline reproduction 