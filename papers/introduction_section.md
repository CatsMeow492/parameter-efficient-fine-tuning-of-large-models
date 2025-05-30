# Introduction - Selective LoRA Placement

Parameter-efficient fine-tuning (PEFT) techniques have emerged as a pragmatic alternative to full model fine-tuning for adapting large language models (LLMs) to downstream tasks.  By updating only a small subset of parameters—often < 3 % of the original model—PEFT methods reduce computational cost, memory footprint, and storage requirements for each adapted model.

Among PEFT approaches, **Low-Rank Adaptation (LoRA)** (Hu et al., 2021) has gained widespread adoption due to its simplicity and effectiveness. LoRA freezes the pre-trained backbone and injects rank-restricted weight updates $\Delta W = BA$ into every linear layer, achieving near-parity with full fine-tuning while training orders-of-magnitude fewer parameters.

Despite LoRA's popularity, **one crucial design decision remains under-explored**: *Where* should LoRA be applied inside the transformer architecture? The original LoRA formulation places adapters in **all linear layers**—namely the attention projections (`W_Q`, `W_K`, `W_V`, `W_O`) and the feed-forward network (FFN) layers. This "full placement" strategy implicitly assumes that *all* components benefit equally from adaptation. 

However, prior work on transformer **layer importance**⁠—e.g., attention head pruning (Michel et al., 2019) and functional dissection (Voita et al., 2019; Geva et al., 2021)—suggests heterogeneous roles for attention and feed-forward components. Consequently, **uniform LoRA placement may be sub-optimal**: allocating scarce adaptation parameters to less-important components wastes capacity, while omitting critical components can degrade performance.

This paper addresses the following research question:

> **Which transformer components derive the greatest benefit from LoRA adaptation, and how can selective placement improve the efficiency-performance trade-off?**

To answer this question we conduct the **first systematic ablation study of LoRA placement strategies**. Using DialoGPT-medium (356 M parameters) fine-tuned on the Alpaca instruction-following dataset, we compare three strategies:

1. **Full Placement (Baseline)** – LoRA applied to *all* linear layers.
2. **Attention-Only** – LoRA applied exclusively to the self-attention projections (`c_attn`, `c_proj`).
3. **Feed-Forward-Only** – LoRA applied exclusively to the FFN expansion layer (`c_fc`).

All experiments share identical hyper-parameters, data splits, and hardware, ensuring a **controlled comparison** in which placement strategy is the only varying factor.

**Contributions**

1. **Empirical Study** – A controlled ablation revealing that attention layers are markedly more important than feed-forward layers for LoRA adaptation.
2. **Efficiency Analysis** – Quantitative evidence that feed-forward-only placement achieves *99.45 % parameter reduction* and *32 % training speed-up* with acceptable performance loss.
3. **Practical Guidelines** – A decision framework for selecting placement strategies based on resource budgets and performance requirements.
4. **Metric Insight** – Discovery of an unexpected divergence between evaluation loss and perplexity, motivating future metric investigations.

By disentangling the contribution of individual transformer components to LoRA's effectiveness, our work guides practitioners toward **principled adapter placement** and lays the foundation for more sophisticated, hybrid PEFT strategies. 