# Figure Captions for arXiv Submission

## Figure 1

**Figure 1: Selective LoRA Placement Analysis.** (a) Parameter efficiency trade-off showing evaluation loss vs trainable parameters. Feed-forward only achieves maximum parameter reduction (1.97M params) with acceptable performance degradation. (b) Pareto frontier analysis demonstrating three optimal strategies for different efficiency-performance requirements. (c) Training speedup factors showing computational benefits of selective placement. (d) Loss vs perplexity relationship revealing unexpected metric behavior, with attention-only achieving lowest perplexity despite higher loss than baseline.

## Figure 2

**Figure 2: Parameter Distribution Across Strategies.** Breakdown of trainable parameters by component type. Baseline LoRA uses all modules (6.29M total), attention-only targets c_attn and c_proj (4.33M), and feed-forward only targets c_fc (1.97M). This distribution analysis guides the selection of placement strategies based on available computational resources.

