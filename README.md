# Parameter-Efficient Fine-Tuning Research Project

A research project investigating **selective LoRA placement strategies** for optimal efficiency-performance trade-offs in large language model fine-tuning.

## Research Question

> **"Can selective LoRA placement in specific transformer layers (attention-only, feed-forward-only, or strategic layer subsets) achieve comparable performance to full-model LoRA placement while using fewer trainable parameters?"**

## Project Structure

```
├── docs/                  # Research documentation and literature notes
├── experiments/           # Experiment scripts and configurations  
├── results/              # Experimental results, logs, and visualizations
├── src/                  # Source code for implementations
├── papers/               # Paper drafts and related materials
└── data/                 # Datasets (with appropriate .gitignore)
```

## Research Roadmap

Following the 10-step research methodology outlined in `goal.md`:

1. ✅ **Define the micro-question** - **COMPLETE** ([docs/research_question.md](docs/research_question.md))
2. ✅ **Skim closest prior work** - **COMPLETE** ([docs/literature_notes.md](docs/literature_notes.md))
3. 🔄 **Reproduce a baseline** - **IN PROGRESS** - Standard LoRA implementation
4. **Implement your twist** - LoRA placement variations (attention-only, FF-only, strategic subsets)
5. **Run controlled experiments** - Comparative analysis across placement strategies
6. **Analyze & visualize** - Performance vs efficiency trade-off analysis
7. **Draft the paper** - Writing and documentation
8. **Collect friendly reviews** - Peer feedback
9. **Pick venue & polish** - Submission preparation
10. **Submit, iterate, release** - Publication and code release

## Current Status

**Phase:** Step 3 - Baseline LoRA reproduction  
**Timeline:** Week 3-4 of 12-week schedule

### Research Progress
- ✅ **Literature Review**: 5 key papers analyzed, gaps identified
- ✅ **Research Question**: Finalized with clear experimental framework
- ✅ **Hypothesis**: Feed-forward-only placement will achieve >95% performance with ~50% fewer parameters

### Next Immediate Steps
1. Set up baseline LoRA implementation using Hugging Face PEFT
2. Reproduce standard LoRA fine-tuning on small dataset (Alpaca/GSM8K)  
3. Establish performance benchmarks and logging infrastructure
4. Implement placement variations for experimental comparison

---

*Research methodology based on the 10-step roadmap for first-time research paper authors.*
