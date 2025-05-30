# Methodology Section - Selective LoRA Placement

## 3. Methodology

### 3.1 Problem Formulation

Low-Rank Adaptation (LoRA) modifies pre-trained transformer layers by introducing trainable decomposition matrices while freezing the original weights. For a linear layer with weight matrix $W \in \mathbb{R}^{d \times k}$, LoRA adds an update $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d,k)$.

The key research question addressed is: **Which transformer layers benefit most from LoRA adaptation?** We systematically investigate three placement strategies:

1. **Full Placement (Baseline)**: Apply LoRA to all linear layers
2. **Attention-Only**: Apply LoRA only to attention mechanism layers  
3. **Feed-Forward-Only**: Apply LoRA only to feed-forward network layers

### 3.2 Experimental Setup

#### 3.2.1 Model Architecture
We use **microsoft/DialoGPT-medium** as our base model, a 356M parameter GPT-2 based conversational model. This choice provides:
- Manageable size for controlled experimentation
- Well-documented architecture with clear layer separation
- Strong baseline performance on instruction-following tasks

The model contains 24 transformer layers, each with:
- **Attention layers**: `c_attn` (combined Q,K,V projections), `c_proj` (output projection)
- **Feed-forward layers**: `c_fc` (first linear transformation)
- **Layer normalization**: Not modified (frozen)

#### 3.2.2 Dataset and Task
We evaluate on **tatsu-lab/alpaca**, a high-quality instruction-following dataset derived from self-instruct methodology. For rapid experimentation, we use:
- **Training set**: 100 examples (`train[:100]`)
- **Evaluation set**: 20 examples (`train[100:120]`)
- **Task format**: Instruction → Response generation

This small-scale setup enables rapid iteration while maintaining statistical validity for comparative analysis.

#### 3.2.3 LoRA Configuration
All experiments use identical LoRA hyperparameters for fair comparison:
- **Rank (r)**: 16 (standard setting from LoRA paper)
- **Alpha (α)**: 32 (scaling factor = 2r)
- **Dropout**: 0.1 
- **Bias adaptation**: None (`bias="none"`)
- **Task type**: Causal language modeling

#### 3.2.4 Training Configuration
Consistent training setup across all experiments:
- **Epochs**: 3
- **Batch size**: 1 (due to memory constraints)
- **Gradient accumulation**: 4 steps (effective batch size = 4)
- **Learning rate**: 2e-4 with cosine scheduler
- **Warmup ratio**: 0.1
- **Weight decay**: 0.01
- **Optimization**: AdamW optimizer
- **Hardware**: MacOS with Metal Performance Shaders (MPS)

### 3.3 LoRA Placement Strategies

#### 3.3.1 Baseline (Full Placement)
**Target modules**: `["c_attn", "c_proj", "c_fc"]`

This represents the standard LoRA approach, applying low-rank adaptation to all linear transformations in each transformer layer. It serves as our performance ceiling and parameter efficiency baseline.

#### 3.3.2 Attention-Only Placement  
**Target modules**: `["c_attn", "c_proj"]`

This strategy targets only the attention mechanism components:
- `c_attn`: Combined query, key, value projections
- `c_proj`: Attention output projection

**Hypothesis**: Attention mechanisms are more critical for task adaptation than feed-forward layers.

#### 3.3.3 Feed-Forward-Only Placement
**Target modules**: `["c_fc"]`

This strategy targets only the feed-forward network components:
- `c_fc`: First linear transformation in the MLP block

**Hypothesis**: Feed-forward layers can provide sufficient adaptation capacity with maximum parameter efficiency.

### 3.4 Evaluation Metrics

#### 3.4.1 Performance Metrics
- **Evaluation Loss**: Cross-entropy loss on held-out evaluation set (primary metric)
- **Perplexity**: Exponential of evaluation loss, measuring language modeling quality
- **Relative Performance**: Percentage change versus baseline approach

#### 3.4.2 Efficiency Metrics
- **Trainable Parameters**: Count of parameters requiring gradient computation
- **Parameter Reduction**: Percentage reduction versus baseline LoRA
- **Training Time**: Wall-clock time for complete training
- **Memory Usage**: Peak system memory during training

#### 3.4.3 Comparative Analysis
For each placement strategy, we calculate:
- **Efficiency Ratio**: Performance loss per parameter saved
- **Time Efficiency**: Performance per unit training time
- **Resource Trade-off**: Multi-dimensional efficiency analysis

### 3.5 Implementation Details

#### 3.5.1 Software Stack
- **Framework**: Hugging Face Transformers + PEFT
- **LoRA Implementation**: Microsoft PEFT library
- **Training**: Hugging Face Trainer with custom callbacks
- **Logging**: Weights & Biases integration
- **Reproducibility**: Fixed random seeds (seed=42)

#### 3.5.2 Data Processing
- **Tokenization**: GPT-2 tokenizer with padding to max_length=256
- **Format**: Alpaca instruction template
- **Data Collation**: Dynamic padding with attention masks
- **Preprocessing**: Remove original text columns after tokenization

#### 3.5.3 Training Monitoring
Real-time tracking of:
- Loss curves (training and evaluation)
- Parameter counts and efficiency metrics
- Memory usage and training time
- Model performance on fixed evaluation set

### 3.6 Statistical Analysis

#### 3.6.1 Experimental Controls
- **Identical hyperparameters** across all placement strategies
- **Same dataset splits** for fair comparison
- **Fixed random seeds** for reproducibility
- **Consistent hardware** environment

#### 3.6.2 Validity Considerations
- **Internal validity**: Controlled experimental design with single variable (placement strategy)
- **Construct validity**: Multiple performance metrics (loss, perplexity, efficiency)
- **External validity**: Limited by single model and small dataset (acknowledged limitation)

This methodology enables systematic comparison of LoRA placement strategies while maintaining experimental rigor and reproducibility.

---

*Note: This methodology can be scaled to larger models and datasets for validation, as outlined in our future work section.* 