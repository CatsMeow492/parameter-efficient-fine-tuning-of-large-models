# Literature Review - Selective LoRA Placement

## 2. Related Work

### 2.1 Parameter-Efficient Fine-Tuning

The challenge of efficiently adapting large pre-trained models to downstream tasks has spawned numerous parameter-efficient fine-tuning (PEFT) approaches. Traditional fine-tuning updates all model parameters, requiring substantial computational resources and storage for each adapted model.

#### 2.1.1 Low-Rank Adaptation (LoRA)

**Hu et al. (2021)** introduced Low-Rank Adaptation (LoRA), which constrains weight updates to low-rank decompositions. For a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA represents the update as $\Delta W = BA$, where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d,k)$. This approach typically achieves comparable performance to full fine-tuning while training only 0.1-3% of the original parameters.

**Key Innovation**: LoRA enables efficient adaptation by exploiting the intrinsic low dimensionality of weight updates during fine-tuning, supported by the hypothesis that learned representations have low "intrinsic dimension."

#### 2.1.2 LoRA Variants and Extensions

Several works have extended the basic LoRA framework:

**AdaLoRA (Zhang et al., 2023)** adaptively allocates parameter budgets among weight matrices based on importance scores, enabling more efficient parameter distribution. However, this work focuses on *adaptive rank allocation* rather than *selective layer placement*.

**QLoRA (Dettmers et al., 2023)** combines LoRA with quantization, enabling fine-tuning of 65B parameter models on single GPUs. While achieving remarkable efficiency, it maintains LoRA application to all linear layers.

**LoRA+ (Hayou et al., 2024)** proposes different learning rates for LoRA matrices A and B, improving convergence speed and final performance.

### 2.2 Transformer Layer Analysis

Understanding the functional roles of different transformer components is crucial for informed placement strategies.

#### 2.2.1 Attention vs Feed-Forward Function

**Rogers et al. (2020)** provide a comprehensive survey of what BERT learns, revealing that different layers capture different linguistic phenomena. However, their analysis focuses on *representational* differences rather than *adaptation* requirements.

**Clark et al. (2019)** analyze attention patterns in BERT, showing that different attention heads specialize in different syntactic relations. This suggests attention layers may be particularly important for task-specific adaptation.

**Geva et al. (2021)** demonstrate that feed-forward layers in transformers function as key-value memories, storing factual knowledge. This functional specialization may impact their importance for different types of downstream tasks.

#### 2.2.2 Layer Importance Studies

**Michel et al. (2019)** investigate attention head importance through pruning studies, finding that many heads can be removed without significant performance loss. However, their work focuses on *pruning* rather than *selective adaptation*.

**Voita et al. (2019)** analyze attention head specialization in machine translation, identifying heads responsible for syntactic functions. Their findings suggest heterogeneous importance across attention components.

### 2.3 Efficient Fine-Tuning Strategies

#### 2.3.1 Selective Parameter Updates

**Diff pruning (Guo et al., 2020)** identifies task-relevant parameters by training a differentiable mask, allowing selective updates. However, this approach requires training additional mask parameters.

**BitFit (Ben-Zaken et al., 2022)** updates only bias terms, achieving surprising effectiveness with minimal parameters. This demonstrates that strategic parameter selection can yield disproportionate benefits.

**Prefix Tuning (Li & Liang, 2021)** prepends trainable continuous prompts to input sequences, avoiding modification of transformer weights entirely.

#### 2.3.2 Layer-wise Adaptation Strategies

**AdapterFusion (Pfeiffer et al., 2021)** combines multiple adapter modules using learned fusion weights, enabling modular task-specific adaptations.

**Compacter (Mahabadi et al., 2021)** uses Kronecker products to achieve even more parameter-efficient adaptations than standard adapters.

### 2.4 Research Gaps

Despite extensive work on parameter-efficient fine-tuning, several gaps remain:

#### 2.4.1 Systematic Placement Analysis

**Gap 1**: No systematic study of LoRA placement strategies across transformer layers. Most works apply LoRA uniformly to all linear layers without theoretical or empirical justification.

**Gap 2**: Limited understanding of layer-specific adaptation requirements. While layer analysis exists for understanding representations, adaptation needs may differ.

**Gap 3**: Lack of efficiency-performance trade-off analysis for selective placement strategies.

#### 2.4.2 Comparative Evaluation

**Gap 4**: Absence of controlled comparisons between attention-only and feed-forward-only LoRA placement strategies.

**Gap 5**: Limited analysis of computational overhead differences between placement strategies.

### 2.5 Motivation for Current Work

Our work addresses these gaps by providing:

1. **Systematic Ablation Study**: Controlled comparison of three placement strategies (full, attention-only, feed-forward-only)
2. **Efficiency Analysis**: Comprehensive evaluation of parameter reduction, training time, and memory usage
3. **Performance Trade-offs**: Quantitative analysis of efficiency-performance Pareto frontier
4. **Practical Guidelines**: Evidence-based recommendations for practitioners

### 2.6 Theoretical Foundation

#### 2.6.1 Low-Rank Hypothesis

The effectiveness of LoRA relies on the hypothesis that weight updates during fine-tuning have low intrinsic rank. **Aghajanyan et al. (2020)** provide empirical evidence for this in their study of intrinsic dimensionality in fine-tuning.

Our work extends this by investigating whether *different transformer components* have different intrinsic dimensionalities for adaptation, potentially explaining why selective placement strategies might be effective.

#### 2.6.2 Functional Specialization

Transformer layers exhibit functional specialization:
- **Lower layers**: Capture syntactic and positional information
- **Middle layers**: Build task-relevant representations  
- **Upper layers**: Perform task-specific reasoning

This specialization suggests that LoRA adaptation requirements may vary systematically across layer types, providing theoretical motivation for our empirical investigation.

### 2.7 Positioning of Current Work

Our research contributes to the parameter-efficient fine-tuning literature by:

**Empirical Contribution**: First systematic comparison of LoRA placement strategies with controlled experimental design.

**Practical Contribution**: Evidence-based guidelines for selecting placement strategies based on efficiency requirements.

**Theoretical Contribution**: Insights into layer-specific adaptation requirements and their implications for efficient fine-tuning.

**Methodological Contribution**: Comprehensive evaluation framework considering multiple efficiency and performance dimensions.

This positions our work as bridging the gap between theoretical understanding of transformer components and practical parameter-efficient fine-tuning strategies, providing actionable insights for the community.

---

*Note: This literature review synthesizes 15+ relevant papers to position our contribution within the broader parameter-efficient fine-tuning landscape.* 