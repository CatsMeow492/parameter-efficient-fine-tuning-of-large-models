#!/usr/bin/env python3
"""
Create publication-ready figures for arXiv submission.
Enhanced formatting with clear captions and professional styling.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_experiment_results():
    """Load results from all three experiments."""
    results = {}
    experiments = ['baseline_lora', 'ff_only_lora', 'attention_only_lora']
    
    for exp in experiments:
        result_path = Path(f'results/{exp}/experiment_summary.yaml')
        if result_path.exists():
            with open(result_path, 'r') as f:
                results[exp] = yaml.safe_load(f)
        else:
            print(f"Warning: {result_path} not found")
    
    return results

def create_arxiv_figure(results):
    """Create publication-ready figure for arXiv submission."""
    print("Creating publication-ready figure for arXiv...")
    
    # Extract data
    experiments = []
    eval_losses = []
    perplexities = []
    trainable_params = []
    training_times = []
    param_reductions = []
    
    # Order: Baseline, Attention-Only, Feed-Forward Only for logical presentation
    order = ['baseline_lora', 'attention_only_lora', 'ff_only_lora']
    labels = ['Baseline (Full)', 'Attention-Only', 'Feed-Forward Only']
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Professional color scheme
    
    for i, exp_name in enumerate(order):
        exp_data = results[exp_name]
        experiments.append(labels[i])
        eval_losses.append(exp_data['performance_metrics']['eval_loss'])
        perplexities.append(exp_data['performance_metrics']['eval_perplexity'])
        trainable_params.append(exp_data['parameter_efficiency']['trainable_parameters'] / 1e6)
        training_times.append(exp_data['efficiency_metrics']['training_time_seconds'])
        param_reductions.append(exp_data['parameter_efficiency']['parameter_reduction'])
    
    # Create figure with 2x2 layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Selective LoRA Placement: Efficiency vs Performance Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: Parameter Efficiency vs Evaluation Loss
    ax1.scatter(trainable_params, eval_losses, c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=1)
    for i, exp in enumerate(experiments):
        ax1.annotate(exp, (trainable_params[i], eval_losses[i]), 
                    xytext=(10, 10), textcoords='offset points', 
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax1.set_xlabel('Trainable Parameters (Millions)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Evaluation Loss', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Parameter Efficiency Trade-off', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 7)
    
    # Plot 2: Parameter Reduction vs Performance Loss
    performance_loss = [(loss - eval_losses[0])/eval_losses[0] * 100 for loss in eval_losses]
    ax2.scatter(param_reductions, performance_loss, c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=1)
    for i, exp in enumerate(experiments):
        ax2.annotate(exp, (param_reductions[i], performance_loss[i]), 
                    xytext=(10, 10), textcoords='offset points', 
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax2.set_xlabel('Parameter Reduction (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Performance Loss (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Efficiency-Performance Pareto Frontier', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(98, 100)
    
    # Plot 3: Training Efficiency
    speedup_factors = [training_times[0] / t for t in training_times]
    ax3.bar(experiments, speedup_factors, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_ylabel('Training Speedup Factor', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Training Efficiency Gains', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for i, v in enumerate(speedup_factors):
        ax3.text(i, v + 0.02, f'{v:.2f}×', ha='center', va='bottom', fontweight='bold')
    ax3.set_ylim(0, 1.6)
    
    # Plot 4: Loss vs Perplexity (the contradiction)
    ax4.scatter(eval_losses, perplexities, c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=1)
    for i, exp in enumerate(experiments):
        ax4.annotate(exp, (eval_losses[i], perplexities[i]), 
                    xytext=(10, 10), textcoords='offset points', 
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add expected relationship line
    x_range = np.linspace(min(eval_losses), max(eval_losses), 100)
    expected_perplexity = np.exp(x_range)
    ax4.plot(x_range, expected_perplexity, 'r--', alpha=0.7, linewidth=2, label='Expected: exp(loss)')
    ax4.set_xlabel('Evaluation Loss', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Loss vs Perplexity Relationship', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save high-resolution version for publication
    output_path = 'results/arxiv_figure_1.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Publication figure saved to: {output_path}")
    
    return fig

def create_parameter_distribution_figure(results):
    """Create a figure showing parameter distribution across strategies."""
    print("Creating parameter distribution figure...")
    
    # Extract parameter data
    order = ['baseline_lora', 'attention_only_lora', 'ff_only_lora']
    labels = ['Baseline\n(Full)', 'Attention-Only', 'Feed-Forward\nOnly']
    
    # Attention parameters (c_attn + c_proj) and Feed-forward parameters (c_fc)
    attention_params = []
    ff_params = []
    
    for exp_name in order:
        exp_data = results[exp_name]
        total_params = exp_data['parameter_efficiency']['trainable_parameters'] / 1e6
        
        if exp_name == 'baseline_lora':
            # Full LoRA: both attention and FF
            attention_params.append(4.33)  # From actual data
            ff_params.append(1.97)
        elif exp_name == 'attention_only_lora':
            # Attention only
            attention_params.append(4.33)
            ff_params.append(0)
        else:  # ff_only_lora
            # Feed-forward only  
            attention_params.append(0)
            ff_params.append(1.97)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    width = 0.6
    x_pos = np.arange(len(labels))
    
    # Create stacked bars
    p1 = ax.bar(x_pos, attention_params, width, label='Attention Parameters (c_attn, c_proj)', 
                color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1)
    p2 = ax.bar(x_pos, ff_params, width, bottom=attention_params, 
                label='Feed-Forward Parameters (c_fc)', 
                color='#F18F01', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (att, ff) in enumerate(zip(attention_params, ff_params)):
        if att > 0:
            ax.text(i, att/2, f'{att:.2f}M', ha='center', va='center', 
                   fontweight='bold', color='white')
        if ff > 0:
            ax.text(i, att + ff/2, f'{ff:.2f}M', ha='center', va='center', 
                   fontweight='bold', color='white')
        
        # Total at top
        total = att + ff
        ax.text(i, total + 0.1, f'{total:.2f}M\nTotal', ha='center', va='bottom', 
               fontweight='bold')
    
    ax.set_xlabel('LoRA Placement Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Trainable Parameters (Millions)', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Distribution Across LoRA Placement Strategies', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'results/arxiv_figure_2.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Parameter distribution figure saved to: {output_path}")
    
    return fig

def create_figure_captions():
    """Create LaTeX-style figure captions for the arXiv paper."""
    captions = {
        "figure_1": """
**Figure 1: Selective LoRA Placement Analysis.** (a) Parameter efficiency trade-off showing evaluation loss vs trainable parameters. Feed-forward only achieves maximum parameter reduction (1.97M params) with acceptable performance degradation. (b) Pareto frontier analysis demonstrating three optimal strategies for different efficiency-performance requirements. (c) Training speedup factors showing computational benefits of selective placement. (d) Loss vs perplexity relationship revealing unexpected metric behavior, with attention-only achieving lowest perplexity despite higher loss than baseline.
        """.strip(),
        
        "figure_2": """
**Figure 2: Parameter Distribution Across Strategies.** Breakdown of trainable parameters by component type. Baseline LoRA uses all modules (6.29M total), attention-only targets c_attn and c_proj (4.33M), and feed-forward only targets c_fc (1.97M). This distribution analysis guides the selection of placement strategies based on available computational resources.
        """.strip()
    }
    
    # Save captions to file
    with open('results/figure_captions.md', 'w') as f:
        f.write("# Figure Captions for arXiv Submission\n\n")
        for fig_name, caption in captions.items():
            f.write(f"## {fig_name.replace('_', ' ').title()}\n\n")
            f.write(caption + "\n\n")
    
    print("✅ Figure captions saved to: results/figure_captions.md")
    return captions

def main():
    """Main function to create all arXiv figures."""
    print("🎨 CREATING ARXIV FIGURES")
    print("=" * 60)
    
    # Load experimental results
    results = load_experiment_results()
    
    if len(results) != 3:
        print(f"❌ Expected 3 experiments, found {len(results)}")
        return
    
    # Create main analysis figure
    fig1 = create_arxiv_figure(results)
    
    # Create parameter distribution figure  
    fig2 = create_parameter_distribution_figure(results)
    
    # Create figure captions
    captions = create_figure_captions()
    
    print("\n" + "=" * 60)
    print("✅ ARXIV FIGURES COMPLETE")
    print("📊 Created: arxiv_figure_1.png (main analysis)")
    print("📊 Created: arxiv_figure_2.png (parameter distribution)")
    print("📝 Created: figure_captions.md (LaTeX captions)")
    print("=" * 60)

if __name__ == "__main__":
    main() 