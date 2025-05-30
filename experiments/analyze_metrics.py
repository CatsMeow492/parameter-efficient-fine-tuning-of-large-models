#!/usr/bin/env python3
"""
Analysis script to investigate the perplexity vs evaluation loss contradiction.

Key question: Why does attention-only have better perplexity (2,272) than baseline (4,283)
despite having higher evaluation loss (3.48 vs 3.09)?
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

def analyze_loss_perplexity_relationship(results):
    """Analyze the relationship between loss and perplexity."""
    print("=" * 60)
    print("LOSS vs PERPLEXITY ANALYSIS")
    print("=" * 60)
    
    data = []
    for exp_name, exp_data in results.items():
        metrics = exp_data['performance_metrics']
        loss = metrics['eval_loss']
        perplexity = metrics['eval_perplexity']
        
        # Calculate expected perplexity from loss
        # Perplexity = exp(loss) for cross-entropy loss
        expected_perplexity = np.exp(loss)
        
        data.append({
            'experiment': exp_name.replace('_', ' ').title(),
            'loss': loss,
            'actual_perplexity': perplexity,
            'expected_perplexity': expected_perplexity,
            'perplexity_ratio': perplexity / expected_perplexity
        })
        
        print(f"\n{exp_name.replace('_', ' ').title()}:")
        print(f"  Evaluation Loss: {loss:.4f}")
        print(f"  Actual Perplexity: {perplexity:.1f}")
        print(f"  Expected Perplexity (exp(loss)): {expected_perplexity:.1f}")
        print(f"  Ratio (actual/expected): {perplexity/expected_perplexity:.4f}")
        
        if abs(perplexity - expected_perplexity) > 1:
            print(f"  ⚠️  SIGNIFICANT DEVIATION: {abs(perplexity - expected_perplexity):.1f}")
    
    return data

def create_efficiency_vs_performance_plot(results):
    """Create a scatter plot showing efficiency vs performance trade-offs."""
    print("\n" + "=" * 60)
    print("CREATING EFFICIENCY vs PERFORMANCE VISUALIZATION")
    print("=" * 60)
    
    # Extract data for plotting
    experiments = []
    eval_losses = []
    perplexities = []
    trainable_params = []
    training_times = []
    
    for exp_name, exp_data in results.items():
        experiments.append(exp_name.replace('_', ' ').title())
        eval_losses.append(exp_data['performance_metrics']['eval_loss'])
        perplexities.append(exp_data['performance_metrics']['eval_perplexity'])
        trainable_params.append(exp_data['parameter_efficiency']['trainable_parameters'] / 1e6)  # Convert to millions
        training_times.append(exp_data['efficiency_metrics']['training_time_seconds'])
    
    # Create subplot with 2x2 layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LoRA Placement Strategy Analysis', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Plot 1: Parameters vs Loss
    ax1.scatter(trainable_params, eval_losses, c=colors, s=150, alpha=0.7)
    for i, exp in enumerate(experiments):
        ax1.annotate(exp, (trainable_params[i], eval_losses[i]), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
    ax1.set_xlabel('Trainable Parameters (Millions)')
    ax1.set_ylabel('Evaluation Loss')
    ax1.set_title('Parameter Efficiency vs Performance')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameters vs Perplexity
    ax2.scatter(trainable_params, perplexities, c=colors, s=150, alpha=0.7)
    for i, exp in enumerate(experiments):
        ax2.annotate(exp, (trainable_params[i], perplexities[i]), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
    ax2.set_xlabel('Trainable Parameters (Millions)')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Parameter Efficiency vs Perplexity')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Time vs Loss
    ax3.scatter(training_times, eval_losses, c=colors, s=150, alpha=0.7)
    for i, exp in enumerate(experiments):
        ax3.annotate(exp, (training_times[i], eval_losses[i]), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
    ax3.set_xlabel('Training Time (seconds)')
    ax3.set_ylabel('Evaluation Loss')
    ax3.set_title('Training Efficiency vs Performance')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Loss vs Perplexity (the contradiction!)
    ax4.scatter(eval_losses, perplexities, c=colors, s=150, alpha=0.7)
    for i, exp in enumerate(experiments):
        ax4.annotate(exp, (eval_losses[i], perplexities[i]), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
    ax4.set_xlabel('Evaluation Loss')
    ax4.set_ylabel('Perplexity')
    ax4.set_title('Loss vs Perplexity (Investigating Contradiction)')
    ax4.grid(True, alpha=0.3)
    
    # Add expected relationship line for Plot 4
    x_range = np.linspace(min(eval_losses), max(eval_losses), 100)
    expected_perplexity = np.exp(x_range)
    ax4.plot(x_range, expected_perplexity, 'r--', alpha=0.5, label='Expected: exp(loss)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('results/lora_placement_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved to: results/lora_placement_analysis.png")
    
    return fig

def print_research_insights(results):
    """Print key research insights from the analysis."""
    print("\n" + "=" * 60)
    print("KEY RESEARCH INSIGHTS")
    print("=" * 60)
    
    # Extract key metrics
    baseline_loss = results['baseline_lora']['performance_metrics']['eval_loss']
    ff_loss = results['ff_only_lora']['performance_metrics']['eval_loss']
    att_loss = results['attention_only_lora']['performance_metrics']['eval_loss']
    
    baseline_perp = results['baseline_lora']['performance_metrics']['eval_perplexity']
    ff_perp = results['ff_only_lora']['performance_metrics']['eval_perplexity']
    att_perp = results['attention_only_lora']['performance_metrics']['eval_perplexity']
    
    baseline_params = results['baseline_lora']['parameter_efficiency']['trainable_parameters']
    ff_params = results['ff_only_lora']['parameter_efficiency']['trainable_parameters']
    att_params = results['attention_only_lora']['parameter_efficiency']['trainable_parameters']
    
    print("\n🎯 LAYER IMPORTANCE HIERARCHY:")
    print(f"   Attention-only loss: {att_loss:.3f} ({((att_loss/baseline_loss-1)*100):+.1f}% vs baseline)")
    print(f"   Feed-forward loss:   {ff_loss:.3f} ({((ff_loss/baseline_loss-1)*100):+.1f}% vs baseline)")
    print("   → Attention layers are MORE CRITICAL for performance")
    
    print("\n💡 EFFICIENCY INSIGHTS:")
    print(f"   Feed-forward params: {ff_params/1e6:.2f}M ({((1-ff_params/baseline_params)*100):.1f}% reduction)")
    print(f"   Attention params:    {att_params/1e6:.2f}M ({((1-att_params/baseline_params)*100):.1f}% reduction)")
    print("   → Feed-forward provides MAXIMUM efficiency")
    
    print("\n❓ PERPLEXITY CONTRADICTION:")
    print(f"   Attention-only: {att_loss:.3f} loss → {att_perp:.0f} perplexity")
    print(f"   Baseline:       {baseline_loss:.3f} loss → {baseline_perp:.0f} perplexity")
    print("   → Better perplexity despite higher loss suggests metric calculation differences")
    
    print("\n🏆 OPTIMAL STRATEGIES:")
    print("   • Maximum efficiency: Feed-forward only (99.45% reduction)")
    print("   • Best balance: Attention only (13% performance hit, 31% param reduction)")
    print("   • Best performance: Full baseline (standard approach)")

def main():
    """Main analysis function."""
    print("🔬 LORA PLACEMENT ANALYSIS")
    print("=" * 60)
    
    # Load experimental results
    results = load_experiment_results()
    
    if len(results) != 3:
        print(f"❌ Expected 3 experiments, found {len(results)}")
        print("Available experiments:", list(results.keys()))
        return
    
    # Analyze loss-perplexity relationship
    loss_perp_data = analyze_loss_perplexity_relationship(results)
    
    # Create visualizations
    fig = create_efficiency_vs_performance_plot(results)
    
    # Print research insights
    print_research_insights(results)
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("📊 Check results/lora_placement_analysis.png for visualizations")
    print("=" * 60)

if __name__ == "__main__":
    main() 