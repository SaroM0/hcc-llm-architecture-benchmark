#!/usr/bin/env python3
"""Generate visualizations for the benchmark results."""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def plot_a1_accuracy_comparison():
    """Plot A1 model accuracy comparison."""
    print("Generating A1 accuracy comparison...")

    a1_report = json.load(open("reports/json/a1_full_report.json"))
    results = a1_report["all_results"]

    # Prepare data
    models = [r['model_key'] for r in results]
    accuracy = [r['accuracy'] * 100 for r in results]
    partial_accuracy = [r['partial_accuracy'] * 100 for r in results]
    model_class = [r['model_class'] for r in results]

    # Create DataFrame
    df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Partial Accuracy': partial_accuracy,
        'Class': model_class
    })

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(models))
    width = 0.35

    colors_class = {'large': '#2E86AB', 'small': '#A23B72'}
    colors = [colors_class[c] for c in model_class]

    bars1 = ax.bar(x - width/2, accuracy, width, label='Exact Accuracy', color=colors, alpha=0.8)
    bars2 = ax.bar(x + width/2, partial_accuracy, width, label='Partial Accuracy', color=colors, alpha=0.5)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('A1: Model Accuracy Comparison (Oneshot, No RAG)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    # Add legend for model classes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='Large Models'),
        Patch(facecolor='#A23B72', label='Small Models')
    ]
    ax.legend(handles=legend_elements + [
        Patch(facecolor='gray', alpha=0.8, label='Exact Accuracy'),
        Patch(facecolor='gray', alpha=0.5, label='Partial Accuracy')
    ], loc='upper right')

    plt.tight_layout()
    plt.savefig('reports/visualizations/a1_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ a1_accuracy_comparison.png")


def plot_consensus_improvement():
    """Plot improvement from A1 to A3/A4 with consensus."""
    print("Generating consensus improvement comparison...")

    comp_report = json.load(open("reports/json/comprehensive_comparison.json"))

    # Data
    experiments = ['A1\n(gpt52)', 'A3\n(gpt52\n+consensus)', 'A1\n(qwen3)', 'A4\n(qwen3\n+consensus)']
    accuracy = [
        comp_report['experiments']['A1']['best_large']['accuracy'] * 100,
        comp_report['experiments']['A3']['accuracy'] * 100,
        comp_report['experiments']['A1']['best_small']['accuracy'] * 100,
        comp_report['experiments']['A4']['accuracy'] * 100,
    ]

    colors = ['#6C757D', '#2E86AB', '#6C757D', '#A23B72']

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(experiments, accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Multi-Agent Consensus on Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels and improvement arrows
    for i, (bar, acc) in enumerate(zip(bars, accuracy)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add improvement annotation
        if i == 1:  # A3
            improvement = comp_report['experiments']['A3']['improvement_over_a1']
            ax.annotate(f'+{improvement:.2f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height/2),
                       fontsize=10, ha='center', color='green', fontweight='bold')
        elif i == 3:  # A4
            improvement = comp_report['experiments']['A4']['improvement_over_a1']
            ax.annotate(f'+{improvement:.2f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height/2),
                       fontsize=10, ha='center', color='green', fontweight='bold')

    # Add arrows showing improvement
    ax.annotate('', xy=(1, accuracy[1]), xytext=(0, accuracy[0]),
                arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.6))
    ax.annotate('', xy=(3, accuracy[3]), xytext=(2, accuracy[2]),
                arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.6))

    plt.tight_layout()
    plt.savefig('reports/visualizations/consensus_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ consensus_improvement.png")


def plot_cost_vs_accuracy():
    """Plot cost vs accuracy tradeoff."""
    print("Generating cost vs accuracy tradeoff...")

    a3_report = json.load(open("reports/json/a3_full_report.json"))
    a4_report = json.load(open("reports/json/a4_full_report.json"))

    # Data
    models = ['A3\n(gpt52)', 'A4\n(qwen3_vl_30b)']
    accuracy = [
        a3_report['results']['accuracy'] * 100,
        a4_report['results']['accuracy'] * 100,
    ]
    cost = [
        a3_report['costs']['total_cost_usd'],
        a4_report['costs']['total_cost_usd'],
    ]

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(cost, accuracy, s=500, c=['#2E86AB', '#A23B72'],
                        alpha=0.6, edgecolors='black', linewidth=2)

    # Add labels
    for i, model in enumerate(models):
        ax.annotate(model, (cost[i], accuracy[i]),
                   textcoords="offset points", xytext=(0, 15),
                   ha='center', fontsize=11, fontweight='bold')
        ax.annotate(f'${cost[i]:.2f}\n{accuracy[i]:.2f}%',
                   (cost[i], accuracy[i]),
                   textcoords="offset points", xytext=(0, -30),
                   ha='center', fontsize=9)

    ax.set_xlabel('Total Cost (USD)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cost vs Accuracy Tradeoff (88 predictions)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 4.5)
    ax.set_ylim(70, 80)

    plt.tight_layout()
    plt.savefig('reports/visualizations/cost_vs_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ cost_vs_accuracy.png")


def plot_latency_comparison():
    """Plot latency comparison."""
    print("Generating latency comparison...")

    a3_report = json.load(open("reports/json/a3_full_report.json"))
    a4_report = json.load(open("reports/json/a4_full_report.json"))

    # Data (avg per prediction in seconds)
    models = ['A3 (gpt52)', 'A4 (qwen3_vl_30b)']
    latency = [
        a3_report['results']['latency_seconds'] / 88,
        a4_report['results']['latency_seconds'] / 88,
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(models, latency, color=['#2E86AB', '#A23B72'], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Average Latency per Prediction (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Latency Comparison: A3 vs A4', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, lat in zip(bars, latency):
        width = bar.get_width()
        ax.text(width + 5, bar.get_y() + bar.get_height()/2.,
                f'{lat:.1f}s', ha='left', va='center', fontsize=11, fontweight='bold')

    # Add speedup annotation
    speedup = latency[1] / latency[0]
    ax.text(0.5, 0.95, f'A4 is {speedup:.1f}x slower',
           transform=ax.transAxes, ha='center', va='top',
           fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig('reports/visualizations/latency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ latency_comparison.png")


def plot_token_usage():
    """Plot token usage comparison."""
    print("Generating token usage comparison...")

    a3_report = json.load(open("reports/json/a3_full_report.json"))
    a4_report = json.load(open("reports/json/a4_full_report.json"))

    # Data
    categories = ['Prompt\nTokens', 'Completion\nTokens', 'Total\nTokens']

    # Load predictions to get token breakdown
    a3_preds = []
    with open("runs/matrix/20260125_002746_A3_gpt52/predictions.jsonl") as f:
        for line in f:
            a3_preds.append(json.loads(line))

    a4_preds = []
    with open("runs/matrix/20260125_083639_A4_qwen3_vl_30b/predictions.jsonl") as f:
        for line in f:
            a4_preds.append(json.loads(line))

    a3_prompt = sum(p.get('debug', {}).get('total_prompt_tokens', 0) for p in a3_preds) / len(a3_preds)
    a3_completion = sum(p.get('debug', {}).get('total_completion_tokens', 0) for p in a3_preds) / len(a3_preds)
    a3_total = a3_report['costs']['avg_tokens_per_prediction']

    a4_prompt = sum(p.get('debug', {}).get('total_prompt_tokens', 0) for p in a4_preds) / len(a4_preds)
    a4_completion = sum(p.get('debug', {}).get('total_completion_tokens', 0) for p in a4_preds) / len(a4_preds)
    a4_total = a4_report['costs']['avg_tokens_per_prediction']

    a3_values = [a3_prompt, a3_completion, a3_total]
    a4_values = [a4_prompt, a4_completion, a4_total]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))

    bars1 = ax.bar(x - width/2, a3_values, width, label='A3 (gpt52)', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, a4_values, width, label='A4 (qwen3_vl_30b)', color='#A23B72', alpha=0.8)

    ax.set_ylabel('Average Tokens per Prediction', fontsize=12, fontweight='bold')
    ax.set_title('Token Usage Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('reports/visualizations/token_usage_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ token_usage_comparison.png")


def plot_consensus_metrics():
    """Plot consensus-related metrics."""
    print("Generating consensus metrics...")

    a3_report = json.load(open("reports/json/a3_full_report.json"))
    a4_report = json.load(open("reports/json/a4_full_report.json"))

    # Data
    metrics = ['Consensus\nRate (%)', 'Avg\nRounds', 'Avg\nConfidence']
    a3_values = [
        a3_report['consensus_metrics']['consensus_rate'] * 100,
        a3_report['consensus_metrics']['avg_rounds'],
        a3_report['consensus_metrics']['avg_confidence'] * 100,
    ]
    a4_values = [
        a4_report['consensus_metrics']['consensus_rate'] * 100,
        a4_report['consensus_metrics']['avg_rounds'],
        a4_report['consensus_metrics']['avg_confidence'] * 100,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))

    bars1 = ax.bar(x - width/2, a3_values, width, label='A3 (gpt52)', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, a4_values, width, label='A4 (qwen3_vl_30b)', color='#A23B72', alpha=0.8)

    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Consensus Process Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('reports/visualizations/consensus_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ consensus_metrics.png")


def plot_comprehensive_dashboard():
    """Create a comprehensive dashboard with key metrics."""
    print("Generating comprehensive dashboard...")

    a1_report = json.load(open("reports/json/a1_full_report.json"))
    a3_report = json.load(open("reports/json/a3_full_report.json"))
    a4_report = json.load(open("reports/json/a4_full_report.json"))

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Accuracy comparison
    ax1 = fig.add_subplot(gs[0, :])
    experiments = ['A1 (gpt52)', 'A3 (gpt52\n+consensus)', 'A1 (qwen3)', 'A4 (qwen3\n+consensus)']
    accuracy = [
        a1_report['best_large_model']['accuracy'] * 100,
        a3_report['results']['accuracy'] * 100,
        a1_report['best_small_model']['accuracy'] * 100,
        a4_report['results']['accuracy'] * 100,
    ]
    bars = ax1.bar(experiments, accuracy, color=['#6C757D', '#2E86AB', '#6C757D', '#A23B72'], alpha=0.8)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Accuracy Comparison Across Experiments', fontweight='bold', fontsize=12)
    ax1.set_ylim(0, 100)
    for bar, acc in zip(bars, accuracy):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 2. Cost comparison
    ax2 = fig.add_subplot(gs[1, 0])
    costs = [a3_report['costs']['total_cost_usd'], a4_report['costs']['total_cost_usd']]
    ax2.bar(['A3', 'A4'], costs, color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax2.set_ylabel('Total Cost (USD)', fontweight='bold')
    ax2.set_title('Total Cost', fontweight='bold')
    for i, cost in enumerate(costs):
        ax2.text(i, cost + 0.1, f'${cost:.2f}', ha='center', fontweight='bold')

    # 3. Latency comparison
    ax3 = fig.add_subplot(gs[1, 1])
    latency = [
        a3_report['results']['latency_minutes'],
        a4_report['results']['latency_minutes'],
    ]
    ax3.bar(['A3', 'A4'], latency, color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax3.set_ylabel('Total Time (minutes)', fontweight='bold')
    ax3.set_title('Total Latency', fontweight='bold')
    for i, lat in enumerate(latency):
        ax3.text(i, lat + 5, f'{lat:.1f}m', ha='center', fontweight='bold')

    # 4. LLM calls per prediction
    ax4 = fig.add_subplot(gs[1, 2])
    llm_calls = [
        a3_report['llm_calls']['avg_per_prediction'],
        a4_report['llm_calls']['avg_per_prediction'],
    ]
    ax4.bar(['A3', 'A4'], llm_calls, color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax4.set_ylabel('Avg LLM Calls', fontweight='bold')
    ax4.set_title('LLM Calls per Prediction', fontweight='bold')
    for i, calls in enumerate(llm_calls):
        ax4.text(i, calls + 0.1, f'{calls:.1f}', ha='center', fontweight='bold')

    # 5. Architecture diagram (text)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    ax5.text(0.5, 0.9, 'Multi-Agent Consensus Architecture', ha='center', va='top',
            fontsize=14, fontweight='bold', transform=ax5.transAxes)
    ax5.text(0.5, 0.7, '3 Medical Specialists → Supervisor → Final Decision', ha='center', va='top',
            fontsize=11, transform=ax5.transAxes)
    ax5.text(0.1, 0.5, '• Hepatologist\n  (Liver disease)', ha='left', va='top',
            fontsize=10, transform=ax5.transAxes)
    ax5.text(0.4, 0.5, '• Oncologist\n  (Cancer)', ha='left', va='top',
            fontsize=10, transform=ax5.transAxes)
    ax5.text(0.7, 0.5, '• Radiologist\n  (Imaging)', ha='left', va='top',
            fontsize=10, transform=ax5.transAxes)
    ax5.text(0.5, 0.2, 'Supervisor coordinates consensus and determines final Likert score (-2 to +2)',
            ha='center', va='top', fontsize=10, style='italic', transform=ax5.transAxes)
    ax5.text(0.5, 0.05, 'RAG: Top-5 retrieval from ChromaDB (HCC clinical guidelines)',
            ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
            transform=ax5.transAxes)

    fig.suptitle('HCC LLM Architecture Benchmark - Comprehensive Dashboard',
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig('reports/visualizations/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ comprehensive_dashboard.png")


if __name__ == "__main__":
    print("="*80)
    print("HCC LLM Architecture Benchmark - Visualization Generation")
    print("="*80)

    Path("reports/visualizations").mkdir(parents=True, exist_ok=True)

    plot_a1_accuracy_comparison()
    plot_consensus_improvement()
    plot_cost_vs_accuracy()
    plot_latency_comparison()
    plot_token_usage()
    plot_consensus_metrics()
    plot_comprehensive_dashboard()

    print("\n" + "="*80)
    print("Visualization generation complete!")
    print("="*80)
    print("\nGenerated visualizations:")
    print("  reports/visualizations/")
    print("    - a1_accuracy_comparison.png")
    print("    - consensus_improvement.png")
    print("    - cost_vs_accuracy.png")
    print("    - latency_comparison.png")
    print("    - token_usage_comparison.png")
    print("    - consensus_metrics.png")
    print("    - comprehensive_dashboard.png")
