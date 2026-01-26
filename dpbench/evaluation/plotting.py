"""Publication-ready plots for DPBench results."""

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CLI use
import matplotlib.pyplot as plt
import numpy as np


# Colorblind-friendly color palette
COLORS = {
    'primary': '#2563eb',    # Blue
    'secondary': '#dc2626',  # Red
    'success': '#16a34a',    # Green
    'warning': '#ca8a04',    # Yellow
    'neutral': '#6b7280',    # Gray
}

# Condition colors for 8 experimental conditions
CONDITION_COLORS = [
    '#2563eb',  # sim-5-comm (blue)
    '#60a5fa',  # sim-5-no-comm (light blue)
    '#dc2626',  # seq-5-comm (red)
    '#f87171',  # seq-5-no-comm (light red)
    '#16a34a',  # sim-3-comm (green)
    '#4ade80',  # sim-3-no-comm (light green)
    '#ca8a04',  # seq-3-comm (yellow)
    '#fcd34d',  # seq-3-no-comm (light yellow)
]


def load_results(filepath: str) -> dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def _extract_condition_label(data: dict) -> str:
    """Extract a short label from result config."""
    config = data.get('config', {})
    mode = config.get('mode', 'sim')[:3]
    n = config.get('num_philosophers', 5)
    comm = 'comm' if config.get('communication', False) else 'no-comm'
    return f"{mode}-{n}-{comm}"


def plot_comparison(results_files: list[str], output: str, title: str = "DPBench Results Comparison"):
    """
    Generate multi-condition comparison bar chart.

    Args:
        results_files: List of JSON result file paths
        output: Output PNG file path
        title: Plot title
    """
    # Load all results
    data_list = []
    labels = []
    for filepath in results_files:
        data = load_results(filepath)
        data_list.append(data)
        # Try to extract label from filename or config
        label = _extract_condition_label(data)
        if label in labels:
            label = Path(filepath).stem
        labels.append(label)

    # Extract metrics
    deadlock_rates = []
    throughputs = []
    throughput_stds = []
    fairness_vals = []
    fairness_stds = []

    for data in data_list:
        metrics = data.get('metrics', {})
        deadlock_rates.append(metrics.get('deadlock_rate', 0) * 100)
        throughputs.append(metrics.get('avg_throughput', 0))
        throughput_stds.append(metrics.get('std_throughput', 0))
        fairness_vals.append(metrics.get('avg_fairness', 0))
        fairness_stds.append(metrics.get('std_fairness', 0))

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    x = np.arange(len(labels))
    colors = CONDITION_COLORS[:len(labels)]

    # Plot 1: Deadlock Rate
    ax1 = axes[0]
    bars1 = ax1.bar(x, deadlock_rates, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Deadlock Rate (%)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.set_title('Deadlock Rate (lower is better)', fontsize=11)
    # Add value labels
    for bar, val in zip(bars1, deadlock_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

    # Plot 2: Throughput
    ax2 = axes[1]
    bars2 = ax2.bar(x, throughputs, yerr=throughput_stds, color=colors,
                    edgecolor='black', linewidth=0.5, capsize=3)
    ax2.set_ylabel('Throughput (meals/step)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.set_title('Throughput (higher is better)', fontsize=11)

    # Plot 3: Fairness
    ax3 = axes[2]
    bars3 = ax3.bar(x, fairness_vals, yerr=fairness_stds, color=colors,
                    edgecolor='black', linewidth=0.5, capsize=3)
    ax3.set_ylabel('Fairness (Gini)', fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax3.set_title('Fairness (higher is better)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return output


def plot_boxplot(results_file: str, output: str, title: str | None = None):
    """
    Generate box plots for metric distributions across episodes.

    Args:
        results_file: JSON result file path
        output: Output PNG file path
        title: Optional plot title
    """
    data = load_results(results_file)
    episodes = data.get('episodes', [])

    if not episodes:
        raise ValueError(f"No episodes found in {results_file}")

    # Extract per-episode metrics
    throughputs = [ep.get('throughput', 0) for ep in episodes]
    fairness_vals = [ep.get('fairness', 0) for ep in episodes]
    timesteps = [ep.get('timesteps', 0) for ep in episodes]

    if title is None:
        label = _extract_condition_label(data)
        title = f"DPBench Metric Distributions: {label}"

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    # Box plot styling
    box_props = dict(facecolor=COLORS['primary'], alpha=0.7)
    median_props = dict(color='black', linewidth=2)
    whisker_props = dict(color='black', linewidth=1)

    # Plot 1: Throughput
    bp1 = axes[0].boxplot([throughputs], patch_artist=True,
                          boxprops=box_props, medianprops=median_props,
                          whiskerprops=whisker_props)
    axes[0].set_ylabel('Throughput (meals/step)', fontsize=10)
    axes[0].set_xticklabels([''])
    axes[0].set_title('Throughput Distribution', fontsize=10)

    # Plot 2: Fairness
    bp2 = axes[1].boxplot([fairness_vals], patch_artist=True,
                          boxprops=box_props, medianprops=median_props,
                          whiskerprops=whisker_props)
    axes[1].set_ylabel('Fairness (Gini)', fontsize=10)
    axes[1].set_xticklabels([''])
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Fairness Distribution', fontsize=10)

    # Plot 3: Timesteps
    bp3 = axes[2].boxplot([timesteps], patch_artist=True,
                          boxprops=box_props, medianprops=median_props,
                          whiskerprops=whisker_props)
    axes[2].set_ylabel('Timesteps', fontsize=10)
    axes[2].set_xticklabels([''])
    axes[2].set_title('Episode Length Distribution', fontsize=10)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return output


def plot_timeline(results_file: str, output: str, title: str | None = None):
    """
    Generate episode timeline showing outcomes.

    Args:
        results_file: JSON result file path
        output: Output PNG file path
        title: Optional plot title
    """
    data = load_results(results_file)
    episodes = data.get('episodes', [])

    if not episodes:
        raise ValueError(f"No episodes found in {results_file}")

    if title is None:
        label = _extract_condition_label(data)
        title = f"DPBench Episode Timeline: {label}"

    # Extract data
    episode_ids = [ep.get('episode_id', i) for i, ep in enumerate(episodes)]
    timesteps = [ep.get('timesteps', 0) for ep in episodes]
    deadlocks = [ep.get('deadlock', False) for ep in episodes]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot completed episodes (green circles)
    completed_x = [i for i, d in zip(episode_ids, deadlocks) if not d]
    completed_y = [t for t, d in zip(timesteps, deadlocks) if not d]
    ax.scatter(completed_x, completed_y, c=COLORS['success'], s=60,
               marker='o', label='Completed', zorder=3, edgecolors='black', linewidth=0.5)

    # Plot deadlocked episodes (red X)
    deadlock_x = [i for i, d in zip(episode_ids, deadlocks) if d]
    deadlock_y = [t for t, d in zip(timesteps, deadlocks) if d]
    ax.scatter(deadlock_x, deadlock_y, c=COLORS['secondary'], s=80,
               marker='X', label='Deadlock', zorder=3, edgecolors='black', linewidth=0.5)

    # Styling
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Timesteps', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add summary stats
    n_deadlock = sum(deadlocks)
    n_total = len(deadlocks)
    deadlock_rate = n_deadlock / n_total * 100 if n_total > 0 else 0
    ax.text(0.02, 0.98, f'Deadlock Rate: {deadlock_rate:.1f}% ({n_deadlock}/{n_total})',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return output


def plot_all(results_file: str, output_dir: str):
    """
    Generate all plot types for a single results file.

    Args:
        results_file: JSON result file path
        output_dir: Output directory for PNG files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stem = Path(results_file).stem

    outputs = []
    outputs.append(plot_boxplot(results_file, str(output_path / f"{stem}_boxplot.png")))
    outputs.append(plot_timeline(results_file, str(output_path / f"{stem}_timeline.png")))

    return outputs
