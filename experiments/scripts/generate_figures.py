"""
Generate figures from DPBench experiment results.

Usage:
    python experiments/scripts/generate_figures.py

Outputs saved to: experiments/figures/
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch


def calc_stderr(p: float, n: int = 20) -> float:
    """Calculate standard error for a proportion."""
    if p <= 0 or p >= 1:
        return 0
    return math.sqrt(p * (1 - p) / n)

# Figure sizes (inches)
SINGLE_COL_WIDTH = 3.25
DOUBLE_COL_WIDTH = 6.875

# Matplotlib defaults
mpl.rcParams['font.family'] = 'DejaVu Serif'
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['legend.fontsize'] = 7
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.02
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['grid.linewidth'] = 0.3
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['patch.linewidth'] = 0.5

# Paths
SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
RESULTS_DIR = EXPERIMENTS_DIR / "results" / "full_20260126_213124"
FIGURES_DIR = EXPERIMENTS_DIR / "figures"

# Okabe-Ito colorblind-friendly palette
OKABE_ITO = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'vermillion': '#D55E00',
    'sky_blue': '#56B4E9',
    'yellow': '#F0E442',
    'purple': '#CC79A7',
    'black': '#000000',
}

COLORS = {
    'gpt-5.2': OKABE_ITO['blue'],
    'claude-opus-4.5': OKABE_ITO['vermillion'],
    'grok-4.1': OKABE_ITO['green'],
    'sim': OKABE_ITO['vermillion'],
    'seq': OKABE_ITO['blue'],
    'no_comm': OKABE_ITO['sky_blue'],
    'comm': OKABE_ITO['purple'],
}


def load_all_metrics() -> dict:
    """Load all metrics.json files from results directory."""
    metrics = {}
    for metrics_file in RESULTS_DIR.glob("*/metrics.json"):
        with open(metrics_file) as f:
            data = json.load(f)
            key = f"{data['model']}_{data['condition']}"
            metrics[key] = data
    return metrics


def fig2_gpt52_all_conditions(metrics: dict) -> None:
    """GPT-5.2 deadlock rates across all 8 conditions."""
    conditions = ['sim3nc', 'sim3c', 'sim5nc', 'sim5c', 'seq3nc', 'seq3c', 'seq5nc', 'seq5c']
    labels = ['S3-', 'S3+', 'S5-', 'S5+', 'Q3-', 'Q3+', 'Q5-', 'Q5+']

    deadlock_rates = []
    errors = []
    colors = []
    for cond in conditions:
        key = f"gpt-5.2_{cond}"
        if key in metrics:
            rate = metrics[key]['deadlock_rate']
            deadlock_rates.append(rate * 100)
            errors.append(calc_stderr(rate) * 100)
            colors.append(COLORS['sim'] if cond.startswith('sim') else COLORS['seq'])
        else:
            deadlock_rates.append(0)
            errors.append(0)
            colors.append('gray')

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 2.2))

    bars = ax.bar(labels, deadlock_rates, color=colors, edgecolor='black', linewidth=0.3,
                  yerr=errors, capsize=2, error_kw={'linewidth': 0.5})

    for bar, val in zip(bars, deadlock_rates):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                   f'{val:.0f}', ha='center', va='bottom', fontsize=6)

    ax.set_ylabel('Deadlock Rate (%)')
    ax.set_xlabel('Condition (S=Simultaneous, Q=Sequential, 3/5=Philosophers, -/+=Comm)')
    ax.set_ylim(0, 105)

    legend_elements = [
        Patch(facecolor=COLORS['sim'], edgecolor='black', linewidth=0.3, label='Simultaneous'),
        Patch(facecolor=COLORS['seq'], edgecolor='black', linewidth=0.3, label='Sequential')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=7)

    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gpt52_deadlock_by_condition.pdf")
    plt.savefig(FIGURES_DIR / "gpt52_deadlock_by_condition.png")
    plt.close()
    print(f"Saved: gpt52_deadlock_by_condition.pdf and .png")


def fig3_cross_model_comparison(metrics: dict) -> None:
    """Cross-model comparison on shared conditions (sim5nc, sim5c, seq5nc)."""
    models = ['gpt-5.2', 'claude-opus-4.5', 'grok-4.1']
    model_labels = ['GPT-5.2', 'Claude 4.5', 'Grok 4.1']
    model_colors = [COLORS['gpt-5.2'], COLORS['claude-opus-4.5'], COLORS['grok-4.1']]
    conditions = ['sim5nc', 'sim5c', 'seq5nc']
    condition_labels = ['Sim 5P\nNo Comm', 'Sim 5P\nComm', 'Seq 5P\nNo Comm']

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.2))

    x = range(len(conditions))
    width = 0.25

    for i, (model, label, color) in enumerate(zip(models, model_labels, model_colors)):
        deadlock_rates = []
        errors = []
        for cond in conditions:
            key = f"{model}_{cond}"
            if key in metrics:
                rate = metrics[key]['deadlock_rate']
                deadlock_rates.append(rate * 100)
                errors.append(calc_stderr(rate) * 100)
            else:
                deadlock_rates.append(0)
                errors.append(0)

        offset = (i - 1) * width
        bars = ax.bar([xi + offset for xi in x], deadlock_rates, width,
                     label=label, color=color,
                     edgecolor='black', linewidth=0.3,
                     yerr=errors, capsize=2, error_kw={'linewidth': 0.5})

        for bar, val in zip(bars, deadlock_rates):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.0f}', ha='center', va='bottom', fontsize=5)

    ax.set_ylabel('Deadlock Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.set_ylim(0, 85)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=7)

    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.pdf")
    plt.savefig(FIGURES_DIR / "model_comparison.png")
    plt.close()
    print(f"Saved: model_comparison.pdf and .png")


def fig4_communication_effect(metrics: dict) -> None:
    """Effect of communication on deadlock rates for GPT-5.2."""
    pairs = [
        ('sim3nc', 'sim3c', '3P Sim'),
        ('sim5nc', 'sim5c', '5P Sim'),
        ('seq3nc', 'seq3c', '3P Seq'),
        ('seq5nc', 'seq5c', '5P Seq'),
    ]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.2))

    x = range(len(pairs))
    width = 0.35

    no_comm_rates = []
    no_comm_errors = []
    comm_rates = []
    comm_errors = []
    labels = []

    for nc, c, label in pairs:
        key_nc = f"gpt-5.2_{nc}"
        key_c = f"gpt-5.2_{c}"
        if key_nc in metrics:
            rate = metrics[key_nc]['deadlock_rate']
            no_comm_rates.append(rate * 100)
            no_comm_errors.append(calc_stderr(rate) * 100)
        else:
            no_comm_rates.append(0)
            no_comm_errors.append(0)
        if key_c in metrics:
            rate = metrics[key_c]['deadlock_rate']
            comm_rates.append(rate * 100)
            comm_errors.append(calc_stderr(rate) * 100)
        else:
            comm_rates.append(0)
            comm_errors.append(0)
        labels.append(label)

    bars1 = ax.bar([xi - width/2 for xi in x], no_comm_rates, width,
                   label='No Comm', color=COLORS['no_comm'],
                   edgecolor='black', linewidth=0.3,
                   yerr=no_comm_errors, capsize=2, error_kw={'linewidth': 0.5})
    bars2 = ax.bar([xi + width/2 for xi in x], comm_rates, width,
                   label='With Comm', color=COLORS['comm'],
                   edgecolor='black', linewidth=0.3,
                   yerr=comm_errors, capsize=2, error_kw={'linewidth': 0.5})

    for bar, val in zip(bars1, no_comm_rates):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                   f'{val:.0f}', ha='center', va='bottom', fontsize=5)
    for bar, val in zip(bars2, comm_rates):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                   f'{val:.0f}', ha='center', va='bottom', fontsize=5)

    ax.set_ylabel('Deadlock Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=7)

    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "communication_effect.pdf")
    plt.savefig(FIGURES_DIR / "communication_effect.png")
    plt.close()
    print(f"Saved: communication_effect.pdf and .png")


def main():
    """Generate all figures."""
    print("=" * 60)
    print("DPBench Figure Generator")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {FIGURES_DIR}")

    if not RESULTS_DIR.exists():
        print(f"Error: Results directory not found: {RESULTS_DIR}")
        return

    print(f"Reading results from: {RESULTS_DIR}")

    metrics = load_all_metrics()
    print(f"Loaded {len(metrics)} experiment results")
    print()

    print("Generating figures...")
    print("-" * 40)

    fig2_gpt52_all_conditions(metrics)
    fig3_cross_model_comparison(metrics)
    fig4_communication_effect(metrics)

    print("-" * 40)
    print("Done! All figures saved to experiments/figures/ (PDF + PNG)")


if __name__ == "__main__":
    main()
