"""Figure 3: three structural variables that determine deadlock outcome.

Three panels with the same chart type (bar with Wilson CI caps):
    (a) Prompt strategy (5 variants, sorted by deadlock rate).
    (b) Communication rounds before commitment (0, 1, 3, 5).
    (c) Group size (N=5 vs N=10) across three agents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from . import _style
from ._data import PER_ANALYSIS, load_csv, find_row, to_float


PROMPT_ORDER = [
    "prompt_minimal",
    "sim_n5_nocomm",
    "prompt_theory_of_mind",
    "prompt_symmetry_breaking",
    "prompt_resource_ordering",
]

PROMPT_DISPLAY = {
    "prompt_minimal": "minimal",
    "sim_n5_nocomm": "default",
    "prompt_theory_of_mind": "theory-of-mind",
    "prompt_symmetry_breaking": "symmetry-breaking",
    "prompt_resource_ordering": "resource-ordering",
}


def _bars_with_ci(ax: plt.Axes, labels: list[str], values: list[float],
                  lows: list[float], highs: list[float], colors: list[str],
                  rotate_labels: bool) -> None:
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, edgecolor=_style.PALETTE["ink"],
           linewidth=0.4, width=0.6, zorder=3)
    ax.errorbar(x, values, yerr=[lows, highs], fmt="none",
                ecolor=_style.PALETTE["ink"], elinewidth=1.0, capsize=3,
                alpha=0.75, zorder=4)
    for xi, v, hi in zip(x, values, highs):
        ax.text(xi, v + hi + 0.04, f"{v * 100:.0f}%",
                ha="center", va="bottom",
                fontsize=_style.FONT_SIZES["annotation"],
                color=_style.PALETTE["ink"], fontweight="bold")
    ax.set_xticks(x)
    if rotate_labels:
        ax.set_xticklabels(labels, rotation=30, ha="right",
                           fontsize=_style.FONT_SIZES["tick"])
    else:
        ax.set_xticklabels(labels, fontsize=_style.FONT_SIZES["tick"])
    ax.set_ylim(0, 1.18)
    _style.percent_axis(ax, axis="y")
    _style.trim_spines(ax)
    _style.horizontal_grid(ax)


def _panel_prompts(ax: plt.Axes, structural: list[dict[str, Any]]) -> None:
    bars: list[tuple[str, float, float, float]] = []
    for cond in PROMPT_ORDER:
        row = find_row(structural, model="gemini-2.5-flash",
                       condition_canonical=cond)
        if row is None:
            continue
        p = to_float(row["deadlock_rate"]) or 0.0
        lo = to_float(row["deadlock_ci_low"]) or 0.0
        hi = to_float(row["deadlock_ci_high"]) or 0.0
        bars.append((PROMPT_DISPLAY[cond], p, lo, hi))

    labels = [b[0] for b in bars]
    values = [b[1] for b in bars]
    lows = [max(0.0, b[1] - b[2]) for b in bars]
    highs = [max(0.0, b[3] - b[1]) for b in bars]
    colors = [_style.deadlock_color_for(v) for v in values]

    _bars_with_ci(ax, labels, values, lows, highs, colors, rotate_labels=True)
    ax.set_title("(a) Prompt strategy", fontsize=_style.FONT_SIZES["title"], pad=8)
    ax.set_ylabel("Deadlock rate")


def _panel_rounds(ax: plt.Axes, structural: list[dict[str, Any]],
                   cross_model: list[dict[str, Any]]) -> None:
    rounds_data: list[tuple[str, float, float, float]] = []

    baseline = find_row(cross_model, model="gemini-2.5-flash",
                        condition_canonical="sim_n5_nocomm")
    if baseline is not None:
        rounds_data.append(("0", to_float(baseline["deadlock_rate"]) or 0.0,
                            to_float(baseline["deadlock_ci_low"]) or 0.0,
                            to_float(baseline["deadlock_ci_high"]) or 0.0))

    single = find_row(cross_model, model="gemini-2.5-flash",
                      condition_canonical="sim_n5_comm")
    if single is not None:
        rounds_data.append(("1", to_float(single["deadlock_rate"]) or 0.0,
                            to_float(single["deadlock_ci_low"]) or 0.0,
                            to_float(single["deadlock_ci_high"]) or 0.0))

    for cond, label in (("comm_rounds_3", "3"), ("comm_rounds_5", "5")):
        row = find_row(structural, model="gemini-2.5-flash",
                       condition_canonical=cond)
        if row is not None:
            rounds_data.append((label, to_float(row["deadlock_rate"]) or 0.0,
                                to_float(row["deadlock_ci_low"]) or 0.0,
                                to_float(row["deadlock_ci_high"]) or 0.0))

    labels = [r[0] for r in rounds_data]
    values = [r[1] for r in rounds_data]
    lows = [max(0.0, r[1] - r[2]) for r in rounds_data]
    highs = [max(0.0, r[3] - r[1]) for r in rounds_data]
    colors = [_style.deadlock_color_for(v) for v in values]

    _bars_with_ci(ax, labels, values, lows, highs, colors, rotate_labels=False)
    ax.set_xlabel("Discussion rounds",
                  fontsize=_style.FONT_SIZES["axis_label"])
    ax.set_title("(b) Communication rounds",
                 fontsize=_style.FONT_SIZES["title"], pad=8)


PANEL_C_SHORT = {
    "gemini-2.5-flash": "Gemini",
    "llama-4-maverick": "Llama",
    "random": "Random",
}


def _panel_group_size(ax: plt.Axes, cross_model: list[dict[str, Any]]) -> None:
    models = ["gemini-2.5-flash", "llama-4-maverick", "random"]
    width = 0.32
    x = np.arange(len(models))

    n5_v, n5_lo, n5_hi = [], [], []
    n10_v, n10_lo, n10_hi = [], [], []
    for model in models:
        for cond, vs, los, his in (
            ("sim_n5_nocomm", n5_v, n5_lo, n5_hi),
            ("sim_n10_nocomm", n10_v, n10_lo, n10_hi),
        ):
            row = find_row(cross_model, model=model, condition_canonical=cond)
            if row is None:
                vs.append(0.0); los.append(0.0); his.append(0.0)
                continue
            p = to_float(row["deadlock_rate"]) or 0.0
            lo = to_float(row["deadlock_ci_low"]) or 0.0
            hi = to_float(row["deadlock_ci_high"]) or 0.0
            vs.append(p)
            los.append(max(0.0, p - lo))
            his.append(max(0.0, hi - p))

    ax.bar(x - width / 2, n5_v, width=width, color=_style.PALETTE["deadlock"],
           edgecolor=_style.PALETTE["ink"], linewidth=0.4, label="N=5", zorder=3)
    ax.bar(x + width / 2, n10_v, width=width, color=_style.PALETTE["success"],
           edgecolor=_style.PALETTE["ink"], linewidth=0.4, label="N=10", zorder=3)
    ax.errorbar(x - width / 2, n5_v, yerr=[n5_lo, n5_hi], fmt="none",
                ecolor=_style.PALETTE["ink"], elinewidth=1.0, capsize=3, alpha=0.75, zorder=4)
    ax.errorbar(x + width / 2, n10_v, yerr=[n10_lo, n10_hi], fmt="none",
                ecolor=_style.PALETTE["ink"], elinewidth=1.0, capsize=3, alpha=0.75, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([PANEL_C_SHORT[m] for m in models],
                       rotation=0, ha="center", fontsize=_style.FONT_SIZES["tick"])
    ax.set_ylim(0, 1.05)
    _style.percent_axis(ax, axis="y")
    _style.trim_spines(ax)
    _style.horizontal_grid(ax)
    ax.set_title("(c) Group size", fontsize=_style.FONT_SIZES["title"], pad=8)
    legend = ax.legend(loc="upper right", frameon=True, framealpha=0.85,
                       edgecolor="none", facecolor=_style.PALETTE["paper"])
    for text in legend.get_texts():
        text.set_color(_style.PALETTE["ink"])


def generate(out_dir: Path, fmt: str = "png", dpi: int = 200) -> Path:
    structural = load_csv(PER_ANALYSIS / "structural_vars.csv")
    cross_model = load_csv(PER_ANALYSIS / "cross_model.csv")

    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.7))
    _panel_prompts(axes[0], structural)
    _panel_rounds(axes[1], structural, cross_model)
    _panel_group_size(axes[2], cross_model)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.27, wspace=0.50)

    out = out_dir / f"structural_vars.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
