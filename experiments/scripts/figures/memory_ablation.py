"""Appendix A1: memory ablation null result.

Dot plot with horizontal CIs. All four conditions fall within the 95% Wilson
interval of the baseline; the figure communicates "no detectable effect"
visually rather than as a contribution-grade negative finding.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from . import _style
from ._data import PER_ANALYSIS, load_csv, find_row, to_float


ORDER = ["sim_n5_nocomm", "memory_3", "memory_5", "memory_3_with_comm"]
DISPLAY = {
    "sim_n5_nocomm": "baseline",
    "memory_3": "memory window = 3",
    "memory_5": "memory window = 5",
    "memory_3_with_comm": "memory + comm",
}


def generate(out_dir: Path, fmt: str = "png", dpi: int = 200) -> Path:
    rows = load_csv(PER_ANALYSIS / "memory_ablation.csv")

    points = []
    for cond in ORDER:
        row = find_row(rows, condition_canonical=cond)
        if row is None:
            continue
        points.append({
            "label": DISPLAY[cond],
            "p": to_float(row["deadlock_rate"]) or 0.0,
            "lo": to_float(row["deadlock_ci_low"]) or 0.0,
            "hi": to_float(row["deadlock_ci_high"]) or 0.0,
        })

    baseline_p = points[0]["p"] if points else None

    fig, ax = plt.subplots(figsize=(3.0, 1.7))
    y_positions = list(range(len(points)))[::-1]

    for y, point in zip(y_positions, points):
        ax.hlines(y, point["lo"], point["hi"], colors=_style.PALETTE["baseline"],
                  linewidth=1.2, alpha=0.6)
        ax.plot(point["p"], y, marker="o", markersize=6,
                color=_style.PALETTE["baseline"],
                markeredgecolor=_style.PALETTE["ink"], markeredgewidth=0.4, zorder=4)

    if baseline_p is not None:
        ax.axvline(baseline_p, color=_style.PALETTE["neutral"], linestyle="--",
                   linewidth=_style.LINEWIDTHS["reference"], alpha=0.7, zorder=2)
        ax.text(baseline_p + 0.01, max(y_positions) + 0.30, "baseline mean",
                fontsize=_style.FONT_SIZES["annotation"],
                color=_style.PALETTE["neutral"], style="italic", va="center")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([p["label"] for p in points])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.45, max(y_positions) + 0.6)
    _style.percent_axis(ax, axis="x")
    ax.set_xlabel("Deadlock rate (95% Wilson CI)")
    _style.trim_spines(ax)
    _style.vertical_grid(ax)

    out = out_dir / f"memory_ablation.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
