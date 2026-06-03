"""Figure 2: cross-model failure pattern with 95% Wilson CI bars.

Six rows (five LLMs and a random baseline). Two markers per row, one for the
simultaneous condition and one for the sequential condition, each accompanied
by a horizontal CI bar so the reader can see overlap directly. Background
shading marks the failure region (deadlock rate >= 0.50) and the success
region (deadlock rate <= 0.25).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from . import _style
from ._data import PER_ANALYSIS, load_csv, find_row, to_float


MODEL_ORDER = [
    "gemini-2.5-flash",
    "llama-4-maverick",
    "grok-4.1",
    "claude-opus-4.5",
    "gpt-5.2",
    "random",
]


def generate(out_dir: Path, fmt: str = "png", dpi: int = 200) -> Path:
    rows = load_csv(PER_ANALYSIS / "cross_model.csv")

    fig, ax = plt.subplots(figsize=(3.3, 3.4))
    _style.trim_spines(ax)

    spacing = 1.0
    y_positions = {model: i * spacing for i, model in enumerate(reversed(MODEL_ORDER))}

    ax.axvspan(0.50, 1.05, color=_style.PALETTE["deadlock"], alpha=0.06, zorder=0)
    ax.axvspan(-0.05, 0.25, color=_style.PALETTE["success"], alpha=0.06, zorder=0)

    drawn_legend = {"sim": False, "seq": False}
    for model in MODEL_ORDER:
        y = y_positions[model]
        color = _style.model_color(model)

        sim_row = find_row(rows, model=model, condition_canonical="sim_n5_nocomm")
        seq_row = find_row(rows, model=model, condition_canonical="seq_n5_nocomm")

        if sim_row is not None:
            p = to_float(sim_row["deadlock_rate"]) or 0.0
            lo = to_float(sim_row["deadlock_ci_low"]) or 0.0
            hi = to_float(sim_row["deadlock_ci_high"]) or 0.0
            ax.hlines(y + 0.18, lo, hi, colors=color, linewidth=1.2, alpha=0.55)
            ax.plot(p, y + 0.18, marker="o", markersize=6, color=color,
                    markeredgecolor=_style.PALETTE["ink"], markeredgewidth=0.4, zorder=4,
                    label="simultaneous" if not drawn_legend["sim"] else None)
            drawn_legend["sim"] = True

        if seq_row is not None:
            p = to_float(seq_row["deadlock_rate"]) or 0.0
            lo = to_float(seq_row["deadlock_ci_low"]) or 0.0
            hi = to_float(seq_row["deadlock_ci_high"]) or 0.0
            ax.hlines(y - 0.18, lo, hi, colors=color, linewidth=1.2, alpha=0.55, linestyles="dashed")
            ax.plot(p, y - 0.18, marker="s", markersize=5.5, color=color,
                    markeredgecolor=_style.PALETTE["ink"], markeredgewidth=0.4, zorder=4,
                    label="sequential" if not drawn_legend["seq"] else None)
            drawn_legend["seq"] = True

    ax.set_yticks([y_positions[m] for m in MODEL_ORDER])
    yticklabels = []
    for m in MODEL_ORDER:
        sim_row = find_row(rows, model=m, condition_canonical="sim_n5_nocomm")
        n = int(sim_row["episodes"]) if sim_row is not None else None
        label = _style.model_display(m)
        if n is not None:
            label = f"{label}  (n={n})"
        yticklabels.append(label)
    ax.set_yticklabels(yticklabels, fontsize=_style.FONT_SIZES["tick"])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.45, max(y_positions.values()) + 0.45)
    _style.percent_axis(ax, axis="x")
    ax.set_xlabel("Deadlock rate (95% Wilson CI)")

    ax.text(0.85, max(y_positions.values()) + 0.30, "failure",
            color=_style.PALETTE["deadlock"], fontsize=_style.FONT_SIZES["annotation"],
            ha="center", style="italic")
    ax.text(0.10, max(y_positions.values()) + 0.30, "success",
            color=_style.PALETTE["success"], fontsize=_style.FONT_SIZES["annotation"],
            ha="center", style="italic")

    legend = ax.legend(loc="lower right", frameon=True, framealpha=0.85,
                       edgecolor="none", facecolor=_style.PALETTE["paper"])
    for text in legend.get_texts():
        text.set_color(_style.PALETTE["ink"])

    _style.vertical_grid(ax)

    out = out_dir / f"cross_model.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
