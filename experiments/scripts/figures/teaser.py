"""Figure 1: schematic + headline result (teaser).

Two panels: the dining philosophers deadlock state on the left, and the bar
chart showing that a single structural intervention (three pre-action
discussion rounds) takes deadlock from 90.0% to 0.0% on the right. Both panels
use the same Gemini 2.5 Flash data summarised elsewhere in the paper; this
figure exists to communicate the research contribution at a glance.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from . import _style
from ._data import PER_ANALYSIS, load_csv, find_row, to_float


N_PHIL = 5
TABLE_RADIUS = 1.45
PHIL_RADIUS = 0.30
FORK_LEN = 0.42
FORK_WIDTH = 0.12
HOLDS_RADIUS = 0.95


def _phil_angle(i: int) -> float:
    return math.pi / 2 - i * 2 * math.pi / N_PHIL


def _fork_angle(i: int) -> float:
    return _phil_angle(i) - math.pi / N_PHIL


def _draw_panel_a(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-2.05, 1.80)
    ax.axis("off")

    phil_centers = []
    for i in range(N_PHIL):
        angle = _phil_angle(i)
        x = TABLE_RADIUS * math.cos(angle)
        y = TABLE_RADIUS * math.sin(angle)
        phil_centers.append((x, y, angle))
        circle = Circle((x, y), PHIL_RADIUS, facecolor=_style.PALETTE["baseline"],
                        edgecolor=_style.PALETTE["ink"], linewidth=0.8, alpha=0.9, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, f"$P_{i}$", ha="center", va="center",
                fontsize=_style.FONT_SIZES["axis_label"],
                color=_style.PALETTE["paper"], fontweight="bold", zorder=4)

    fork_centers = []
    for i in range(N_PHIL):
        angle = _fork_angle(i)
        x = HOLDS_RADIUS * math.cos(angle)
        y = HOLDS_RADIUS * math.sin(angle)
        rotation = math.degrees(angle) - 90
        rect = Rectangle(
            (x - FORK_WIDTH / 2, y - FORK_LEN / 2),
            FORK_WIDTH, FORK_LEN,
            facecolor=_style.PALETTE["highlight"],
            edgecolor=_style.PALETTE["ink"],
            linewidth=0.6, zorder=3,
        )
        transform = (
            plt.matplotlib.transforms.Affine2D()
            .rotate_deg_around(x, y, rotation)
            + ax.transData
        )
        rect.set_transform(transform)
        ax.add_patch(rect)
        fork_centers.append((x, y))

    for i in range(N_PHIL):
        px, py, _ = phil_centers[i]
        fx, fy = fork_centers[i]
        ax.plot([px, fx], [py, fy],
                color=_style.PALETTE["ink"],
                linewidth=1.6, zorder=2, solid_capstyle="round", alpha=0.85)

    for i in range(N_PHIL):
        px, py, _ = phil_centers[i]
        target_idx = (i - 1) % N_PHIL
        fx, fy = fork_centers[target_idx]
        arrow = FancyArrowPatch(
            (px, py), (fx, fy),
            connectionstyle="arc3,rad=0.32",
            arrowstyle="-|>",
            mutation_scale=8,
            linestyle=(0, (3, 2)),
            color=_style.PALETTE["deadlock"],
            linewidth=1.0,
            alpha=0.85,
            zorder=2,
        )
        ax.add_patch(arrow)

    legend_y = -1.90
    ax.plot([-1.30, -1.00], [legend_y, legend_y],
            color=_style.PALETTE["ink"], linewidth=1.6, alpha=0.85,
            solid_capstyle="round")
    ax.text(-0.95, legend_y, "holds", ha="left", va="center",
            fontsize=_style.FONT_SIZES["annotation"], color=_style.PALETTE["ink"])
    ax.annotate("", xy=(0.30, legend_y), xytext=(0.00, legend_y),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=7,
                                color=_style.PALETTE["deadlock"],
                                linewidth=0.9, linestyle=(0, (3, 2))))
    ax.text(0.35, legend_y, "needs", ha="left", va="center",
            fontsize=_style.FONT_SIZES["annotation"], color=_style.PALETTE["ink"])


def _draw_panel_b(ax: plt.Axes) -> None:
    rows = load_csv(PER_ANALYSIS / "structural_vars.csv")
    baseline = find_row(rows, model="gemini-2.5-flash",
                       condition_canonical="sim_n5_nocomm")
    intervention = find_row(rows, model="gemini-2.5-flash",
                           condition_canonical="comm_rounds_3")

    points = []
    for label, row, color in (
        ("default\n(0 rounds)", baseline, _style.PALETTE["deadlock"]),
        ("3 rounds", intervention, _style.PALETTE["success"]),
    ):
        p = to_float(row["deadlock_rate"]) or 0.0
        lo = to_float(row["deadlock_ci_low"]) or 0.0
        hi = to_float(row["deadlock_ci_high"]) or 0.0
        points.append({"label": label, "p": p, "lo": lo, "hi": hi, "color": color})

    x = np.arange(len(points))
    values = [p["p"] for p in points]
    lows = [max(0.0, p["p"] - p["lo"]) for p in points]
    highs = [max(0.0, p["hi"] - p["p"]) for p in points]
    colors = [p["color"] for p in points]

    ax.bar(x, values, color=colors, edgecolor=_style.PALETTE["ink"],
           linewidth=0.5, width=0.55, zorder=3)
    ax.errorbar(x, values, yerr=[lows, highs], fmt="none",
                ecolor=_style.PALETTE["ink"], elinewidth=1.0,
                capsize=4, alpha=0.8, zorder=4)

    for xi, point in zip(x, points):
        if point["p"] >= 0.5:
            text_y = point["p"] - 0.04
            color = _style.PALETTE["paper"]
            va = "top"
        else:
            text_y = max(point["hi"], point["p"]) + 0.05
            color = _style.PALETTE["ink"]
            va = "bottom"
        ax.text(xi, text_y, f"{point['p'] * 100:.1f}%",
                ha="center", va=va,
                fontsize=_style.FONT_SIZES["annotation"],
                color=color, fontweight="bold", zorder=5)

    arrow = FancyArrowPatch(
        (0.30, 0.93), (0.738, 0.30),
        connectionstyle="arc3,rad=-0.30",
        arrowstyle="-|>", mutation_scale=10,
        color=_style.PALETTE["highlight"], linewidth=1.4,
        alpha=0.9, zorder=4,
        transform=ax.transAxes,
    )
    ax.add_patch(arrow)
    ax.text(0.78, 0.78, "structural\nintervention",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=_style.FONT_SIZES["annotation"], fontstyle="italic",
            color=_style.PALETTE["highlight"], fontweight="bold", zorder=5)

    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks(x)
    ax.set_xticklabels([p["label"] for p in points],
                       fontsize=_style.FONT_SIZES["tick"])
    sample_sizes = [int(baseline["episodes"]), int(intervention["episodes"])]
    for xi, n in zip(x, sample_sizes):
        ax.text(xi, -0.18, f"n = {n}",
                ha="center", va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=_style.FONT_SIZES["annotation"],
                color=_style.PALETTE["neutral"], fontstyle="italic")
    ax.set_ylim(0, 1.05)
    _style.percent_axis(ax, axis="y")
    ax.set_ylabel("Deadlock rate")
    _style.trim_spines(ax)
    _style.horizontal_grid(ax)


def generate(out_dir: Path, fmt: str = "png", dpi: int = 200) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4),
                              gridspec_kw={"width_ratios": [1.0, 1.05]})
    _draw_panel_a(axes[0])
    _draw_panel_b(axes[1])

    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.18, wspace=0.18)

    out = out_dir / f"teaser.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
