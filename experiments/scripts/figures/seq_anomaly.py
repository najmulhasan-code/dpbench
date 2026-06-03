"""Sequential anomaly investigation figure.

Two-panel custom figure surfacing the qualitative difference between Claude
Opus 4.5 (60% deadlock in sequential mode) and Gemini 2.5 Flash (0%):
    (a) Action distribution per model across all timesteps.
    (b) Per-episode outcome marker for every episode.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import _style
from ._data import PER_ANALYSIS, load_csv


MODELS = ["claude-opus-4.5", "gemini-2.5-flash"]
ACTION_ORDER = ["grab_left", "grab_right", "release", "wait"]
ACTION_DISPLAY = {"grab_left": "GRAB_LEFT", "grab_right": "GRAB_RIGHT",
                  "release": "RELEASE", "wait": "WAIT"}


def generate(out_dir: Path, fmt: str = "png", dpi: int = 200) -> Path:
    rows = load_csv(PER_ANALYSIS / "seq_anomaly.csv")
    actions = [r for r in rows if r["row_type"] == "action"]
    episodes = [r for r in rows if r["row_type"] == "episode"]

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3),
                              gridspec_kw={"width_ratios": [1.05, 1.4]})

    ax_actions = axes[0]
    width = 0.36
    x = np.arange(len(ACTION_ORDER))
    for offset, model in zip([-width / 2, width / 2], MODELS):
        shares = []
        for action in ACTION_ORDER:
            row = next((r for r in actions
                        if r["model"] == model and r["action_label"] == action), None)
            shares.append(float(row["action_share"]) if row else 0.0)
        ax_actions.bar(x + offset, shares, width=width,
                       color=_style.model_color(model),
                       edgecolor=_style.PALETTE["ink"], linewidth=0.4,
                       label=_style.model_display(model), zorder=3)

    ax_actions.set_xticks(x)
    ax_actions.set_xticklabels([ACTION_DISPLAY[a] for a in ACTION_ORDER],
                                rotation=20, ha="right",
                                fontsize=_style.FONT_SIZES["tick"])
    ax_actions.set_ylim(0, 0.45)
    ax_actions.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v * 100)}%"))
    ax_actions.set_ylabel("Share of actions")
    ax_actions.set_title("(a) Action distribution",
                          fontsize=_style.FONT_SIZES["title"], pad=6)
    _style.trim_spines(ax_actions)
    _style.horizontal_grid(ax_actions)
    legend = ax_actions.legend(loc="upper right", frameon=True, framealpha=0.85,
                                edgecolor="none", facecolor=_style.PALETTE["paper"])
    for text in legend.get_texts():
        text.set_color(_style.PALETTE["ink"])

    ax_eps = axes[1]
    for row_idx, model in enumerate(MODELS):
        model_eps = sorted(
            (e for e in episodes if e["model"] == model),
            key=lambda r: int(r["episode_id"]) if r["episode_id"] else 0,
        )
        y = (len(MODELS) - 1 - row_idx)
        for ep_idx, ep in enumerate(model_eps):
            color = (_style.PALETTE["deadlock"] if int(ep["deadlock"])
                     else _style.PALETTE["success"])
            ax_eps.plot(ep_idx, y, marker="o", markersize=7, color=color,
                        markeredgecolor=_style.PALETTE["ink"], markeredgewidth=0.4)

    ax_eps.set_yticks(list(range(len(MODELS))))
    ax_eps.set_yticklabels(
        [_style.model_display(MODELS[len(MODELS) - 1 - i]) for i in range(len(MODELS))]
    )
    ax_eps.set_xlim(-0.6, max(len(episodes) // len(MODELS), 30) - 0.4)
    ax_eps.set_ylim(-0.7, len(MODELS) - 0.3)
    ax_eps.set_xlabel("Episode index")
    ax_eps.set_title("(b) Per-episode outcomes",
                      fontsize=_style.FONT_SIZES["title"], pad=6)
    _style.trim_spines(ax_eps)
    ax_eps.tick_params(left=True, bottom=True)
    ax_eps.set_xticks(np.arange(0, 31, 5))

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_style.PALETTE["deadlock"],
                   markersize=7, label="deadlock",
                   markeredgecolor=_style.PALETTE["ink"], markeredgewidth=0.4),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_style.PALETTE["success"],
                   markersize=7, label="completed",
                   markeredgecolor=_style.PALETTE["ink"], markeredgewidth=0.4),
    ]
    legend = ax_eps.legend(handles=legend_handles, loc="lower right", frameon=True,
                            framealpha=0.85, edgecolor="none",
                            facecolor=_style.PALETTE["paper"])
    for text in legend.get_texts():
        text.set_color(_style.PALETTE["ink"])

    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.20, wspace=0.30)

    out = out_dir / f"seq_anomaly.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
