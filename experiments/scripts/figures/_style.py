"""Shared visual design system for the DPBench paper figures.

All figure scripts import constants and helpers from this module so that any
change to the palette, typography, or layout discipline propagates to every
figure consistently.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager


SCRIPTS_DIR = Path(__file__).resolve().parent.parent
FONTS_DIR = SCRIPTS_DIR / "fonts"
REPO_ROOT = SCRIPTS_DIR.parent.parent

PALETTE = {
    "deadlock": "#C9442B",
    "success": "#2D6A47",
    "baseline": "#5B7B97",
    "highlight": "#D4A55B",
    "neutral": "#7B8794",
    "random": "#A89F91",
    "ink": "#1F2937",
    "paper": "#FFFFFF",
}

MODEL_COLORS = {
    "GPT-5.2": "#3D5A80",
    "gpt-5.2": "#3D5A80",
    "Claude Opus 4.5": "#C77B58",
    "claude-opus-4.5": "#C77B58",
    "Grok 4.1": "#5C7457",
    "grok-4.1": "#5C7457",
    "Gemini 2.5 Flash": "#7B5C8A",
    "gemini-2.5-flash": "#7B5C8A",
    "Llama 4 Maverick": "#9D7A38",
    "llama-4-maverick": "#9D7A38",
    "Random": "#A89F91",
    "random": "#A89F91",
}

MODEL_DISPLAY_NAMES = {
    "gpt-5.2": "GPT-5.2",
    "claude-opus-4.5": "Claude Opus 4.5",
    "grok-4.1": "Grok 4.1",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "llama-4-maverick": "Llama 4 Maverick",
    "random": "Random",
}

FONT_SIZES = {
    "title": 10,
    "axis_label": 9,
    "tick": 8,
    "annotation": 7.5,
    "legend": 8,
}

LINEWIDTHS = {
    "spine": 0.6,
    "tick": 0.6,
    "grid": 0.5,
    "ci_cap": 1.0,
    "reference": 0.8,
}


def setup_style() -> str:
    """Configure matplotlib globally for the paper. Returns the active font family."""
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    inter_files = sorted(FONTS_DIR.glob("Inter*.ttf"))
    for path in inter_files:
        font_manager.fontManager.addfont(str(path))

    available = {f.name for f in font_manager.fontManager.ttflist}
    if "Inter" in available:
        family = "Inter"
    elif "STIXGeneral" in available:
        family = "STIXGeneral"
    else:
        family = "DejaVu Sans"

    matplotlib.rcParams["font.family"] = family
    matplotlib.rcParams["font.size"] = FONT_SIZES["tick"]
    matplotlib.rcParams["axes.titlesize"] = FONT_SIZES["title"]
    matplotlib.rcParams["axes.labelsize"] = FONT_SIZES["axis_label"]
    matplotlib.rcParams["xtick.labelsize"] = FONT_SIZES["tick"]
    matplotlib.rcParams["ytick.labelsize"] = FONT_SIZES["tick"]
    matplotlib.rcParams["legend.fontsize"] = FONT_SIZES["legend"]

    matplotlib.rcParams["axes.edgecolor"] = PALETTE["ink"]
    matplotlib.rcParams["axes.labelcolor"] = PALETTE["ink"]
    matplotlib.rcParams["xtick.color"] = PALETTE["ink"]
    matplotlib.rcParams["ytick.color"] = PALETTE["ink"]
    matplotlib.rcParams["text.color"] = PALETTE["ink"]
    matplotlib.rcParams["axes.linewidth"] = LINEWIDTHS["spine"]
    matplotlib.rcParams["xtick.major.width"] = LINEWIDTHS["tick"]
    matplotlib.rcParams["ytick.major.width"] = LINEWIDTHS["tick"]
    matplotlib.rcParams["xtick.major.size"] = 3
    matplotlib.rcParams["ytick.major.size"] = 3
    matplotlib.rcParams["xtick.direction"] = "out"
    matplotlib.rcParams["ytick.direction"] = "out"

    matplotlib.rcParams["figure.facecolor"] = PALETTE["paper"]
    matplotlib.rcParams["axes.facecolor"] = PALETTE["paper"]
    matplotlib.rcParams["savefig.facecolor"] = PALETTE["paper"]
    matplotlib.rcParams["savefig.edgecolor"] = "none"
    matplotlib.rcParams["savefig.bbox"] = "tight"
    matplotlib.rcParams["savefig.pad_inches"] = 0.02

    return family


def trim_spines(ax: plt.Axes, keep: tuple[str, ...] = ("left", "bottom")) -> None:
    """Remove top and right spines; leave left and bottom as the default."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def horizontal_grid(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(
        True,
        color=PALETTE["neutral"],
        alpha=0.30,
        linewidth=LINEWIDTHS["grid"],
        linestyle="-",
    )
    ax.xaxis.grid(False)


def vertical_grid(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.xaxis.grid(
        True,
        color=PALETTE["neutral"],
        alpha=0.30,
        linewidth=LINEWIDTHS["grid"],
        linestyle="-",
    )
    ax.yaxis.grid(False)


def percent_axis(ax: plt.Axes, axis: str = "y") -> None:
    """Format an axis as percentages assuming values are in [0, 1]."""
    from matplotlib.ticker import PercentFormatter

    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def model_color(model: str) -> str:
    return MODEL_COLORS.get(model, PALETTE["neutral"])


def model_display(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def deadlock_color_for(rate: float) -> str:
    """Return the band color a bar of the given deadlock rate should take."""
    if rate >= 0.50:
        return PALETTE["deadlock"]
    if rate <= 0.10:
        return PALETTE["success"]
    return PALETTE["baseline"]
