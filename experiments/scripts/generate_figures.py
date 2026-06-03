"""Orchestrate generation of every paper figure.

Each figure module under ``experiments/scripts/figures/`` exposes a
``generate(out_dir, fmt, dpi) -> Path`` function. This script calls them in
sequence and writes the outputs to ``experiments/figures/``.

Usage::

    python experiments/scripts/generate_figures.py            # writes PNG
    python experiments/scripts/generate_figures.py --pdf      # writes PDF
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "figures"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures import _style  # noqa: E402
from figures import (  # noqa: E402
    teaser,
    cross_model,
    structural_vars,
    memory_ablation,
    seq_anomaly,
)


MODULES = [
    teaser,
    cross_model,
    structural_vars,
    memory_ablation,
    seq_anomaly,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory (default: experiments/figures/)")
    parser.add_argument("--pdf", action="store_true",
                        help="Render PDF instead of PNG (paper-time output)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Raster DPI for PNG output (default: 200)")
    args = parser.parse_args()

    fmt = "pdf" if args.pdf else "png"
    args.out.mkdir(parents=True, exist_ok=True)

    _style.setup_style()

    for module in MODULES:
        path = module.generate(args.out, fmt=fmt, dpi=args.dpi)
        size = path.stat().st_size if path.exists() else 0
        print(f"  {path.relative_to(REPO_ROOT)}  ({size} bytes)")
    print(f"Wrote {len(MODULES)} figures to {args.out.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
