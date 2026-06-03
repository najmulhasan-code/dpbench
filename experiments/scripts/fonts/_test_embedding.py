"""Render a small figure with Inter and report the embedded font types.

Run from the repository root::

    python experiments/scripts/fonts/_test_embedding.py

The script prints the font information of the produced PDF. Type 3 fonts must
be avoided for publication-quality output; the result must show TrueType
(Type 42) or Type 1 only.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager


REPO_ROOT = Path(__file__).resolve().parents[3]
FONTS_DIR = REPO_ROOT / "experiments" / "scripts" / "fonts"
OUT_PDF = REPO_ROOT / "experiments" / "scripts" / "fonts" / "_embedding_test.pdf"


def main() -> int:
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    inter_paths = sorted(FONTS_DIR.glob("Inter*.ttf"))
    if not inter_paths:
        print("No Inter*.ttf in fonts/ directory.", file=sys.stderr)
        return 1
    for p in inter_paths:
        font_manager.fontManager.addfont(str(p))

    family = "Inter"
    available = {f.name for f in font_manager.fontManager.ttflist}
    if family not in available:
        print(f"Inter not in matplotlib font list after addfont. Available: "
              f"{sorted(n for n in available if 'Inter' in n)}", file=sys.stderr)
        return 1

    matplotlib.rcParams["font.family"] = family

    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.bar(["A", "B", "C"], [1, 2, 3], color="#5B7B97")
    ax.set_title("Inter font test", fontsize=10)
    ax.set_xlabel("Category", fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_PDF}")

    pdffonts = shutil.which("pdffonts")
    if pdffonts:
        result = subprocess.run([pdffonts, str(OUT_PDF)], capture_output=True, text=True)
        print("\npdffonts output:")
        print(result.stdout or "(empty)")
        if "Type 3" in result.stdout:
            print("FAIL: Type 3 font detected.", file=sys.stderr)
            return 1
        return 0

    print("\npdffonts not on PATH; falling back to header inspection.")
    raw = OUT_PDF.read_bytes()
    type3 = raw.count(b"/Type3")
    truetype = raw.count(b"/TrueType")
    type1 = raw.count(b"/Type1")
    print(f"  /Type3 occurrences:    {type3}")
    print(f"  /TrueType occurrences: {truetype}")
    print(f"  /Type1 occurrences:    {type1}")
    if type3 > 0:
        print("FAIL: Type 3 font present.", file=sys.stderr)
        return 1
    if truetype == 0 and type1 == 0:
        print("WARNING: no TrueType or Type1 fonts found; manual verification needed.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
