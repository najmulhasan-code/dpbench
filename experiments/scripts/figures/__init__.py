"""DPBench paper figures.

Each module exposes a ``generate(out_dir, fmt="png", dpi=200) -> Path`` function.
The orchestrator in ``experiments/scripts/generate_figures.py`` calls each
module and writes outputs to ``experiments/figures/`` for codebase preview;
when the paper is being written the same modules can render PDFs into the
Overleaf-cloned paper folder.
"""
