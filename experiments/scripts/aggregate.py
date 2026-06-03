"""Aggregate experimental results into a single canonical schema.

Walks every ``metrics.json`` and ``log.jsonl`` under ``experiments/results/``,
normalises the v0.1.0 and v0.2.0 schemas to a common form, computes 95%
confidence intervals (Wilson for binomial proportions, t-based for continuous
metrics), and writes a master CSV plus per-analysis subset CSVs and one
per-episode CSV for every (model, condition) cell.

Usage:
    python experiments/scripts/aggregate.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import t as student_t


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "experiments" / "results"
OUTPUT_ROOT = RESULTS_ROOT / "aggregated"

SOURCE_DIRS = [
    "v0.1.0_full_20260126_213124",
    "v0.2.0_20260330_100109",
    "v0.2.0_20260330_100459",
    "v0.2.0_20260330_175114",
    "v0.2.0_20260403_180607",
    "v0.2.0_20260406_200140",
]

V01_CONDITION_PATTERN = re.compile(r"^(sim|seq)(\d+)(nc|c)$")

PROMPT_VARIANT_PREFIX = "prompt_"
COMM_ROUNDS_PREFIX = "comm_rounds_"
MEMORY_PREFIX = "memory_"

MASTER_COLUMNS = [
    "source_dir",
    "model",
    "condition_canonical",
    "condition_original",
    "prompt_variant",
    "mode",
    "communication",
    "philosophers",
    "episodes",
    "max_timesteps",
    "temperature",
    "memory_window",
    "communication_rounds",
    "deadlock_terminal",
    "deadlock_rate",
    "deadlock_count",
    "deadlock_ci_low",
    "deadlock_ci_high",
    "avg_throughput",
    "std_throughput",
    "throughput_ci_half",
    "avg_fairness",
    "std_fairness",
    "fairness_ci_half",
    "avg_starvation_count",
    "avg_time_to_deadlock",
    "avg_timesteps",
    "total_deadlock_events",
    "message_action_consistency",
    "total_llm_calls",
    "avg_latency_ms",
    "total_tokens_in",
    "total_tokens_out",
    "total_tokens",
    "runtime_seconds",
    "dpbench_version",
]

EPISODE_COLUMNS = [
    "episode_id",
    "total_timesteps",
    "deadlock",
    "total_meals",
    "throughput",
    "fairness",
    "meals_per_philosopher",
]


def wilson_interval(deadlock_count: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion.

    Reduces to (0, 0) for n <= 0 to keep downstream CSVs writable. The Wald
    interval collapses at p=0 and p=1, which is why the Wilson form is used.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = deadlock_count / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def t_based_half_width(std: float | None, n: int, alpha: float = 0.05) -> float | None:
    """Half-width of the 95% CI for a sample mean, using the t critical value."""
    if std is None or n <= 1:
        return None
    t_crit = student_t.ppf(1.0 - alpha / 2.0, df=n - 1)
    return float(t_crit * std / math.sqrt(n))


def canonical_condition(condition: str) -> str:
    """Map v0.1.0 short codes to v0.2.0 long codes; pass through everything else."""
    match = V01_CONDITION_PATTERN.match(condition)
    if not match:
        return condition
    mode, n, comm_flag = match.groups()
    comm = "comm" if comm_flag == "c" else "nocomm"
    return f"{mode}_n{n}_{comm}"


def derive_prompt_variant(condition: str) -> str:
    if condition.startswith(PROMPT_VARIANT_PREFIX):
        return condition[len(PROMPT_VARIANT_PREFIX):]
    return "default"


def derive_memory_window(condition: str) -> int:
    if condition.startswith(MEMORY_PREFIX):
        rest = condition[len(MEMORY_PREFIX):]
        first = rest.split("_", 1)[0]
        try:
            return int(first)
        except ValueError:
            return 0
    return 0


def derive_communication_rounds(condition: str, communication_enabled: bool) -> int:
    if condition.startswith(COMM_ROUNDS_PREFIX):
        try:
            return int(condition[len(COMM_ROUNDS_PREFIX):])
        except ValueError:
            return 1 if communication_enabled else 0
    if "with_comm" in condition:
        return 1
    return 1 if communication_enabled else 0


def derive_deadlock_terminal(condition: str) -> bool:
    return "nonterminal" not in condition


def find_metrics_files(source_dir: Path) -> list[Path]:
    return sorted(source_dir.rglob("metrics.json"))


def normalise_metrics(raw: dict[str, Any], source_dir: str, metrics_path: Path) -> dict[str, Any]:
    """Flatten a v0.1.0 or v0.2.0 ``metrics.json`` into the canonical schema.

    Some keys were renamed between versions and several v0.2.0 ablation runs
    omit fields entirely; the function returns ``None`` for any missing value
    so the master CSV ends up rectangular. The metrics file path is retained
    in a leading-underscore field so per-episode reconstruction can locate the
    sibling ``log.jsonl`` without re-globbing.
    """
    condition_original = raw["condition"]
    condition_can = canonical_condition(condition_original)

    n = raw.get("num_episodes") or raw.get("episodes")
    deadlock_rate = raw.get("deadlock_rate", 0.0)
    deadlock_count = raw.get("deadlock_count")
    if deadlock_count is None and n is not None:
        deadlock_count = round(deadlock_rate * n)

    ci_low, ci_high = wilson_interval(int(deadlock_count or 0), int(n or 0))

    std_throughput = raw.get("std_throughput")
    std_fairness = raw.get("std_fairness")

    communication = bool(raw.get("communication", False))

    return {
        "_metrics_path": str(metrics_path),
        "source_dir": source_dir,
        "model": raw.get("model"),
        "condition_canonical": condition_can,
        "condition_original": condition_original,
        "prompt_variant": derive_prompt_variant(condition_original),
        "mode": raw.get("mode"),
        "communication": communication,
        "philosophers": raw.get("philosophers"),
        "episodes": n,
        "max_timesteps": raw.get("max_timesteps"),
        "temperature": raw.get("temperature"),
        "memory_window": derive_memory_window(condition_original),
        "communication_rounds": derive_communication_rounds(condition_original, communication),
        "deadlock_terminal": derive_deadlock_terminal(condition_original),
        "deadlock_rate": deadlock_rate,
        "deadlock_count": deadlock_count,
        "deadlock_ci_low": ci_low,
        "deadlock_ci_high": ci_high,
        "avg_throughput": raw.get("avg_throughput"),
        "std_throughput": std_throughput,
        "throughput_ci_half": t_based_half_width(std_throughput, int(n or 0)),
        "avg_fairness": raw.get("avg_fairness"),
        "std_fairness": std_fairness,
        "fairness_ci_half": t_based_half_width(std_fairness, int(n or 0)),
        "avg_starvation_count": raw.get("avg_starvation_count"),
        "avg_time_to_deadlock": raw.get("avg_time_to_deadlock"),
        "avg_timesteps": raw.get("avg_timesteps"),
        "total_deadlock_events": raw.get("total_deadlock_events"),
        "message_action_consistency": raw.get("message_action_consistency"),
        "total_llm_calls": raw.get("total_llm_calls"),
        "avg_latency_ms": raw.get("avg_latency_ms"),
        "total_tokens_in": raw.get("total_tokens_in"),
        "total_tokens_out": raw.get("total_tokens_out"),
        "total_tokens": raw.get("total_tokens"),
        "runtime_seconds": raw.get("runtime_seconds"),
        "dpbench_version": raw.get("dpbench_version", "0.1.0"),
    }


def load_master_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name in SOURCE_DIRS:
        source_dir = RESULTS_ROOT / source_name
        if not source_dir.is_dir():
            continue
        for metrics_path in find_metrics_files(source_dir):
            with metrics_path.open(encoding="utf-8") as f:
                raw = json.load(f)
            rows.append(normalise_metrics(raw, source_name, metrics_path))
    rows.sort(key=lambda r: (r["source_dir"], r["model"] or "", r["condition_canonical"]))
    return rows


def write_master_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in MASTER_COLUMNS})


def filter_rows(rows: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if all(row.get(k) == v for k, v in filters.items()):
            out.append(row)
    return out


def write_per_analysis(rows: list[dict[str, Any]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    cross_model_conditions = {"sim_n5_nocomm", "sim_n5_comm", "seq_n5_nocomm", "sim_n10_nocomm"}
    cross_model = [r for r in rows if r["condition_canonical"] in cross_model_conditions]
    write_subset(cross_model, directory / "cross_model.csv")

    structural_conditions = {
        "sim_n5_nocomm",
        "sim_n5_comm",
        "prompt_minimal",
        "prompt_resource_ordering",
        "prompt_symmetry_breaking",
        "prompt_theory_of_mind",
        "comm_rounds_3",
        "comm_rounds_5",
        "sim_n10_nocomm",
    }
    structural = [
        r for r in rows
        if r["condition_canonical"] in structural_conditions
        and (r["model"] == "gemini-2.5-flash" or r["condition_canonical"] in {"sim_n5_nocomm", "sim_n10_nocomm"})
    ]
    write_subset(structural, directory / "structural_vars.csv")

    memory_conditions = {"sim_n5_nocomm", "memory_3", "memory_5", "memory_3_with_comm"}
    memory = [
        r for r in rows
        if r["model"] == "gemini-2.5-flash" and r["condition_canonical"] in memory_conditions
    ]
    write_subset(memory, directory / "memory_ablation.csv")

    main_table_conditions = {
        "sim_n5_nocomm",
        "sim_n5_comm",
        "seq_n5_nocomm",
        "sim_n10_nocomm",
    }
    main_table = [r for r in rows if r["condition_canonical"] in main_table_conditions]
    write_subset(main_table, directory / "main_results.csv")


def write_subset(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in MASTER_COLUMNS})


def reconstruct_episodes(log_path: Path) -> list[dict[str, Any]]:
    """Read ``episode_end`` records from a ``log.jsonl`` and emit one row each."""
    episodes: list[dict[str, Any]] = []
    with log_path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "episode_end":
                continue
            meals = obj.get("meals_per_philosopher") or []
            episodes.append({
                "episode_id": obj.get("episode_id"),
                "total_timesteps": obj.get("total_timesteps"),
                "deadlock": bool(obj.get("deadlock", False)),
                "total_meals": obj.get("total_meals"),
                "throughput": obj.get("throughput"),
                "fairness": obj.get("fairness"),
                "meals_per_philosopher": ";".join(str(m) for m in meals),
            })
    episodes.sort(key=lambda e: (e["episode_id"] is None, e["episode_id"]))
    return episodes


def write_per_episode(rows: list[dict[str, Any]], directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in rows:
        metrics_path = Path(row["_metrics_path"])
        log_path = metrics_path.parent / "log.jsonl"
        if not log_path.exists():
            continue
        episodes = reconstruct_episodes(log_path)
        if not episodes:
            continue
        safe_model = (row["model"] or "unknown").replace("/", "_")
        out_path = directory / f"{row['source_dir']}__{safe_model}__{row['condition_original']}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPISODE_COLUMNS)
            writer.writeheader()
            for ep in episodes:
                writer.writerow(ep)
        written += 1
    return written


def write_manifest(rows: list[dict[str, Any]], episodes_written: int, path: Path) -> None:
    sources = sorted({r["source_dir"] for r in rows})
    models = sorted({r["model"] for r in rows if r["model"]})
    conditions = sorted({r["condition_canonical"] for r in rows})

    lines = [
        f"generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"schema_version: 1",
        f"total_cells: {len(rows)}",
        f"per_episode_files: {episodes_written}",
        "",
        "source_directories:",
    ]
    for src in sources:
        n_rows = sum(1 for r in rows if r["source_dir"] == src)
        lines.append(f"  - name: {src}")
        lines.append(f"    cells: {n_rows}")
    lines.append("")
    lines.append("models:")
    for m in models:
        n_rows = sum(1 for r in rows if r["model"] == m)
        lines.append(f"  - name: {m}")
        lines.append(f"    cells: {n_rows}")
    lines.append("")
    lines.append("canonical_conditions:")
    for c in conditions:
        n_rows = sum(1 for r in rows if r["condition_canonical"] == c)
        lines.append(f"  - name: {c}")
        lines.append(f"    cells: {n_rows}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(rows: list[dict[str, Any]], path: Path) -> None:
    body = [
        "# DPBench aggregated results",
        "",
        f"Total cells: {len(rows)}",
        "",
        "## Files",
        "",
        "- `master.csv` — every (model, condition) cell with confidence intervals.",
        "- `per_analysis/` — filtered views aligned to paper figures and tables.",
        "- `per_episode/` — one CSV per cell with per-episode outcomes.",
        "- `manifest.yaml` — generation metadata and counts.",
        "",
        "## Confidence intervals",
        "",
        "- Deadlock rate: Wilson score 95% interval.",
        "- Throughput, fairness: t-based 95% interval (`t.ppf(0.975, df=n-1) * std / sqrt(n)`).",
        "- Time-to-deadlock: not interval-estimated (small subsample).",
        "",
        "## Schema",
        "",
        "Master CSV columns: " + ", ".join(MASTER_COLUMNS) + ".",
        "",
        "## Regeneration",
        "",
        "Run `python experiments/scripts/aggregate.py` from the repository root.",
    ]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def main() -> int:
    rows = load_master_rows()
    if not rows:
        print("No metrics.json files found under experiments/results/", file=sys.stderr)
        return 1

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_master_csv(rows, OUTPUT_ROOT / "master.csv")
    write_per_analysis(rows, OUTPUT_ROOT / "per_analysis")
    episodes_written = write_per_episode(rows, OUTPUT_ROOT / "per_episode")
    write_manifest(rows, episodes_written, OUTPUT_ROOT / "manifest.yaml")
    write_readme(rows, OUTPUT_ROOT / "README.md")

    print(f"Wrote {len(rows)} master rows to {OUTPUT_ROOT / 'master.csv'}")
    print(f"Wrote {episodes_written} per-episode CSVs to {OUTPUT_ROOT / 'per_episode'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
