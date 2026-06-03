"""Compute confidence intervals and significance tests from experiment results.

Reads metrics.json files from one or more result directories, computes
95% confidence intervals for all metrics, runs significance tests,
and saves everything to files.

Usage:
    python experiments/scripts/statistics.py --results-dir path/to/results1 path/to/results2
    python experiments/scripts/statistics.py --results-dir path/to/results1 path/to/results2 --output-dir path/to/output
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_all_metrics(results_dirs: list[Path]) -> list[dict]:
    metrics = []
    seen = set()
    for results_dir in results_dirs:
        for f in results_dir.rglob("metrics.json"):
            with open(f) as fh:
                m = json.load(fh)
                key = (m["model"], m["condition"])
                if key not in seen:
                    seen.add(key)
                    metrics.append(m)
    return metrics


def deadlock_ci_95(deadlock_rate: float, num_episodes: int) -> float:
    """95% confidence interval for a proportion (Wald interval).

    For a binomial proportion p estimated from n trials:
        CI = 1.96 * sqrt(p * (1-p) / n)
    """
    p = deadlock_rate
    n = num_episodes
    if n == 0:
        return 0.0
    return 1.96 * np.sqrt(p * (1 - p) / n)


def throughput_ci_95(std: float, num_episodes: int) -> float:
    """95% confidence interval from standard deviation and sample size.

    CI = 1.96 * std / sqrt(n)
    """
    if num_episodes < 2:
        return 0.0
    return 1.96 * std / np.sqrt(num_episodes)


def mann_whitney_test(values_a: list[float], values_b: list[float]) -> dict:
    """Two-sided Mann-Whitney U test."""
    if len(values_a) < 2 or len(values_b) < 2:
        return {"u_statistic": None, "p_value": None, "significant": None}

    u_stat, p_val = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
    return {
        "u_statistic": float(u_stat),
        "p_value": float(p_val),
        "significant": bool(p_val < 0.05),
    }


def write_results_table(all_metrics: list[dict], output_path: Path):
    """Write full results table with confidence intervals."""
    rows = []
    for m in sorted(all_metrics, key=lambda x: (x["condition"], x["model"])):
        n = m.get("num_episodes", m.get("episodes", 30))

        dl_rate = m["deadlock_rate"]
        dl_ci = deadlock_ci_95(dl_rate, n)

        tp_mean = m["avg_throughput"]
        tp_std = m.get("std_throughput", 0)
        tp_ci = throughput_ci_95(tp_std, n)

        fr_mean = m["avg_fairness"]
        fr_std = m.get("std_fairness", 0)
        fr_ci = throughput_ci_95(fr_std, n)

        sc_mean = m["avg_starvation_count"]
        sc_std = m.get("std_starvation_count", 0)
        sc_ci = throughput_ci_95(sc_std, n)

        ttd = m.get("avg_time_to_deadlock")

        rows.append({
            "model": m["model"],
            "condition": m["condition"],
            "episodes": n,
            "deadlock_rate": round(dl_rate, 3),
            "deadlock_ci_95": round(dl_ci, 3),
            "deadlock_count": m.get("deadlock_count", 0),
            "avg_throughput": round(tp_mean, 3),
            "throughput_ci_95": round(tp_ci, 3),
            "avg_fairness": round(fr_mean, 3),
            "fairness_ci_95": round(fr_ci, 3),
            "avg_time_to_deadlock": round(ttd, 1) if ttd else "",
            "avg_starvation_count": round(sc_mean, 2),
            "starvation_ci_95": round(sc_ci, 2),
            "avg_timesteps": round(m.get("avg_timesteps", 0), 1),
            "total_llm_calls": m.get("total_llm_calls", 0),
            "avg_latency_ms": round(m.get("avg_latency_ms", 0), 0),
            "total_tokens": m.get("total_tokens", 0),
            "runtime_seconds": round(m.get("runtime_seconds", 0), 0),
        })

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def write_summary_text(rows: list[dict], tests: dict, output_path: Path):
    """Write human-readable summary to a text file."""
    lines = []
    lines.append("DPBench Results Summary")
    lines.append("=" * 80)
    lines.append("")

    conditions = sorted(set(r["condition"] for r in rows))

    for cond in conditions:
        lines.append(f"Condition: {cond}")
        lines.append("-" * 60)
        lines.append(f"  {'Model':<22} {'DL Rate':>8} {'DL CI':>8} {'Throughput':>11} {'TP CI':>8} {'Fairness':>9} {'FR CI':>8} {'TTD':>6} {'Starv':>6}")

        cond_rows = [r for r in rows if r["condition"] == cond]
        for r in sorted(cond_rows, key=lambda x: x["deadlock_rate"], reverse=True):
            ttd_str = f"{r['avg_time_to_deadlock']}" if r["avg_time_to_deadlock"] != "" else "—"
            lines.append(
                f"  {r['model']:<22} "
                f"{r['deadlock_rate']*100:7.1f}% "
                f"+-{r['deadlock_ci_95']*100:5.1f}% "
                f"{r['avg_throughput']:>10.3f} "
                f"+-{r['throughput_ci_95']:>5.3f} "
                f"{r['avg_fairness']:>8.3f} "
                f"+-{r['fairness_ci_95']:>5.3f} "
                f"{ttd_str:>6} "
                f"{r['avg_starvation_count']:>5.1f}"
            )
        lines.append("")

    lines.append("=" * 80)
    lines.append("Significance Tests (Mann-Whitney U, two-sided, alpha=0.05)")
    lines.append("=" * 80)
    lines.append("")

    for name, test in tests.items():
        sig = "YES" if test.get("significant") else "NO"
        p = test.get("p_value")
        u = test.get("u_statistic")
        if p is not None:
            lines.append(f"  {name}:")
            lines.append(f"    U = {u}, p = {p:.6f}, significant = {sig}")
        else:
            lines.append(f"  {name}: insufficient data")
        lines.append("")

    lines.append("=" * 80)
    lines.append("Computational Costs")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  {'Model':<22} {'Condition':<18} {'LLM Calls':>10} {'Avg Latency':>12} {'Total Tokens':>13} {'Runtime':>10}")

    for r in sorted(rows, key=lambda x: (x["model"], x["condition"])):
        if r["total_llm_calls"] > 0:
            lines.append(
                f"  {r['model']:<22} "
                f"{r['condition']:<18} "
                f"{r['total_llm_calls']:>10} "
                f"{r['avg_latency_ms']:>10.0f} ms "
                f"{r['total_tokens']:>13} "
                f"{r['runtime_seconds']:>8.0f}s"
            )
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_significance_tests(all_metrics: list[dict]) -> dict:
    tests = {}

    sim5 = [m for m in all_metrics if m["condition"] == "sim_n5_nocomm" and m["model"] != "random"]
    seq5 = [m for m in all_metrics if m["condition"] == "seq_n5_nocomm" and m["model"] != "random"]
    if sim5 and seq5:
        tests["simultaneous_vs_sequential_n5"] = mann_whitney_test(
            [m["deadlock_rate"] for m in sim5],
            [m["deadlock_rate"] for m in seq5],
        )

    comm = [m for m in all_metrics if m["condition"] == "sim_n5_comm" and m["model"] != "random"]
    nocomm = [m for m in all_metrics if m["condition"] == "sim_n5_nocomm" and m["model"] != "random"]
    if comm and nocomm:
        tests["communication_vs_no_communication_n5"] = mann_whitney_test(
            [m["deadlock_rate"] for m in comm],
            [m["deadlock_rate"] for m in nocomm],
        )

    n5 = [m for m in all_metrics if m["condition"] == "sim_n5_nocomm" and m["model"] != "random"]
    n10 = [m for m in all_metrics if m["condition"] == "sim_n10_nocomm" and m["model"] != "random"]
    if n5 and n10:
        tests["n5_vs_n10_simultaneous"] = mann_whitney_test(
            [m["deadlock_rate"] for m in n5],
            [m["deadlock_rate"] for m in n10],
        )

    return tests


def main():
    parser = argparse.ArgumentParser(description="Compute statistics from experiment results")
    parser.add_argument("--results-dir", nargs="+", required=True, help="One or more result directories")
    parser.add_argument("--output-dir", default=None, help="Where to save output")
    args = parser.parse_args()

    results_dirs = [Path(d) for d in args.results_dir]
    for d in results_dirs:
        if not d.exists():
            print(f"Directory not found: {d}")
            return

    all_metrics = load_all_metrics(results_dirs)
    print(f"Loaded {len(all_metrics)} experiment results from {len(results_dirs)} directories")

    if not all_metrics:
        print("No metrics.json files found.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else results_dirs[0].parent / "combined_statistics"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = write_results_table(all_metrics, output_dir / "results_with_ci.csv")
    print(f"Wrote {output_dir / 'results_with_ci.csv'}")

    tests = run_significance_tests(all_metrics)
    with open(output_dir / "significance_tests.json", "w") as f:
        json.dump(tests, f, indent=2)
    print(f"Wrote {output_dir / 'significance_tests.json'}")

    write_summary_text(rows, tests, output_dir / "summary.txt")
    print(f"Wrote {output_dir / 'summary.txt'}")

    print(f"\n{'=' * 80}")
    print(open(output_dir / "summary.txt").read())


if __name__ == "__main__":
    main()
