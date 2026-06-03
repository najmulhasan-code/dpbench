"""Extract per-action distributions and per-episode outcomes for the
Claude / Grok sequential anomaly. Output feeds Figure A3.

Reads ``log.jsonl`` files for the v0.1.0 sequential conditions and writes
``experiments/results/aggregated/per_analysis/seq_anomaly.csv``.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "experiments" / "results"
OUTPUT_PATH = RESULTS_ROOT / "aggregated" / "per_analysis" / "seq_anomaly.csv"

CONDITION = "seq5nc"

CASES = [
    ("claude-opus-4.5", "Claude Opus 4.5"),
    ("gemini-2.5-flash", "Gemini 2.5 Flash"),
]

SEQUENCES = [
    ("v0.1.0_full_20260126_213124", "claude-opus-4.5_seq5nc"),
    ("v0.2.0_20260330_100459", "gemini-2.5-flash/seq_n5_nocomm"),
]

ACTIONS = ["grab_left", "grab_right", "release", "wait"]


def collect(log_path: Path) -> tuple[Counter[str], list[dict[str, object]]]:
    actions: Counter[str] = Counter()
    episode_outcomes: list[dict[str, object]] = []
    with log_path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "timestep":
                for d in obj.get("decisions", []):
                    action = d.get("action")
                    if action:
                        actions[action] += 1
            elif obj.get("type") == "episode_end":
                episode_outcomes.append({
                    "episode_id": obj.get("episode_id"),
                    "deadlock": bool(obj.get("deadlock", False)),
                    "total_timesteps": obj.get("total_timesteps"),
                })
    return actions, episode_outcomes


def main() -> int:
    rows: list[dict[str, object]] = []
    for (model_id, model_display), (source_dir, leaf) in zip(CASES, SEQUENCES):
        log_path = RESULTS_ROOT / source_dir / leaf / "log.jsonl"
        if not log_path.exists():
            print(f"Missing log: {log_path}")
            continue
        actions, episodes = collect(log_path)
        total_actions = sum(actions.values())

        for episode in episodes:
            rows.append({
                "model": model_id,
                "model_display": model_display,
                "row_type": "episode",
                "episode_id": episode["episode_id"],
                "deadlock": int(bool(episode["deadlock"])),
                "total_timesteps": episode["total_timesteps"],
                "action_label": "",
                "action_count": "",
                "action_share": "",
                "total_actions": "",
            })

        for action in ACTIONS:
            count = actions.get(action, 0)
            share = count / total_actions if total_actions else 0.0
            rows.append({
                "model": model_id,
                "model_display": model_display,
                "row_type": "action",
                "episode_id": "",
                "deadlock": "",
                "total_timesteps": "",
                "action_label": action,
                "action_count": count,
                "action_share": share,
                "total_actions": total_actions,
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "model_display", "row_type", "episode_id", "deadlock",
            "total_timesteps", "action_label", "action_count", "action_share",
            "total_actions",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    action_rows = [r for r in rows if r["row_type"] == "action"]
    episode_rows = [r for r in rows if r["row_type"] == "episode"]
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")
    print(f"  action distributions: {len(action_rows)}")
    print(f"  episode outcomes:     {len(episode_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
