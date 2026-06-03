"""Analyze communication content from experiment logs.

Categorizes messages by type and analyzes communication strategies.

Usage:
    python experiments/scripts/analyze_messages.py --results-dir experiments/results/v0.2.0_20260329_171521
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


INTENT_PATTERNS = {
    "grab_left": ["grab left", "take left", "pick up left", "grab my left"],
    "grab_right": ["grab right", "take right", "pick up right", "grab my right"],
    "release": ["release", "put down", "let go", "give up"],
    "wait": ["wait", "will wait", "i'll wait", "going to wait"],
    "request_wait": ["please wait", "you wait", "could you wait", "don't grab"],
    "announce_plan": ["i plan to", "i will", "i'm going to", "my plan"],
    "coordinate": ["take turns", "alternate", "your turn", "my turn", "let me"],
}


def classify_message(message: str) -> list[str]:
    msg = message.lower()
    categories = []
    for category, patterns in INTENT_PATTERNS.items():
        if any(p in msg for p in patterns):
            categories.append(category)
    return categories if categories else ["uncategorized"]


def analyze_log_file(log_path: Path) -> dict:
    messages = []
    with open(log_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("type") != "timestep":
                continue
            for call in record.get("llm_calls", {}).values():
                if not isinstance(call, dict):
                    continue
                response = call.get("response", "")
                import re
                match = re.search(r"MESSAGE:\s*(.+?)(?=\n|ACTION:|$)", response, re.IGNORECASE)
                if match:
                    msg = match.group(1).strip()
                    if msg.lower() not in ("none", "n/a", "-", ""):
                        messages.append(msg)

    category_counts = Counter()
    for msg in messages:
        for cat in classify_message(msg):
            category_counts[cat] += 1

    return {
        "total_messages": len(messages),
        "categories": dict(category_counts),
        "sample_messages": messages[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze communication from logs")
    parser.add_argument("--results-dir", required=True, help="Path to results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = results_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    comm_conditions = ["sim_n5_comm", "sim_n10_comm"]
    all_analysis = []

    for log_file in results_dir.rglob("log.jsonl"):
        parts = log_file.relative_to(results_dir).parts
        if len(parts) < 2:
            continue

        condition = parts[-2]
        if not any(c in condition for c in ["comm"]):
            continue

        model = parts[-3] if len(parts) >= 3 else "unknown"

        analysis = analyze_log_file(log_file)
        analysis["model"] = model
        analysis["condition"] = condition
        all_analysis.append(analysis)

        if analysis["total_messages"] > 0:
            print(f"  {model}/{condition}: {analysis['total_messages']} messages")
            for cat, count in sorted(analysis["categories"].items(), key=lambda x: -x[1]):
                print(f"    {cat}: {count}")

    if all_analysis:
        with open(output_dir / "communication_analysis.json", "w") as f:
            json.dump(all_analysis, f, indent=2)

        rows = []
        for a in all_analysis:
            row = {"model": a["model"], "condition": a["condition"], "total_messages": a["total_messages"]}
            row.update(a["categories"])
            rows.append(row)

        if rows:
            all_keys = set()
            for r in rows:
                all_keys.update(r.keys())
            fieldnames = ["model", "condition", "total_messages"] + sorted(all_keys - {"model", "condition", "total_messages"})

            with open(output_dir / "communication_summary.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

        print(f"\nAnalysis saved to {output_dir}")
    else:
        print("No communication logs found.")


if __name__ == "__main__":
    main()
