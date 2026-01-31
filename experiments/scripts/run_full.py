"""Run full experiment matrix: all models × all conditions."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dpbench import Benchmark
from experiments.models import create_model


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run full DPBench experiment matrix")
    parser.add_argument("--config", default="experiment.yaml", help="Experiment config file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without running")
    args = parser.parse_args()

    configs_dir = Path(__file__).parent.parent / "configs"
    experiment_config = load_config(configs_dir / args.config)
    models_config = load_config(configs_dir / "models.yaml")
    conditions_config = load_config(configs_dir / "conditions.yaml")

    exp = experiment_config["experiment"]
    params = exp["parameters"]

    total_experiments = len(exp["models"]) * len(exp["conditions"])
    total_episodes = total_experiments * params["episodes"]

    print("=" * 60)
    print(f"DPBench Full Experiment: {exp['name']}")
    print("=" * 60)
    print(f"Models: {len(exp['models'])}")
    print(f"Conditions: {len(exp['conditions'])}")
    print(f"Episodes per condition: {params['episodes']}")
    print(f"Total experiments: {total_experiments}")
    print(f"Total episodes: {total_episodes}")
    print("=" * 60)

    if args.dry_run:
        print("\nDRY RUN - Would run:")
        for model_name in exp["models"]:
            for cond_name in exp["conditions"]:
                print(f"  {model_name} + {cond_name}")
        print(f"\nTotal: {total_experiments} experiments")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = Path(__file__).parent.parent / "results" / f"full_{timestamp}"
    base_results_dir.mkdir(parents=True, exist_ok=True)

    with open(base_results_dir / "experiment_config.yaml", "w") as f:
        yaml.dump(experiment_config, f)

    prompts_dir = Path(__file__).parent.parent / "prompts"

    completed = 0
    failed = []

    for model_name in exp["models"]:
        model_cfg = models_config["models"][model_name]

        print(f"\nCreating model: {model_name}")
        try:
            model_fn = create_model(
                provider=model_cfg["provider"],
                model_id=model_cfg["model_id"],
                temperature=params["temperature"],
                max_tokens=model_cfg["max_tokens"],
            )
        except Exception as e:
            print(f"  ERROR creating model: {e}")
            for cond_name in exp["conditions"]:
                failed.append(f"{model_name}_{cond_name}")
            continue

        for cond_name in exp["conditions"]:
            cond_cfg = conditions_config["conditions"][cond_name]

            if cond_cfg["communication"]:
                system_prompt = (prompts_dir / "system_comm.txt").read_text()
                decision_prompt = (prompts_dir / "decision_comm.txt").read_text()
            else:
                system_prompt = (prompts_dir / "system.txt").read_text()
                decision_prompt = (prompts_dir / "decision.txt").read_text()

            run_results_dir = base_results_dir / f"{model_name}_{cond_name}"
            run_results_dir.mkdir(parents=True, exist_ok=True)

            log_file = run_results_dir / "log.jsonl"
            transcript_file = run_results_dir / "transcript.txt"

            print(f"\n  Running: {model_name} + {cond_name}")
            print(f"    Episodes: {params['episodes']}")

            run_start = datetime.now()
            try:
                metrics = Benchmark.run(
                    model_fn=model_fn,
                    system_prompt=system_prompt,
                    decision_prompt=decision_prompt,
                    philosophers=cond_cfg["philosophers"],
                    episodes=params["episodes"],
                    max_timesteps=params["max_timesteps"],
                    mode=cond_cfg["mode"],
                    communication=cond_cfg["communication"],
                    log_file=str(log_file),
                    transcript_file=str(transcript_file),
                    verbose=False,
                    seed=params.get("seed"),
                )
                run_end = datetime.now()
                runtime_seconds = (run_end - run_start).total_seconds()
                completed += 1
                print(f"    Completed: {metrics['num_episodes']} episodes in {runtime_seconds:.1f}s")
                print(f"    Deadlock: {metrics['deadlock_rate']*100:.1f}%")
                print(f"    Throughput: {metrics['avg_throughput']:.3f}")
                print(f"    Fairness: {metrics['avg_fairness']:.3f}")
                if "total_tokens" in metrics:
                    print(f"    Tokens: {metrics['total_tokens']:,} ({metrics.get('avg_latency_ms', 0):.0f}ms avg)")

                # Save metrics to JSON file
                metrics_data = {
                    "model": model_name,
                    "condition": cond_name,
                    "model_id": model_cfg["model_id"],
                    "provider": model_cfg["provider"],
                    "philosophers": cond_cfg["philosophers"],
                    "mode": cond_cfg["mode"],
                    "communication": cond_cfg["communication"],
                    "episodes": params["episodes"],
                    "max_timesteps": params["max_timesteps"],
                    "seed": params.get("seed"),
                    "temperature": params["temperature"],
                    "runtime_seconds": runtime_seconds,
                    "timestamp": run_start.isoformat(),
                    **metrics,
                }
                metrics_file = run_results_dir / "metrics.json"
                with open(metrics_file, "w") as f:
                    json.dump(metrics_data, f, indent=2)
            except Exception as e:
                print(f"    ERROR: {e}")
                failed.append(f"{model_name}_{cond_name}")

    # Generate summary CSV
    all_metrics = []
    for metrics_file in base_results_dir.glob("*/metrics.json"):
        with open(metrics_file) as f:
            all_metrics.append(json.load(f))

    if all_metrics:
        summary_file = base_results_dir / "summary.csv"
        fieldnames = [
            "model", "condition", "mode", "communication", "philosophers",
            "episodes", "deadlock_rate", "avg_throughput", "avg_fairness",
            "avg_time_to_deadlock", "avg_starvation_count", "message_action_consistency",
            "total_tokens", "total_tokens_in", "total_tokens_out",
            "avg_latency_ms", "runtime_seconds",
        ]
        with open(summary_file, "w", newline="") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for m in sorted(all_metrics, key=lambda x: (x["model"], x["condition"])):
                writer.writerow(m)
        print(f"\nSummary CSV: {summary_file}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Completed: {completed}/{total_experiments}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed experiments: {failed}")
    print(f"Results: {base_results_dir}")

    if all_metrics:
        print("\n--- QUICK SUMMARY ---")
        for m in sorted(all_metrics, key=lambda x: (x["model"], x["condition"])):
            print(f"{m['model']:>18} + {m['condition']:<8}: "
                  f"DL={m['deadlock_rate']*100:5.1f}% "
                  f"TP={m['avg_throughput']:.3f} "
                  f"FR={m['avg_fairness']:.3f}")


if __name__ == "__main__":
    main()
