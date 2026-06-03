"""DPBench experiment runner.

Usage:
    python experiments/scripts/run.py                           # run everything
    python experiments/scripts/run.py --layer 1                 # layer 1 only
    python experiments/scripts/run.py --models gpt-5.2          # one model
    python experiments/scripts/run.py --conditions sim_n5_nocomm # one condition
    python experiments/scripts/run.py --dry-run                 # show plan without running
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dpbench
from dpbench.core.types import BenchmarkConfig
from dpbench.runner import run_experiment
from dpbench.evaluation.metrics import compute_aggregate_metrics
from experiments.models.openrouter import create_openrouter_model
from experiments.models.random_baseline import create_random_model


SCALING_MODELS = ["gpt-5.2", "claude-opus-4.6", "llama-4-maverick", "qwen3-235b-a22b"]
ABLATION_MODELS = ["gpt-5.2", "claude-opus-4.6"]


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_prompts(prompts_dir: Path, variant: str, communication: bool) -> tuple[str, str]:
    """Load system and decision prompts for a given variant."""
    variant_dir = prompts_dir / variant

    if communication:
        system_file = variant_dir / "system_comm.txt"
        decision_file = variant_dir / "decision_comm.txt"
        if not system_file.exists():
            system_file = prompts_dir / "default" / "system_comm.txt"
        if not decision_file.exists():
            decision_file = prompts_dir / "default" / "decision_comm.txt"
    else:
        system_file = variant_dir / "system.txt"
        decision_file = variant_dir / "decision.txt"
        if not system_file.exists():
            system_file = prompts_dir / "default" / "system.txt"
        if not decision_file.exists():
            decision_file = prompts_dir / "default" / "decision.txt"

    return system_file.read_text(), decision_file.read_text()


def create_model(model_name: str, model_cfg: dict, temperature: float | None = None):
    """Create a model function from config."""
    if model_name == "random":
        return create_random_model()

    temp = temperature if temperature is not None else model_cfg.get("temperature", 0.7)
    return create_openrouter_model(
        model_id=model_cfg["model_id"],
        temperature=temp,
        max_tokens=model_cfg.get("max_tokens", 1024),
    )


def run_single(
    model_name: str,
    model_fn,
    condition_name: str,
    condition: dict,
    prompts_dir: Path,
    output_dir: Path,
) -> dict | None:
    """Run one model on one condition. Returns metrics dict or None on failure."""
    variant = condition.get("prompt_variant", "default")
    system_prompt, decision_prompt = load_prompts(
        prompts_dir, variant, condition["communication"],
    )

    run_dir = output_dir / condition_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "log.jsonl"
    transcript_file = run_dir / "transcript.txt"

    total_ep = condition["episodes"]
    print(f"  {condition_name}: {total_ep} episodes, "
          f"N={condition['philosophers']}, {condition['mode']}")

    def show_progress(ep_id, total, result):
        pct = (ep_id + 1) / total * 100
        dl = "DL" if result.deadlock else "ok"
        print(f"\r    [{ep_id + 1}/{total}] {pct:.0f}% ({dl})", end="", flush=True)

    config = BenchmarkConfig(
        model_fn=model_fn,
        system_prompt=system_prompt,
        decision_prompt=decision_prompt,
        num_philosophers=condition["philosophers"],
        num_episodes=total_ep,
        max_timesteps=condition["max_timesteps"],
        mode=condition["mode"],
        communication=condition["communication"],
        log_file=str(log_file),
        transcript_file=str(transcript_file),
        verbose=False,
        random_seed=condition.get("seed", 42),
        deadlock_terminal=condition.get("deadlock_terminal", True),
        communication_rounds=condition.get("communication_rounds", 1),
        memory_window=condition.get("memory_window", 0),
        backoff_threshold=condition.get("backoff_threshold", 0),
        backoff_probability=condition.get("backoff_probability", 0.5),
    )

    start = datetime.now()
    try:
        results_list, _ = run_experiment(config, on_episode_complete=show_progress)
        metrics = compute_aggregate_metrics(results_list, config.communication)
        runtime = (datetime.now() - start).total_seconds()

        print(f"\r    deadlock={metrics['deadlock_rate']*100:.0f}% "
              f"throughput={metrics['avg_throughput']:.3f} "
              f"fairness={metrics['avg_fairness']:.3f} "
              f"({runtime:.0f}s)          ")

        result = {
            "model": model_name,
            "condition": condition_name,
            "philosophers": condition["philosophers"],
            "mode": condition["mode"],
            "communication": condition["communication"],
            "episodes": condition["episodes"],
            "runtime_seconds": runtime,
            "timestamp": start.isoformat(),
            "dpbench_version": dpbench.__version__,
            **metrics,
        }

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as e:
        print(f"    FAILED: {e}")
        return None


def write_summary_csv(results: list[dict], output_path: Path):
    """Write summary CSV from collected results."""
    if not results:
        return

    fields = [
        "model", "condition", "mode", "communication", "philosophers",
        "episodes", "deadlock_rate", "avg_throughput", "avg_fairness",
        "avg_time_to_deadlock", "avg_starvation_count", "total_deadlock_events",
        "message_action_consistency", "total_tokens", "avg_latency_ms",
        "runtime_seconds", "dpbench_version",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x["model"], x["condition"])):
            writer.writerow(r)

    print(f"\nSummary: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DPBench experiment runner")
    parser.add_argument("--layer", type=int, choices=[1, 2, 3], help="Run specific layer only")
    parser.add_argument("--models", nargs="+", help="Run specific models only")
    parser.add_argument("--conditions", nargs="+", help="Run specific conditions only")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without running")
    parser.add_argument("--resume", action="store_true", help="Skip conditions with existing results")
    args = parser.parse_args()

    load_dotenv()

    root = Path(__file__).parent.parent
    configs_dir = root / "configs"
    prompts_dir = root / "prompts"

    models_config = load_yaml(configs_dir / "models.yaml")
    conditions_config = load_yaml(configs_dir / "conditions.yaml")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = dpbench.__version__
    results_dir = root / "results" / f"v{version}_{timestamp}"

    runs = []

    for layer_name in ["layer1", "layer2", "layer3"]:
        layer_num = int(layer_name[-1])
        if args.layer and layer_num != args.layer:
            continue

        layer_conditions = conditions_config.get(layer_name, {})

        if args.models:
            model_names = list(args.models)
        elif layer_num == 2:
            model_names = SCALING_MODELS
        elif layer_num == 3:
            model_names = ABLATION_MODELS
        else:
            model_names = list(models_config.keys()) + ["random"]

        for model_name in model_names:
            for cond_name, cond in layer_conditions.items():
                if args.conditions and cond_name not in args.conditions:
                    continue

                subdir = "" if layer_num == 1 else f"_{'scaling' if layer_num == 2 else 'ablations'}"
                if model_name == "random":
                    subdir = "_random_baseline"

                runs.append({
                    "model_name": model_name,
                    "condition_name": cond_name,
                    "condition": cond,
                    "output_subdir": subdir,
                })

    total = len(runs)
    print(f"DPBench v{version}")
    print(f"Experiments: {total}")
    print(f"Results: {results_dir}")
    print()

    if args.dry_run:
        for r in runs:
            print(f"  {r['model_name']:>20} + {r['condition_name']}")
        print(f"\nTotal: {total} experiments")
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "dpbench_version": version,
        "timestamp": timestamp,
        "total_experiments": total,
        "models": list(models_config.keys()),
        "layers": list(conditions_config.keys()),
    }
    with open(results_dir / "run_config.yaml", "w") as f:
        yaml.dump(run_config, f)

    all_results = []
    completed = 0
    failed = 0

    current_model = None
    current_model_fn = None

    for i, run in enumerate(runs):
        model_name = run["model_name"]
        cond_name = run["condition_name"]
        cond = run["condition"]
        subdir = run["output_subdir"]

        if model_name == "random":
            output_dir = results_dir / "random_baseline"
        elif subdir:
            output_dir = results_dir / subdir.lstrip("_") / model_name
        else:
            output_dir = results_dir / model_name

        if args.resume and (output_dir / cond_name / "metrics.json").exists():
            print(f"  [{i+1}/{total}] {model_name} + {cond_name}: skipped (exists)")
            completed += 1
            continue

        if model_name != current_model:
            current_model = model_name
            print(f"\n[{model_name}]")
            try:
                if model_name == "random":
                    current_model_fn = create_random_model(seed=42)
                else:
                    model_cfg = models_config[model_name]
                    temp_override = cond.get("temperature")
                    current_model_fn = create_model(model_name, model_cfg, temp_override)
            except Exception as e:
                print(f"  Failed to create model: {e}")
                current_model_fn = None

        if current_model_fn is None:
            failed += 1
            continue

        temp_override = cond.get("temperature")
        if temp_override and model_name != "random":
            model_cfg = models_config[model_name]
            model_fn = create_model(model_name, model_cfg, temp_override)
        else:
            model_fn = current_model_fn

        result = run_single(
            model_name, model_fn, cond_name, cond, prompts_dir, output_dir,
        )

        if result:
            all_results.append(result)
            completed += 1
        else:
            failed += 1

    write_summary_csv(all_results, results_dir / "summary.csv")

    print(f"\nDone: {completed}/{total} completed, {failed} failed")
    print(f"Results: {results_dir}")


if __name__ == "__main__":
    main()
