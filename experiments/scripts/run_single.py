"""Run single experiment: one model + one condition."""

import argparse
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
    parser = argparse.ArgumentParser(description="Run single DPBench experiment")
    parser.add_argument("--model", required=True, help="Model name from configs/models.yaml")
    parser.add_argument("--condition", required=True, help="Condition name from configs/conditions.yaml")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes")
    parser.add_argument("--max-timesteps", type=int, default=50, help="Max timesteps per episode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    configs_dir = Path(__file__).parent.parent / "configs"
    models_config = load_config(configs_dir / "models.yaml")
    conditions_config = load_config(configs_dir / "conditions.yaml")

    if args.model not in models_config["models"]:
        print(f"Error: Model '{args.model}' not found in models.yaml")
        print(f"Available models: {list(models_config['models'].keys())}")
        sys.exit(1)
    model_cfg = models_config["models"][args.model]

    if args.condition not in conditions_config["conditions"]:
        print(f"Error: Condition '{args.condition}' not found in conditions.yaml")
        print(f"Available conditions: {list(conditions_config['conditions'].keys())}")
        sys.exit(1)
    cond_cfg = conditions_config["conditions"][args.condition]

    prompts_dir = Path(__file__).parent.parent / "prompts"
    if cond_cfg["communication"]:
        system_prompt = (prompts_dir / "system_comm.txt").read_text()
        decision_prompt = (prompts_dir / "decision_comm.txt").read_text()
    else:
        system_prompt = (prompts_dir / "system.txt").read_text()
        decision_prompt = (prompts_dir / "decision.txt").read_text()

    print(f"Creating model: {args.model} ({model_cfg['provider']})")
    model_fn = create_model(
        provider=model_cfg["provider"],
        model_id=model_cfg["model_id"],
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
    )

    # User controls file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    log_file = results_dir / f"{args.model}_{args.condition}_{timestamp}.jsonl"
    transcript_file = results_dir / f"{args.model}_{args.condition}_{timestamp}_transcript.txt"

    print(f"Running: {args.model} + {args.condition}")
    print(f"Episodes: {args.episodes}, Max timesteps: {args.max_timesteps}")
    print(f"Log file: {log_file}")
    print("-" * 60)

    metrics = Benchmark.run(
        model_fn=model_fn,
        system_prompt=system_prompt,
        decision_prompt=decision_prompt,
        philosophers=cond_cfg["philosophers"],
        episodes=args.episodes,
        max_timesteps=args.max_timesteps,
        mode=cond_cfg["mode"],
        communication=cond_cfg["communication"],
        log_file=str(log_file),
        transcript_file=str(transcript_file),
        verbose=args.verbose,
    )

    print("-" * 60)
    print("Results:")
    print(f"  Deadlock Rate: {metrics['deadlock_rate']*100:.1f}%")
    print(f"  Throughput: {metrics['avg_throughput']:.3f} meals/step")
    print(f"  Fairness: {metrics['avg_fairness']:.3f}")
    print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
