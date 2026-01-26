"""CLI entry point for DPBench."""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dpbench.core.types import BenchmarkConfig
from dpbench.runner import run_experiment
from dpbench.evaluation.metrics import compute_aggregate_metrics
from dpbench.ui import Console


def _create_openai_model(model: str, temperature: float):
    """Create OpenAI model function."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: pip install openai")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    def fn(system_prompt: str, user_prompt: str) -> str:
        r = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return r.choices[0].message.content

    return fn


def _write_csv(results, filepath: str, num_philosophers: int):
    """Write results to CSV file."""
    with open(filepath, 'w', newline='') as f:
        # Build header
        headers = ['episode_id', 'timesteps', 'deadlock', 'throughput', 'fairness']
        headers += [f'meals_p{i}' for i in range(num_philosophers)]

        writer = csv.writer(f)
        writer.writerow(headers)

        for r in results:
            row = [
                r.episode_id,
                r.total_timesteps,
                r.deadlock,
                f'{r.throughput:.4f}',
                f'{r.fairness_gini:.4f}',
            ]
            row += list(r.meals_per_philosopher)
            writer.writerow(row)


def _run_command(args) -> int:
    """Run benchmark experiment."""
    console = Console(no_color=args.no_color)

    sys_path = Path(args.system_prompt)
    dec_path = Path(args.decision_prompt)

    if not sys_path.exists():
        console.error(f"{sys_path} not found")
        return 1
    if not dec_path.exists():
        console.error(f"{dec_path} not found")
        return 1

    # Print header and config
    console.header()
    console.config(
        mode=args.mode,
        philosophers=args.philosophers,
        episodes=args.episodes,
        max_timesteps=args.max_timesteps,
        communication=args.communication,
        model=args.model,
        log_dir=args.log_dir,
    )

    config = BenchmarkConfig(
        model_fn=_create_openai_model(args.model, args.temperature),
        system_prompt=sys_path.read_text(),
        decision_prompt=dec_path.read_text(),
        mode=args.mode,
        communication=args.communication,
        num_philosophers=args.philosophers,
        num_episodes=args.episodes,
        max_timesteps=args.max_timesteps,
        verbose=args.show_table or args.verbose,
        show_reasoning=args.show_reasoning or args.verbose,
        log_dir=args.log_dir,
        random_seed=args.seed,
        save_transcript=args.save_transcript,
    )

    try:
        results, logger = run_experiment(config, console)
    except KeyboardInterrupt:
        console.newline()
        console.warning("Interrupted")
        return 1

    # Print results
    metrics = compute_aggregate_metrics(results, config.communication)
    console.results(metrics, config.communication)

    # Determine output paths (auto-generate if --log-dir but no explicit path)
    run_id = logger.get_run_id() if logger else None
    log_dir = Path(args.log_dir) if args.log_dir else None

    json_path = args.output
    if not json_path and log_dir and run_id:
        json_path = str(log_dir / f"{run_id}_results.json")

    csv_path = args.csv
    if not csv_path and log_dir and run_id:
        csv_path = str(log_dir / f"{run_id}_results.csv")

    # Save JSON output
    if json_path:
        data = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "config": {
                "mode": config.mode,
                "num_philosophers": config.num_philosophers,
                "num_episodes": config.num_episodes,
                "max_timesteps": config.max_timesteps,
                "communication": config.communication,
                "model": args.model,
                "temperature": args.temperature,
            },
            "metrics": metrics,
            "episodes": [
                {
                    "episode_id": r.episode_id,
                    "timesteps": r.total_timesteps,
                    "deadlock": r.deadlock,
                    "meals": list(r.meals_per_philosopher),
                    "throughput": r.throughput,
                    "fairness": r.fairness_gini,
                }
                for r in results
            ],
        }
        Path(json_path).write_text(json.dumps(data, indent=2))
        console.saved(json_path)

    # Save CSV output
    if csv_path:
        _write_csv(results, csv_path, config.num_philosophers)
        console.saved(csv_path)

    return 0


def _plot_command(args) -> int:
    """Generate plots from results."""
    from dpbench.evaluation.plotting import (
        plot_comparison,
        plot_boxplot,
        plot_timeline,
        plot_all,
    )

    console = Console(no_color=getattr(args, 'no_color', False))

    plot_type = args.plot_type

    if plot_type == 'comparison':
        if len(args.files) < 2:
            console.error("Comparison plot requires at least 2 result files")
            return 1
        output = args.output or 'comparison.png'
        plot_comparison(args.files, output, title=args.title)
        console.success(f"Generated comparison plot: {output}")

    elif plot_type == 'boxplot':
        if len(args.files) != 1:
            console.error("Boxplot requires exactly 1 result file")
            return 1
        output = args.output or 'boxplot.png'
        plot_boxplot(args.files[0], output, title=args.title)
        console.success(f"Generated box plot: {output}")

    elif plot_type == 'timeline':
        if len(args.files) != 1:
            console.error("Timeline plot requires exactly 1 result file")
            return 1
        output = args.output or 'timeline.png'
        plot_timeline(args.files[0], output, title=args.title)
        console.success(f"Generated timeline plot: {output}")

    elif plot_type == 'all':
        if len(args.files) != 1:
            console.error("'all' plot type requires exactly 1 result file")
            return 1
        output_dir = args.output or './figures'
        outputs = plot_all(args.files[0], output_dir)
        for out in outputs:
            console.success(f"Generated: {out}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="DPBench: Benchmark for LLM Multi-Agent Coordination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ========== RUN COMMAND ==========
    run_parser = subparsers.add_parser(
        'run',
        help='Run benchmark experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dpbench run --system-prompt system.txt --decision-prompt decision.txt
  dpbench run --system-prompt s.txt --decision-prompt d.txt --episodes 10 -v
  dpbench run --system-prompt s.txt --decision-prompt d.txt --csv results.csv
        """
    )
    # Required
    run_parser.add_argument("--system-prompt", type=str, required=True, help="Path to system prompt file")
    run_parser.add_argument("--decision-prompt", type=str, required=True, help="Path to decision prompt file")
    # Experiment config
    run_parser.add_argument("--episodes", type=int, default=30, help="Number of episodes (default: 30)")
    run_parser.add_argument("--mode", choices=["simultaneous", "sequential"], default="simultaneous", help="Decision mode (default: simultaneous)")
    run_parser.add_argument("--philosophers", type=int, default=5, help="Number of philosophers (default: 5, min: 2)")
    run_parser.add_argument("--communication", action="store_true", default=False, help="Enable inter-agent messaging")
    run_parser.add_argument("--max-timesteps", type=int, default=50, help="Max timesteps per episode (default: 50)")
    # Model config
    run_parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name (default: gpt-4o)")
    run_parser.add_argument("--temperature", type=float, default=0.7, help="Model temperature (default: 0.7)")
    # Output options
    run_parser.add_argument("--show-table", action="store_true", help="Show table state each timestep")
    run_parser.add_argument("--show-reasoning", action="store_true", help="Show agent thinking and messages")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Show both table and reasoning")
    run_parser.add_argument("--output", "-o", type=str, help="Save results to JSON file")
    run_parser.add_argument("--csv", type=str, help="Save results to CSV file")
    run_parser.add_argument("--log-dir", type=str, help="Directory for JSONL trace logs")
    run_parser.add_argument("--save-transcript", action="store_true", help="Save human-readable transcript (requires --log-dir)")
    run_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    run_parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    run_parser.set_defaults(func=_run_command)

    # ========== PLOT COMMAND ==========
    plot_parser = subparsers.add_parser(
        'plot',
        help='Generate plots from results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dpbench plot comparison results1.json results2.json -o comparison.png
  dpbench plot boxplot results.json -o boxplot.png
  dpbench plot timeline results.json -o timeline.png
  dpbench plot all results.json -o ./figures/
        """
    )
    plot_parser.add_argument("plot_type", choices=["comparison", "boxplot", "timeline", "all"],
                             help="Type of plot to generate")
    plot_parser.add_argument("files", nargs="+", help="Result JSON file(s)")
    plot_parser.add_argument("-o", "--output", type=str, help="Output file/directory")
    plot_parser.add_argument("--title", type=str, help="Custom plot title")
    plot_parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    plot_parser.set_defaults(func=_plot_command)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # If no command specified, show help
    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
