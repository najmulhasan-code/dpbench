"""Main benchmark interface for DPBench."""

from typing import Callable, Optional

from dpbench.core.types import BenchmarkConfig, EpisodeResult
from dpbench.runner import run_experiment
from dpbench.evaluation.metrics import compute_aggregate_metrics


class Benchmark:
    """Primary interface for running DPBench experiments."""

    @staticmethod
    def run(
        model_fn: Callable[[str, str], str],
        system_prompt: str,
        decision_prompt: str,
        philosophers: int = 5,
        episodes: int = 30,
        max_timesteps: int = 50,
        mode: str = "simultaneous",
        communication: bool = False,
        verbose: bool = False,
        show_reasoning: bool = False,
        log_file: Optional[str] = None,
        transcript_file: Optional[str] = None,
        seed: Optional[int] = None,
        deadlock_terminal: bool = True,
        communication_rounds: int = 1,
        memory_window: int = 0,
        backoff_threshold: int = 0,
        backoff_probability: float = 0.5,
    ) -> dict:
        """Run benchmark and return metrics.

        Args:
            model_fn: Function (system_prompt, user_prompt) -> str or ModelResponse
            system_prompt: System prompt text
            decision_prompt: Decision prompt template
            philosophers: Number of philosophers (>= 2)
            episodes: Number of episodes
            max_timesteps: Maximum timesteps per episode
            mode: 'simultaneous' or 'sequential'
            communication: Enable inter-agent messaging
            verbose: Print detailed output
            show_reasoning: Print agent reasoning
            log_file: Path for JSONL log file
            transcript_file: Path for transcript file
            seed: Random seed
            deadlock_terminal: If False, recover from deadlock and continue
            communication_rounds: Number of discussion rounds before action (requires communication=True)
            memory_window: Number of past timesteps visible to agents (0 = memoryless)
            backoff_threshold: Auto-release fork after this many steps holding one (0 = disabled)
            backoff_probability: Probability of release when backoff triggers
        """
        config = BenchmarkConfig(
            model_fn=model_fn,
            system_prompt=system_prompt,
            decision_prompt=decision_prompt,
            mode=mode,
            communication=communication,
            num_philosophers=philosophers,
            num_episodes=episodes,
            max_timesteps=max_timesteps,
            verbose=verbose,
            show_reasoning=show_reasoning,
            log_file=log_file,
            transcript_file=transcript_file,
            random_seed=seed,
            deadlock_terminal=deadlock_terminal,
            communication_rounds=communication_rounds,
            memory_window=memory_window,
            backoff_threshold=backoff_threshold,
            backoff_probability=backoff_probability,
        )
        results, _ = run_experiment(config)
        metrics = compute_aggregate_metrics(results, config.communication)
        return metrics

    @staticmethod
    def run_with_config(config: BenchmarkConfig) -> tuple[list[EpisodeResult], dict]:
        """Run benchmark with a BenchmarkConfig object."""
        results, _ = run_experiment(config)
        metrics = compute_aggregate_metrics(results, config.communication)
        return results, metrics
