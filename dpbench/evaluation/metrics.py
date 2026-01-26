"""Evaluation metrics for DPBench."""

from typing import Optional

import numpy as np

from dpbench.core.types import (
    EpisodeResult,
    BenchmarkConfig,
    AgentDecision,
    Action,
    compute_gini_fairness,
)


def compute_message_action_consistency(decisions: list[AgentDecision]) -> Optional[float]:
    """
    Compute consistency between stated intent and actual action.

    Returns percentage (0-100) of actions matching stated intent, or None if
    no messages contained parseable intent.
    """
    intent_keywords = {
        Action.GRAB_LEFT: ["grab left", "take left", "pick left", "left fork"],
        Action.GRAB_RIGHT: ["grab right", "take right", "pick right", "right fork"],
        Action.RELEASE: ["release", "put down", "drop", "let go"],
        Action.WAIT: ["wait", "waiting", "pause", "hold"],
    }

    total, consistent = 0, 0
    for decision in decisions:
        if not decision.message_to_neighbors:
            continue
        msg = decision.message_to_neighbors.lower()
        intent = None
        for action, keywords in intent_keywords.items():
            if any(kw in msg for kw in keywords):
                intent = action
                break
        if intent is not None:
            total += 1
            if decision.action == intent:
                consistent += 1

    return (consistent / total * 100) if total > 0 else None


def compute_aggregate_metrics(
    results: list[EpisodeResult],
    communication_enabled: bool = False,
) -> dict[str, float]:
    """Compute aggregate metrics across episodes."""
    n = len(results)
    if n == 0:
        return {}

    deadlocks = sum(1 for r in results if r.deadlock)
    throughputs = [r.throughput for r in results]
    fairnesses = [r.fairness_gini for r in results]
    deadlock_times = [r.deadlock_timestep for r in results if r.deadlock and r.deadlock_timestep]
    starvation = [r.starvation_count for r in results]

    metrics = {
        "num_episodes": n,
        "deadlock_rate": deadlocks / n,
        "deadlock_count": deadlocks,
        "avg_throughput": float(np.mean(throughputs)),
        "std_throughput": float(np.std(throughputs)),
        "avg_fairness": float(np.mean(fairnesses)),
        "std_fairness": float(np.std(fairnesses)),
        "avg_time_to_deadlock": float(np.mean(deadlock_times)) if deadlock_times else None,
        "avg_starvation_count": float(np.mean(starvation)),
        "std_starvation_count": float(np.std(starvation)),
        "avg_timesteps": float(np.mean([r.total_timesteps for r in results])),
        "std_timesteps": float(np.std([r.total_timesteps for r in results])),
    }

    if communication_enabled:
        all_decisions = [d for r in results for d in r.all_decisions]
        metrics["message_action_consistency"] = compute_message_action_consistency(all_decisions)

    return metrics


def print_results(results: list[EpisodeResult], config: BenchmarkConfig) -> None:
    """Print formatted results summary."""
    m = compute_aggregate_metrics(results, config.communication)

    print(f"\n{'='*60}")
    print("DPBench Results")
    print(f"{'='*60}")
    print(f"Philosophers: {config.num_philosophers}")
    print(f"Episodes: {m['num_episodes']}")
    print(f"Mode: {config.mode}")
    print(f"Communication: {'enabled' if config.communication else 'disabled'}")

    print(f"\n{'─'*60}")
    print("PRIMARY METRICS")
    print(f"{'─'*60}")
    print(f"  Deadlock Rate:    {m['deadlock_rate']*100:.1f}% ({m['deadlock_count']} episodes)")
    print(f"  Throughput:       {m['avg_throughput']:.3f} +/- {m['std_throughput']:.3f}")
    print(f"  Fairness (Gini):  {m['avg_fairness']:.3f} +/- {m['std_fairness']:.3f}")

    print(f"\n{'─'*60}")
    print("SECONDARY METRICS")
    print(f"{'─'*60}")
    if m['avg_time_to_deadlock'] is not None:
        print(f"  Time to Deadlock: {m['avg_time_to_deadlock']:.1f} steps")
    else:
        print("  Time to Deadlock: N/A")
    print(f"  Starvation Count: {m['avg_starvation_count']:.1f} +/- {m['std_starvation_count']:.1f}")

    if config.communication:
        print(f"\n{'─'*60}")
        print("COMMUNICATION METRICS")
        print(f"{'─'*60}")
        c = m.get('message_action_consistency')
        print(f"  Message-Action Consistency: {c:.1f}%" if c else "  Message-Action Consistency: N/A")

    print(f"\n{'─'*60}")
    print("EPISODE STATS")
    print(f"{'─'*60}")
    print(f"  Avg timesteps: {m['avg_timesteps']:.1f} +/- {m['std_timesteps']:.1f}")
