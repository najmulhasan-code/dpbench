"""Evaluation metrics for DPBench."""

from typing import Optional

import numpy as np

from dpbench.core.types import (
    EpisodeResult,
    AgentDecision,
    Action,
)


def compute_message_action_consistency(decisions: list[AgentDecision]) -> Optional[float]:
    """
    Compute consistency between stated intent and actual action.

    Returns percentage (0-100) of actions matching stated intent, or None if
    no messages contained parseable intent.
    """
    intent_keywords = {
        Action.GRAB_LEFT: ["grab left", "take left", "pick up left", "grab my left"],
        Action.GRAB_RIGHT: ["grab right", "take right", "pick up right", "grab my right"],
        Action.RELEASE: ["release", "put down", "drop", "let go"],
        Action.WAIT: ["wait", "will wait", "i'll wait", "going to wait"],
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

    total_deadlock_events = sum(r.deadlock_events for r in results)

    metrics = {
        "num_episodes": n,
        "deadlock_rate": deadlocks / n,
        "deadlock_count": deadlocks,
        "total_deadlock_events": total_deadlock_events,
        "avg_deadlock_events": total_deadlock_events / n,
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

    # Aggregate LLM call metrics (tokens and latency)
    all_llm_calls = [call for r in results for call in r.all_llm_calls]
    if all_llm_calls:
        latencies = [call.latency_ms for call in all_llm_calls]
        tokens_in = [call.tokens_in for call in all_llm_calls if call.tokens_in is not None]
        tokens_out = [call.tokens_out for call in all_llm_calls if call.tokens_out is not None]

        metrics["total_llm_calls"] = len(all_llm_calls)
        metrics["avg_latency_ms"] = float(np.mean(latencies))
        metrics["std_latency_ms"] = float(np.std(latencies))
        metrics["min_latency_ms"] = float(np.min(latencies))
        metrics["max_latency_ms"] = float(np.max(latencies))

        if tokens_in:
            metrics["total_tokens_in"] = int(sum(tokens_in))
            metrics["avg_tokens_in"] = float(np.mean(tokens_in))
        if tokens_out:
            metrics["total_tokens_out"] = int(sum(tokens_out))
            metrics["avg_tokens_out"] = float(np.mean(tokens_out))
        if tokens_in and tokens_out:
            metrics["total_tokens"] = int(sum(tokens_in)) + int(sum(tokens_out))

    return metrics
