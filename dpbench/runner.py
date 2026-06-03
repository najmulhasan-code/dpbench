"""Experiment runner using LangGraph orchestration."""

from typing import TYPE_CHECKING

from dpbench.core.types import (
    BenchmarkConfig,
    EpisodeResult,
)
from dpbench.core.environment import create_initial_state
from dpbench.graph.builder import build_graph
from dpbench.evaluation.logger import ExperimentLogger

if TYPE_CHECKING:
    from dpbench.ui import Console


def run_episode(
    config: BenchmarkConfig,
    episode_id: int,
    verbose: bool = False,
    logger: ExperimentLogger | None = None,
    console: "Console | None" = None,
) -> EpisodeResult:
    """Run a single episode and return metrics."""
    graph = build_graph(config)
    recursion_limit = config.max_timesteps * (config.num_philosophers * config.communication_rounds + config.num_philosophers + 5)
    app = graph.compile()

    table = create_initial_state(config.num_philosophers)
    state = {
        "table_state": table,
        "observations": {},
        "decisions": {},
        "llm_calls": {},
        "messages": {},
        "episode_complete": False,
        "deadlock": False,
        "timestep": 0,
        "communication_round": 0,
        "history": [],
        "deadlock_events": 0,
    }

    if logger:
        logger.log_episode_start(episode_id)

    if verbose and console:
        console.episode_start(episode_id, config.num_episodes)
        console.table_state(table)

    final_state = state
    current_decisions = {}
    current_llm_calls = {}
    all_decisions = []
    all_llm_calls = []
    last_timestep = -1

    for step_output in app.stream(state, config={"recursion_limit": recursion_limit}):
        for node_state in step_output.values():
            if "decisions" in node_state and node_state["decisions"]:
                current_decisions.update(node_state["decisions"])
            if "llm_calls" in node_state and node_state["llm_calls"]:
                current_llm_calls.update(node_state["llm_calls"])
            if "table_state" in node_state:
                final_state = node_state
                t = node_state["table_state"].timestep
                if t > last_timestep:
                    if verbose and console:
                        console.table_state(node_state["table_state"])
                    if len(current_decisions) == config.num_philosophers:
                        if config.show_reasoning and console:
                            console.agent_reasoning(t, current_decisions, config.num_philosophers)
                        all_decisions.extend(current_decisions.values())
                        all_llm_calls.extend(current_llm_calls.values())
                        if logger:
                            logger.log_timestep(episode_id, t, node_state["table_state"], current_decisions, current_llm_calls)
                        current_decisions = {}
                        current_llm_calls = {}
                    last_timestep = t

    final_table = final_state["table_state"]
    meals = tuple(p.meals_eaten for p in final_table.philosophers)
    deadlock = final_state.get("deadlock", False)

    result = EpisodeResult(
        episode_id=episode_id,
        total_timesteps=final_table.timestep,
        deadlock=deadlock,
        deadlock_timestep=final_table.timestep if deadlock else None,
        meals_per_philosopher=meals,
        total_meals=sum(meals),
        deadlock_events=final_state.get("deadlock_events", 0),
        all_decisions=all_decisions,
        all_llm_calls=all_llm_calls,
    )

    if verbose and console:
        console.episode_summary(result)

    if logger:
        logger.log_episode_end(
            episode_id, result.total_timesteps, deadlock,
            meals, result.throughput, result.fairness_gini
        )

    return result


def run_experiment(
    config: BenchmarkConfig,
    console: "Console | None" = None,
    on_episode_complete=None,
) -> tuple[list[EpisodeResult], ExperimentLogger | None]:
    """Run all episodes and return results with logger for consistent naming.

    Args:
        on_episode_complete: Optional callback(episode_id, total_episodes, result)
            called after each episode finishes. Use for progress reporting.
    """
    results = []
    logger = ExperimentLogger(config) if config.log_file or config.transcript_file else None

    if console:
        console.newline()
        console.info(f"Running {config.num_episodes} episodes...")
        if logger:
            console.info(f"Logging to: {logger.get_log_path()}")

    for i in range(config.num_episodes):
        result = run_episode(
            config, i,
            config.verbose or config.show_reasoning,
            logger,
            console,
        )
        results.append(result)
        if on_episode_complete:
            on_episode_complete(i, config.num_episodes, result)
        if not config.verbose and not config.show_reasoning and console:
            console.episode_marker(result.deadlock)

    if not config.verbose and not config.show_reasoning and console:
        console.newline()

    if logger and console:
        console.saved(logger.get_log_path())
        if logger.get_transcript_path():
            console.saved(logger.get_transcript_path())

    return results, logger
