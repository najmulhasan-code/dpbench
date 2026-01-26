"""LangGraph builder functions."""

from typing import Any
from langgraph.graph import StateGraph, START, END
from dpbench.core.types import BenchmarkConfig
from dpbench.core.environment import get_observation, step
from dpbench.agents.philosopher import get_philosopher_decision
from dpbench.graph.state import GraphState
from dpbench.graph.nodes import (
    build_observations_node,
    build_philosopher_node,
    build_apply_actions_node,
    should_continue,
)


def build_simultaneous_graph(config: BenchmarkConfig) -> StateGraph:
    """Build graph for simultaneous mode: all philosophers decide in parallel."""
    graph = StateGraph(GraphState)
    graph.add_node("observations", build_observations_node(config))
    graph.add_node("apply_actions", build_apply_actions_node(config))

    nodes = []
    for i in range(config.num_philosophers):
        name = f"philosopher_{i}"
        graph.add_node(name, build_philosopher_node(i, config))
        nodes.append(name)

    graph.add_edge(START, "observations")
    for n in nodes:
        graph.add_edge("observations", n)
        graph.add_edge(n, "apply_actions")
    graph.add_conditional_edges("apply_actions", should_continue, {"continue": "observations", "end": END})

    return graph


def build_sequential_graph(config: BenchmarkConfig) -> StateGraph:
    """Build graph for sequential mode: philosophers decide one at a time."""
    graph = StateGraph(GraphState)

    def build_step(pid: int):
        def step_fn(state: GraphState) -> dict[str, Any]:
            table = state["table_state"]
            messages = state.get("messages", {}) if config.communication else None
            obs = get_observation(table, pid, messages)
            decision = get_philosopher_decision(config.model_fn, obs, config)
            result = step(table, [decision], mode="sequential")

            new_messages = dict(state.get("messages", {}))
            if config.communication and decision.message_to_neighbors:
                new_messages[pid] = decision.message_to_neighbors

            all_decisions = dict(state.get("decisions", {}))
            all_decisions[pid] = decision

            return {
                "table_state": result.new_state,
                "messages": new_messages,
                "decisions": all_decisions,
                "deadlock": result.deadlock,
            }
        return step_fn

    for i in range(config.num_philosophers):
        graph.add_node(f"step_{i}", build_step(i))

    def finalize(state: GraphState) -> dict[str, Any]:
        table = state["table_state"]
        done = state.get("deadlock", False) or table.timestep >= config.max_timesteps
        return {"episode_complete": done, "timestep": table.timestep}

    graph.add_node("finalize", finalize)
    graph.add_edge(START, "step_0")
    for i in range(config.num_philosophers - 1):
        graph.add_edge(f"step_{i}", f"step_{i + 1}")
    graph.add_edge(f"step_{config.num_philosophers - 1}", "finalize")
    graph.add_conditional_edges("finalize", should_continue, {"continue": "step_0", "end": END})

    return graph


def build_graph(config: BenchmarkConfig) -> StateGraph:
    """Build graph for the specified mode."""
    return build_simultaneous_graph(config) if config.mode == "simultaneous" else build_sequential_graph(config)
