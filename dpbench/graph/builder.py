"""LangGraph builder functions."""

from typing import Any
from langgraph.graph import StateGraph, START, END
from dpbench.core.types import BenchmarkConfig, sanitize_message
from dpbench.core.environment import get_observation, step, resolve_deadlock
from dpbench.agents.philosopher import get_philosopher_decision
from dpbench.graph.state import GraphState
from dpbench.graph.nodes import (
    build_observations_node,
    build_philosopher_node,
    build_communication_node,
    build_apply_actions_node,
    build_increment_comm_round_node,
    should_continue,
)


def build_simultaneous_graph(config: BenchmarkConfig) -> StateGraph:
    """Build graph for simultaneous mode.

    All philosophers observe the same state, then each decides independently.
    Actions are applied simultaneously — no philosopher sees another's decision
    before choosing their own.

    With multi-round communication (communication_rounds > 1), philosophers
    exchange messages for several rounds before the action round.
    """
    graph = StateGraph(GraphState)
    graph.add_node("observations", build_observations_node(config))
    graph.add_node("apply_actions", build_apply_actions_node(config))

    if config.communication and config.communication_rounds > 1:
        for r in range(1, config.communication_rounds):
            round_prefix = f"comm_r{r}"
            graph.add_node(f"{round_prefix}_inc", build_increment_comm_round_node())
            graph.add_node(f"{round_prefix}_obs", build_observations_node(config))
            for i in range(config.num_philosophers):
                node_name = f"{round_prefix}_p{i}"
                graph.add_node(node_name, build_communication_node(i, config))

        for i in range(config.num_philosophers):
            graph.add_node(f"action_p{i}", build_philosopher_node(i, config))

        graph.add_edge(START, "observations")
        first_round = "comm_r1"
        graph.add_edge("observations", f"{first_round}_inc")
        graph.add_edge(f"{first_round}_inc", f"{first_round}_obs")

        for r in range(1, config.communication_rounds):
            round_prefix = f"comm_r{r}"
            for i in range(config.num_philosophers):
                graph.add_edge(f"{round_prefix}_obs", f"{round_prefix}_p{i}")

            if r < config.communication_rounds - 1:
                next_prefix = f"comm_r{r + 1}"
                for i in range(config.num_philosophers):
                    graph.add_edge(f"{round_prefix}_p{i}", f"{next_prefix}_inc")
                graph.add_edge(f"{next_prefix}_inc", f"{next_prefix}_obs")
            else:
                for i in range(config.num_philosophers):
                    graph.add_edge(f"{round_prefix}_p{i}", f"action_p{i}")

        for i in range(config.num_philosophers):
            graph.add_edge(f"action_p{i}", "apply_actions")

    else:
        action_nodes = []
        for i in range(config.num_philosophers):
            name = f"philosopher_{i}"
            graph.add_node(name, build_philosopher_node(i, config))
            action_nodes.append(name)

        graph.add_edge(START, "observations")
        graph.add_edge("observations", action_nodes[0])
        for i in range(len(action_nodes) - 1):
            graph.add_edge(action_nodes[i], action_nodes[i + 1])
        graph.add_edge(action_nodes[-1], "apply_actions")

    graph.add_conditional_edges(
        "apply_actions", should_continue,
        {"continue": "observations", "end": END},
    )
    return graph


def build_sequential_graph(config: BenchmarkConfig) -> StateGraph:
    """Build graph for sequential mode: philosophers decide one at a time."""
    graph = StateGraph(GraphState)

    def build_step(pid: int):
        def step_fn(state: GraphState) -> dict[str, Any]:
            table = state["table_state"]
            messages = state.get("messages", {}) if config.communication else None
            obs = get_observation(table, pid, messages)
            history = state.get("history", [])
            decision, llm_record = get_philosopher_decision(
                config.model_fn, obs, config, history=history,
            )
            result = step(table, [decision], mode="sequential")

            new_messages = dict(state.get("messages", {}))
            if config.communication and decision.message_to_neighbors:
                new_messages[pid] = sanitize_message(decision.message_to_neighbors)

            all_decisions = dict(state.get("decisions", {}))
            all_decisions[pid] = decision

            all_llm_calls = dict(state.get("llm_calls", {}))
            all_llm_calls[pid] = llm_record

            return {
                "table_state": result.new_state,
                "messages": new_messages,
                "decisions": all_decisions,
                "llm_calls": all_llm_calls,
                "deadlock": result.deadlock,
            }
        return step_fn

    for i in range(config.num_philosophers):
        graph.add_node(f"step_{i}", build_step(i))

    def finalize(state: GraphState) -> dict[str, Any]:
        table = state["table_state"]
        is_dead = state.get("deadlock", False)
        deadlock_events = state.get("deadlock_events", 0)

        if is_dead and not config.deadlock_terminal:
            table = resolve_deadlock(table)
            deadlock_events += 1
            done = table.timestep >= config.max_timesteps
        else:
            done = is_dead or table.timestep >= config.max_timesteps

        history = list(state.get("history", []))
        snapshot = {
            "timestep": table.timestep,
            "philosophers": [
                {"id": p.id, "state": p.state.value, "meals": p.meals_eaten,
                 "left_fork": p.has_left_fork, "right_fork": p.has_right_fork}
                for p in table.philosophers
            ],
            "deadlock": is_dead,
        }
        history.append(snapshot)
        if config.memory_window > 0:
            history = history[-config.memory_window:]

        return {
            "table_state": table,
            "episode_complete": done,
            "timestep": table.timestep,
            "history": history,
            "deadlock_events": deadlock_events,
        }

    graph.add_node("finalize", finalize)
    graph.add_edge(START, "step_0")
    for i in range(config.num_philosophers - 1):
        graph.add_edge(f"step_{i}", f"step_{i + 1}")
    graph.add_edge(f"step_{config.num_philosophers - 1}", "finalize")
    graph.add_conditional_edges(
        "finalize", should_continue,
        {"continue": "step_0", "end": END},
    )
    return graph


def build_graph(config: BenchmarkConfig) -> StateGraph:
    """Build graph for the specified mode."""
    if config.mode == "simultaneous":
        return build_simultaneous_graph(config)
    return build_sequential_graph(config)
