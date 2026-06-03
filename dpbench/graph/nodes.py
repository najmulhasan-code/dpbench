"""LangGraph node functions."""

from typing import Any
from dpbench.core.types import BenchmarkConfig, sanitize_message
from dpbench.core.environment import (
    get_observation,
    step,
    resolve_deadlock,
)
from dpbench.agents.philosopher import get_philosopher_decision, get_philosopher_message
from dpbench.graph.state import GraphState


def build_observations_node(config: BenchmarkConfig):
    """Build node that creates observations for all philosophers."""
    def node(state: GraphState) -> dict[str, Any]:
        table = state["table_state"]
        messages = state.get("messages", {}) if config.communication else None
        observations = {i: get_observation(table, i, messages) for i in range(config.num_philosophers)}
        return {"observations": observations, "decisions": {}, "llm_calls": {}}
    return node


def build_philosopher_node(philosopher_id: int, config: BenchmarkConfig):
    """Build node for one philosopher's decision."""
    def node(state: GraphState) -> dict[str, Any]:
        obs = state["observations"][philosopher_id]
        history = state.get("history", [])
        decision, llm_record = get_philosopher_decision(
            config.model_fn, obs, config, history=history,
        )
        return {
            "decisions": {philosopher_id: decision},
            "llm_calls": {philosopher_id: llm_record},
        }
    return node


def build_communication_node(philosopher_id: int, config: BenchmarkConfig):
    """Build node for one philosopher's communication-only round (no action)."""
    def node(state: GraphState) -> dict[str, Any]:
        obs = state["observations"][philosopher_id]
        comm_round = state.get("communication_round", 1)
        total_rounds = config.communication_rounds
        message, llm_record = get_philosopher_message(
            config.model_fn, obs, config,
            round_number=comm_round,
            total_rounds=total_rounds,
        )
        sanitized = sanitize_message(message)
        return {
            "messages": {philosopher_id: sanitized},
            "llm_calls": {philosopher_id: llm_record},
        }
    return node


def build_apply_actions_node(config: BenchmarkConfig):
    """Build node that applies all decisions to the environment."""
    def node(state: GraphState) -> dict[str, Any]:
        table = state["table_state"]
        decisions = [state["decisions"][i] for i in range(config.num_philosophers)]
        result = step(table, decisions, mode=config.mode)

        new_state = result.new_state
        deadlock_events = state.get("deadlock_events", 0)

        if result.deadlock and not config.deadlock_terminal:
            new_state = resolve_deadlock(new_state)
            deadlock_events += 1
            episode_complete = new_state.timestep >= config.max_timesteps
        else:
            episode_complete = result.deadlock or new_state.timestep >= config.max_timesteps

        messages = {}
        if config.communication:
            for d in decisions:
                if d.message_to_neighbors:
                    messages[d.philosopher_id] = sanitize_message(d.message_to_neighbors)

        history = list(state.get("history", []))
        snapshot = {
            "timestep": new_state.timestep,
            "philosophers": [
                {"id": p.id, "state": p.state.value, "meals": p.meals_eaten,
                 "left_fork": p.has_left_fork, "right_fork": p.has_right_fork}
                for p in new_state.philosophers
            ],
            "deadlock": result.deadlock,
        }
        history.append(snapshot)
        if config.memory_window > 0:
            history = history[-config.memory_window:]

        return {
            "table_state": new_state,
            "messages": messages,
            "deadlock": result.deadlock if config.deadlock_terminal else False,
            "episode_complete": episode_complete,
            "timestep": new_state.timestep,
            "history": history,
            "deadlock_events": deadlock_events,
            "communication_round": 0,
        }
    return node


def build_increment_comm_round_node():
    """Build node that increments the communication round counter."""
    def node(state: GraphState) -> dict[str, Any]:
        return {"communication_round": state.get("communication_round", 0) + 1}
    return node


def should_continue(state: GraphState) -> str:
    """Router: 'continue' or 'end'."""
    return "end" if state.get("episode_complete", False) else "continue"
