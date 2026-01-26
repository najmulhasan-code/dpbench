"""LangGraph node functions."""

from typing import Any
from dpbench.core.types import BenchmarkConfig
from dpbench.core.environment import get_observation, step
from dpbench.agents.philosopher import get_philosopher_decision
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
        decision, llm_record = get_philosopher_decision(config.model_fn, obs, config)
        return {
            "decisions": {philosopher_id: decision},
            "llm_calls": {philosopher_id: llm_record},
        }
    return node


def build_apply_actions_node(config: BenchmarkConfig):
    """Build node that applies all decisions to the environment."""
    def node(state: GraphState) -> dict[str, Any]:
        table = state["table_state"]
        decisions = [state["decisions"][i] for i in range(config.num_philosophers)]
        result = step(table, decisions, mode=config.mode)

        messages = {}
        if config.communication:
            for d in decisions:
                if d.message_to_neighbors:
                    messages[d.philosopher_id] = d.message_to_neighbors

        return {
            "table_state": result.new_state,
            "messages": messages,
            "deadlock": result.deadlock,
            "episode_complete": result.deadlock or result.new_state.timestep >= config.max_timesteps,
            "timestep": result.new_state.timestep,
        }
    return node


def should_continue(state: GraphState) -> str:
    """Router: 'continue' or 'end'."""
    return "end" if state.get("episode_complete", False) else "continue"
