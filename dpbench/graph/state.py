"""LangGraph state definition."""

from typing import Annotated, TypedDict
from dpbench.core.types import TableState, Observation, AgentDecision, LLMCallRecord


def merge_dicts(left: dict, right: dict) -> dict:
    """Reducer for merging parallel updates to dict fields."""
    if left is None:
        return right
    if right is None:
        return left
    return {**left, **right}


class GraphState(TypedDict):
    """State passed between LangGraph nodes."""
    table_state: TableState
    observations: dict[int, Observation]
    decisions: Annotated[dict[int, AgentDecision], merge_dicts]
    llm_calls: Annotated[dict[int, LLMCallRecord], merge_dicts]
    messages: dict[int, str]
    episode_complete: bool
    deadlock: bool
    timestep: int
