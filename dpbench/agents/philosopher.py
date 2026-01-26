"""Philosopher agent: LLM-based decision making."""

import re
import time
from typing import Callable

from dpbench.core.types import Action, Observation, AgentDecision, BenchmarkConfig, LLMCallRecord


def parse_action(response: str) -> Action:
    """Extract action from LLM response. Defaults to WAIT on parse failure."""
    text = response.upper()
    match = re.search(r"ACTION:\s*(\w+)", text)
    action_str = match.group(1) if match else text

    if "GRAB_LEFT" in action_str or "GRABLEFT" in action_str:
        return Action.GRAB_LEFT
    if "GRAB_RIGHT" in action_str or "GRABRIGHT" in action_str:
        return Action.GRAB_RIGHT
    if "RELEASE" in action_str:
        return Action.RELEASE
    return Action.WAIT


def parse_message(response: str) -> str | None:
    """Extract message from LLM response."""
    match = re.search(r"MESSAGE:\s*(.+?)(?=\n|ACTION:|$)", response, re.IGNORECASE)
    if match:
        msg = match.group(1).strip()
        if msg.lower() not in ("none", "n/a", "-", ""):
            return msg
    return None


def parse_reasoning(response: str) -> str | None:
    """Extract reasoning from LLM response."""
    match = re.search(r"THINKING:\s*(.+?)(?=\nMESSAGE:|\nACTION:|$)", response, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def _build_observation_prompt(obs: Observation, template: str, comm: bool) -> str:
    """Build decision prompt from template and observation."""
    left_fork = "AVAILABLE" if obs.left_fork_available else "TAKEN"
    right_fork = "AVAILABLE" if obs.right_fork_available else "TAKEN"

    holding = []
    if obs.holding_left:
        holding.append("left fork")
    if obs.holding_right:
        holding.append("right fork")
    holding_status = ", ".join(holding) if holding else "nothing"

    left_msg = f'"{obs.left_neighbor_message}"' if comm and obs.left_neighbor_message else "(no message)"
    right_msg = f'"{obs.right_neighbor_message}"' if comm and obs.right_neighbor_message else "(no message)"

    return template.format(
        philosopher_name=obs.philosopher_name,
        state=obs.state.value,
        meals_eaten=obs.meals_eaten,
        left_fork_status=left_fork,
        right_fork_status=right_fork,
        holding_status=holding_status,
        left_message=left_msg,
        right_message=right_msg,
    )


def _build_system_prompt(template: str, name: str, n: int) -> str:
    """Build system prompt from template."""
    return template.format(philosopher_name=name, num_philosophers=n)


def get_philosopher_decision(
    model_fn: Callable[[str, str], str],
    observation: Observation,
    config: BenchmarkConfig,
) -> tuple[AgentDecision, LLMCallRecord]:
    """Get decision from one philosopher agent.

    Returns:
        Tuple of (AgentDecision, LLMCallRecord) for logging.
    """
    system = _build_system_prompt(config.system_prompt, observation.philosopher_name, config.num_philosophers)
    user = _build_observation_prompt(observation, config.decision_prompt, config.communication)

    start_time = time.perf_counter()
    response = model_fn(system, user)
    latency_ms = (time.perf_counter() - start_time) * 1000

    decision = AgentDecision(
        philosopher_id=observation.philosopher_id,
        action=parse_action(response),
        message_to_neighbors=parse_message(response) if config.communication else None,
        reasoning=parse_reasoning(response),
    )

    llm_record = LLMCallRecord(
        philosopher_id=observation.philosopher_id,
        system_prompt=system,
        user_prompt=user,
        response=response,
        latency_ms=latency_ms,
    )

    return decision, llm_record
