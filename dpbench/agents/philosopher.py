"""Philosopher agent: LLM-based decision making."""

import re
import time
from typing import Callable

from dpbench.core.types import (
    Action, Observation, AgentDecision, BenchmarkConfig,
    LLMCallRecord, ModelResponse, truncate_response, sanitize_message,
)


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
            return sanitize_message(msg)
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

    left_msg = f'[Neighbor Message]: "{obs.left_neighbor_message}"' if comm and obs.left_neighbor_message else "(no message)"
    right_msg = f'[Neighbor Message]: "{obs.right_neighbor_message}"' if comm and obs.right_neighbor_message else "(no message)"

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
    return template.format(
        philosopher_name=name,
        num_philosophers=n,
        num_philosophers_minus_one=n - 1,
    )


def _build_history_section(history: list[dict], window: int) -> str:
    """Build a text section showing recent timestep history."""
    if not history or window <= 0:
        return ""

    recent = history[-window:]
    lines = ["\nRECENT HISTORY:"]
    for entry in recent:
        t = entry["timestep"]
        dead = " [DEADLOCK]" if entry.get("deadlock") else ""
        lines.append(f"  Timestep {t}{dead}:")
        for p in entry["philosophers"]:
            forks = []
            if p["left_fork"]:
                forks.append("L")
            if p["right_fork"]:
                forks.append("R")
            forks_str = "+".join(forks) if forks else "-"
            lines.append(f"    P{p['id']}: {p['state']} [{forks_str}] meals={p['meals']}")
    return "\n".join(lines)


def _call_model(
    model_fn: Callable,
    system: str,
    user: str,
    philosopher_id: int,
) -> tuple[str, float, int | None, int | None]:
    """Call the model function and extract response text, latency, and token counts."""
    start_time = time.perf_counter()
    result = model_fn(system, user)
    latency_ms = (time.perf_counter() - start_time) * 1000

    if isinstance(result, ModelResponse):
        text = truncate_response(result.text)
        return text, latency_ms, result.tokens_in, result.tokens_out

    text = truncate_response(str(result))
    return text, latency_ms, None, None


def get_philosopher_decision(
    model_fn: Callable[[str, str], str | ModelResponse],
    observation: Observation,
    config: BenchmarkConfig,
    history: list[dict] | None = None,
) -> tuple[AgentDecision, LLMCallRecord]:
    """Get decision from one philosopher agent."""
    system = _build_system_prompt(config.system_prompt, observation.philosopher_name, config.num_philosophers)
    user = _build_observation_prompt(observation, config.decision_prompt, config.communication)

    if history and config.memory_window > 0:
        user += _build_history_section(history, config.memory_window)

    response_text, latency_ms, tokens_in, tokens_out = _call_model(
        model_fn, system, user, observation.philosopher_id,
    )

    decision = AgentDecision(
        philosopher_id=observation.philosopher_id,
        action=parse_action(response_text),
        message_to_neighbors=parse_message(response_text) if config.communication else None,
        reasoning=parse_reasoning(response_text),
    )

    llm_record = LLMCallRecord(
        philosopher_id=observation.philosopher_id,
        system_prompt=system,
        user_prompt=user,
        response=response_text,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )

    return decision, llm_record


def get_philosopher_message(
    model_fn: Callable[[str, str], str | ModelResponse],
    observation: Observation,
    config: BenchmarkConfig,
    round_number: int = 1,
    total_rounds: int = 1,
) -> tuple[str, LLMCallRecord]:
    """Get a communication-only message from a philosopher (no action taken).

    Used during multi-round communication before the action round.
    """
    system = _build_system_prompt(config.system_prompt, observation.philosopher_name, config.num_philosophers)

    left_fork = "AVAILABLE" if observation.left_fork_available else "TAKEN"
    right_fork = "AVAILABLE" if observation.right_fork_available else "TAKEN"
    left_msg = f'[Neighbor Message]: "{observation.left_neighbor_message}"' if observation.left_neighbor_message else "(no message)"
    right_msg = f'[Neighbor Message]: "{observation.right_neighbor_message}"' if observation.right_neighbor_message else "(no message)"

    user = (
        f"This is discussion round {round_number} of {total_rounds}. "
        f"You will act after all discussion rounds are complete.\n\n"
        f"Your state: {observation.state.value}, meals eaten: {observation.meals_eaten}\n"
        f"Left fork: {left_fork}, Right fork: {right_fork}\n"
        f"From left neighbor: {left_msg}\n"
        f"From right neighbor: {right_msg}\n\n"
        f"Share your plan with your neighbors. What do you intend to do?\n\n"
        f"THINKING: [Your reasoning about coordination]\n"
        f"MESSAGE: [Your message to neighbors]"
    )

    response_text, latency_ms, tokens_in, tokens_out = _call_model(
        model_fn, system, user, observation.philosopher_id,
    )

    message = parse_message(response_text) or ""

    llm_record = LLMCallRecord(
        philosopher_id=observation.philosopher_id,
        system_prompt=system,
        user_prompt=user,
        response=response_text,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )

    return message, llm_record
