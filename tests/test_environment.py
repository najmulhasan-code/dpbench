"""Tests for the dining philosophers environment."""

import pytest

from dpbench.core.types import (
    Action,
    PhilosopherState,
    AgentDecision,
)
from dpbench.core.environment import (
    create_initial_state,
    get_observation,
    step,
    is_deadlock,
)


def test_create_initial_state():
    """Initial state should have all philosophers hungry with no forks."""
    state = create_initial_state(5)

    assert len(state.philosophers) == 5
    assert len(state.forks) == 5
    assert state.timestep == 0

    for phil in state.philosophers:
        assert phil.state == PhilosopherState.HUNGRY
        assert phil.meals_eaten == 0
        assert not phil.has_left_fork
        assert not phil.has_right_fork

    for fork in state.forks:
        assert fork.is_free


def test_get_observation():
    """Observation should show local state correctly."""
    state = create_initial_state(5)
    obs = get_observation(state, 0)

    assert obs.philosopher_id == 0
    assert obs.philosopher_name == "P0"
    assert obs.state == PhilosopherState.HUNGRY
    assert obs.left_fork_available
    assert obs.right_fork_available


def test_grab_left_fork():
    """Grabbing a free fork should succeed."""
    state = create_initial_state(5)

    decisions = [
        AgentDecision(philosopher_id=0, action=Action.GRAB_LEFT),
        AgentDecision(philosopher_id=1, action=Action.WAIT),
        AgentDecision(philosopher_id=2, action=Action.WAIT),
        AgentDecision(philosopher_id=3, action=Action.WAIT),
        AgentDecision(philosopher_id=4, action=Action.WAIT),
    ]

    result = step(state, decisions, mode="simultaneous")

    assert result.new_state.philosophers[0].has_left_fork
    assert not result.new_state.philosophers[0].has_right_fork


def test_deadlock_detection():
    """Deadlock should be detected when all hold one fork."""
    state = create_initial_state(5)

    # Everyone grabs left fork
    decisions = [
        AgentDecision(philosopher_id=i, action=Action.GRAB_LEFT)
        for i in range(5)
    ]

    result = step(state, decisions, mode="simultaneous")

    assert result.deadlock
    assert is_deadlock(result.new_state)


def test_no_deadlock_with_wait():
    """If someone waits, no deadlock."""
    state = create_initial_state(5)

    decisions = [
        AgentDecision(philosopher_id=0, action=Action.GRAB_LEFT),
        AgentDecision(philosopher_id=1, action=Action.GRAB_LEFT),
        AgentDecision(philosopher_id=2, action=Action.GRAB_LEFT),
        AgentDecision(philosopher_id=3, action=Action.GRAB_LEFT),
        AgentDecision(philosopher_id=4, action=Action.WAIT),  # One waits
    ]

    result = step(state, decisions, mode="simultaneous")

    assert not result.deadlock
