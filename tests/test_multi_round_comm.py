"""Tests for deadlock recovery, backoff, memory, and multi-round communication."""

from dpbench.core.types import (
    BenchmarkConfig,
    EpisodeResult,
    sanitize_message,
    truncate_response,
    MAX_MESSAGE_LENGTH,
    MAX_RESPONSE_LENGTH,
)
from dpbench.core.environment import (
    create_initial_state,
    resolve_deadlock,
    apply_backoff,
    is_deadlock,
    step,
)
from dpbench.core.types import Action, AgentDecision, PhilosopherState


class TestSanitization:
    def test_sanitize_short_message(self):
        assert sanitize_message("hello") == "hello"

    def test_sanitize_long_message(self):
        long = "x" * 500
        result = sanitize_message(long)
        assert len(result) == MAX_MESSAGE_LENGTH

    def test_sanitize_empty(self):
        assert sanitize_message("") == ""
        assert sanitize_message(None) == ""

    def test_truncate_response_normal(self):
        assert truncate_response("short") == "short"

    def test_truncate_response_long(self):
        long = "y" * 20000
        result = truncate_response(long)
        assert len(result) == MAX_RESPONSE_LENGTH


class TestDeadlockRecovery:
    def test_resolve_deadlock_clears_forks(self):
        state = create_initial_state(3)
        decisions = [
            AgentDecision(philosopher_id=0, action=Action.GRAB_LEFT),
            AgentDecision(philosopher_id=1, action=Action.GRAB_LEFT),
            AgentDecision(philosopher_id=2, action=Action.GRAB_LEFT),
        ]
        result = step(state, decisions, mode="simultaneous")
        assert result.deadlock

        recovered = resolve_deadlock(result.new_state)
        assert not is_deadlock(recovered)
        for p in recovered.philosophers:
            assert not p.has_left_fork
            assert not p.has_right_fork
        for f in recovered.forks:
            assert f.is_free

    def test_resolve_preserves_meals(self):
        state = create_initial_state(3)
        decisions = [
            AgentDecision(philosopher_id=0, action=Action.GRAB_LEFT),
            AgentDecision(philosopher_id=1, action=Action.GRAB_LEFT),
            AgentDecision(philosopher_id=2, action=Action.GRAB_LEFT),
        ]
        result = step(state, decisions, mode="simultaneous")
        recovered = resolve_deadlock(result.new_state)
        for p in recovered.philosophers:
            assert p.meals_eaten == 0


class TestBackoff:
    def test_backoff_releases_after_threshold(self):
        state = create_initial_state(3)
        decisions = [
            AgentDecision(philosopher_id=0, action=Action.GRAB_LEFT),
            AgentDecision(philosopher_id=1, action=Action.WAIT),
            AgentDecision(philosopher_id=2, action=Action.WAIT),
        ]
        result = step(state, decisions, mode="simultaneous")
        new_state = result.new_state

        hold_counts = {0: 2, 1: 0, 2: 0}
        released_state, new_counts = apply_backoff(new_state, hold_counts, threshold=2, probability=1.0)

        p0 = released_state.philosophers[0]
        assert not p0.has_left_fork
        assert new_counts[0] == 0

    def test_backoff_no_release_below_threshold(self):
        state = create_initial_state(3)
        decisions = [
            AgentDecision(philosopher_id=0, action=Action.GRAB_LEFT),
            AgentDecision(philosopher_id=1, action=Action.WAIT),
            AgentDecision(philosopher_id=2, action=Action.WAIT),
        ]
        result = step(state, decisions, mode="simultaneous")

        hold_counts = {0: 0, 1: 0, 2: 0}
        released_state, new_counts = apply_backoff(result.new_state, hold_counts, threshold=3, probability=1.0)

        p0 = released_state.philosophers[0]
        assert p0.has_left_fork
        assert new_counts[0] == 1


class TestNewConfigFields:
    def _dummy_model(self, sys, usr):
        return "ACTION: WAIT"

    def test_default_values(self):
        config = BenchmarkConfig(
            model_fn=self._dummy_model,
            system_prompt="test",
            decision_prompt="test",
        )
        assert config.deadlock_terminal is True
        assert config.communication_rounds == 1
        assert config.memory_window == 0
        assert config.backoff_threshold == 0
        assert config.backoff_probability == 0.5

    def test_new_experiment_code_format(self):
        config = BenchmarkConfig(
            model_fn=self._dummy_model,
            system_prompt="test",
            decision_prompt="test",
            mode="simultaneous",
            num_philosophers=5,
            communication=False,
        )
        assert config.experiment_code == "sim_n5_nocomm"

    def test_invalid_communication_rounds(self):
        import pytest
        with pytest.raises(ValueError):
            BenchmarkConfig(
                model_fn=self._dummy_model,
                system_prompt="test",
                decision_prompt="test",
                communication_rounds=0,
            )

    def test_invalid_backoff_probability(self):
        import pytest
        with pytest.raises(ValueError):
            BenchmarkConfig(
                model_fn=self._dummy_model,
                system_prompt="test",
                decision_prompt="test",
                backoff_probability=1.5,
            )


class TestEpisodeResultDeadlockEvents:
    def test_deadlock_events_default(self):
        result = EpisodeResult(
            episode_id=0,
            total_timesteps=10,
            deadlock=False,
            deadlock_timestep=None,
            meals_per_philosopher=(1, 1, 1),
            total_meals=3,
        )
        assert result.deadlock_events == 0

    def test_deadlock_events_set(self):
        result = EpisodeResult(
            episode_id=0,
            total_timesteps=10,
            deadlock=False,
            deadlock_timestep=None,
            meals_per_philosopher=(1, 1, 1),
            total_meals=3,
            deadlock_events=5,
        )
        assert result.deadlock_events == 5
