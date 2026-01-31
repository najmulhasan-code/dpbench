"""Tests for metric calculations."""

import pytest
from dpbench.core.types import compute_gini_fairness, EpisodeResult


class TestGiniFairness:
    """Test Gini coefficient fairness calculation."""

    def test_perfect_equality(self):
        """All philosophers eat same amount = perfect fairness."""
        assert compute_gini_fairness((1, 1, 1, 1, 1)) == 1.0
        assert compute_gini_fairness((5, 5, 5, 5, 5)) == 1.0
        assert compute_gini_fairness((10, 10, 10)) == 1.0

    def test_maximum_inequality(self):
        """One philosopher eats everything = zero fairness."""
        # After normalization, max inequality should give fairness = 0.0
        assert compute_gini_fairness((5, 0, 0, 0, 0)) == 0.0
        assert compute_gini_fairness((10, 0, 0)) == 0.0
        assert compute_gini_fairness((100, 0, 0, 0, 0, 0, 0)) == 0.0

    def test_all_zeros(self):
        """No meals eaten = fair (no one is worse off)."""
        assert compute_gini_fairness((0, 0, 0, 0, 0)) == 1.0

    def test_empty_tuple(self):
        """Empty input = fair (edge case)."""
        assert compute_gini_fairness(()) == 1.0

    def test_single_philosopher(self):
        """Single philosopher = perfect equality."""
        assert compute_gini_fairness((5,)) == 1.0
        assert compute_gini_fairness((0,)) == 1.0

    def test_partial_inequality(self):
        """Partial inequality should be between 0 and 1."""
        fairness = compute_gini_fairness((3, 2, 1, 1, 0))
        assert 0.0 < fairness < 1.0

        fairness2 = compute_gini_fairness((5, 4, 3, 2, 1))
        assert 0.0 < fairness2 < 1.0

    def test_ordering_invariant(self):
        """Fairness should not depend on order."""
        f1 = compute_gini_fairness((5, 0, 0, 0, 0))
        f2 = compute_gini_fairness((0, 5, 0, 0, 0))
        f3 = compute_gini_fairness((0, 0, 0, 0, 5))
        assert f1 == f2 == f3

    def test_realistic_scenarios(self):
        """Test realistic meal distributions."""
        # Moderately fair distribution
        fairness = compute_gini_fairness((3, 3, 2, 2, 1))
        assert fairness > 0.5, f"Expected >0.5, got {fairness}"

        # Very unequal distribution
        fairness_unequal = compute_gini_fairness((10, 1, 0, 0, 0))
        assert fairness_unequal < 0.3, f"Expected <0.3, got {fairness_unequal}"


class TestEpisodeResultMetrics:
    """Test EpisodeResult computed properties."""

    def test_throughput_calculation(self):
        """Throughput = total_meals / total_timesteps."""
        result = EpisodeResult(
            episode_id=0,
            total_timesteps=50,
            deadlock=False,
            deadlock_timestep=None,
            meals_per_philosopher=(2, 3, 2, 1, 2),
            total_meals=10,
        )
        assert result.throughput == 10 / 50  # 0.2

    def test_throughput_zero_timesteps(self):
        """Throughput with zero timesteps should not crash."""
        result = EpisodeResult(
            episode_id=0,
            total_timesteps=0,
            deadlock=True,
            deadlock_timestep=0,
            meals_per_philosopher=(0, 0, 0, 0, 0),
            total_meals=0,
        )
        assert result.throughput == 0.0

    def test_starvation_count(self):
        """Starvation count = philosophers with 0 meals."""
        result = EpisodeResult(
            episode_id=0,
            total_timesteps=50,
            deadlock=False,
            deadlock_timestep=None,
            meals_per_philosopher=(5, 0, 3, 0, 2),
            total_meals=10,
        )
        assert result.starvation_count == 2

    def test_starvation_count_all_starving(self):
        """All philosophers starving (deadlock at start)."""
        result = EpisodeResult(
            episode_id=0,
            total_timesteps=5,
            deadlock=True,
            deadlock_timestep=5,
            meals_per_philosopher=(0, 0, 0, 0, 0),
            total_meals=0,
        )
        assert result.starvation_count == 5

    def test_starvation_count_none_starving(self):
        """No philosophers starving."""
        result = EpisodeResult(
            episode_id=0,
            total_timesteps=50,
            deadlock=False,
            deadlock_timestep=None,
            meals_per_philosopher=(2, 1, 3, 1, 2),
            total_meals=9,
        )
        assert result.starvation_count == 0

    def test_fairness_gini_uses_normalized(self):
        """Verify fairness_gini property uses normalized Gini."""
        # Maximum inequality
        result = EpisodeResult(
            episode_id=0,
            total_timesteps=50,
            deadlock=False,
            deadlock_timestep=None,
            meals_per_philosopher=(10, 0, 0, 0, 0),
            total_meals=10,
        )
        assert result.fairness_gini == 0.0

        # Perfect equality
        result_equal = EpisodeResult(
            episode_id=0,
            total_timesteps=50,
            deadlock=False,
            deadlock_timestep=None,
            meals_per_philosopher=(2, 2, 2, 2, 2),
            total_meals=10,
        )
        assert result_equal.fairness_gini == 1.0
