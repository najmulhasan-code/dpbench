"""Core type definitions for DPBench."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

import numpy as np


class Action(Enum):
    """Actions available to each philosopher per timestep."""
    GRAB_LEFT = "grab_left"
    GRAB_RIGHT = "grab_right"
    RELEASE = "release"
    WAIT = "wait"


class PhilosopherState(Enum):
    """Philosopher states in the 'always hungry' variant."""
    HUNGRY = "hungry"
    EATING = "eating"


def compute_gini_fairness(meals: tuple[int, ...]) -> float:
    """
    Compute fairness as 1 - normalized Gini coefficient.

    G = (2 * sum(i * x_i)) / (n * sum(x)) - (n + 1) / n

    where x_i are meal counts sorted ascending, i is 1-indexed rank.

    The Gini coefficient is normalized so that maximum inequality
    (one person has everything) gives Gini = 1.0, not (n-1)/n.

    Returns:
        1.0 for perfect equality, 0.0 for maximum inequality.
    """
    arr = np.array(meals)
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 1.0
    if n == 1:
        return 1.0  # Single philosopher = perfect equality
    sorted_arr = np.sort(arr)
    idx = np.arange(1, n + 1)
    gini = (2 * np.sum(idx * sorted_arr)) / (n * np.sum(sorted_arr)) - (n + 1) / n
    # Normalize: max Gini for n discrete samples is (n-1)/n, scale to [0, 1]
    gini_normalized = gini * n / (n - 1)
    return max(0.0, min(1.0, 1.0 - gini_normalized))


def _generate_philosopher_names(n: int) -> tuple[str, ...]:
    """Generate philosopher identifiers (P0, P1, P2, ...)."""
    return tuple(f"P{i}" for i in range(n))


@dataclass(frozen=True)
class ForkState:
    """State of a single fork. holder=None means the fork is free."""
    holder: Optional[int] = None

    @property
    def is_free(self) -> bool:
        return self.holder is None


@dataclass(frozen=True)
class Philosopher:
    """State of a single philosopher."""
    id: int
    name: str
    state: PhilosopherState = PhilosopherState.HUNGRY
    meals_eaten: int = 0
    has_left_fork: bool = False
    has_right_fork: bool = False


@dataclass(frozen=True)
class Observation:
    """Local observation visible to a philosopher (partial observability)."""
    philosopher_id: int
    philosopher_name: str
    state: PhilosopherState
    meals_eaten: int
    left_fork_available: bool
    right_fork_available: bool
    holding_left: bool
    holding_right: bool
    left_neighbor_message: Optional[str] = None
    right_neighbor_message: Optional[str] = None


@dataclass(frozen=True)
class TableState:
    """Global state of the dining table."""
    philosophers: tuple[Philosopher, ...]
    forks: tuple[ForkState, ...]
    timestep: int = 0

    @property
    def num_philosophers(self) -> int:
        return len(self.philosophers)


@dataclass(frozen=True)
class AgentDecision:
    """Decision made by an agent for one timestep."""
    philosopher_id: int
    action: Action
    message_to_neighbors: Optional[str] = None
    reasoning: Optional[str] = None


@dataclass(frozen=True)
class LLMCallRecord:
    """Record of a single LLM API call for logging."""
    philosopher_id: int
    system_prompt: str
    user_prompt: str
    response: str
    latency_ms: float
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None


@dataclass(frozen=True)
class StepResult:
    """Result of applying actions for one timestep."""
    new_state: TableState
    deadlock: bool
    meals_this_step: int


@dataclass
class EpisodeResult:
    """Metrics collected from one episode."""
    episode_id: int
    total_timesteps: int
    deadlock: bool
    deadlock_timestep: Optional[int]
    meals_per_philosopher: tuple[int, ...]
    total_meals: int = 0
    all_decisions: list = field(default_factory=list)

    @property
    def throughput(self) -> float:
        """Meals per timestep."""
        if self.total_timesteps == 0:
            return 0.0
        return self.total_meals / self.total_timesteps

    @property
    def fairness_gini(self) -> float:
        """Fairness computed via Gini coefficient."""
        return compute_gini_fairness(self.meals_per_philosopher)

    @property
    def starvation_count(self) -> int:
        """Number of philosophers with zero meals."""
        return sum(1 for m in self.meals_per_philosopher if m == 0)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    model_fn: Callable[[str, str], str]
    system_prompt: str
    decision_prompt: str
    mode: str = "simultaneous"
    communication: bool = False
    num_philosophers: int = 5
    num_episodes: int = 30
    max_timesteps: int = 50
    verbose: bool = False
    show_reasoning: bool = False
    log_dir: Optional[str] = None
    random_seed: Optional[int] = None
    save_transcript: bool = False

    @property
    def experiment_code(self) -> str:
        """Generate short experiment code: sim5c, seq3nc, etc."""
        mode = "sim" if self.mode == "simultaneous" else "seq"
        comm = "c" if self.communication else "nc"
        return f"{mode}{self.num_philosophers}{comm}"

    def __post_init__(self):
        if self.num_philosophers < 2:
            raise ValueError(f"num_philosophers must be >= 2, got {self.num_philosophers}")
        if self.mode not in ("simultaneous", "sequential"):
            raise ValueError(f"mode must be 'simultaneous' or 'sequential', got {self.mode}")
        if not self.system_prompt or not self.system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")
        if not self.decision_prompt or not self.decision_prompt.strip():
            raise ValueError("decision_prompt cannot be empty")
        if not callable(self.model_fn):
            raise TypeError(f"model_fn must be callable, got {type(self.model_fn).__name__}")
