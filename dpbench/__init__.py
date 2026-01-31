"""DPBench: Benchmark for LLM multi-agent coordination."""

__version__ = "0.1.0"

from dpbench.benchmark import Benchmark
from dpbench.core.types import (
    Action,
    PhilosopherState,
    Philosopher,
    ForkState,
    TableState,
    Observation,
    AgentDecision,
    StepResult,
    EpisodeResult,
    BenchmarkConfig,
    compute_gini_fairness,
)
from dpbench.core.environment import (
    create_initial_state,
    get_observation,
    step,
    is_deadlock,
    format_table_state,
)
from dpbench.models.base import ModelFunction, validate_model_function
from dpbench.config.prompts import TEMPLATE_VARIABLES, get_template_variables
from dpbench.evaluation.metrics import compute_aggregate_metrics
from dpbench.runner import run_episode, run_experiment

__all__ = [
    "__version__",
    "Benchmark",
    "Action",
    "PhilosopherState",
    "Philosopher",
    "ForkState",
    "TableState",
    "Observation",
    "AgentDecision",
    "StepResult",
    "EpisodeResult",
    "BenchmarkConfig",
    "compute_gini_fairness",
    "create_initial_state",
    "get_observation",
    "step",
    "is_deadlock",
    "format_table_state",
    "ModelFunction",
    "validate_model_function",
    "TEMPLATE_VARIABLES",
    "get_template_variables",
    "compute_aggregate_metrics",
    "run_episode",
    "run_experiment",
]
