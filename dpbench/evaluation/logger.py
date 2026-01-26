"""JSONL logging for experiment traces."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import dpbench
from dpbench.core.types import (
    TableState,
    AgentDecision,
    BenchmarkConfig,
    LLMCallRecord,
    _generate_philosopher_names,
)


def _get_git_commit() -> str | None:
    """Get short git commit hash."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return result.stdout.strip()[:8] if result.returncode == 0 else None
    except Exception:
        return None


class ExperimentLogger:
    """JSONL logger for timestep-by-timestep experiment traces."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.log_path: Path | None = None
        self.transcript_path: Path | None = None
        self._names = _generate_philosopher_names(config.num_philosophers)
        self._start_time = datetime.now()
        self._run_id: str | None = None
        if config.log_dir:
            self._init_log(config.log_dir)

    def _init_log(self, log_dir: str) -> None:
        """Initialize log file with header."""
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Generate run ID: {code}_{timestamp} e.g., sim5c_20260126_143000
        timestamp = self._start_time.strftime("%Y%m%d_%H%M%S")
        self._run_id = f"{self.config.experiment_code}_{timestamp}"

        # All files share same base name for easy linking
        self.log_path = path / f"{self._run_id}.jsonl"

        # Initialize transcript file if enabled
        if self.config.save_transcript:
            self.transcript_path = path / f"{self._run_id}_transcript.txt"
            self._write_transcript_header()

        self._write({
            "type": "header",
            "timestamp": self._start_time.isoformat(),
            "config": {
                "mode": self.config.mode,
                "num_philosophers": self.config.num_philosophers,
                "num_episodes": self.config.num_episodes,
                "max_timesteps": self.config.max_timesteps,
                "communication": self.config.communication,
                "random_seed": self.config.random_seed,
            },
            "environment": {
                "dpbench_version": dpbench.__version__,
                "python_version": sys.version.split()[0],
                "git_commit": _get_git_commit(),
            },
        })

    def _write(self, data: dict[str, Any]) -> None:
        """Append JSON line to log file."""
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _write_transcript(self, text: str) -> None:
        """Append text to transcript file."""
        if self.transcript_path:
            with open(self.transcript_path, "a", encoding="utf-8") as f:
                f.write(text)

    def _write_transcript_header(self) -> None:
        """Write transcript file header."""
        comm = "enabled" if self.config.communication else "disabled"
        self._write_transcript(f"""{'='*80}
DPBench Experiment Transcript
Started: {self._start_time.isoformat()}
Config: {self.config.mode}, {self.config.num_philosophers} philosophers, communication={comm}
{'='*80}

""")

    def log_timestep(
        self,
        episode_id: int,
        timestep: int,
        state: TableState,
        decisions: dict[int, AgentDecision],
        llm_calls: dict[int, LLMCallRecord] | None = None,
    ) -> None:
        """Log one timestep."""
        if not self.log_path:
            return

        # Build JSONL record
        record = {
            "type": "timestep",
            "episode_id": episode_id,
            "timestep": timestep,
            "table_state": {
                "philosophers": [
                    {"id": p.id, "name": p.name, "state": p.state.value,
                     "meals_eaten": p.meals_eaten, "has_left_fork": p.has_left_fork, "has_right_fork": p.has_right_fork}
                    for p in state.philosophers
                ],
                "forks": [{"holder": f.holder} for f in state.forks],
            },
            "decisions": [
                {"philosopher_id": d.philosopher_id, "philosopher_name": self._names[d.philosopher_id],
                 "action": d.action.value, "message": d.message_to_neighbors, "reasoning": d.reasoning}
                for d in decisions.values()
            ],
        }

        # Add LLM call records if available
        if llm_calls:
            record["llm_calls"] = [
                {
                    "philosopher_id": r.philosopher_id,
                    "system_prompt": r.system_prompt,
                    "user_prompt": r.user_prompt,
                    "response": r.response,
                    "latency_ms": r.latency_ms,
                    "tokens_in": r.tokens_in,
                    "tokens_out": r.tokens_out,
                }
                for r in llm_calls.values()
            ]

        self._write(record)

        # Write to transcript
        if self.transcript_path:
            self._write_timestep_transcript(timestep, state, decisions, llm_calls)

    def _write_timestep_transcript(
        self,
        timestep: int,
        state: TableState,
        decisions: dict[int, AgentDecision],
        llm_calls: dict[int, LLMCallRecord] | None,
    ) -> None:
        """Write timestep to transcript in human-readable format."""
        lines = [f"Timestep {timestep}:", ""]

        # Table state
        phil_states = []
        for p in state.philosophers:
            forks = ""
            if p.has_left_fork:
                forks += "L"
            if p.has_right_fork:
                forks += "R"
            forks_str = f"({p.meals_eaten},{forks})" if forks else f"({p.meals_eaten})"
            phil_states.append(f"{p.name}={p.state.value.upper()}{forks_str}")
        lines.append(f"  Table: {' '.join(phil_states)}")

        fork_states = []
        for i, f in enumerate(state.forks):
            fork_states.append(f"[{self._names[f.holder] if f.holder is not None else 'free'}]")
        lines.append(f"  Forks: {' '.join(fork_states)}")
        lines.append("")

        # Agent decisions
        for i in range(self.config.num_philosophers):
            d = decisions.get(i)
            if d is None:
                continue
            lines.append(f"  [{self._names[i]}] Action: {d.action.value.upper()}")
            if d.reasoning:
                reasoning = d.reasoning.replace("\n", " ").strip()
                if len(reasoning) > 100:
                    reasoning = reasoning[:97] + "..."
                lines.append(f"       Reasoning: \"{reasoning}\"")
            if d.message_to_neighbors:
                lines.append(f"       Message: \"{d.message_to_neighbors}\"")

            # Add latency if available
            if llm_calls and i in llm_calls:
                lines.append(f"       Latency: {llm_calls[i].latency_ms:.0f}ms")
            lines.append("")

        self._write_transcript("\n".join(lines) + "\n")

    def log_episode_start(self, episode_id: int) -> None:
        """Log episode start (for transcript)."""
        if self.transcript_path:
            self._write_transcript(f"""
{'='*80}
Episode {episode_id}
{'-'*80}
""")

    def log_episode_end(
        self,
        episode_id: int,
        total_timesteps: int,
        deadlock: bool,
        meals: tuple[int, ...],
        throughput: float,
        fairness: float,
    ) -> None:
        """Log episode summary."""
        if not self.log_path:
            return
        self._write({
            "type": "episode_end",
            "episode_id": episode_id,
            "total_timesteps": total_timesteps,
            "deadlock": deadlock,
            "meals_per_philosopher": list(meals),
            "total_meals": sum(meals),
            "throughput": throughput,
            "fairness": fairness,
        })

        # Write to transcript
        if self.transcript_path:
            status = "DEADLOCK" if deadlock else "COMPLETED"
            self._write_transcript(f"""
{'-'*80}
Episode {episode_id} Result: {status} at timestep {total_timesteps}
  Meals: {list(meals)}
  Throughput: {throughput:.3f}
  Fairness: {fairness:.3f}

""")

    def get_log_path(self) -> Path | None:
        """Return log file path."""
        return self.log_path

    def get_transcript_path(self) -> Path | None:
        """Return transcript file path."""
        return self.transcript_path

    def get_run_id(self) -> str | None:
        """Return run ID for consistent file naming."""
        return self._run_id
