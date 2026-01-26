"""Console output handler for DPBench."""

import sys
from typing import TYPE_CHECKING

from .colors import Colors
from .components import (
    box,
    progress_bar,
    table,
    mini_table,
    section_header,
    status_badge,
)

if TYPE_CHECKING:
    from dpbench.core.types import TableState, AgentDecision, EpisodeResult


class Console:
    """Handle all terminal output for DPBench."""

    VERSION = "0.1.0"

    def __init__(self, no_color: bool = False):
        """
        Initialize console.

        Args:
            no_color: Force disable colors
        """
        # Ensure UTF-8 encoding for Windows compatibility
        if sys.platform == "win32":
            try:
                sys.stdout.reconfigure(encoding="utf-8")
                sys.stderr.reconfigure(encoding="utf-8")
            except (AttributeError, OSError):
                pass

        if no_color:
            Colors.disable()
        else:
            Colors.auto_configure()

    def header(self):
        """Print DPBench header banner."""
        content = [
            f"  {Colors.DIM}Benchmark for LLM Multi-Agent Coordination{Colors.RESET}",
            f"  {Colors.DIM}Based on Dijkstra's Dining Philosophers (1965){Colors.RESET}",
        ]
        print(box(f"DPBench v{self.VERSION}", content))

    def config(
        self,
        mode: str,
        philosophers: int,
        episodes: int,
        max_timesteps: int,
        communication: bool,
        model: str,
        log_dir: str | None = None,
    ):
        """Print experiment configuration."""
        print(f"\n{section_header('Configuration')}")
        data = {
            "Mode": mode,
            "Philosophers": str(philosophers),
            "Episodes": str(episodes),
            "Max Timesteps": str(max_timesteps),
            "Communication": status_badge("enabled", "success") if communication else status_badge("disabled", "warning"),
            "Model": model,
        }
        if log_dir:
            data["Log Directory"] = log_dir
        print(mini_table(data))

    def episode_start(self, episode_id: int, total_episodes: int):
        """Print episode start marker."""
        print(f"\n{section_header(f'Episode {episode_id + 1}/{total_episodes}', '═')}")

    def progress(self, current: int, total: int, inline: bool = True):
        """
        Update progress indicator.

        Args:
            current: Current episode number
            total: Total episodes
            inline: Whether to print inline (carriage return)
        """
        bar = progress_bar(current, total)
        if inline:
            print(f"\r{bar}", end="", flush=True)
        else:
            print(bar)

    def episode_marker(self, deadlock: bool):
        """Print single character episode marker (for non-verbose mode)."""
        if deadlock:
            print(f"{Colors.RED}D{Colors.RESET}", end="", flush=True)
        else:
            print(f"{Colors.GREEN}.{Colors.RESET}", end="", flush=True)

    def newline(self):
        """Print a newline."""
        print()

    def table_state(self, state: "TableState"):
        """
        Print the current table state.

        Args:
            state: Current TableState
        """
        n = state.num_philosophers
        print(f"\n{Colors.DIM}Timestep {state.timestep}{Colors.RESET}")

        headers = ["Agent", "State", "Left Fork", "Right Fork", "Meals"]
        rows = []

        for i, phil in enumerate(state.philosophers):
            left_fork_idx = i
            right_fork_idx = (i + 1) % n

            # State with color
            if phil.state.value == "eating":
                state_str = f"{Colors.GREEN}EATING{Colors.RESET}"
            else:
                state_str = f"{Colors.YELLOW}HUNGRY{Colors.RESET}"

            # Fork status
            left_fork = state.forks[left_fork_idx]
            right_fork = state.forks[right_fork_idx]

            if phil.has_left_fork:
                left_str = f"{Colors.GREEN}holding{Colors.RESET}"
            elif left_fork.is_free:
                left_str = f"{Colors.DIM}free{Colors.RESET}"
            else:
                left_str = f"{Colors.RED}taken{Colors.RESET}"

            if phil.has_right_fork:
                right_str = f"{Colors.GREEN}holding{Colors.RESET}"
            elif right_fork.is_free:
                right_str = f"{Colors.DIM}free{Colors.RESET}"
            else:
                right_str = f"{Colors.RED}taken{Colors.RESET}"

            rows.append([phil.name, state_str, left_str, right_str, str(phil.meals_eaten)])

        print(table(headers, rows))

    def agent_reasoning(self, timestep: int, decisions: dict[int, "AgentDecision"], num_philosophers: int):
        """
        Print agent reasoning and decisions.

        Args:
            timestep: Current timestep
            decisions: Dictionary of philosopher_id -> AgentDecision
            num_philosophers: Total number of philosophers
        """
        print(f"\n{section_header(f'Agent Decisions (t={timestep})')}")

        for i in range(num_philosophers):
            decision = decisions.get(i)
            if decision is None:
                continue

            name = f"P{i}"
            print(f"\n  {Colors.BOLD}[{name}]{Colors.RESET}")

            if decision.reasoning:
                reasoning = decision.reasoning.replace("\n", " ").strip()
                if len(reasoning) > 80:
                    reasoning = reasoning[:77] + "..."
                print(f"    {Colors.DIM}Thinking:{Colors.RESET} \"{reasoning}\"")

            if decision.message_to_neighbors:
                print(f"    {Colors.CYAN}Message:{Colors.RESET} \"{decision.message_to_neighbors}\"")

            # Action with color
            action = decision.action.value.upper()
            if action == "WAIT":
                action_str = f"{Colors.YELLOW}{action}{Colors.RESET}"
            elif action in ("GRAB_LEFT", "GRAB_RIGHT"):
                action_str = f"{Colors.CYAN}{action}{Colors.RESET}"
            elif action == "RELEASE":
                action_str = f"{Colors.GREEN}{action}{Colors.RESET}"
            else:
                action_str = action

            print(f"    {Colors.BOLD}Action:{Colors.RESET} {action_str}")

    def episode_summary(self, result: "EpisodeResult"):
        """
        Print episode summary.

        Args:
            result: Episode result
        """
        if result.deadlock:
            status = status_badge("DEADLOCK", "error")
        else:
            status = status_badge("COMPLETED", "success")

        print(f"\n{section_header('')}")
        print(f"  Episode {result.episode_id + 1}: {status}")
        print(f"    Timesteps: {result.total_timesteps}")
        print(f"    Total Meals: {result.total_meals}")
        print(f"    Throughput: {result.throughput:.3f} meals/step")
        print(f"    Fairness: {result.fairness_gini:.3f}")

    def results(self, metrics: dict, communication: bool = False):
        """
        Print final experiment results.

        Args:
            metrics: Aggregated metrics dictionary
            communication: Whether communication was enabled
        """
        content = [
            f"  {Colors.BOLD}PRIMARY METRICS{Colors.RESET}",
            "",
            f"    Deadlock Rate    {self._format_metric(metrics['deadlock_rate'] * 100, '%', lower_better=True)}",
            f"    Throughput       {self._format_metric(metrics['throughput'], ' meals/step')} {Colors.DIM}(std: {metrics['throughput_std']:.2f}){Colors.RESET}",
            f"    Fairness (Gini)  {self._format_metric(metrics['fairness'], '')} {Colors.DIM}(std: {metrics['fairness_std']:.2f}){Colors.RESET}",
            "",
            f"  {Colors.BOLD}SECONDARY METRICS{Colors.RESET}",
            "",
            f"    Time to Deadlock {metrics['time_to_deadlock']:.1f} steps",
            f"    Starvation Count {metrics['starvation_count']:.1f} agents",
        ]

        if communication and "message_consistency" in metrics:
            content.append(f"    Msg Consistency  {metrics['message_consistency']:.2f}")

        print(f"\n{box('Results', content)}")

    def _format_metric(self, value: float, suffix: str, lower_better: bool = False) -> str:
        """Format a metric value with color based on quality."""
        # Simple thresholds for coloring
        if lower_better:
            if value <= 20:
                color = Colors.GREEN
            elif value <= 50:
                color = Colors.YELLOW
            else:
                color = Colors.RED
        else:
            if value >= 0.8:
                color = Colors.GREEN
            elif value >= 0.5:
                color = Colors.YELLOW
            else:
                color = Colors.RED

        return f"{color}{value:.2f}{suffix}{Colors.RESET}"

    def error(self, message: str):
        """Print an error message."""
        print(f"{Colors.RED}Error:{Colors.RESET} {message}", file=sys.stderr)

    def warning(self, message: str):
        """Print a warning message."""
        print(f"{Colors.YELLOW}Warning:{Colors.RESET} {message}")

    def info(self, message: str):
        """Print an info message."""
        print(f"{Colors.CYAN}Info:{Colors.RESET} {message}")

    def success(self, message: str):
        """Print a success message."""
        print(f"{Colors.GREEN}Success:{Colors.RESET} {message}")

    def saved(self, path: str):
        """Print a 'saved to' message."""
        print(f"\n{Colors.DIM}Saved to:{Colors.RESET} {path}")
