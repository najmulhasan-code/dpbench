"""Console output handler for DPBench."""

import sys
from typing import TYPE_CHECKING

from .colors import Colors
from .components import (
    progress_bar,
    table,
    section_header,
    status_badge,
)

if TYPE_CHECKING:
    from dpbench.core.types import TableState, AgentDecision, EpisodeResult


class Console:
    """Handle all terminal output for DPBench experiments."""

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
