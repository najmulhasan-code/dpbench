"""Dining philosophers environment: state transitions and deadlock detection."""

from dpbench.core.types import (
    Action,
    PhilosopherState,
    Philosopher,
    ForkState,
    TableState,
    Observation,
    AgentDecision,
    StepResult,
    _generate_philosopher_names,
)


def create_initial_state(num_philosophers: int = 5) -> TableState:
    """Create initial table state with all philosophers hungry and all forks free."""
    names = _generate_philosopher_names(num_philosophers)
    philosophers = tuple(
        Philosopher(id=i, name=names[i], state=PhilosopherState.HUNGRY)
        for i in range(num_philosophers)
    )
    forks = tuple(ForkState() for _ in range(num_philosophers))
    return TableState(philosophers=philosophers, forks=forks, timestep=0)


def get_left_fork_index(philosopher_id: int, num_philosophers: int) -> int:
    """Return index of the fork to the philosopher's left."""
    return philosopher_id


def get_right_fork_index(philosopher_id: int, num_philosophers: int) -> int:
    """Return index of the fork to the philosopher's right."""
    return (philosopher_id + 1) % num_philosophers


def get_left_neighbor_id(philosopher_id: int, num_philosophers: int) -> int:
    """Return ID of the philosopher's left neighbor."""
    return (philosopher_id - 1) % num_philosophers


def get_right_neighbor_id(philosopher_id: int, num_philosophers: int) -> int:
    """Return ID of the philosopher's right neighbor."""
    return (philosopher_id + 1) % num_philosophers


def get_observation(
    state: TableState,
    philosopher_id: int,
    messages: dict[int, str] | None = None,
) -> Observation:
    """Build the local observation for a philosopher."""
    n = state.num_philosophers
    phil = state.philosophers[philosopher_id]
    left_idx = get_left_fork_index(philosopher_id, n)
    right_idx = get_right_fork_index(philosopher_id, n)
    left_fork = state.forks[left_idx]
    right_fork = state.forks[right_idx]

    left_available = left_fork.is_free or left_fork.holder == philosopher_id
    right_available = right_fork.is_free or right_fork.holder == philosopher_id

    left_msg, right_msg = None, None
    if messages:
        left_msg = messages.get(get_left_neighbor_id(philosopher_id, n))
        right_msg = messages.get(get_right_neighbor_id(philosopher_id, n))

    return Observation(
        philosopher_id=philosopher_id,
        philosopher_name=phil.name,
        state=phil.state,
        meals_eaten=phil.meals_eaten,
        left_fork_available=left_available,
        right_fork_available=right_available,
        holding_left=phil.has_left_fork,
        holding_right=phil.has_right_fork,
        left_neighbor_message=left_msg,
        right_neighbor_message=right_msg,
    )


def _apply_single_action(
    state: TableState,
    philosopher_id: int,
    action: Action,
) -> TableState:
    """Apply one philosopher's action. Actions on taken forks fail silently."""
    n = state.num_philosophers
    phil = state.philosophers[philosopher_id]
    forks = list(state.forks)
    philosophers = list(state.philosophers)
    left_idx = get_left_fork_index(philosopher_id, n)
    right_idx = get_right_fork_index(philosopher_id, n)

    has_left = phil.has_left_fork
    has_right = phil.has_right_fork
    new_state = phil.state
    new_meals = phil.meals_eaten

    if action == Action.GRAB_LEFT and forks[left_idx].is_free:
        forks[left_idx] = ForkState(holder=philosopher_id)
        has_left = True
    elif action == Action.GRAB_RIGHT and forks[right_idx].is_free:
        forks[right_idx] = ForkState(holder=philosopher_id)
        has_right = True
    elif action == Action.RELEASE:
        if phil.has_left_fork:
            forks[left_idx] = ForkState(holder=None)
            has_left = False
        if phil.has_right_fork:
            forks[right_idx] = ForkState(holder=None)
            has_right = False

    # Transition to eating if both forks acquired
    if has_left and has_right and phil.state == PhilosopherState.HUNGRY:
        new_state = PhilosopherState.EATING
        new_meals += 1

    # Auto-release after eating
    if phil.state == PhilosopherState.EATING:
        new_state = PhilosopherState.HUNGRY
        forks[left_idx] = ForkState(holder=None)
        forks[right_idx] = ForkState(holder=None)
        has_left = False
        has_right = False

    philosophers[philosopher_id] = Philosopher(
        id=phil.id,
        name=phil.name,
        state=new_state,
        meals_eaten=new_meals,
        has_left_fork=has_left,
        has_right_fork=has_right,
    )
    return TableState(
        philosophers=tuple(philosophers),
        forks=tuple(forks),
        timestep=state.timestep,
    )


def _apply_actions_simultaneous(
    state: TableState,
    decisions: list[AgentDecision],
) -> TableState:
    """Apply all actions simultaneously. Conflicts resolved by lower ID winning."""
    n = state.num_philosophers
    forks = list(state.forks)
    philosophers = list(state.philosophers)
    fork_requests: dict[int, list[int]] = {i: [] for i in range(n)}

    # Collect grab requests and process releases
    for decision in decisions:
        pid = decision.philosopher_id
        phil = state.philosophers[pid]
        left_idx = get_left_fork_index(pid, n)
        right_idx = get_right_fork_index(pid, n)

        if decision.action == Action.GRAB_LEFT and forks[left_idx].is_free:
            fork_requests[left_idx].append(pid)
        elif decision.action == Action.GRAB_RIGHT and forks[right_idx].is_free:
            fork_requests[right_idx].append(pid)
        elif decision.action == Action.RELEASE:
            if phil.has_left_fork:
                forks[left_idx] = ForkState(holder=None)
            if phil.has_right_fork:
                forks[right_idx] = ForkState(holder=None)

    # Resolve conflicts: lowest ID wins
    for fork_idx, requesters in fork_requests.items():
        if requesters:
            forks[fork_idx] = ForkState(holder=min(requesters))

    # Update philosopher states
    for i in range(n):
        phil = state.philosophers[i]
        left_idx = get_left_fork_index(i, n)
        right_idx = get_right_fork_index(i, n)
        has_left = forks[left_idx].holder == i
        has_right = forks[right_idx].holder == i
        new_state = phil.state
        new_meals = phil.meals_eaten

        if phil.state == PhilosopherState.EATING:
            forks[left_idx] = ForkState(holder=None)
            forks[right_idx] = ForkState(holder=None)
            has_left = False
            has_right = False
            new_state = PhilosopherState.HUNGRY
        elif has_left and has_right and phil.state == PhilosopherState.HUNGRY:
            new_state = PhilosopherState.EATING
            new_meals += 1

        philosophers[i] = Philosopher(
            id=phil.id,
            name=phil.name,
            state=new_state,
            meals_eaten=new_meals,
            has_left_fork=has_left,
            has_right_fork=has_right,
        )

    return TableState(
        philosophers=tuple(philosophers),
        forks=tuple(forks),
        timestep=state.timestep + 1,
    )


def _apply_actions_sequential(
    state: TableState,
    decisions: list[AgentDecision],
) -> TableState:
    """Apply actions sequentially in order."""
    current = state
    for decision in decisions:
        current = _apply_single_action(current, decision.philosopher_id, decision.action)
    return TableState(
        philosophers=current.philosophers,
        forks=current.forks,
        timestep=state.timestep + 1,
    )


def is_deadlock(state: TableState) -> bool:
    """Check for deadlock: all hungry, each holding exactly one fork."""
    n = state.num_philosophers
    for phil in state.philosophers:
        if phil.state != PhilosopherState.HUNGRY:
            return False
        if phil.has_left_fork and phil.has_right_fork:
            return False
        if not phil.has_left_fork and not phil.has_right_fork:
            left_idx = get_left_fork_index(phil.id, n)
            right_idx = get_right_fork_index(phil.id, n)
            if state.forks[left_idx].is_free or state.forks[right_idx].is_free:
                return False

    for phil in state.philosophers:
        if not (phil.has_left_fork != phil.has_right_fork):
            return False
    return True


def step(
    state: TableState,
    decisions: list[AgentDecision],
    mode: str = "simultaneous",
) -> StepResult:
    """Execute one timestep and return the result."""
    if mode == "simultaneous":
        new_state = _apply_actions_simultaneous(state, decisions)
    else:
        new_state = _apply_actions_sequential(state, decisions)

    meals = sum(
        1 for new_p, old_p in zip(new_state.philosophers, state.philosophers)
        if new_p.meals_eaten > old_p.meals_eaten
    )
    return StepResult(
        new_state=new_state,
        deadlock=is_deadlock(new_state),
        meals_this_step=meals,
    )


def format_table_state(state: TableState) -> str:
    """Format table state for display."""
    lines = [f"Timestep {state.timestep}"]
    for phil in state.philosophers:
        forks = []
        if phil.has_left_fork:
            forks.append("L")
        if phil.has_right_fork:
            forks.append("R")
        forks_str = "+".join(forks) if forks else "-"
        lines.append(f"  {phil.name}: {phil.state.value} [{forks_str}] meals={phil.meals_eaten}")
    return "\n".join(lines)
