"""Terminal UI components: boxes, tables, progress bars."""

from .colors import Colors


def box(title: str, content: list[str], width: int = 60) -> str:
    """
    Draw a box with title and content.

    Args:
        title: Box title (centered in header)
        content: List of content lines
        width: Total box width

    Returns:
        Formatted box string
    """
    inner_width = width - 2
    lines = []

    # Top border with title
    lines.append(f"┌{'─' * inner_width}┐")
    lines.append(f"│{Colors.BOLD}{title.center(inner_width)}{Colors.RESET}│")
    lines.append(f"├{'─' * inner_width}┤")

    # Content lines
    for line in content:
        # Strip ANSI codes for length calculation
        stripped = _strip_ansi(line)
        padding = inner_width - len(stripped)
        if padding < 0:
            # Truncate if too long
            lines.append(f"│{line[:inner_width]}│")
        else:
            lines.append(f"│{line}{' ' * padding}│")

    # Bottom border
    lines.append(f"└{'─' * inner_width}┘")

    return "\n".join(lines)


def progress_bar(
    current: int,
    total: int,
    width: int = 30,
    show_percent: bool = True,
    show_count: bool = True,
) -> str:
    """
    Draw a progress bar.

    Args:
        current: Current progress value
        total: Total value
        width: Bar width in characters
        show_percent: Show percentage
        show_count: Show current/total count

    Returns:
        Formatted progress bar string
    """
    if total == 0:
        pct = 0.0
    else:
        pct = current / total

    filled = int(width * pct)
    remaining = width - filled

    # Build bar
    if filled == width:
        bar = f"{Colors.GREEN}{'━' * filled}{Colors.RESET}"
    elif filled > 0:
        bar = f"{Colors.GREEN}{'━' * filled}╸{Colors.RESET}{Colors.DIM}{'─' * (remaining - 1)}{Colors.RESET}"
    else:
        bar = f"{Colors.DIM}{'─' * width}{Colors.RESET}"

    parts = [bar]
    if show_percent:
        parts.append(f"{pct * 100:5.1f}%")
    if show_count:
        parts.append(f"({current}/{total})")

    return " ".join(parts)


def table(headers: list[str], rows: list[list[str]], colors: list[str] | None = None) -> str:
    """
    Draw a table with headers and rows.

    Args:
        headers: Column headers
        rows: List of rows (each row is a list of cell values)
        colors: Optional list of ANSI color codes for each column

    Returns:
        Formatted table string
    """
    if not headers:
        return ""

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(_strip_ansi(str(cell))))

    # Build separators
    top_sep = "┬".join("─" * (w + 2) for w in widths)
    mid_sep = "┼".join("─" * (w + 2) for w in widths)
    bot_sep = "┴".join("─" * (w + 2) for w in widths)

    lines = []

    # Top border
    lines.append(f"┌{top_sep}┐")

    # Header row
    header_cells = []
    for i, (h, w) in enumerate(zip(headers, widths)):
        cell = f" {Colors.BOLD}{h.ljust(w)}{Colors.RESET} "
        header_cells.append(cell)
    lines.append(f"│{'│'.join(header_cells)}│")

    # Header separator
    lines.append(f"├{mid_sep}┤")

    # Data rows
    for row in rows:
        cells = []
        for i, w in enumerate(widths):
            if i < len(row):
                cell_value = str(row[i])
                stripped_len = len(_strip_ansi(cell_value))
                padding = w - stripped_len
                cell = f" {cell_value}{' ' * padding} "
            else:
                cell = f" {' ' * w} "
            cells.append(cell)
        lines.append(f"│{'│'.join(cells)}│")

    # Bottom border
    lines.append(f"└{bot_sep}┘")

    return "\n".join(lines)


def mini_table(data: dict[str, str], label_width: int = 15) -> str:
    """
    Draw a simple key-value table (no borders).

    Args:
        data: Dictionary of label -> value
        label_width: Width for labels

    Returns:
        Formatted key-value pairs
    """
    lines = []
    for label, value in data.items():
        lines.append(f"  {Colors.DIM}{label.ljust(label_width)}{Colors.RESET} {value}")
    return "\n".join(lines)


def section_header(title: str, char: str = "─", width: int = 60) -> str:
    """
    Draw a section header line.

    Args:
        title: Section title
        char: Character for the line
        width: Total width

    Returns:
        Formatted header string
    """
    if title:
        padding = (width - len(title) - 2) // 2
        return f"{char * padding} {Colors.BOLD}{title}{Colors.RESET} {char * padding}"
    return char * width


def status_badge(text: str, status: str = "info") -> str:
    """
    Create a colored status badge.

    Args:
        text: Badge text
        status: One of 'success', 'error', 'warning', 'info'

    Returns:
        Colored badge string
    """
    color_map = {
        "success": Colors.GREEN,
        "error": Colors.RED,
        "warning": Colors.YELLOW,
        "info": Colors.CYAN,
    }
    color = color_map.get(status, Colors.WHITE)
    return f"{color}{text}{Colors.RESET}"


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text for length calculation."""
    import re
    ansi_pattern = re.compile(r"\033\[[0-9;]*m")
    return ansi_pattern.sub("", text)
