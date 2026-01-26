"""ANSI color codes for terminal output."""

import sys


class Colors:
    """ANSI escape codes for terminal colors and styles."""

    # Reset
    RESET = "\033[0m"

    # Basic colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Status colors (semantic)
    SUCCESS = GREEN
    ERROR = RED
    WARNING = YELLOW
    INFO = CYAN

    _disabled = False

    @classmethod
    def disable(cls):
        """Disable all colors (for non-TTY or --no-color)."""
        if cls._disabled:
            return
        cls._disabled = True
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith("_"):
                setattr(cls, attr, "")

    @classmethod
    def is_tty(cls) -> bool:
        """Check if stdout is a TTY."""
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    @classmethod
    def auto_configure(cls):
        """Auto-disable colors if not running in a TTY."""
        if not cls.is_tty():
            cls.disable()
