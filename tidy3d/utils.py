"""General utility functions."""

from __future__ import annotations

import sys


def is_running_pytest() -> bool:
    """Return True if the code is currently running under pytest."""
    return "pytest" in sys.modules
