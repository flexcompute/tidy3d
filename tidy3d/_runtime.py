"""Runtime environment detection for tidy3d.

This module must have ZERO dependencies on other tidy3d modules to avoid
circular imports. It is imported very early in the initialization chain.
"""

from __future__ import annotations

import sys
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

# Detect WASM/Pyodide environment where web and filesystem features are unavailable
WASM_BUILD = "pyodide" in sys.modules or sys.platform == "emscripten"


# --- validation mode -------------------------------------------------------------------------
# Kept here, with no tidy3d imports, so both the config handler and Tidy3dBaseModel can use it
# without depending on each other's import order.

VALIDATION_MODES = ("full", "fast")
FAST_VALIDATION_DEFAULT = False  # config.validation.mode, process-wide
VALIDATE_OVERRIDE: ContextVar[str | None] = ContextVar("tidy3d_validate_override", default=None)


def set_validation_mode(mode: str) -> None:
    """Set the process-wide default from ``config.validation.mode``: ``"full"`` or ``"fast"``."""
    global FAST_VALIDATION_DEFAULT
    if mode not in VALIDATION_MODES:
        raise ValueError(f"Unknown validation mode {mode!r}; expected one of {VALIDATION_MODES}.")
    FAST_VALIDATION_DEFAULT = mode == "fast"


def fast_validation_active() -> bool:
    """True when tidy3d's check validators are skipped in the current context."""
    override = VALIDATE_OVERRIDE.get()
    return FAST_VALIDATION_DEFAULT if override is None else override == "fast"


def always_validate(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a validator that derives or changes a value, so ``fast`` mode keeps it.

    Place it directly above the ``def``, below the pydantic decorator, including inside
    validator factories. ``before`` and ``plain`` validators never need it; ``after`` and
    ``wrap`` validators that only check do not want it. The rule: mark an ``after`` validator
    if it returns something other than the value it was given, or sets an attribute on
    ``self``. An unmarked deriving validator would leave its derived field unset in ``fast``
    mode. The current markers were placed from an AST scan of the package, kept with the
    benchmark exploration in the flex repository.
    """
    func._tidy3d_always_validate = True  # type: ignore[attr-defined]
    return func
