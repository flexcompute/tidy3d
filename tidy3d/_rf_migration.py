"""Helpers for RF APIs that moved from Tidy3D to Flexcompute RF.

Every helper reports through the Tidy3D logger, the channel the rest of the
package uses for deprecations (see ``plugins/dispersion/fit_web.py``) and the
one ``Simulation`` already uses for its RF notice, so the messages stay under
the logging configuration users control.

Each distinct message is logged once per process, and the messages are worded so
that one import line yields one message: a module that moved wholesale blames
itself on both import and attribute access, so the two texts match and collapse,
while a module that survived blames the individual name that left it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tidy3d.log import log

if TYPE_CHECKING:
    from collections.abc import Callable

RF_MIGRATION_MESSAGE = (
    "RF functionality has moved from Tidy3D to Flexcompute RF. Install "
    "'flexcompute-rf' and import from 'flexcompute.rf.tidy3d' instead of "
    "'{path}'."
)

# Only worth saying to someone leaving a path that used to sit alongside the
# modal modeler; on a 'plugins.microwave' path it answers a question nobody
# asked. Spelled out, because the bare note reads as a contradiction next to a
# sentence telling the user to stop importing from the same module.
RF_MODAL_SURVIVES_SUFFIX = (
    " Only the terminal component modeler moved; the modal one stayed in Tidy3D "
    "and is still available from 'tidy3d.plugins.smatrix'."
)

_MODAL_SURVIVES_PATHS = ("tidy3d.plugins.smatrix", "tidy3d.rf")

RF_DEPRECATION_MESSAGE = (
    "The module '{path}' is deprecated and will be removed in Tidy3D 3.0. RF "
    "development continues in Flexcompute RF; install 'flexcompute-rf' and "
    "import from 'flexcompute.rf.tidy3d' instead."
)

RF_RELOCATION_MESSAGE = "'{path}' has moved to '{new_path}'."


def warn_rf_migration(path: str) -> None:
    """Log that an RF import path is now owned by Flexcompute RF."""
    message = RF_MIGRATION_MESSAGE.format(path=path)
    if path.startswith(_MODAL_SURVIVES_PATHS):
        message += RF_MODAL_SURVIVES_SUFFIX
    log.warning(message, log_once=True)


def warn_rf_deprecated(path: str) -> None:
    """Log that an RF import path still works but is scheduled for removal."""
    log.warning(RF_DEPRECATION_MESSAGE.format(path=path), log_once=True)


def missing_rf_attribute(path: str, name: str, *, report_path: str | None = None) -> None:
    """Log and fail access to an attribute now owned by Flexcompute RF.

    ``report_path`` overrides the path the message blames. A module that moved
    wholesale has already reported itself on import, so its attribute access
    names the module again rather than a longer, unique path that ``log_once``
    cannot match against what was already said.
    """
    full_path = f"{path}.{name}"
    warn_rf_migration(full_path if report_path is None else report_path)
    raise AttributeError(f"'{full_path}' has moved to 'flexcompute.rf.tidy3d'.")


def relocated_rf_attribute(path: str, name: str, new_path: str) -> None:
    """Log and fail access to an attribute that moved elsewhere within Tidy3D.

    For names that stayed in Tidy3D and are still the supported spelling: the
    shared photonics S-matrix classes that ``tidy3d.rf`` used to re-export. A
    name that only survives on paper, kept for SemVer and wanting no new
    callers, is reported as migrated instead, so nobody is sent to it.
    """
    full_path = f"{path}.{name}"
    log.warning(RF_RELOCATION_MESSAGE.format(path=full_path, new_path=new_path), log_once=True)
    raise AttributeError(f"'{full_path}' has moved to '{new_path}'.")


def migrated_rf_module(path: str) -> Callable[[str], None]:
    """Log on module import and return an informative module ``__getattr__``.

    A shim belongs where a removal starts: the package, where a whole namespace
    went, otherwise the module itself. Python imports every parent before it
    resolves a child, so one shim speaks for everything under it -- a path
    deleted beneath it still reports the move, then fails with the
    ``ModuleNotFoundError`` naming the level that is missing.

    Importing the shim reports, so a removed module says so even if nothing is
    read from it, and attribute access reports the same fact and so names the
    same path. Identical text is all ``log_once`` needs to leave one import line
    with one message; which attribute was asked for is left to the error.
    """
    warn_rf_migration(path)

    def _missing_attribute(name: str) -> None:
        # The import machinery, pytest and IPython all probe module dunders
        # (``__path__``, ``__all__``, ``__wrapped__``, ...). Those never moved
        # anywhere, so answer them plainly instead of claiming a migration.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"module {path!r} has no attribute {name!r}")
        missing_rf_attribute(path, name, report_path=path)

    return _missing_attribute
