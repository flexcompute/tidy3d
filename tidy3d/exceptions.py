"""Custom Tidy3D exceptions"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .log import log

if TYPE_CHECKING:
    from typing import Optional


def _format_exception_detail(exc: BaseException) -> str:
    """Format an exception detail for inclusion in a user-facing chained message."""

    detail = str(exc).strip()
    return detail or type(exc).__name__


def format_chained_exception_message(message: str, exc: BaseException) -> str:
    """Append a standardized cause description to a user-facing exception message."""

    message = message.rstrip()
    if message.endswith(":"):
        message = message[:-1].rstrip()

    if isinstance(exc, KeyError):
        return message

    exc_type = type(exc).__name__
    detail = _format_exception_detail(exc)
    if detail == exc_type:
        return f"{message} (cause: {exc_type})"

    return f"{message} (cause: {exc_type}: {detail})"


class Tidy3dError(ValueError):
    """Any error in tidy3d"""

    def __init__(self, message: Optional[str] = None, log_error: bool = True) -> None:
        """Log just the error message and then raise the Exception."""
        super().__init__(message)
        if log_error:
            log.error(message)


class ConfigError(Tidy3dError):
    """Error when configuring Tidy3d."""


class Tidy3dKeyError(Tidy3dError):
    """Could not find a key in a Tidy3d dictionary."""


class ValidationError(Tidy3dError):
    """Error when constructing Tidy3d components."""


class SetupError(Tidy3dError):
    """Error regarding the setup of the components (outside of domains, etc)."""


class FileError(Tidy3dError):
    """Error reading or writing to file."""


class WebError(Tidy3dError):
    """Error with the webAPI."""


class AuthenticationError(Tidy3dError):
    """Error authenticating a user through webapi webAPI."""


class DataError(Tidy3dError):
    """Error accessing data."""


class Tidy3dImportError(Tidy3dError):
    """Error importing a package needed for tidy3d."""


class Tidy3dNotImplementedError(Tidy3dError):
    """Error when a functionality is not (yet) supported."""


class AdjointError(Tidy3dError):
    """An error in setting up the adjoint solver."""
