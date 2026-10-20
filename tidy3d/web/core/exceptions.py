"""Custom Tidy3D exceptions"""

from __future__ import annotations

from .core_config import get_logger


class WebError(Exception):
    """Any error in tidy3d"""

    def __init__(
        self,
        message: str | None = None,
        *,
        status_code: int | None = None,
        log_error: bool = True,
    ) -> None:
        """Log the message unless the caller will report an aggregate error."""
        super().__init__(message)
        self.status_code = status_code
        if log_error:
            get_logger().error(message)


class WebNotFoundError(WebError):
    """A generic error indicating an HTTP 404 (resource not found)."""
