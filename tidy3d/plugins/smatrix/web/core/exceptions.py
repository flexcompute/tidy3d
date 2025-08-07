"""Custom Tidy3D exceptions"""

from __future__ import annotations

from typing import Optional

from .core_config import get_logger


class WebError(Exception):
    """Any error in tidy3d"""

    def __init__(self, message: Optional[str] = None):
        """Log just the error message and then raise the Exception."""
        log = get_logger()
        super().__init__(message)
        log.error(message)


class WebNotFoundError(WebError):
    """A generic error indicating an HTTP 404 (resource not found)."""
