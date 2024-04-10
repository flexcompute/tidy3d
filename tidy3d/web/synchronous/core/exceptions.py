"""Custom Tidy3D exceptions"""
from .core_config import get_logger


class WebError(Exception):
    """Any error in tidy3d"""

    def __init__(self, message: str = None):
        """Log just the error message and then raise the Exception."""
        log = get_logger()
        super().__init__(message)
        log.error(message)
