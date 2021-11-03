"""Logging and error-handling for Tidy3d."""
import logging
from rich.logging import RichHandler

# TODO: more logging features (to file, etc).

FORMAT = "%(message)s"

logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

log = logging.getLogger("rich")

level_map = {
    "error": 40,
    "warning": 30,
    "info": 20,
    "debug": 10,
}


class Tidy3DError(Exception):
    """Any error in tidy3d"""

    def __init__(self, message: str = None):
        """Log just the error message and then raise the Exception."""
        log.error(message)
        super().__init__(self, message)


class ConfigError(Tidy3DError):
    """Error when configuring Tidy3d."""


class ValidationError(Tidy3DError):
    """eError when constructing Tidy3d components."""


class SetupError(Tidy3DError):
    """Error regarding the setup of the components (outside of domains, etc)."""


class FileError(Tidy3DError):
    """Error reading or writing to file."""


class WebError(Tidy3DError):
    """Error with the webAPI."""


class AuthenticationError(Tidy3DError):
    """Error authenticating a user through webapi webAPI."""


class DataError(Tidy3DError):
    """Error accessing data."""


def logging_level(level: str = "info") -> None:
    """Set tidy3d logging level priority.

    Parameters
    ----------
    level : str
        One of ``{'debug', 'info', 'warning', 'error'}`` (listed in increasing priority).
        the lowest priority level of logging messages to display.
    """
    if level not in level_map:
        raise ConfigError(f"logging level {level} not supported, must be in {level_map.keys()}.")
    level_int = level_map[level]
    log.setLevel(level_int)
