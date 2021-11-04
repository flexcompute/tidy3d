"""Logging and error-handling for Tidy3d."""
import logging
from rich.logging import RichHandler

# TODO: more logging features (to file, etc).

FORMAT = "%(message)s"

DEFAULT_LEVEL = "INFO"

logging.basicConfig(level=DEFAULT_LEVEL, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

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


class Tidy3dKeyError(Tidy3DError):
    """Could not find a key in a Tidy3d dictionary."""


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


def _get_level_int(level: str) -> int:
    """Get the integer corresponding to the level string."""
    level = level.lower()
    if level not in level_map:
        raise ConfigError(
            f"logging level {level} not supported, " f"must be in {list(level_map.keys())}."
        )
    return level_map[level]


def set_logging_level(level: str = DEFAULT_LEVEL.lower()) -> None:
    """Set tidy3d logging level priority.

    Parameters
    ----------
    level : str = 'info'
        The lowest priority level of logging messages to display.
        One of ``{'debug', 'info', 'warning', 'error'}`` (listed in increasing priority).

    Example
    -------
    >>> log.debug('this message should not appear (default logging level = INFO')
    >>> set_logging_level('debug')
    >>> log.debug('this message should appear now')
    """

    level_int = _get_level_int(level)
    log.setLevel(level_int)


def set_logging_file(fname: str, filemode="w", level=DEFAULT_LEVEL.lower()):
    """Set a file to write log to, independently from the stdout and stderr
    output chosen using :meth:`logging_level`.

    Parameters
    ----------
    fname : str
        Path to file to direct the output to.
    filemode : str = 'w'
        'w' or 'a', defining if the file should be overwritten or appended.
    level : str = 'info'
        One of 'debug', 'info', 'warning', 'error', 'critical'. This is
        set for the file independently of the console output level set by
        :meth:`logging_level`.

    Example
    -------
    >>> set_logging_file('tidy3d_log.log)
    >>> log.warning('this warning will appear in the tidy3d_log.log')
    """

    file_handler = logging.FileHandler(fname, filemode)
    level_int = _get_level_int(level)
    file_handler.setLevel(level_int)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
