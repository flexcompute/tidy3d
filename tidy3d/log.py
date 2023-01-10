"""Logging and error-handling for Tidy3d."""
import logging
from rich.logging import RichHandler

# TODO: more logging features (to file, etc).

FORMAT = "%(message)s"

DEFAULT_LEVEL = "INFO"
LOGGER_NAME = "tidy3d_logger"

logging.basicConfig(level=DEFAULT_LEVEL, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

# maps level string to level integer for python logging package
LEVEL_MAP = {
    "error": 40,
    "warning": 30,
    "info": 20,
    "debug": 10,
}

# importable logger
log = logging.getLogger(LOGGER_NAME)


""" Tidy3d custom exceptions """


class Tidy3dError(Exception):
    """Any error in tidy3d"""

    def __init__(self, message: str = None):
        """Log just the error message and then raise the Exception."""
        super().__init__(message)
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


""" Logging functions """


def _get_level_int(level: str) -> int:
    """Get the integer corresponding to the level string."""
    level = level.lower()
    if level not in LEVEL_MAP:
        raise ConfigError(
            f"logging level {level} not supported, must be in {list(LEVEL_MAP.keys())}."
        )
    return LEVEL_MAP[level]


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
    output chosen using :meth:`set_logging_level`.

    Parameters
    ----------
    fname : str
        Path to file to direct the output to.
    filemode : str = 'w'
        'w' or 'a', defining if the file should be overwritten or appended.
    level : str = 'info'
        One of 'debug', 'info', 'warning', 'error', 'critical'. This is
        set for the file independently of the console output level set by
        :meth:`set_logging_level`.

    Example
    -------
    >>> set_logging_file('tidy3d_log.log')
    >>> log.warning('this warning will appear in the tidy3d_log.log')
    """

    file_handler = logging.FileHandler(fname, filemode)
    level_int = _get_level_int(level)
    file_handler.setLevel(level_int)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
