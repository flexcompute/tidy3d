"""Logging for Tidy3d."""

import inspect

from typing import Union
from typing_extensions import Literal

from rich.console import Console


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogValue = Union[int, LogLevel]

# Logging levels compatible with logging module
_level_value = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

_level_name = {v: k for k, v in _level_value.items()}

DEFAULT_LEVEL = "WARNING"


def _get_level_int(level: LogValue) -> int:
    """Get the integer corresponding to the level string."""
    if isinstance(level, int):
        return level

    if level not in _level_value:
        # We don't want to import ConfigError to avoid a circular dependency
        raise ValueError(
            f"logging level {level} not supported, must be "
            "'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'"
        )
    return _level_value[level]


class LogHandler:
    """Handle log messages depending on log level"""

    def __init__(self, console: Console, level: LogValue):
        self.level = _get_level_int(level)
        self.console = console

    def handle(self, level, level_name, message):
        """Output log messages depending on log level"""
        if level >= self.level:
            stack = inspect.stack()
            offset = 4
            if stack[offset - 1].filename.endswith("exceptions.py"):
                # We want the calling site for exceptions.py
                offset += 1
            self.console.log(level_name, message, sep=": ", _stack_offset=offset)


class Logger:
    """Custom logger to avoid the complexities of the logging module"""

    def __init__(self):
        self.handlers = {}

    def _log(self, level: int, level_name: str, message: str, *args) -> None:
        """Distribute log messages to all handlers"""
        if len(args) > 0:
            try:
                composed_message = str(message) % args
            # pylint: disable=broad-exception-caught
            except Exception as e:
                composed_message = f"{message} % {args}\n{e}"
        else:
            composed_message = str(message)
        for handler in self.handlers.values():
            handler.handle(level, level_name, composed_message)

    def log(self, level: LogValue, message: str, *args) -> None:
        """Log (message) % (args) with given level"""
        if isinstance(level, str):
            level_name = level
            level = _get_level_int(level)
        else:
            level_name = _level_name.get(level, "unknown")
        self._log(level, level_name, message, *args)

    def debug(self, message: str, *args) -> None:
        """Log (message) % (args) at debug level"""
        self._log(_level_value["DEBUG"], "DEBUG", message, *args)

    def info(self, message: str, *args) -> None:
        """Log (message) % (args) at info level"""
        self._log(_level_value["INFO"], "INFO", message, *args)

    def warning(self, message: str, *args) -> None:
        """Log (message) % (args) at warning level"""
        self._log(_level_value["WARNING"], "WARNING", message, *args)

    def error(self, message: str, *args) -> None:
        """Log (message) % (args) at error level"""
        self._log(_level_value["ERROR"], "ERROR", message, *args)

    def critical(self, message: str, *args) -> None:
        """Log (message) % (args) at critical level"""
        self._log(_level_value["CRITICAL"], "CRITICAL", message, *args)


# Initialize Tidy3d's logger
log = Logger()


def set_logging_level(level: LogValue = DEFAULT_LEVEL) -> None:
    """Set tidy3d console logging level priority.

    Parameters
    ----------
    level : str
        The lowest priority level of logging messages to display. One of ``{'DEBUG', 'INFO',
        'WARNING', 'ERROR', 'CRITICAL'}`` (listed in increasing priority).
    """
    if "console" in log.handlers:
        log.handlers["console"].level = _get_level_int(level)


def set_logging_console(stderr: bool = False) -> None:
    """Set stdout or stderr as console output

    Parameters
    ----------
    stderr : bool
        If False, logs are directed to stdout, otherwise to stderr.
    """
    if "console" in log.handlers:
        previous_level = log.handlers["console"].level
    else:
        previous_level = DEFAULT_LEVEL
    log.handlers["console"] = LogHandler(Console(stderr=stderr), previous_level)


def set_logging_file(
    fname: str,
    filemode: str = "w",
    level: LogValue = DEFAULT_LEVEL,
) -> None:
    """Set a file to write log to, independently from the stdout and stderr
    output chosen using :meth:`set_logging_level`.

    Parameters
    ----------
    fname : str
        Path to file to direct the output to.
    filemode : str
        'w' or 'a', defining if the file should be overwritten or appended.
    level : str
        One of ``{'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}``. This is set for the file
        independently of the console output level set by :meth:`set_logging_level`.
    """
    if filemode not in "wa":
        raise ValueError("filemode must be either 'w' or 'a'")

    # Close previous handler, if any
    if "file" in log.handlers:
        try:
            log.handlers["file"].file.close()
        except:  # pylint: disable=bare-except
            del log.handlers["file"]
            log.warning("Log file could not be closed")

    try:
        # pylint: disable=consider-using-with
        file = open(fname, filemode)
    except:  # pylint: disable=bare-except
        log.error(f"File {fname} could not be opened")
        return

    log.handlers["file"] = LogHandler(Console(file=file), level)


# Set default logging output
set_logging_console()
