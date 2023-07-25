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

DEFAULT_LOG_STYLES = {
    "DEBUG": None,
    "INFO": None,
    "WARNING": "red",
    "ERROR": "red bold",
    "CRITICAL": "red bold",
}


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
            self.console.log(
                level_name,
                message,
                sep=": ",
                style=DEFAULT_LOG_STYLES[level_name],
                _stack_offset=offset,
            )


class Logger:
    """Custom logger to avoid the complexities of the logging module

    The logger can be used in a context manager to avoid the emission of multiple messages. In this
    case, the first message in the context is emitted normally, but any others are discarded. When
    the context is exited, the number of discarded messages of each level is displayed with the
    highest level of the captures messages.

    Messages can also be captured for post-processing. That can be enabled through 'set_capture' to
    record all warnings emitted during model validation. A structured copy of all validation
    messages can then be recovered through 'captured_warnings'.
    """

    _static_cache = set()

    def __init__(self):
        self.handlers = {}
        self.suppression = True
        self._counts = None
        self._stack = None
        self._capture = False
        self._captured_warnings = []

    def set_capture(self, capture: bool):
        """Turn on/off tree-like capturing of log messages."""
        self._capture = capture

    def captured_warnings(self):
        """Get the formatted list of captured log messages."""
        captured_warnings = self._captured_warnings
        self._captured_warnings = []
        return captured_warnings

    def __enter__(self):
        """If suppression is enables, enter a consolidation context (only a single message is
        emitted)."""
        if self.suppression and self._counts is None:
            self._counts = {}
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exist a consolidation context (report the number of messages discarded)."""
        if self._counts is not None:
            total = sum(v for v in self._counts.values())
            if total > 0:
                max_level = max(k for k, v in self._counts.items() if v > 0)
                counts = [f"{v} {_level_name[k]}" for k, v in self._counts.items() if v > 0]
            self._counts = None
            if total > 0:
                noun = " messages." if total > 1 else " message."
                # Temporarily prevent capturing messages to emit consolidated summary
                stack = self._stack
                self._stack = None
                self.log(max_level, "Suppressed " + ", ".join(counts) + noun)
                self._stack = stack
        return False

    def begin_capture(self):
        """Start capturing log stack for consolidated validation log.

        This method is used before any model validation starts and is included in the initialization
        of 'BaseModel'. It must be followed by a corresponding 'end_capture'.
        """
        if not self._capture:
            return

        stack_item = {"messages": [], "children": {}}
        if self._stack:
            self._stack.append(stack_item)
        else:
            self._stack = [stack_item]

    def end_capture(self, model):
        """End capturing log stack for consolidated validation log.

        This method is used after all model validations and is included in the initialization of
        'BaseModel'. It must follow a corresponding 'begin_capture'.
        """
        if not self._stack:
            return

        stack_item = self._stack.pop()
        if len(self._stack) == 0:
            self._stack = None

        # Check if this stack item contains any messages or children
        if len(stack_item["messages"]) > 0 or len(stack_item["children"]) > 0:
            stack_item["type"] = model.__class__.__name__

            # Set the path for each children
            model_fields = model.get_submodels_by_hash()
            for child_hash, child_dict in stack_item["children"].items():
                child_dict["parent_fields"] = model_fields.get(child_hash, [])

            # Are we at the bottom of the stack?
            if self._stack is None:
                # Yes, we're root
                self._parse_warning_capture(current_loc=[], stack_item=stack_item)
            else:
                # No, we're someone else's child
                hash_ = hash(model)
                self._stack[-1]["children"][hash_] = stack_item

    def _parse_warning_capture(self, current_loc, stack_item):
        """Process capture tree to compile formatted captured warnings."""

        if "parent_fields" in stack_item:
            for field in stack_item["parent_fields"]:
                if isinstance(field, tuple):
                    # array field
                    new_loc = current_loc + list(field)
                else:
                    # single field
                    new_loc = current_loc + [field]

                # process current level warnings
                for level, msg in stack_item["messages"]:
                    if level == "WARNING":
                        self._captured_warnings.append({"loc": new_loc, "msg": msg})

                # initialize processing at children level
                for child_stack in stack_item["children"].values():
                    self._parse_warning_capture(current_loc=new_loc, stack_item=child_stack)

        else:  # for root object
            # process current level warnings
            for level, msg in stack_item["messages"]:
                if level == "WARNING":
                    self._captured_warnings.append({"loc": current_loc, "msg": msg})

            # initialize processing at children level
            for child_stack in stack_item["children"].values():
                self._parse_warning_capture(current_loc=current_loc, stack_item=child_stack)

    def _log(
        self, level: int, level_name: str, message: str, *args, log_once: bool = False
    ) -> None:
        """Distribute log messages to all handlers"""

        # Compose message
        if len(args) > 0:
            try:
                composed_message = str(message) % args
            # pylint: disable=broad-exception-caught
            except Exception as e:
                composed_message = f"{message} % {args}\n{e}"
        else:
            composed_message = str(message)

        # Capture all messages (even if suppressed later)
        if self._stack:
            self._stack[-1]["messages"].append((level_name, composed_message))

        # Check global cache if requested
        if log_once:
            # Use the message body before composition as key
            if message in self._static_cache:
                return
            self._static_cache.add(message)

        # Context-local logger emits a single message and consolidates the rest
        if self._counts is not None:
            if len(self._counts) > 0:
                self._counts[level] = 1 + self._counts.get(level, 0)
                return
            self._counts[level] = 0

        # Forward message to handlers
        for handler in self.handlers.values():
            handler.handle(level, level_name, composed_message)

    def log(self, level: LogValue, message: str, *args, log_once: bool = False) -> None:
        """Log (message) % (args) with given level"""
        if isinstance(level, str):
            level_name = level
            level = _get_level_int(level)
        else:
            level_name = _level_name.get(level, "unknown")
        self._log(level, level_name, message, *args, log_once=log_once)

    def debug(self, message: str, *args, log_once: bool = False) -> None:
        """Log (message) % (args) at debug level"""
        self._log(_level_value["DEBUG"], "DEBUG", message, *args, log_once=log_once)

    def info(self, message: str, *args, log_once: bool = False) -> None:
        """Log (message) % (args) at info level"""
        self._log(_level_value["INFO"], "INFO", message, *args, log_once=log_once)

    def warning(self, message: str, *args, log_once: bool = False) -> None:
        """Log (message) % (args) at warning level"""
        self._log(_level_value["WARNING"], "WARNING", message, *args, log_once=log_once)

    def error(self, message: str, *args, log_once: bool = False) -> None:
        """Log (message) % (args) at error level"""
        self._log(_level_value["ERROR"], "ERROR", message, *args, log_once=log_once)

    def critical(self, message: str, *args, log_once: bool = False) -> None:
        """Log (message) % (args) at critical level"""
        self._log(_level_value["CRITICAL"], "CRITICAL", message, *args, log_once=log_once)


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


def set_log_suppression(value: bool) -> None:
    """Control log suppression for repeated messages."""
    log.suppression = value


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
            log.warning("Log file could not be closed")
        finally:
            del log.handlers["file"]

    try:
        # pylint: disable=consider-using-with
        file = open(fname, filemode)
    except:  # pylint: disable=bare-except
        log.error(f"File {fname} could not be opened")
        return

    log.handlers["file"] = LogHandler(Console(file=file), level)


# Initialize Tidy3d's logger
log = Logger()

# Set default logging output
set_logging_console()
