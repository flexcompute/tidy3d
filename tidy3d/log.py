"""Logging for Tidy3d."""

import inspect
from datetime import datetime
from typing import Callable, List, Tuple, Union

from rich.console import Console
from rich.text import Text
from typing_extensions import Literal

# Note: "SUPPORT" and "USER" levels are meant for backend runs only.
# Logging in frontend code should just use the standard debug/info/warning/error/critical.
LogLevel = Literal["DEBUG", "SUPPORT", "USER", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogValue = Union[int, LogLevel]

# Logging levels compatible with logging module
_level_value = {
    "DEBUG": 10,
    "SUPPORT": 12,
    "USER": 15,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

_level_name = {v: k for k, v in _level_value.items()}

DEFAULT_LEVEL = "WARNING"

DEFAULT_LOG_STYLES = {
    "DEBUG": None,
    "SUPPORT": None,
    "USER": None,
    "INFO": None,
    "WARNING": "red",
    "ERROR": "red bold",
    "CRITICAL": "red bold",
}

# Width of the console used for rich logging (in characters).
CONSOLE_WIDTH = 80


def _default_log_level_format(level: str, message: str) -> Tuple[str, str]:
    """By default just return unformatted prefix and message."""
    return level, message


def _get_level_int(level: LogValue) -> int:
    """Get the integer corresponding to the level string."""
    if isinstance(level, int):
        return level

    if level not in _level_value:
        # We don't want to import ConfigError to avoid a circular dependency
        raise ValueError(
            f"logging level {level} not supported, must be "
            "'DEBUG', 'SUPPORT', 'USER', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'"
        )
    return _level_value[level]


class LogHandler:
    """Handle log messages depending on log level"""

    def __init__(
        self,
        console: Console,
        level: LogValue,
        log_level_format: Callable = _default_log_level_format,
        prefix_every_line: bool = False,
    ):
        self.level = _get_level_int(level)
        self.console = console
        self.log_level_format = log_level_format
        self.prefix_every_line = prefix_every_line

    def handle(self, level, level_name, message):
        """Output log messages depending on log level"""
        if level >= self.level:
            stack = inspect.stack()
            console = self.console
            offset = 4
            if stack[offset - 1].filename.endswith("exceptions.py"):
                # We want the calling site for exceptions.py
                offset += 1
            prefix, msg = self.log_level_format(level_name, message)
            if self.prefix_every_line:
                wrapped_text = Text(msg, style="default")
                msgs = wrapped_text.wrap(console=console, width=console.width - len(prefix) - 2)
            else:
                msgs = [msg]
            for msg in msgs:
                console.log(
                    prefix,
                    msg,
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
        """If suppression is enabled, enter a consolidation context (only a single message is
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
                for level, msg, custom_loc in stack_item["messages"]:
                    if level == "WARNING":
                        self._captured_warnings.append({"loc": new_loc + custom_loc, "msg": msg})

                # initialize processing at children level
                for child_stack in stack_item["children"].values():
                    self._parse_warning_capture(current_loc=new_loc, stack_item=child_stack)

        else:  # for root object
            # process current level warnings
            for level, msg, custom_loc in stack_item["messages"]:
                if level == "WARNING":
                    self._captured_warnings.append({"loc": current_loc + custom_loc, "msg": msg})

            # initialize processing at children level
            for child_stack in stack_item["children"].values():
                self._parse_warning_capture(current_loc=current_loc, stack_item=child_stack)

    def _log(
        self,
        level: int,
        level_name: str,
        message: str,
        *args,
        log_once: bool = False,
        custom_loc: List = None,
        capture: bool = True,
    ) -> None:
        """Distribute log messages to all handlers"""

        # Compose message
        if len(args) > 0:
            try:
                composed_message = str(message) % args

            except Exception as e:
                composed_message = f"{message} % {args}\n{e}"
        else:
            composed_message = str(message)

        # Capture all messages (even if suppressed later)
        if self._stack and capture:
            if custom_loc is None:
                custom_loc = []
            self._stack[-1]["messages"].append((level_name, composed_message, custom_loc))

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

    def support(self, message: str, *args, log_once: bool = False) -> None:
        """Log (message) % (args) at support level"""
        self._log(_level_value["SUPPORT"], "SUPPORT", message, *args, log_once=log_once)

    def user(self, message: str, *args, log_once: bool = False) -> None:
        """Log (message) % (args) at user level"""
        self._log(_level_value["USER"], "USER", message, *args, log_once=log_once)

    def info(self, message: str, *args, log_once: bool = False) -> None:
        """Log (message) % (args) at info level"""
        self._log(_level_value["INFO"], "INFO", message, *args, log_once=log_once)

    def warning(
        self,
        message: str,
        *args,
        log_once: bool = False,
        custom_loc: List = None,
        capture: bool = True,
    ) -> None:
        """Log (message) % (args) at warning level"""
        self._log(
            _level_value["WARNING"],
            "WARNING",
            message,
            *args,
            log_once=log_once,
            custom_loc=custom_loc,
            capture=capture,
        )

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
        The lowest priority level of logging messages to display. One of ``{'DEBUG', 'SUPPORT',
        'USER', INFO', 'WARNING', 'ERROR', 'CRITICAL'}`` (listed in increasing priority).
    """
    if "console" in log.handlers:
        log.handlers["console"].level = _get_level_int(level)


def set_log_suppression(value: bool) -> None:
    """Control log suppression for repeated messages."""
    log.suppression = value


def get_aware_datetime() -> datetime:
    """Get an aware current local datetime(with local timezone info)"""
    return datetime.now().astimezone()


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
    log.handlers["console"] = LogHandler(
        Console(
            stderr=stderr,
            width=CONSOLE_WIDTH,
            log_path=False,
            get_datetime=get_aware_datetime,
            log_time_format="%X %Z",
        ),
        previous_level,
    )


def set_logging_file(
    fname: str,
    filemode: str = "w",
    level: LogValue = DEFAULT_LEVEL,
    log_path: bool = False,
) -> None:
    """Set a file to write log to, independently from the stdout and stderr
    output chosen using :meth:`set_logging_level`.

    Parameters
    ----------
    fname : str
        Path to file to direct the output to. If empty string, a previously set logging file will
        be closed, if any, but nothing else happens.
    filemode : str
        'w' or 'a', defining if the file should be overwritten or appended.
    level : str
        One of ``{'DEBUG', 'SUPPORT', 'USER', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}``. This is set
        for the file independently of the console output level set by :meth:`set_logging_level`.
    log_path : bool = False
        Whether to log the path to the file that issued the message.
    """
    if filemode not in "wa":
        raise ValueError("filemode must be either 'w' or 'a'")

    # Close previous handler, if any
    if "file" in log.handlers:
        try:
            log.handlers["file"].console.file.close()
        except Exception:  # TODO: catch specific exception
            log.warning("Log file could not be closed")
        finally:
            del log.handlers["file"]

    if fname == "":
        # Empty string can be passed to just stop previously opened file handler
        return

    try:
        file = open(fname, filemode)
    except Exception:  # TODO: catch specific exception
        log.error(f"File {fname} could not be opened")
        return

    log.handlers["file"] = LogHandler(
        Console(file=file, force_jupyter=False, log_path=log_path), level
    )


# Initialize Tidy3d's logger
log = Logger()

# Set default logging output
set_logging_console()


def get_logging_console() -> Console:
    """Get console from logging handlers."""
    if "console" not in log.handlers:
        set_logging_console()
    return log.handlers["console"].console
