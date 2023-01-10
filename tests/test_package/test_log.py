"""Test the logging."""

import pytest
import pydantic as pd
import tidy3d as td
from tidy3d.log import Tidy3dError, ConfigError, set_logging_level
from tidy3d.log import DEFAULT_LEVEL, _get_level_int

log = td.log


def test_log():
    log.debug("test")
    log.info("test")
    log.warning("test")
    log.error("test")


def test_log_config():
    td.config.logging_level = "debug"
    td.set_logging_file("test.log")


def test_log_level_not_found():
    with pytest.raises(ConfigError):
        set_logging_level("NOT_A_LEVEL")


def test_set_logging_level_deprecated():
    with pytest.raises(DeprecationWarning):
        td.set_logging_level("warning")


def test_exception_message():
    MESSAGE = "message"
    e = Tidy3dError(MESSAGE)
    assert str(e) == MESSAGE


def test_reset_logging_level():

    import logging
    from rich.logging import RichHandler

    td.config.logging_level = DEFAULT_LEVEL.lower()
    LEVEL = logging.ERROR
    assert DEFAULT_LEVEL != LEVEL, "set LEVEL something different from the DEFAULT_LEVEL for tidy3d"

    logging.basicConfig(
        level=LEVEL,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    log_test = logging.getLogger("rich")
    log_test.setLevel(LEVEL)
    assert log_test.level == LEVEL, "test logger level wasnt set correctly."
    assert log.level == _get_level_int(
        DEFAULT_LEVEL.lower()
    ), "tidy3d log level was reset by making a new logger"
