"""Test the logging."""

import pytest
import pydantic as pd
import tidy3d as td
from tidy3d.log import Tidy3dError, ConfigError, set_logging_level

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
