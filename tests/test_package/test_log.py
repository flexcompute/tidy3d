"""Test the logging."""

import pytest

import pydantic as pd
import tidy3d as td
from tidy3d.exceptions import Tidy3dError
from tidy3d.log import DEFAULT_LEVEL, _get_level_int, set_logging_level
from ..utils import log_capture, assert_log_level


def test_log():
    td.log.debug("debug test")
    td.log.info("info test")
    td.log.warning("warning test")
    td.log.error("error test")
    td.log.critical("critical test")
    td.log.log(0, "zero test")


def test_log_config():
    td.config.logging_level = "DEBUG"
    td.set_logging_file("tests/tmp/test.log")
    assert len(td.log.handlers) == 2
    assert td.log.handlers["console"].level == _get_level_int("DEBUG")
    assert td.log.handlers["file"].level == _get_level_int(DEFAULT_LEVEL)


def test_log_level_not_found():
    with pytest.raises(ValueError):
        set_logging_level("NOT_A_LEVEL")


def test_set_logging_level_deprecated():
    with pytest.raises(DeprecationWarning):
        td.set_logging_level("WARNING")


def test_exception_message():
    MESSAGE = "message"
    e = Tidy3dError(MESSAGE)
    assert str(e) == MESSAGE


def test_logging_upper():
    """Make sure we get an error if lowercase."""
    td.config.logging_level = "WARNING"
    with pytest.raises(ValueError):
        td.config.logging_level = "warning"


def test_logging_unrecognized():
    """If unrecognized option, raise validation errorr."""
    with pytest.raises(pd.ValidationError):
        td.config.logging_level = "blah"
