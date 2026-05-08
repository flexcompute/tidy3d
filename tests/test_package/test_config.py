"""test the grid operations"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import tidy3d as td
from tidy3d.log import DEFAULT_LEVEL, _level_value


def test_logging_level():
    """Make sure setting the logging level in config affects the log.level"""

    # default level
    assert td.log.handlers["console"].level == _level_value[DEFAULT_LEVEL]

    # check setting all levels
    for key, val in _level_value.items():
        td.config.logging.level = key
        assert td.log.handlers["console"].level == val


def test_log_level_not_found():
    with pytest.raises(ValidationError):
        td.config.logging.level = "NOT_A_LEVEL"
