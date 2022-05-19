""" test the grid operations """

import pytest

import tidy3d as td
from tidy3d.web.config import DEFAULT_CONFIG, WEB_CONFIGS
from tidy3d.log import log, DEFAULT_LEVEL, LEVEL_MAP

from .utils import assert_log_level


def test_logging_level():
    """Make sure setting the logging level in config affects the log.level"""

    # default level
    assert log.level == LEVEL_MAP[DEFAULT_LEVEL.lower()]

    # check etting all levels
    for key, val in LEVEL_MAP.items():
        td.config.logging_level = key
        assert log.level == val


def test_config():
    """Make sure setting the web_config in config affects the DEFAULT_CONFIG"""

    for config_key, target_config in WEB_CONFIGS.items():

        td.config.web_config = config_key

        for key in DEFAULT_CONFIG.dict():
            assert DEFAULT_CONFIG.dict()[key] == target_config.dict()[key]


def test_frozen():
    """Make sure you can dynamically freeze tidy3d components."""

    # make sure it's mutable
    b = td.Box(size=(1, 1, 1))
    b.center = (1, 2, 3)

    # freeze and make sure it's immutable
    td.config.frozen = True
    with pytest.raises(TypeError):
        b.center = (2, 2, 2)

    # unfreeze and make sure it's mutable again
    td.config.frozen = False
    b.center = (1, 2, 3)
