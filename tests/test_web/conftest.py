from __future__ import annotations

import pytest

from tidy3d import config


@pytest.fixture
def use_dev_profile():
    with config:
        config.switch_profile("dev")
        yield


@pytest.fixture
def use_test_profile():
    with config:
        config.switch_profile("test")
        yield
