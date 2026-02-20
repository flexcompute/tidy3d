"""Shared fixtures for the configuration test suite."""

from __future__ import annotations

import os

import pytest

from tidy3d.config.__init__ import get_manager, reload_config
from tidy3d.config.registry import attach_manager
from tidy3d.config.registry import get_manager as get_registry_manager

_ENV_VARS_TO_CLEAR = {
    "TIDY3D_PROFILE",
    "TIDY3D_CONFIG_PROFILE",
    "TIDY3D_ENV",
    "TIDY3D_CONFIG_AUTO_MIGRATE",
    "TIDY3D_CONFIG_FORWARD_COMPAT",
    "SIMCLOUD_APIKEY",
    "TIDY3D_AUTH__APIKEY",
    "TIDY3D_WEB__APIKEY",
    "TIDY3D_BASE_DIR",
}


@pytest.fixture(autouse=True)
def _isolate_registry_manager():
    """Prevent ConfigManager instances created during a test from leaking
    into subsequent tests via the global ``_MANAGER`` in the registry."""
    original = get_registry_manager()
    yield
    attach_manager(original)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure configuration-related env vars do not leak between tests."""

    original: dict[str, str | None] = {var: os.environ.get(var) for var in _ENV_VARS_TO_CLEAR}
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)
    try:
        yield
    finally:
        for var, value in original.items():
            if value is None:
                monkeypatch.delenv(var, raising=False)
            else:
                monkeypatch.setenv(var, value)


@pytest.fixture
def mock_config_dir(tmp_path, monkeypatch):
    """Point the config system at a temporary directory."""

    base_dir = tmp_path / "config_home"
    monkeypatch.setenv("TIDY3D_BASE_DIR", str(base_dir))
    return base_dir / "config"


@pytest.fixture
def config_manager(mock_config_dir):
    """Return a freshly initialized configuration manager."""

    from tidy3d.config import config as config_wrapper

    reload_config(profile="default")
    config_wrapper.switch_profile("default")
    manager = get_manager()
    return manager
