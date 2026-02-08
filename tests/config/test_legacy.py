from __future__ import annotations

import importlib

import pytest

from tidy3d.config.__init__ import get_manager, reload_config


def test_legacy_logging_level(config_manager):
    cfg = reload_config(profile=config_manager.profile)
    with pytest.warns(
        DeprecationWarning,
        match=r"config\.logging_level.*deprecated",
    ):
        cfg.logging_level = "DEBUG"
    manager = get_manager()
    assert manager.get_section("logging").level == "DEBUG"


def test_env_switch(config_manager):
    config_module = importlib.import_module("tidy3d.config.__init__")
    with pytest.warns(DeprecationWarning, match="tidy3d.config.Env"):
        config_module.Env.dev.active()
    assert get_manager().profile == "dev"
    with pytest.warns(DeprecationWarning, match="tidy3d.config.Env"):
        config_module.Env.set_current(config_module.Env.prod)
    assert get_manager().profile == "prod"


def test_legacy_wrapper_str(config_manager):
    from tidy3d.config import config

    text = str(config)
    assert "Config (profile='default')" in text
    assert "├── microwave" in text
    assert "'api_endpoint': 'https://tidy3d-api.simulation.cloud'" in text
