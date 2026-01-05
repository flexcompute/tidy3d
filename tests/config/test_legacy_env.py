from __future__ import annotations

import importlib
import os
import ssl
import warnings

import pytest

from tidy3d.config import Env, get_manager, reload_config
from tidy3d.config import config as config_wrapper


@pytest.fixture
def suppress_legacy_env_deprecated_warning(monkeypatch):
    """Patch Env deprecation helper to a no-op for tests that don't want the warning."""
    legacy_module = importlib.import_module("tidy3d.config.legacy")
    monkeypatch.setattr(legacy_module, "_warn_env_deprecated", lambda: None)


def test_env_tracks_profile_switch(config_manager):
    """Regression: Env should mirror manager profile switches."""

    del config_manager  # ensure fixture runs, avoid lint for unused variable
    try:
        config_wrapper.switch_profile("dev")
        assert config_wrapper.profile == "dev"
        assert Env.current.name == "dev"
        assert str(Env.current.web_api_endpoint) == str(config_wrapper.web.api_endpoint)
    finally:
        reload_config(profile="default")


def test_env_pending_overrides_apply_on_activation(
    mock_config_dir,
    config_manager,
    suppress_legacy_env_deprecated_warning,
):
    """Queued overrides should land once the corresponding profile is activated."""

    del mock_config_dir
    manager = config_manager
    try:
        config_wrapper.switch_profile("default")
        assert manager.profile == "default"

        Env.dev.enable_caching = False
        Env.dev.ssl_version = ssl.TLSVersion.TLSv1_2

        # Pending overrides should not touch the active profile yet.
        default_web = manager.get_section("web")
        assert default_web.enable_caching is True
        assert default_web.ssl_version is None

        Env.dev.active()
        current_manager = get_manager()
        assert current_manager.profile == "dev"
        dev_web = current_manager.get_section("web")
        assert dev_web.enable_caching is False
        assert dev_web.ssl_version == "TLSv1_2"
        assert Env.current.enable_caching is False
        assert Env.current.ssl_version == "TLSv1_2"
    finally:
        reload_config(profile="default")


def test_env_vars_follow_profile_switch(
    mock_config_dir,
    monkeypatch,
    config_manager,
    suppress_legacy_env_deprecated_warning,
):
    """Environment variables applied via Env should restore previous values on switch."""

    del mock_config_dir
    del config_manager  # ensure fixture executes without lint complaints
    try:
        config_wrapper.switch_profile("default")
        monkeypatch.setenv("TIDY3D_TEST_VAR", "previous")

        Env.default.env_vars = {"TIDY3D_TEST_VAR": "applied"}
        assert os.environ["TIDY3D_TEST_VAR"] == "applied"

        Env.dev.env_vars = {}
        Env.dev.active()
        assert os.environ["TIDY3D_TEST_VAR"] == "previous"

        Env.default.active()
        assert os.environ["TIDY3D_TEST_VAR"] == "applied"
    finally:
        reload_config(profile="default")


def test_web_core_environment_reexports():
    """Legacy `tidy3d.web.core.environment` exports remain available via config shim."""

    import tidy3d.web as web
    from tidy3d.config import Env as ConfigEnv

    environment = web.core.environment
    assert environment.Env is ConfigEnv

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        dev = environment.dev
        uat = environment.uat

    assert dev is ConfigEnv.dev
    assert uat is ConfigEnv.uat
    assert len(caught) == 2
    assert "tidy3d.web.core.environment.dev" in str(caught[0].message)
    assert "tidy3d.web.core.environment.uat" in str(caught[1].message)
    assert "2.12" in str(caught[0].message)
