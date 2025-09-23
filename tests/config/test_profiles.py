from __future__ import annotations

import toml


def test_save_default_profile(config_manager):
    config_manager.update_section("web", apikey="token")
    config_manager.update_section("web", timeout=30)
    config_manager.save()

    config_path = config_manager.config_dir / "config.toml"
    assert config_path.exists()
    data = toml.load(config_path)
    assert data["web"]["apikey"] == "token"


def test_save_custom_profile(config_manager):
    config_manager.switch_profile("customer")
    config_manager.update_section("logging", level="DEBUG")
    config_manager.save()

    profile_path = config_manager.config_dir / "profiles" / "customer.toml"
    assert profile_path.exists()
    data = toml.load(profile_path)
    assert data["logging"]["level"] == "DEBUG"
