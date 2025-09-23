from __future__ import annotations

from pydantic import Field

from tidy3d.config.__init__ import get_manager, reload_config
from tidy3d.config.registry import get_sections, register_plugin
from tidy3d.config.sections import ConfigSection


def ensure_dummy_plugin():
    if "plugins.dummy" in get_sections():
        return

    @register_plugin("dummy")
    class DummyPlugin(ConfigSection):
        enabled: bool = Field(False, json_schema_extra={"persist": True})
        precision: int = Field(1, json_schema_extra={"persist": True})


def test_plugin_defaults_available(mock_config_dir):
    ensure_dummy_plugin()
    assert "plugins.dummy" in get_sections()
    reload_config(profile="default")
    manager = get_manager()
    plugin = manager.plugins.dummy
    assert plugin.enabled is False
    assert plugin.precision == 1


def test_plugin_updates_persist(mock_config_dir):
    ensure_dummy_plugin()
    assert "plugins.dummy" in get_sections()
    reload_config(profile="default")
    manager = get_manager()
    manager.update_section("plugins.dummy", enabled=True, precision=4)
    manager.save()
    config_path = manager.config_dir / "config.toml"
    assert config_path.exists()

    reload_config(profile=manager.profile)
    new_manager = get_manager()
    plugin = new_manager.plugins.dummy
    assert plugin.enabled is True
    assert plugin.precision == 4
