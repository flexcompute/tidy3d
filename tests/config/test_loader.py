from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
from pydantic import Field

from tidy3d.config import get_manager, reload_config
from tidy3d.config import registry as config_registry
from tidy3d.config.sections import ConfigSection
from tidy3d.web.cli.app import tidy3d_cli


def _config_path(config_dir: Path) -> Path:
    return config_dir / "config.toml"


def test_loads_legacy_flat_config(mock_config_dir):
    legacy_path = mock_config_dir / "config"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text('apikey = "legacy-key"\n', encoding="utf-8")

    reload_config(profile="default")
    manager = get_manager()
    web = manager.get_section("web")
    assert web.apikey is not None
    assert web.apikey.get_secret_value() == "legacy-key"


def test_save_includes_descriptions(config_manager, mock_config_dir):
    manager = config_manager
    manager.save(include_defaults=True)

    content = _config_path(mock_config_dir).read_text(encoding="utf-8")
    assert "# Web/HTTP configuration." in content


def test_preserves_user_comments(config_manager, mock_config_dir):
    manager = config_manager
    manager.save(include_defaults=True)

    config_path = _config_path(mock_config_dir)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace(
        "Web/HTTP configuration.",
        "user-modified comment",
    )
    config_path.write_text(text, encoding="utf-8")

    reload_config(profile="default")
    manager = get_manager()
    manager.save(include_defaults=True)

    updated = config_path.read_text(encoding="utf-8")
    assert "user-modified comment" in updated
    assert "Web/HTTP configuration." not in updated


def test_profile_preserves_comments(config_manager, mock_config_dir):
    @config_registry.register_plugin("profile_comment")
    class ProfileComment(ConfigSection):
        """Profile comment plugin."""

        knob: int = Field(
            1,
            description="Profile knob description.",
            json_schema_extra={"persist": True},
        )

    try:
        manager = config_manager
        manager.switch_profile("custom")
        manager.update_section("plugins.profile_comment", knob=5)
        manager.save()

        profile_path = mock_config_dir / "profiles" / "custom.toml"
        text = profile_path.read_text(encoding="utf-8")
        assert "Profile knob description." in text
        text = text.replace("Profile knob description.", "user comment")
        profile_path.write_text(text, encoding="utf-8")

        manager.update_section("plugins.profile_comment", knob=7)
        manager.save()

        updated = profile_path.read_text(encoding="utf-8")
        assert "user comment" in updated
        assert "Profile knob description." not in updated
    finally:
        config_registry._SECTIONS.pop("plugins.profile_comment", None)
        reload_config(profile="default")


def test_cli_reset_config(mock_config_dir):
    @config_registry.register_plugin("cli_comment")
    class CLIPlugin(ConfigSection):
        """CLI plugin configuration."""

        knob: int = Field(
            3,
            description="CLI knob description.",
            json_schema_extra={"persist": True},
        )

    try:
        reload_config(profile="default")
        manager = get_manager()
        manager.update_section("web", apikey="secret")
        manager.save(include_defaults=True)
        manager.switch_profile("custom")
        manager.update_section("plugins.cli_comment", knob=42)
        manager.save()

        profiles_dir = mock_config_dir / "profiles"
        assert profiles_dir.exists()

        runner = CliRunner()
        result = runner.invoke(tidy3d_cli, ["config", "reset", "--yes"])
        assert result.exit_code == 0, result.output

        config_text = _config_path(mock_config_dir).read_text(encoding="utf-8")
        assert "Web/HTTP configuration." in config_text
        assert "[web]" in config_text
        assert "secret" not in config_text
        assert not profiles_dir.exists()
    finally:
        config_registry._SECTIONS.pop("plugins.cli_comment", None)
        reload_config(profile="default")


def test_plugin_descriptions(mock_config_dir):
    @config_registry.register_plugin("comment_test")
    class CommentPlugin(ConfigSection):
        """Comment plugin configuration."""

        knob: int = Field(
            3,
            description="Plugin knob description.",
            json_schema_extra={"persist": True},
        )

    try:
        reload_config(profile="default")
        manager = get_manager()
        manager.save(include_defaults=True)
        content = _config_path(mock_config_dir).read_text(encoding="utf-8")
        assert "Comment plugin configuration." in content
        assert "Plugin knob description." in content
    finally:
        config_registry._SECTIONS.pop("plugins.comment_test", None)
        reload_config(profile="default")
