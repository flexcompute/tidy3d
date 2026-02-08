from __future__ import annotations

import pathlib
from pathlib import Path

from click.testing import CliRunner
from pydantic import Field

from tidy3d.config import get_manager, reload_config
from tidy3d.config import loader as config_loader
from tidy3d.config import registry as config_registry
from tidy3d.config.legacy import finalize_legacy_migration
from tidy3d.config.loader import migrate_legacy_config
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
    assert "Lowest logging level that will be emitted." in content


def test_preserves_user_comments(config_manager, mock_config_dir):
    manager = config_manager
    manager.save(include_defaults=True)

    config_path = _config_path(mock_config_dir)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("Lowest logging level that will be emitted.", "user-modified comment")
    config_path.write_text(text, encoding="utf-8")

    reload_config(profile="default")
    manager = get_manager()
    manager.save(include_defaults=True)

    updated = config_path.read_text(encoding="utf-8")
    assert "user-modified comment" in updated
    assert "Lowest logging level that will be emitted." not in updated


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
        assert "Lowest logging level that will be emitted." in config_text
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
        assert "Plugin knob description." in content
    finally:
        config_registry._SECTIONS.pop("plugins.comment_test", None)
        reload_config(profile="default")


def test_finalize_legacy_migration_promotes_flat_file(tmp_path):
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir()
    legacy_file = canonical_dir / "config"
    legacy_file.write_text('apikey = "legacy-key"\n', encoding="utf-8")
    extra_file = canonical_dir / "extra.txt"
    extra_file.write_text("keep", encoding="utf-8")

    finalize_legacy_migration(canonical_dir)

    config_toml = canonical_dir / "config.toml"
    assert config_toml.exists()
    content = config_toml.read_text(encoding="utf-8")
    assert "[web]" in content
    assert "[logging]" in content
    assert "Lowest logging level that will be emitted." in content
    assert "legacy-key" in content
    assert not legacy_file.exists()
    assert extra_file.exists()
    assert extra_file.read_text(encoding="utf-8") == "keep"


def test_migrate_legacy_config_promotes_structured_config(tmp_path, monkeypatch):
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy_file = legacy_dir / "config"
    legacy_file.write_text('apikey = "legacy-key"\n', encoding="utf-8")
    (legacy_dir / "extra.txt").write_text("keep", encoding="utf-8")

    canonical_dir = tmp_path / "canonical"

    monkeypatch.setattr(config_loader, "legacy_config_directory", lambda: legacy_dir)
    monkeypatch.setattr(config_loader, "canonical_config_directory", lambda: canonical_dir)

    destination = migrate_legacy_config()

    assert destination == canonical_dir
    config_toml = canonical_dir / "config.toml"
    assert config_toml.exists()
    content = config_toml.read_text(encoding="utf-8")
    assert "[web]" in content
    assert "[logging]" in content
    assert "Lowest logging level that will be emitted." in content
    assert "legacy-key" in content
    assert not (canonical_dir / "config").exists()
    extra_file = canonical_dir / "extra.txt"
    assert extra_file.exists()
    assert extra_file.read_text(encoding="utf-8") == "keep"
    assert legacy_dir.exists()


def test_is_writable_ignores_file_not_found_cleanup(tmp_path, monkeypatch):
    original_unlink = pathlib.Path.unlink

    def flaky_unlink(self, *args, **kwargs):
        if self.name.startswith(".tidy3d_write_test_"):
            raise FileNotFoundError
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", flaky_unlink)
    assert config_loader._is_writable(tmp_path)


def test_resolve_config_directory_prefers_existing_canonical_dir(tmp_path, monkeypatch):
    canonical_dir = tmp_path / "xdg" / "tidy3d"
    canonical_dir.mkdir(parents=True)
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()

    monkeypatch.setattr(config_loader, "canonical_config_directory", lambda: canonical_dir)
    monkeypatch.setattr(config_loader, "legacy_config_directory", lambda: legacy_dir)

    def _boom(_):
        raise AssertionError(
            "_is_writable should not be called when canonical config directory exists"
        )

    monkeypatch.setattr(config_loader, "_is_writable", _boom)

    assert config_loader.resolve_config_directory() == canonical_dir


def test_resolve_config_directory_prefers_existing_base_dir(tmp_path, monkeypatch):
    base_dir = tmp_path / "base"
    config_dir = base_dir / "config"
    config_dir.mkdir(parents=True)

    monkeypatch.setenv("TIDY3D_BASE_DIR", str(base_dir))

    def _boom(_):
        raise AssertionError("_is_writable should not be called when base config directory exists")

    monkeypatch.setattr(config_loader, "_is_writable", _boom)

    assert config_loader.resolve_config_directory() == config_dir
