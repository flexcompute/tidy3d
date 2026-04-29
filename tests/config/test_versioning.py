"""Tests for configuration schema versioning and migrations."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap

import pytest
import toml
import tomlkit
from click.testing import CliRunner
from pydantic import Field

import tidy3d as td
from tests.utils import AssertLogStr
from tidy3d.config import ConfigManager
from tidy3d.config import loader as config_loader
from tidy3d.config import manager as config_manager_module
from tidy3d.config import migrations as config_migrations
from tidy3d.config import profiles as config_profiles
from tidy3d.config import registry as config_registry
from tidy3d.config.loader import ConfigLoader, build_validated_models
from tidy3d.config.migrations import CURRENT_CONFIG_VERSION
from tidy3d.config.sections import ConfigSection
from tidy3d.web.cli.app import tidy3d_cli
from tidy3d.web.core.types import PayType


def test_config_version_written_on_save(config_manager, mock_config_dir):
    assert td.config.profile == config_manager.profile
    config_manager.save(include_defaults=True)

    config_path = mock_config_dir / "config.toml"
    data = toml.loads(config_path.read_text(encoding="utf-8"))
    assert data["config_version"] == CURRENT_CONFIG_VERSION


def test_auto_migrate_disabled_does_not_write_back(mock_config_dir, monkeypatch):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('[logging]\nlevel = "INFO"\n', encoding="utf-8")

    monkeypatch.setenv("TIDY3D_CONFIG_AUTO_MIGRATE", "0")

    manager = ConfigManager(config_dir=mock_config_dir)
    assert manager.logging.level == "INFO"
    content = config_path.read_text(encoding="utf-8")
    assert "config_version" not in content


def test_auto_migrate_write_back_keeps_backup(mock_config_dir):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('[logging]\nlevel = "INFO"\n', encoding="utf-8")

    manager = ConfigManager(config_dir=mock_config_dir)
    assert manager.logging.level == "INFO"

    migrated = toml.loads(config_path.read_text(encoding="utf-8"))
    assert migrated["config_version"] == CURRENT_CONFIG_VERSION

    backup_path = config_path.with_suffix(".toml.bak")
    assert backup_path.exists()
    backup_data = toml.loads(backup_path.read_text(encoding="utf-8"))
    assert "config_version" not in backup_data
    assert backup_data["logging"]["level"] == "INFO"


def test_loader_load_base_auto_migrate_write_back_keeps_backup(mock_config_dir):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('[logging]\nlevel = "INFO"\n', encoding="utf-8")

    loader = ConfigLoader(mock_config_dir)
    data = loader.load_base()
    assert data["logging"]["level"] == "INFO"

    migrated = toml.loads(config_path.read_text(encoding="utf-8"))
    assert migrated["config_version"] == CURRENT_CONFIG_VERSION

    backup_path = config_path.with_suffix(".toml.bak")
    assert backup_path.exists()
    backup_data = toml.loads(backup_path.read_text(encoding="utf-8"))
    assert "config_version" not in backup_data
    assert backup_data["logging"]["level"] == "INFO"


@pytest.mark.parametrize("version_offset", [0, 10])
def test_stale_pending_write_is_cleared_when_file_is_not_behind(mock_config_dir, version_offset):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('[logging]\nlevel = "INFO"\n', encoding="utf-8")

    loader = ConfigLoader(mock_config_dir)
    loader.load_base(commit_writes=False, queue_migration_write=True)
    assert config_path in loader._pending_writes

    config_path.write_text(
        "\n".join(
            [
                f"config_version = {CURRENT_CONFIG_VERSION + version_offset}",
                "[logging]",
                'level = "ERROR"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    data = loader.load_base(commit_writes=True)
    assert data["logging"]["level"] == "ERROR"
    assert config_path not in loader._pending_writes

    persisted = toml.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["config_version"] == CURRENT_CONFIG_VERSION + version_offset
    assert persisted["logging"]["level"] == "ERROR"


def test_loader_load_user_profile_auto_migrate_write_back_keeps_backup(mock_config_dir):
    profile_path = mock_config_dir / "profiles" / "dev.toml"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text('[logging]\nlevel = "DEBUG"\n', encoding="utf-8")

    loader = ConfigLoader(mock_config_dir)
    data = loader.load_user_profile("dev")
    assert data["logging"]["level"] == "DEBUG"

    migrated = toml.loads(profile_path.read_text(encoding="utf-8"))
    assert migrated["config_version"] == CURRENT_CONFIG_VERSION

    backup_path = profile_path.with_suffix(".toml.bak")
    assert backup_path.exists()
    backup_data = toml.loads(backup_path.read_text(encoding="utf-8"))
    assert "config_version" not in backup_data
    assert backup_data["logging"]["level"] == "DEBUG"


def test_loader_profile_metadata_validation_failure_does_not_trigger_write_back(mock_config_dir):
    profile_path = mock_config_dir / "profiles" / "dev.toml"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        'default_profile = "prod"\n[logging]\nlevel = "INFO"\n', encoding="utf-8"
    )

    loader = ConfigLoader(mock_config_dir)
    with pytest.raises(ValueError, match=r"only allowed in 'config.toml'"):
        loader.load_user_profile("dev")

    loader.load_base()
    content = profile_path.read_text(encoding="utf-8")
    assert "config_version" not in content
    assert not profile_path.with_suffix(".toml.bak").exists()


def test_auto_migrate_skips_write_back_when_file_payload_is_invalid(mock_config_dir, monkeypatch):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('[logging]\nlevel = "NOT_A_LEVEL"\n', encoding="utf-8")

    monkeypatch.setenv("TIDY3D_LOGGING__LEVEL", "WARNING")

    with AssertLogStr(
        log_level_expected="WARNING",
        contains_str="Skipping auto-migration write-back",
    ):
        manager = ConfigManager(config_dir=mock_config_dir)

    assert manager.logging.level == "WARNING"
    content = config_path.read_text(encoding="utf-8")
    assert "config_version" not in content
    assert 'level = "NOT_A_LEVEL"' in content
    assert not config_path.with_suffix(".toml.bak").exists()


def test_unknown_env_override_is_ignored(mock_config_dir, monkeypatch):
    monkeypatch.setenv("TIDY3D_FOO__BAR", "1")

    manager = ConfigManager(config_dir=mock_config_dir)
    assert "foo" not in manager._env_overrides


def test_simcloud_apikey_ignored_when_web_section_missing(monkeypatch):
    monkeypatch.setenv("SIMCLOUD_APIKEY", "secret")
    sections_without_web = {
        name: schema
        for name, schema in config_registry.get_sections().items()
        if name.split(".", 1)[0] != "web"
    }
    monkeypatch.setattr(config_loader, "get_sections", lambda: sections_without_web)

    overrides = config_loader.load_environment_overrides()
    assert "web" not in overrides


def test_forward_compat_best_effort_drops_unknown_keys(mock_config_dir):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    original = "\n".join(
        [
            f"config_version = {CURRENT_CONFIG_VERSION + 1}",
            "[logging]",
            'level = "INFO"',
            'extra_key = "ignored"',
            "",
        ]
    )
    config_path.write_text(original, encoding="utf-8")

    manager = ConfigManager(config_dir=mock_config_dir)
    assert manager.logging.level == "INFO"
    assert config_path.read_text(encoding="utf-8") == original
    assert not config_path.with_suffix(".toml.bak").exists()


def test_forward_compat_best_effort_coerces_non_dict(mock_config_dir):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                f"config_version = {CURRENT_CONFIG_VERSION + 1}",
                'logging = "oops"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    with AssertLogStr(
        log_level_expected="WARNING",
        contains_str="Configuration section 'logging' should be a table",
    ):
        manager = ConfigManager(config_dir=mock_config_dir)
    assert manager.logging.level == "WARNING"


def test_forward_compat_best_effort_drops_unknown_sections_and_plugins():
    data = {
        "config_version": CURRENT_CONFIG_VERSION + 1,
        "logging": {"level": "INFO", "extra_key": "ignored"},
        "future_section": {"key": "value"},
        "plugins": {"future_plugin": {"enabled": True}},
    }

    filtered = config_migrations.best_effort_filter(data)
    assert filtered["logging"] == {"level": "INFO"}
    assert "future_section" not in filtered
    assert filtered.get("plugins", {}) == {}


def test_forward_compat_best_effort_preserves_default_profile_metadata():
    data = {
        "config_version": CURRENT_CONFIG_VERSION + 1,
        "default_profile": "dev",
        "logging": {"level": "INFO"},
    }

    filtered = config_migrations.best_effort_filter(data)
    assert filtered["default_profile"] == "dev"
    assert filtered["logging"] == {"level": "INFO"}


def test_current_version_unknown_top_level_section_raises(mock_config_dir):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                f"config_version = {CURRENT_CONFIG_VERSION}",
                "[loging]",
                'level = "INFO"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Unknown configuration section 'loging'"):
        ConfigManager(config_dir=mock_config_dir)


def test_current_version_plugins_must_be_table(mock_config_dir):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                f"config_version = {CURRENT_CONFIG_VERSION}",
                'plugins = "oops"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match=r"Configuration section 'plugins' should be a table"):
        ConfigManager(config_dir=mock_config_dir)


def test_current_version_registered_plugin_payload_must_be_table(mock_config_dir):
    original_manager = config_registry._MANAGER
    config_registry._MANAGER = None
    try:

        @config_registry.register_section("plugins.strict_demo")
        class StrictDemoPlugin(ConfigSection):
            enabled: bool = Field(False, json_schema_extra={"persist": True})

        config_path = mock_config_dir / "config.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "\n".join(
                [
                    f"config_version = {CURRENT_CONFIG_VERSION}",
                    "[plugins]",
                    'strict_demo = "oops"',
                    "",
                ]
            ),
            encoding="utf-8",
        )

        with AssertLogStr(
            log_level_expected="ERROR",
            contains_str="plugin 'strict_demo'",
        ):
            with pytest.raises(TypeError, match=r"must be a mapping"):
                ConfigManager(config_dir=mock_config_dir)
    finally:
        config_registry._SECTIONS.pop("plugins.strict_demo", None)
        config_registry._MANAGER = original_manager


def test_current_version_unknown_plugin_section_is_ignored(mock_config_dir):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                f"config_version = {CURRENT_CONFIG_VERSION}",
                "[plugins.future_plugin]",
                "enabled = true",
                "[logging]",
                'level = "INFO"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    manager = ConfigManager(config_dir=mock_config_dir)
    assert manager.logging.level == "INFO"


def test_unregistered_optional_core_sections_are_ignored(mock_config_dir, monkeypatch):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                f"config_version = {CURRENT_CONFIG_VERSION}",
                "[web]",
                'apikey = "token"',
                "[logging]",
                'level = "INFO"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    sections_without_optional = {
        name: schema
        for name, schema in config_registry.get_sections().items()
        if name.split(".", 1)[0] not in {"web", "local_cache", "batch_data_cache"}
    }
    monkeypatch.setattr(config_loader, "get_sections", lambda: sections_without_optional)
    monkeypatch.setattr(config_manager_module, "get_sections", lambda: sections_without_optional)

    with AssertLogStr(
        log_level_expected="WARNING",
        contains_str="not available in this build",
    ):
        manager = ConfigManager(config_dir=mock_config_dir)
    assert manager.logging.level == "INFO"


def test_default_profile_metadata_key_is_accepted(mock_config_dir):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                'default_profile = "dev"',
                f"config_version = {CURRENT_CONFIG_VERSION}",
                "[logging]",
                'level = "INFO"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    manager = ConfigManager(config_dir=mock_config_dir)
    assert manager.logging.level == "INFO"
    assert manager.profile == "dev"


def test_default_profile_metadata_key_is_rejected_in_profile_file(mock_config_dir):
    profile_path = mock_config_dir / "profiles" / "dev.toml"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text('default_profile = "prod"\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"only allowed in 'config.toml'"):
        ConfigManager(profile="dev", config_dir=mock_config_dir)


def test_forward_compat_strict_raises(mock_config_dir, monkeypatch):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f'config_version = {CURRENT_CONFIG_VERSION + 1}\n[logging]\nlevel = "INFO"\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("TIDY3D_CONFIG_FORWARD_COMPAT", "strict")

    with pytest.raises(ValueError, match=r"config_version"):
        ConfigManager(config_dir=mock_config_dir)


def test_config_upgrade_dry_run(mock_config_dir):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('[web]\napikey = "token"\n', encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(tidy3d_cli, ["config", "upgrade", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "config_version" in result.output
    assert "config_version" not in config_path.read_text(encoding="utf-8")


def test_config_upgrade_profile_only_targets_profiles(mock_config_dir):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    base_text = '[logging]\nlevel = "INFO"\n'
    config_path.write_text(base_text, encoding="utf-8")

    profile_path = mock_config_dir / "profiles" / "dev.toml"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text('[logging]\nlevel = "DEBUG"\n', encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(tidy3d_cli, ["config", "upgrade", "--profile", "dev"])
    assert result.exit_code == 0, result.output

    assert config_path.read_text(encoding="utf-8") == base_text
    assert "config_version" in profile_path.read_text(encoding="utf-8")


def test_config_upgrade_rolls_back_when_write_fails(mock_config_dir, monkeypatch):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    base_text = '[logging]\nlevel = "INFO"\n'
    config_path.write_text(base_text, encoding="utf-8")
    base_backup_path = config_path.with_suffix(".toml.bak")
    base_backup_text = "pre-upgrade backup snapshot\n"
    base_backup_path.write_text(base_backup_text, encoding="utf-8")

    profile_path = mock_config_dir / "profiles" / "dev.toml"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_text = '[logging]\nlevel = "DEBUG"\n'
    profile_path.write_text(profile_text, encoding="utf-8")

    original_atomic_write_document = ConfigLoader._atomic_write_document
    calls = {"count": 0}

    def fail_on_second_write(self, path, document, *, keep_backup=False):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated write failure")
        return original_atomic_write_document(self, path, document, keep_backup=keep_backup)

    monkeypatch.setattr(ConfigLoader, "_atomic_write_document", fail_on_second_write)

    runner = CliRunner()
    result = runner.invoke(tidy3d_cli, ["config", "upgrade"])
    assert result.exit_code != 0
    assert "Failed to apply configuration upgrade atomically" in result.output
    assert config_path.read_text(encoding="utf-8") == base_text
    assert base_backup_path.read_text(encoding="utf-8") == base_backup_text
    assert profile_path.read_text(encoding="utf-8") == profile_text
    assert not profile_path.with_suffix(".toml.bak").exists()


def test_config_upgrade_check_uses_merged_validation_for_profiles(mock_config_dir):
    original_manager = config_registry._MANAGER
    config_registry._MANAGER = None
    try:

        @config_registry.register_section("required_example")
        class RequiredExample(ConfigSection):
            token: str = Field(..., json_schema_extra={"persist": True})

        base_path = mock_config_dir / "config.toml"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(
            "\n".join(
                [
                    f"config_version = {CURRENT_CONFIG_VERSION}",
                    "[required_example]",
                    'token = "from-base"',
                    "",
                ]
            ),
            encoding="utf-8",
        )

        profile_path = mock_config_dir / "profiles" / "dev.toml"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(
            "\n".join(
                [
                    f"config_version = {CURRENT_CONFIG_VERSION}",
                    "[required_example]",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(tidy3d_cli, ["config", "upgrade", "--check"])
        assert result.exit_code == 0, result.output
        assert "Configuration files are up to date." in result.output
    finally:
        config_registry._SECTIONS.pop("required_example", None)
        config_registry._MANAGER = original_manager


def test_config_upgrade_check_uses_merged_validation_for_base(mock_config_dir):
    original_manager = config_registry._MANAGER
    config_registry._MANAGER = None
    try:

        @config_registry.register_section("required_example")
        class RequiredExample(ConfigSection):
            token: str = Field(..., json_schema_extra={"persist": True})

        base_path = mock_config_dir / "config.toml"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(
            "\n".join(
                [
                    'default_profile = "dev"',
                    f"config_version = {CURRENT_CONFIG_VERSION}",
                    "[required_example]",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        profile_path = mock_config_dir / "profiles" / "dev.toml"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(
            "\n".join(
                [
                    f"config_version = {CURRENT_CONFIG_VERSION}",
                    "[required_example]",
                    'token = "from-profile"',
                    "",
                ]
            ),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(tidy3d_cli, ["config", "upgrade", "--check"])
        assert result.exit_code == 0, result.output
        assert "Configuration files are up to date." in result.output
    finally:
        config_registry._SECTIONS.pop("required_example", None)
        config_registry._MANAGER = original_manager


def test_config_upgrade_check_uses_builtin_defaults_for_default_profile(
    mock_config_dir, monkeypatch
):
    original_manager = config_registry._MANAGER
    config_registry._MANAGER = None
    try:

        @config_registry.register_section("required_example")
        class RequiredExample(ConfigSection):
            token: str = Field(..., json_schema_extra={"persist": True})

        monkeypatch.setitem(
            config_profiles.BUILTIN_PROFILES,
            "default",
            {"required_example": {"token": "from-builtin-default"}},
        )

        base_path = mock_config_dir / "config.toml"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(
            "\n".join(
                [
                    f"config_version = {CURRENT_CONFIG_VERSION}",
                    "[required_example]",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(tidy3d_cli, ["config", "upgrade", "--check"])
        assert result.exit_code == 0, result.output
        assert "Configuration files are up to date." in result.output
    finally:
        config_registry._SECTIONS.pop("required_example", None)
        config_registry._MANAGER = original_manager


def test_config_upgrade_check_fails_when_default_runtime_invalid_but_other_profile_valid(
    mock_config_dir,
):
    original_manager = config_registry._MANAGER
    config_registry._MANAGER = None
    try:

        @config_registry.register_section("required_example")
        class RequiredExample(ConfigSection):
            token: str = Field(..., json_schema_extra={"persist": True})

        base_path = mock_config_dir / "config.toml"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(
            "\n".join(
                [
                    f"config_version = {CURRENT_CONFIG_VERSION}",
                    "[required_example]",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        profile_path = mock_config_dir / "profiles" / "dev.toml"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(
            "\n".join(
                [
                    f"config_version = {CURRENT_CONFIG_VERSION}",
                    "[required_example]",
                    'token = "from-profile"',
                    "",
                ]
            ),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(tidy3d_cli, ["config", "upgrade", "--check"])
        assert result.exit_code != 0
        assert "Configuration files are up to date." not in result.output
    finally:
        config_registry._SECTIONS.pop("required_example", None)
        config_registry._MANAGER = original_manager


def test_profile_auto_migration_write_back_uses_merged_validation(mock_config_dir):
    original_manager = config_registry._MANAGER
    config_registry._MANAGER = None
    try:

        @config_registry.register_section("required_example")
        class RequiredExample(ConfigSection):
            token: str = Field(..., json_schema_extra={"persist": True})

        base_path = mock_config_dir / "config.toml"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(
            "\n".join(
                [
                    "[required_example]",
                    'token = "from-base"',
                    "",
                ]
            ),
            encoding="utf-8",
        )

        profile_path = mock_config_dir / "profiles" / "dev.toml"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text("[required_example]\n", encoding="utf-8")

        manager = ConfigManager(profile="dev", config_dir=mock_config_dir)
        assert manager.get_section("required_example").token == "from-base"

        profile_data = toml.loads(profile_path.read_text(encoding="utf-8"))
        assert profile_data["config_version"] == CURRENT_CONFIG_VERSION
    finally:
        config_registry._SECTIONS.pop("required_example", None)
        config_registry._MANAGER = original_manager


def test_failed_manager_init_does_not_write_back_base_auto_migration(mock_config_dir):
    original_manager = config_registry._MANAGER
    config_registry._MANAGER = None
    try:

        @config_registry.register_section("required_example")
        class RequiredExample(ConfigSection):
            token: str = Field(..., json_schema_extra={"persist": True})

        base_path = mock_config_dir / "config.toml"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        original_base = "[required_example]\n"
        base_path.write_text(original_base, encoding="utf-8")

        profile_path = mock_config_dir / "profiles" / "dev.toml"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(
            '[required_example]\ntoken = "from-profile"\n',
            encoding="utf-8",
        )

        with pytest.raises(ValueError):
            ConfigManager(config_dir=mock_config_dir)

        assert base_path.read_text(encoding="utf-8") == original_base
        assert not base_path.with_suffix(".toml.bak").exists()
    finally:
        config_registry._SECTIONS.pop("required_example", None)
        config_registry._MANAGER = original_manager


def test_base_auto_migration_write_back_uses_merged_validation(mock_config_dir):
    original_manager = config_registry._MANAGER
    config_registry._MANAGER = None
    try:

        @config_registry.register_section("required_example")
        class RequiredExample(ConfigSection):
            token: str = Field(..., json_schema_extra={"persist": True})

        base_path = mock_config_dir / "config.toml"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text("[required_example]\n", encoding="utf-8")

        profile_path = mock_config_dir / "profiles" / "dev.toml"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(
            '[required_example]\ntoken = "from-profile"\n',
            encoding="utf-8",
        )

        manager = ConfigManager(profile="dev", config_dir=mock_config_dir)
        assert manager.get_section("required_example").token == "from-profile"

        base_data = toml.loads(base_path.read_text(encoding="utf-8"))
        assert base_data["config_version"] == CURRENT_CONFIG_VERSION
    finally:
        config_registry._SECTIONS.pop("required_example", None)
        config_registry._MANAGER = original_manager


def test_base_auto_migration_write_back_uses_builtin_defaults_for_default_profile(
    mock_config_dir, monkeypatch
):
    original_manager = config_registry._MANAGER
    config_registry._MANAGER = None
    try:

        @config_registry.register_section("required_example")
        class RequiredExample(ConfigSection):
            token: str = Field(..., json_schema_extra={"persist": True})

        monkeypatch.setitem(
            config_profiles.BUILTIN_PROFILES,
            "default",
            {"required_example": {"token": "from-builtin-default"}},
        )

        base_path = mock_config_dir / "config.toml"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text("[required_example]\n", encoding="utf-8")

        manager = ConfigManager(config_dir=mock_config_dir)
        assert manager.get_section("required_example").token == "from-builtin-default"

        base_data = toml.loads(base_path.read_text(encoding="utf-8"))
        assert base_data["config_version"] == CURRENT_CONFIG_VERSION
    finally:
        config_registry._SECTIONS.pop("required_example", None)
        config_registry._MANAGER = original_manager


def test_preview_schema_upgrade_rejects_non_table_section(mock_config_dir):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f'config_version = {CURRENT_CONFIG_VERSION}\nlogging = "oops"\n',
        encoding="utf-8",
    )

    loader = ConfigLoader(mock_config_dir)
    with pytest.raises(TypeError, match=r"must be a mapping"):
        loader.preview_schema_upgrade(config_path)


def test_config_upgrade_check_fails_for_invalid_non_table_section(mock_config_dir):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f'config_version = {CURRENT_CONFIG_VERSION}\nlogging = "oops"\n',
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match=r"must be a mapping"):
        ConfigManager(config_dir=mock_config_dir)

    runner = CliRunner()
    result = runner.invoke(tidy3d_cli, ["config", "upgrade", "--check"])
    assert result.exit_code != 0
    assert "Configuration files are up to date." not in result.output


def test_config_upgrade_check_does_not_crash_on_import_with_invalid_config(mock_config_dir):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f'config_version = {CURRENT_CONFIG_VERSION}\nlogging = "oops"\n',
        encoding="utf-8",
    )

    script = textwrap.dedent(
        """
        from click.testing import CliRunner
        from tidy3d.web.cli.app import tidy3d_cli

        result = CliRunner().invoke(tidy3d_cli, ["config", "upgrade", "--check"])
        if result.exit_code == 0:
            raise SystemExit("expected non-zero exit code")
        if "must be a mapping" not in result.output:
            raise SystemExit(result.output)
        """
    )
    env = os.environ.copy()
    env["TIDY3D_BASE_DIR"] = str(mock_config_dir.parent)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"


def test_config_upgrade_check_fails_for_unknown_top_level_section(mock_config_dir):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                f"config_version = {CURRENT_CONFIG_VERSION}",
                "[loging]",
                'level = "INFO"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(tidy3d_cli, ["config", "upgrade", "--check"])
    assert result.exit_code != 0
    assert "Unknown configuration section 'loging'" in result.output


def test_config_upgrade_check_fails_for_forward_version(mock_config_dir):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                f"config_version = {CURRENT_CONFIG_VERSION + 1}",
                "[logging]",
                'level = "INFO"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(tidy3d_cli, ["config", "upgrade", "--check"])
    assert result.exit_code != 0
    assert "newer schema version" in result.output


def test_config_upgrade_check_fails_for_default_profile_in_profile_file(mock_config_dir):
    profile_path = mock_config_dir / "profiles" / "dev.toml"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text('default_profile = "prod"\n', encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(tidy3d_cli, ["config", "upgrade", "--check"])
    assert result.exit_code != 0
    assert "only allowed in 'config.toml'" in result.output


def test_get_config_version_rejects_non_integer_numeric():
    with AssertLogStr(
        log_level_expected="WARNING",
        contains_str="Invalid 'config_version' value 1.9",
    ):
        assert config_migrations.get_config_version({"config_version": 1.9}) == 0


def test_tomlkit_parse_failure_is_tolerated(mock_config_dir, monkeypatch):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('[logging]\nlevel = "INFO"\n', encoding="utf-8")

    original_parse = tomlkit.parse
    calls = {"count": 0}

    def flaky_parse(text: str):
        calls["count"] += 1
        if calls["count"] == 1:
            raise Exception("boom")
        return original_parse(text)

    monkeypatch.setattr(tomlkit, "parse", flaky_parse)

    loader = ConfigLoader(mock_config_dir)
    with AssertLogStr(
        log_level_expected="WARNING", contains_str="Failed to parse configuration file"
    ):
        data = loader.load_base()
    assert data == {}


def test_malformed_base_config_does_not_fail_manager_init(mock_config_dir):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('[logging\nlevel = "INFO"\n', encoding="utf-8")

    with AssertLogStr(
        log_level_expected="WARNING", contains_str="Failed to parse configuration file"
    ):
        manager = ConfigManager(config_dir=mock_config_dir)

    assert manager.logging.level == "WARNING"


def test_backward_auto_migration_failure_raises_and_logs(mock_config_dir, monkeypatch):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('[logging]\nlevel = "INFO"\n', encoding="utf-8")

    def fail_migration(*args, **kwargs):
        raise RuntimeError("migration failed")

    monkeypatch.setattr(config_loader, "apply_migrations", fail_migration)

    with AssertLogStr(log_level_expected="ERROR", contains_str="tidy3d config upgrade"):
        with pytest.raises(ValueError, match=r"Automatic configuration migration failed"):
            ConfigManager(config_dir=mock_config_dir)
    content = config_path.read_text(encoding="utf-8")
    assert "config_version" not in content


def test_backward_auto_migration_failure_does_not_use_best_effort_fallback(
    mock_config_dir, monkeypatch
):
    original_manager = config_registry._MANAGER
    config_registry._MANAGER = None
    try:

        @config_registry.register_section("required_example")
        class RequiredExample(ConfigSection):
            token: str = Field("default", json_schema_extra={"persist": True})

        config_path = mock_config_dir / "config.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "\n".join(
                [
                    "[required_example]",
                    'legacy_token = "from-v0"',
                    "",
                ]
            ),
            encoding="utf-8",
        )

        def fail_migration(*args, **kwargs):
            raise RuntimeError("migration failed")

        monkeypatch.setattr(config_loader, "apply_migrations", fail_migration)

        with AssertLogStr(log_level_expected="ERROR", contains_str="tidy3d config upgrade"):
            with pytest.raises(ValueError, match=r"Automatic configuration migration failed"):
                ConfigManager(config_dir=mock_config_dir)
        content = config_path.read_text(encoding="utf-8")
        assert "config_version" not in content
        assert "legacy_token" in content
    finally:
        config_registry._SECTIONS.pop("required_example", None)
        config_registry._MANAGER = original_manager


def test_backward_auto_migration_failure_restores_cached_document(mock_config_dir, monkeypatch):
    config_path = mock_config_dir / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('[logging]\n# keep\nlevel = "INFO"\n', encoding="utf-8")

    def fail_migration(document, *_args, **_kwargs):
        document["partially_migrated"] = True
        raise RuntimeError("migration failed")

    monkeypatch.setattr(config_loader, "apply_migrations", fail_migration)

    loader = ConfigLoader(mock_config_dir)
    with AssertLogStr(log_level_expected="ERROR", contains_str="tidy3d config upgrade"):
        with pytest.raises(ValueError, match=r"Automatic configuration migration failed"):
            loader.load_base()
    cached_text = tomlkit.dumps(loader._docs[config_path])
    assert "partially_migrated" not in cached_text
    assert "# keep" in cached_text


def test_migration_chain_applies_and_validates(tmp_path, monkeypatch):
    original_manager = config_registry._MANAGER
    config_registry._MANAGER = None
    try:

        @config_registry.register_section("example")
        class ExampleConfig(ConfigSection):
            token: str = Field("default", json_schema_extra={"persist": True})

        calls: list[int] = []

        @config_migrations.register_migration(1)
        def _migrate_v1_to_v2(document: tomlkit.TOMLDocument) -> None:
            calls.append(1)
            table = document.get("example")
            if not isinstance(table, tomlkit.items.Table):
                table = tomlkit.table()
                document["example"] = table
            if "legacy_token" in table:
                table["token"] = table["legacy_token"]
                del table["legacy_token"]
            if "token" not in table:
                table["token"] = "migrated"

        monkeypatch.setattr(config_migrations, "CURRENT_CONFIG_VERSION", 2)
        config_loader = importlib.import_module("tidy3d.config.loader")

        monkeypatch.setattr(config_loader, "CURRENT_CONFIG_VERSION", 2)

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_path = config_dir / "config.toml"
        config_path.write_text('[example]\nlegacy_token = "from-v0"\n', encoding="utf-8")

        manager = ConfigManager(config_dir=config_dir)
        assert manager.get_section("example").token == "from-v0"
        assert calls == [1]

        migrated = toml.loads(config_path.read_text(encoding="utf-8"))
        assert migrated["config_version"] == 2
        assert migrated["example"]["token"] == "from-v0"
        assert "legacy_token" not in migrated["example"]
    finally:
        config_registry._SECTIONS.pop("example", None)
        config_registry._MANAGER = original_manager
        migrations = config_migrations._MIGRATIONS.get(1, [])
        if "_migrate_v1_to_v2" in locals() and _migrate_v1_to_v2 in migrations:
            migrations.remove(_migrate_v1_to_v2)
        if not migrations:
            config_migrations._MIGRATIONS.pop(1, None)
        config_migrations._MIGRATION_CHAIN_VALIDATED_UP_TO = 0


def test_migration_chain_gap_raises(monkeypatch):
    def _noop_migration(document: tomlkit.TOMLDocument) -> None:
        return None

    monkeypatch.setattr(config_migrations, "_MIGRATIONS", {0: [_noop_migration]})
    monkeypatch.setattr(config_migrations, "_MIGRATION_CHAIN_VALIDATED_UP_TO", 0)

    document = tomlkit.parse("")
    with pytest.raises(RuntimeError, match=r"v1 -> v2"):
        config_migrations.apply_migrations(document, 0, 2)


def test_migrations_are_idempotent():
    document = tomlkit.parse('[logging]\nlevel = "INFO"\n')
    config_migrations.apply_migrations(document, 0, CURRENT_CONFIG_VERSION)
    first = tomlkit.dumps(document)

    config_migrations.apply_migrations(document, 0, CURRENT_CONFIG_VERSION)
    second = tomlkit.dumps(document)

    assert first == second


def test_migration_moves_vgpu_pay_type_to_run(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.toml"
    config_path.write_text(
        'config_version = 1\n[vgpu]\npay_type = "CREDITS"\npriority = 3\n',
        encoding="utf-8",
    )

    manager = ConfigManager(config_dir=config_dir)
    assert manager.run.pay_type == PayType.CREDITS.value
    assert manager.vgpu.priority == 3
    assert not hasattr(manager.vgpu, "pay_type")

    migrated = toml.loads(config_path.read_text(encoding="utf-8"))
    assert migrated["config_version"] == CURRENT_CONFIG_VERSION
    assert migrated["run"]["pay_type"] == "CREDITS"
    assert "pay_type" not in migrated["vgpu"]


def test_deprecated_field_warns(tmp_path, monkeypatch):
    original_manager = config_registry._MANAGER
    config_registry._MANAGER = None
    try:

        @config_registry.register_section("deprecated_example")
        class DeprecatedExample(ConfigSection):
            old: str = Field(
                "default",
                json_schema_extra={
                    "persist": True,
                    "deprecated_in": 1,
                    "replaced_by": "deprecated_example.new",
                },
            )

        monkeypatch.setattr(config_migrations, "CURRENT_CONFIG_VERSION", 1)
        monkeypatch.setattr(config_loader, "CURRENT_CONFIG_VERSION", 1)

        config_dir = tmp_path / "config_dir"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            '[deprecated_example]\nold = "value"\n', encoding="utf-8"
        )

        with AssertLogStr(
            log_level_expected="WARNING",
            contains_str="deprecated_example.old",
        ):
            ConfigManager(config_dir=config_dir)
    finally:
        config_registry._SECTIONS.pop("deprecated_example", None)
        config_registry._MANAGER = original_manager


def test_invalid_deprecation_window_raises(monkeypatch):
    original_manager = config_registry._MANAGER
    config_registry._MANAGER = None
    try:

        @config_registry.register_section("bad_window")
        class BadWindow(ConfigSection):
            old: str = Field(
                "default",
                json_schema_extra={
                    "persist": True,
                    "deprecated_in": 2,
                    "removed_in": 3,
                },
            )

        monkeypatch.setattr(config_migrations, "CURRENT_CONFIG_VERSION", 2)
        with pytest.raises(ValueError, match=r"violates the minimum window"):
            build_validated_models({"bad_window": {"old": "value"}}, error_context="validate")
    finally:
        config_registry._SECTIONS.pop("bad_window", None)
        config_registry._MANAGER = original_manager


def test_removed_field_raises(tmp_path, monkeypatch):
    original_manager = config_registry._MANAGER
    config_registry._MANAGER = None
    try:

        @config_registry.register_section("removed_example")
        class RemovedExample(ConfigSection):
            old: str = Field(
                "default",
                json_schema_extra={"persist": True, "removed_in": 1},
            )

        monkeypatch.setattr(config_migrations, "CURRENT_CONFIG_VERSION", 1)
        monkeypatch.setattr(config_loader, "CURRENT_CONFIG_VERSION", 1)

        config_dir = tmp_path / "config_dir"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            '[removed_example]\nold = "value"\n', encoding="utf-8"
        )

        with AssertLogStr(
            log_level_expected="ERROR",
            contains_str="Failed to load configuration for section 'removed_example'",
        ):
            with pytest.raises(ValueError, match=r"removed in config schema v1"):
                ConfigManager(config_dir=config_dir)
    finally:
        config_registry._SECTIONS.pop("removed_example", None)
        config_registry._MANAGER = original_manager


def test_validate_config_data_preserves_earlier_errors_when_removed_field_present(monkeypatch):
    original_manager = config_registry._MANAGER
    config_registry._MANAGER = None
    try:

        @config_registry.register_section("error_first")
        class ErrorFirst(ConfigSection):
            value: int = Field(0, json_schema_extra={"persist": True})

        @config_registry.register_section("removed_later")
        class RemovedLater(ConfigSection):
            old: str = Field(
                "default",
                json_schema_extra={"persist": True, "removed_in": 1},
            )

        monkeypatch.setattr(config_migrations, "CURRENT_CONFIG_VERSION", 1)

        data = {
            "error_first": {"value": "not-an-int"},
            "removed_later": {"old": "value"},
        }
        with AssertLogStr(
            log_level_expected="ERROR",
            contains_str="Failed to validate configuration for section 'removed_later'",
        ):
            with pytest.raises(ValueError) as exc_info:
                build_validated_models(data, error_context="validate")
        message = str(exc_info.value)
        assert "valid integer" in message
        assert "removed in config schema" not in message
    finally:
        config_registry._SECTIONS.pop("error_first", None)
        config_registry._SECTIONS.pop("removed_later", None)
        config_registry._MANAGER = original_manager
