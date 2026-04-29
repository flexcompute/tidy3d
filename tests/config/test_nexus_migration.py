"""Tests for Nexus configuration migration from old to new format."""

from __future__ import annotations

import pytest
import toml

from tidy3d.config.legacy import load_legacy_flat_config
from tidy3d.config.loader import ConfigLoader
from tidy3d.config.manager import ConfigManager


@pytest.fixture
def old_nexus_config(tmp_path):
    """Create an old-style Nexus configuration."""
    config_dir = tmp_path / ".tidy3d"
    config_dir.mkdir()
    old_config = config_dir / "config"

    old_config.write_text("""
apikey = "test-nexus-key"
web_api_endpoint = "http://nexus.company.com:5000"
website_endpoint = "http://nexus.company.com/tidy3d"
s3_region = "us-west-2"
s3_endpoint = "http://nexus.company.com:9000"
ssl_verify = false
enable_caching = true
""")

    return config_dir, old_config


@pytest.fixture
def old_minimal_config(tmp_path):
    """Create an old-style config with just API key."""
    config_dir = tmp_path / ".tidy3d"
    config_dir.mkdir()
    old_config = config_dir / "config"

    old_config.write_text('apikey = "test-key"\n')

    return config_dir, old_config


def test_load_legacy_nexus_config(old_nexus_config):
    """Test that load_legacy_flat_config parses all Nexus fields correctly."""
    config_dir, _old_config = old_nexus_config

    legacy_data = load_legacy_flat_config(config_dir)

    assert "web" in legacy_data
    web = legacy_data["web"]

    # Check all fields are migrated
    assert web["apikey"] == "test-nexus-key"
    assert web["api_endpoint"] == "http://nexus.company.com:5000"
    assert web["website_endpoint"] == "http://nexus.company.com/tidy3d"
    assert web["s3_region"] == "us-west-2"
    assert web["ssl_verify"] is False
    assert web["enable_caching"] is True

    # S3 endpoint should be in env_vars
    assert "env_vars" in web
    assert web["env_vars"]["AWS_ENDPOINT_URL_S3"] == "http://nexus.company.com:9000"


def test_load_legacy_minimal_config(old_minimal_config):
    """Test that minimal old config (just API key) still works."""
    config_dir, _old_config = old_minimal_config

    legacy_data = load_legacy_flat_config(config_dir)

    assert "web" in legacy_data
    assert legacy_data["web"]["apikey"] == "test-key"
    assert len(legacy_data["web"]) == 1  # Only apikey


def test_auto_migration_on_load(old_nexus_config):
    """Test that ConfigLoader auto-migrates legacy config on first load."""
    config_dir, old_config = old_nexus_config

    # Verify old config exists
    assert old_config.exists()

    # Load via ConfigLoader (should trigger auto-migration)
    loader = ConfigLoader(config_dir)
    data = loader.load_base()

    # New config.toml should be created
    new_config = config_dir / "config.toml"
    assert new_config.exists()

    # Old config should be backed up
    backup = config_dir / "config.migrated"
    assert backup.exists()
    assert not old_config.exists()

    # Verify data is correct
    assert "web" in data
    assert data["web"]["api_endpoint"] == "http://nexus.company.com:5000"


def test_migrated_config_format(old_nexus_config):
    """Test that migrated config is in correct nested TOML format."""
    config_dir, _old_config = old_nexus_config

    # Trigger migration
    loader = ConfigLoader(config_dir)
    loader.load_base()

    # Read the new config file
    new_config = config_dir / "config.toml"
    parsed = toml.loads(new_config.read_text())

    # Should have nested structure
    assert "web" in parsed
    assert "api_endpoint" in parsed["web"]
    assert "website_endpoint" in parsed["web"]
    assert "env_vars" in parsed["web"]
    assert "AWS_ENDPOINT_URL_S3" in parsed["web"]["env_vars"]

    # Old flat keys should NOT exist
    assert "web_api_endpoint" not in parsed
    assert "s3_endpoint" not in parsed


def test_config_manager_with_legacy_nexus(old_nexus_config):
    """Test that ConfigManager can load and use legacy Nexus config."""
    config_dir, old_config = old_nexus_config

    manager = ConfigManager(config_dir=config_dir)

    # Manager load should persist the migrated file and back up the legacy flat config.
    new_config = config_dir / "config.toml"
    backup = config_dir / "config.migrated"
    assert new_config.exists()
    assert backup.exists()
    assert not old_config.exists()

    # Should auto-load Nexus settings
    web = manager.get_section("web")

    assert str(web.api_endpoint) == "http://nexus.company.com:5000"
    assert str(web.website_endpoint) == "http://nexus.company.com/tidy3d"
    assert web.s3_region == "us-west-2"
    assert web.ssl_verify is False
    assert web.enable_caching is True

    # Check env_vars
    assert "AWS_ENDPOINT_URL_S3" in web.env_vars
    assert web.env_vars["AWS_ENDPOINT_URL_S3"] == "http://nexus.company.com:9000"


def test_config_manager_legacy_invalid_payload_skips_write_back(tmp_path, monkeypatch):
    """Invalid legacy payloads should load with env overrides but not be persisted."""
    from tests.utils import AssertLogStr

    config_dir = tmp_path / ".tidy3d"
    config_dir.mkdir()
    legacy_file = config_dir / "config"
    legacy_file.write_text('apikey = "test-key"\nenable_caching = "not-a-bool"\n', encoding="utf-8")

    monkeypatch.setenv("TIDY3D_WEB__ENABLE_CACHING", "false")

    with AssertLogStr(
        log_level_expected="WARNING",
        contains_str="Skipping auto-migration write-back",
    ):
        manager = ConfigManager(config_dir=config_dir)

    assert manager.get_section("web").enable_caching is False
    assert legacy_file.exists()
    assert not (config_dir / "config.toml").exists()
    assert not (config_dir / "config.migrated").exists()


def test_no_migration_if_new_config_exists(tmp_path):
    """Test that migration doesn't happen if config.toml already exists."""
    config_dir = tmp_path / ".tidy3d"
    config_dir.mkdir()

    # Create both old and new config
    old_config = config_dir / "config"
    old_config.write_text('apikey = "old-key"\n')

    new_config = config_dir / "config.toml"
    new_config.write_text('[web]\napikey = "new-key"\n')

    # Load - should use new config, not migrate
    loader = ConfigLoader(config_dir)
    data = loader.load_base()

    # Should have loaded from new config
    assert data["web"]["apikey"] == "new-key"

    # Old config should still exist (not backed up)
    assert old_config.exists()
    backup = config_dir / "config.migrated"
    assert not backup.exists()


def test_partial_nexus_config(tmp_path):
    """Test migration with partial Nexus settings."""
    config_dir = tmp_path / ".tidy3d"
    config_dir.mkdir()
    old_config = config_dir / "config"

    # Only some Nexus fields
    old_config.write_text("""
apikey = "test-key"
web_api_endpoint = "http://custom:5000"
website_endpoint = "http://custom/web"
""")

    legacy_data = load_legacy_flat_config(config_dir)

    assert legacy_data["web"]["apikey"] == "test-key"
    assert legacy_data["web"]["api_endpoint"] == "http://custom:5000"
    assert legacy_data["web"]["website_endpoint"] == "http://custom/web"

    # Fields not provided shouldn't be in the dict
    assert "s3_region" not in legacy_data["web"]
    assert "env_vars" not in legacy_data["web"]


def test_save_after_migration(old_nexus_config):
    """Test that saving after migration works correctly.

    After migration, when save() is called on the default profile,
    only persisted fields are written to base config. Non-persisted
    fields like api_endpoint are filtered out.
    """
    config_dir, _old_config = old_nexus_config

    manager = ConfigManager(config_dir=config_dir)

    # Verify that before save, the manager has access to migrated values
    assert str(manager.web.api_endpoint) == "http://nexus.company.com:5000"

    # Modify a persisted setting
    manager.update_section("web", enable_caching=False)
    manager.save()

    # Read back the config
    new_config = config_dir / "config.toml"
    parsed = toml.loads(new_config.read_text())

    # Only persisted fields remain in base config after save
    # apikey and enable_caching are persisted
    assert "apikey" in parsed["web"]
    assert parsed["web"]["enable_caching"] is False

    # Non-persisted fields like api_endpoint and timeout are filtered out
    assert "api_endpoint" not in parsed.get("web", {})
    assert "timeout" not in parsed.get("web", {})


def test_no_migration_if_no_legacy_config(tmp_path):
    """Test that loader handles missing legacy config gracefully."""
    config_dir = tmp_path / ".tidy3d"
    config_dir.mkdir()

    loader = ConfigLoader(config_dir)
    data = loader.load_base()

    # Should return empty dict
    assert data == {}

    # No files should be created
    assert not (config_dir / "config").exists()
    assert not (config_dir / "config.toml").exists()
    assert not (config_dir / "config.migrated").exists()


def test_legacy_payload_migration_failure_falls_back_without_legacy_file(tmp_path, monkeypatch):
    from tests.utils import AssertLogStr
    from tidy3d.config import legacy as config_legacy

    config_dir = tmp_path / ".tidy3d"
    config_dir.mkdir()

    legacy_payload = {"web": {"apikey": "legacy-key"}}

    monkeypatch.setattr(
        config_legacy, "load_legacy_flat_config", lambda _config_dir: legacy_payload
    )

    def fail_migration(self, _data):
        raise RuntimeError("boom")

    monkeypatch.setattr(ConfigLoader, "_migrate_legacy_payload", fail_migration)

    loader = ConfigLoader(config_dir)
    with AssertLogStr(
        log_level_expected="WARNING", contains_str="Using legacy data without migration"
    ):
        data = loader.load_base()

    assert data == legacy_payload
    assert not (config_dir / "config.toml").exists()


def test_migration_preserves_comments_when_possible(old_nexus_config):
    """Test that migration creates a clean, well-formatted config."""
    config_dir, _old_config = old_nexus_config

    # Trigger migration
    loader = ConfigLoader(config_dir)
    loader.load_base()

    # Read the new config as text
    new_config = config_dir / "config.toml"
    content = new_config.read_text()

    # Should have section headers
    assert "[web]" in content

    # Should not have old flat format
    assert "web_api_endpoint" not in content


def test_configure_nexus_saves_to_profile(tmp_path):
    """Test that configuring nexus saves to profiles/nexus.toml, not base config."""
    from tidy3d.config import ConfigManager

    # Create a fresh config manager with temp directory
    manager = ConfigManager(config_dir=tmp_path)

    # Save API key to base config first (simulating normal configure flow)
    manager.update_section("web", apikey="test-key")
    manager.save()

    # Switch to nexus profile and configure custom nexus settings
    manager.switch_profile("nexus")
    manager.update_section(
        "web",
        api_endpoint="http://custom-nexus.company.com/tidy3d-api",
        website_endpoint="http://custom-nexus.company.com/tidy3d",
        env_vars={"AWS_ENDPOINT_URL_S3": "http://custom-nexus.company.com:9000"},
    )
    manager.save()

    # Set nexus as default
    manager.set_default_profile("nexus")

    # Check that profiles/nexus.toml was created
    nexus_profile = tmp_path / "profiles" / "nexus.toml"
    assert nexus_profile.exists(), "Nexus profile file should be created"

    # Read the nexus profile
    nexus_data = toml.loads(nexus_profile.read_text())

    # Should contain the custom nexus settings
    assert "web" in nexus_data
    assert nexus_data["web"]["api_endpoint"] == "http://custom-nexus.company.com/tidy3d-api"
    assert nexus_data["web"]["website_endpoint"] == "http://custom-nexus.company.com/tidy3d"
    assert (
        nexus_data["web"]["env_vars"]["AWS_ENDPOINT_URL_S3"]
        == "http://custom-nexus.company.com:9000"
    )

    # Check base config
    base_config = tmp_path / "config.toml"
    assert base_config.exists()
    base_data = toml.loads(base_config.read_text())

    # Base config should have apikey and may have default endpoints (not custom ones)
    assert "web" in base_data
    assert base_data["web"]["apikey"] == "test-key"
    # Should NOT have the custom nexus endpoint
    if "api_endpoint" in base_data["web"]:
        assert base_data["web"]["api_endpoint"] != "http://custom-nexus.company.com/tidy3d-api"

    # Verify default profile is set
    assert "default_profile" in base_data
    assert base_data["default_profile"] == "nexus"


def test_configure_nexus_loads_correctly(tmp_path):
    """Test that after configuring nexus, loading the profile works correctly."""
    from tidy3d.config import ConfigManager

    # Create a fresh config manager and configure nexus
    manager = ConfigManager(config_dir=tmp_path)

    # Save API key to base
    manager.update_section("web", apikey="test-key")
    manager.save()

    # Configure nexus settings
    manager.switch_profile("nexus")
    manager.update_section(
        "web",
        api_endpoint="http://my-nexus.example.com/tidy3d-api",
        website_endpoint="http://my-nexus.example.com/tidy3d",
        env_vars={"AWS_ENDPOINT_URL_S3": "http://my-nexus.example.com:9000"},
    )
    manager.save()
    manager.set_default_profile("nexus")

    # Create a NEW manager instance to simulate a fresh load
    new_manager = ConfigManager(config_dir=tmp_path)

    # Should automatically load nexus profile (because default_profile is set)
    assert new_manager.profile == "nexus"
    assert str(new_manager.web.api_endpoint) == "http://my-nexus.example.com/tidy3d-api"
    assert str(new_manager.web.website_endpoint) == "http://my-nexus.example.com/tidy3d"
    assert new_manager.web.env_vars["AWS_ENDPOINT_URL_S3"] == "http://my-nexus.example.com:9000"

    # Verify we can manually switch to production and back
    new_manager.switch_profile("prod")
    assert str(new_manager.web.api_endpoint) == "https://tidy3d-api.simulation.cloud"

    # Switch back to nexus
    new_manager.switch_profile("nexus")
    assert str(new_manager.web.api_endpoint) == "http://my-nexus.example.com/tidy3d-api"


def test_nexus_url_derivation():
    """Test that nexus URL derivation handles edge cases correctly."""
    from urllib.parse import urlparse, urlunparse

    test_cases = [
        # (input_url, expected_api, expected_website, expected_s3)
        (
            "http://localhost",
            "http://localhost/tidy3d-api",
            "http://localhost/tidy3d",
            "http://localhost:9000",
        ),
        (
            "http://localhost/",
            "http://localhost/tidy3d-api",
            "http://localhost/tidy3d",
            "http://localhost:9000",
        ),
        (
            "http://localhost:8080",
            "http://localhost:8080/tidy3d-api",
            "http://localhost:8080/tidy3d",
            "http://localhost:9000",
        ),
        (
            "http://nexus.company.com",
            "http://nexus.company.com/tidy3d-api",
            "http://nexus.company.com/tidy3d",
            "http://nexus.company.com:9000",
        ),
        (
            "https://nexus.company.com/",
            "https://nexus.company.com/tidy3d-api",
            "https://nexus.company.com/tidy3d",
            "https://nexus.company.com:9000",
        ),
    ]

    for nexus_url, expected_api, expected_website, expected_s3 in test_cases:
        # Replicate the logic from configure_fn
        base_url = nexus_url.rstrip("/")
        api_endpoint = f"{base_url}/tidy3d-api"
        website_endpoint = f"{base_url}/tidy3d"

        parsed = urlparse(nexus_url)
        hostname = parsed.hostname or parsed.netloc.split(":")[0]
        s3_netloc = f"{hostname}:9000"
        s3_endpoint = urlunparse((parsed.scheme, s3_netloc, "", "", "", ""))

        assert api_endpoint == expected_api, f"Failed for {nexus_url}: api_endpoint"
        assert website_endpoint == expected_website, f"Failed for {nexus_url}: website_endpoint"
        assert s3_endpoint == expected_s3, f"Failed for {nexus_url}: s3_endpoint"


def test_configure_fn_with_nexus_url(tmp_path, monkeypatch):
    """Test configure_fn with nexus_url parameter."""
    from unittest.mock import Mock

    from tidy3d.config import ConfigManager
    from tidy3d.web.cli.config import configure_fn

    # Create a fresh config manager
    manager = ConfigManager(config_dir=tmp_path)

    # Monkeypatch the global config and requests
    import tidy3d.web.cli.config as cli_config

    monkeypatch.setattr(cli_config, "config", manager)

    # Mock successful API key validation
    mock_response = Mock()
    mock_response.status_code = 200
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: mock_response)

    # Test with nexus_url
    configure_fn(
        apikey="test-api-key",
        nexus_url="http://localhost:8080",
    )

    # Verify profile was created
    nexus_profile = tmp_path / "profiles" / "nexus.toml"
    assert nexus_profile.exists()

    # Verify endpoints were derived correctly
    nexus_data = toml.loads(nexus_profile.read_text())
    assert nexus_data["web"]["api_endpoint"] == "http://localhost:8080/tidy3d-api"
    assert nexus_data["web"]["website_endpoint"] == "http://localhost:8080/tidy3d"
    assert nexus_data["web"]["env_vars"]["AWS_ENDPOINT_URL_S3"] == "http://localhost:9000"


def test_configure_fn_with_manual_endpoints(tmp_path, monkeypatch):
    """Test configure_fn with manual endpoint parameters."""
    from unittest.mock import Mock

    from tidy3d.config import ConfigManager
    from tidy3d.web.cli.config import configure_fn

    manager = ConfigManager(config_dir=tmp_path)

    import tidy3d.web.cli.config as cli_config

    monkeypatch.setattr(cli_config, "config", manager)

    mock_response = Mock()
    mock_response.status_code = 200
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: mock_response)

    # Test with manual endpoints
    # Note: Using ssl_verify=True to differ from builtin nexus default (False)
    configure_fn(
        apikey="test-key",
        api_endpoint="http://custom:5000/api",
        website_endpoint="http://custom:5000/web",
        s3_endpoint="http://custom:9000",
        ssl_verify=True,  # Different from builtin nexus default
        enable_caching=True,  # Different from builtin nexus default
    )

    nexus_profile = tmp_path / "profiles" / "nexus.toml"
    assert nexus_profile.exists()

    nexus_data = toml.loads(nexus_profile.read_text())
    assert nexus_data["web"]["api_endpoint"] == "http://custom:5000/api"
    assert nexus_data["web"]["website_endpoint"] == "http://custom:5000/web"
    # These should be saved since they differ from builtin nexus defaults
    assert nexus_data["web"]["ssl_verify"] is True
    assert nexus_data["web"]["enable_caching"] is True


def test_configure_fn_validation_error(tmp_path, monkeypatch, capsys):
    """Test configure_fn with incomplete endpoint specification."""
    from tidy3d.config import ConfigManager
    from tidy3d.web.cli.config import configure_fn

    manager = ConfigManager(config_dir=tmp_path)

    import tidy3d.web.cli.config as cli_config

    monkeypatch.setattr(cli_config, "config", manager)

    # Only provide api_endpoint without website_endpoint (should fail)
    configure_fn(
        apikey="test-key",
        api_endpoint="http://custom:5000/api",
    )

    captured = capsys.readouterr()
    assert "Both --api-endpoint and --website-endpoint must be provided together" in captured.out


def test_configure_fn_restore_defaults(tmp_path, monkeypatch, capsys):
    """Test configure_fn with restore_defaults flag."""
    from unittest.mock import Mock

    from tidy3d.config import ConfigManager
    from tidy3d.web.cli.config import configure_fn

    manager = ConfigManager(config_dir=tmp_path)

    import tidy3d.web.cli.config as cli_config

    monkeypatch.setattr(cli_config, "config", manager)

    # First configure nexus
    mock_response = Mock()
    mock_response.status_code = 200
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: mock_response)

    configure_fn(apikey="test-key", nexus_url="http://localhost")

    # Verify profile exists
    nexus_profile = tmp_path / "profiles" / "nexus.toml"
    assert nexus_profile.exists()

    # Now restore defaults
    configure_fn(apikey=None, restore_defaults=True)

    # Verify profile was removed
    assert not nexus_profile.exists()

    # Verify message was printed
    captured = capsys.readouterr()
    assert "Successfully restored production defaults" in captured.out
    assert "Cleared default_profile setting" in captured.out


def test_get_default_profile_error_handling(tmp_path):
    """Test get_default_profile handles corrupted config gracefully."""
    from tidy3d.config.loader import ConfigLoader

    config_dir = tmp_path / ".tidy3d"
    config_dir.mkdir()
    config_file = config_dir / "config.toml"

    # Write invalid TOML
    config_file.write_text("invalid toml {{{")

    loader = ConfigLoader(config_dir)
    result = loader.get_default_profile()

    # Should return None instead of crashing
    assert result is None


def test_legacy_migration_error_handling(tmp_path, monkeypatch):
    """Test that legacy migration errors are handled gracefully."""
    from tidy3d.config import ConfigManager
    from tidy3d.config.loader import ConfigLoader

    config_dir = tmp_path / ".tidy3d"
    config_dir.mkdir()

    # Create a legacy config file
    legacy_file = config_dir / "config"
    legacy_file.write_text("apikey = test-key")

    # Mock save_base to raise an exception during migration
    def mock_save_failure(self, data):
        raise RuntimeError("Migration save failed")

    monkeypatch.setattr(ConfigLoader, "save_base", mock_save_failure)

    # This should not crash, even though migration fails
    # ConfigManager uses ConfigLoader internally
    manager = ConfigManager(config_dir=config_dir)

    # Should still be able to access config (falls back to legacy data)
    # The key point is that no exception is raised
    assert manager is not None


def test_api_key_validation_failure(tmp_path, monkeypatch, capsys):
    """Test configure_fn handles API key validation failure."""
    from unittest.mock import Mock

    from tidy3d.config import ConfigManager
    from tidy3d.web.cli.config import configure_fn

    manager = ConfigManager(config_dir=tmp_path)

    import tidy3d.web.cli.config as cli_config

    monkeypatch.setattr(cli_config, "config", manager)

    # Mock failed API key validation
    mock_response = Mock()
    mock_response.status_code = 401  # Unauthorized
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: mock_response)

    # Try to configure with invalid API key
    configure_fn(
        apikey="invalid-key",
        nexus_url="http://localhost",
    )

    # Verify error message was printed
    captured = capsys.readouterr()
    assert "API key validation failed" in captured.out
    assert "401" in captured.out

    # Verify no profile was created
    nexus_profile = tmp_path / "profiles" / "nexus.toml"
    assert not nexus_profile.exists()


def test_profile_notification_on_init(tmp_path):
    """Test that non-default profile usage is logged on initialization."""
    from tests.utils import AssertLogStr
    from tidy3d.config import ConfigManager

    # Test 1: Default profile should not log
    with AssertLogStr(log_level_expected="INFO", excludes_str="Using configuration profile"):
        manager = ConfigManager(config_dir=tmp_path, profile="default")

    # Test 2: Nexus profile should log
    with AssertLogStr(
        log_level_expected="INFO", contains_str="Using configuration profile: 'nexus'"
    ):
        manager = ConfigManager(config_dir=tmp_path, profile="nexus")


def test_profile_notification_on_switch(tmp_path):
    """Test that profile switching is logged."""
    from tests.utils import AssertLogStr
    from tidy3d.config import ConfigManager

    manager = ConfigManager(config_dir=tmp_path)

    # Switch to nexus profile - should log
    with AssertLogStr(
        log_level_expected="INFO", contains_str="Switched to configuration profile: 'nexus'"
    ):
        manager.switch_profile("nexus")

    # Switch back to default - should not log
    with AssertLogStr(log_level_expected="INFO", excludes_str="Switched to configuration profile"):
        manager.switch_profile("default")
