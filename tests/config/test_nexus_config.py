"""Tests for Nexus configuration."""

from __future__ import annotations

import toml


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
