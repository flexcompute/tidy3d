"""Tests for CLI commands including nexus configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import toml
from click.testing import CliRunner

from tidy3d.web.cli import tidy3d_cli
from tidy3d.web.cli.app import configure_fn
from tidy3d.web.core.environment import Environment


@pytest.fixture
def runner():
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


@pytest.fixture
def temp_config_dir(tmp_path, monkeypatch):
    """Create a temporary config directory."""
    config_dir = tmp_path / ".tidy3d"
    config_dir.mkdir()
    config_file = config_dir / "config"

    # Patch the CONFIG_FILE and TIDY3D_DIR at module level
    monkeypatch.setattr("tidy3d.web.cli.app.CONFIG_FILE", str(config_file))
    monkeypatch.setattr("tidy3d.web.cli.app.TIDY3D_DIR", str(config_dir))

    return tmp_path, config_file


# CLI Command Tests


def test_nexus_command_minimal(runner, temp_config_dir):
    """Test configure with nexus endpoints."""
    tmp_path, config_file = temp_config_dir

    result = runner.invoke(
        tidy3d_cli,
        [
            "configure",
            "--api-endpoint",
            "http://test-api:5000",
            "--website-endpoint",
            "http://test-website/tidy3d",
        ],
    )

    assert result.exit_code == 0
    assert "Nexus environment configured successfully" in result.output

    config = toml.loads(config_file.read_text())
    assert config["web_api_endpoint"] == "http://test-api:5000"
    assert config["website_endpoint"] == "http://test-website/tidy3d"


def test_nexus_command_full_options(runner, temp_config_dir):
    """Test configure with all nexus options."""
    tmp_path, config_file = temp_config_dir

    result = runner.invoke(
        tidy3d_cli,
        [
            "configure",
            "--api-endpoint",
            "http://api:5000",
            "--website-endpoint",
            "http://web/tidy3d",
            "--s3-region",
            "eu-west-1",
            "--s3-endpoint",
            "http://s3:9000",
            "--ssl-verify",
            "--enable-caching",
        ],
    )

    assert result.exit_code == 0

    config = toml.loads(config_file.read_text())
    assert config["s3_region"] == "eu-west-1"
    assert config["s3_endpoint"] == "http://s3:9000"
    assert config["ssl_verify"] is True
    assert config["enable_caching"] is True


def test_nexus_command_preserves_apikey(runner, temp_config_dir):
    """Test that configure with nexus options preserves existing API key."""
    tmp_path, config_file = temp_config_dir
    config_file.write_text('apikey = "existing-key-123"\n')

    result = runner.invoke(
        tidy3d_cli,
        [
            "configure",
            "--api-endpoint",
            "http://api:5000",
            "--website-endpoint",
            "http://web/tidy3d",
        ],
    )

    assert result.exit_code == 0

    config = toml.loads(config_file.read_text())
    assert config["apikey"] == "existing-key-123"
    assert config["web_api_endpoint"] == "http://api:5000"


def test_nexus_command_help(runner):
    """Test configure command help shows nexus options."""
    result = runner.invoke(tidy3d_cli, ["configure", "--help"])

    assert result.exit_code == 0
    assert "--api-endpoint" in result.output
    assert "--website-endpoint" in result.output
    assert "--s3-region" in result.output


# configure_nexus_fn Tests


def test_configure_nexus_creates_config(temp_config_dir):
    """Test that configure_fn creates config file with nexus settings."""
    tmp_path, config_file = temp_config_dir

    configure_fn(
        apikey=None,
        api_endpoint="http://test:5000",
        website_endpoint="http://test/web",
        s3_region="us-west-2",
        s3_endpoint="http://s3:9000",
        ssl_verify=True,
        enable_caching=False,
    )

    assert config_file.exists()
    config = toml.loads(config_file.read_text())
    assert config["web_api_endpoint"] == "http://test:5000"
    assert config["s3_region"] == "us-west-2"


def test_configure_nexus_updates_existing(temp_config_dir):
    """Test that configure_fn updates existing config."""
    tmp_path, config_file = temp_config_dir
    config_file.write_text('apikey = "key123"\nold_field = "old"\n')

    configure_fn(
        apikey=None,
        api_endpoint="http://new:5000",
        website_endpoint="http://new/web",
        s3_region="us-east-1",
        s3_endpoint="http://127.0.0.1:9000",
        ssl_verify=False,
        enable_caching=False,
    )

    config = toml.loads(config_file.read_text())
    assert config["apikey"] == "key123"
    assert config["web_api_endpoint"] == "http://new:5000"


# Environment Loading Tests


def test_load_custom_env_from_config(temp_config_dir):
    """Test _load_custom_env_from_config method."""
    tmp_path, config_file = temp_config_dir

    config_content = """
web_api_endpoint = "http://customer:5000"
website_endpoint = "http://customer/tidy3d"
s3_region = "eu-west-1"
s3_endpoint = "http://customer-s3:9000"
ssl_verify = true
enable_caching = true
"""
    config_file.write_text(config_content)

    with patch.dict(os.environ, {"TIDY3D_BASE_DIR": str(tmp_path)}):
        env = Environment()
        custom_env = env._load_custom_env_from_config()

    assert custom_env is not None
    assert custom_env.name == "nexus_custom"
    assert custom_env.web_api_endpoint == "http://customer:5000"
    assert custom_env.s3_region == "eu-west-1"
    assert custom_env.env_vars["AWS_ENDPOINT_URL_S3"] == "http://customer-s3:9000"


def test_load_custom_env_no_config(temp_config_dir):
    """Test _load_custom_env_from_config returns None when no config."""
    tmp_path, config_file = temp_config_dir

    if config_file.exists():
        config_file.unlink()

    with patch.dict(os.environ, {"TIDY3D_BASE_DIR": str(tmp_path)}):
        env = Environment()
        custom_env = env._load_custom_env_from_config()

    assert custom_env is None


def test_load_custom_env_missing_endpoints(temp_config_dir):
    """Test _load_custom_env_from_config returns None if endpoints missing."""
    tmp_path, config_file = temp_config_dir
    config_file.write_text('web_api_endpoint = "http://test:5000"\n')

    with patch.dict(os.environ, {"TIDY3D_BASE_DIR": str(tmp_path)}):
        env = Environment()
        custom_env = env._load_custom_env_from_config()

    assert custom_env is None


def test_environment_uses_custom_config(temp_config_dir):
    """Test Environment initialization uses custom config."""
    tmp_path, config_file = temp_config_dir

    config_content = """
web_api_endpoint = "http://custom:5000"
website_endpoint = "http://custom/web"
"""
    config_file.write_text(config_content)

    with patch.dict(os.environ, {"TIDY3D_BASE_DIR": str(tmp_path)}, clear=False):
        if "TIDY3D_ENV" in os.environ:
            del os.environ["TIDY3D_ENV"]
        env = Environment()

    assert env.current.name == "nexus_custom"
    assert env.current.web_api_endpoint == "http://custom:5000"


def test_tidy3d_env_overrides_custom_config(temp_config_dir):
    """Test that TIDY3D_ENV environment variable overrides custom config."""
    tmp_path, config_file = temp_config_dir

    config_content = """
web_api_endpoint = "http://custom:5000"
website_endpoint = "http://custom/web"
"""
    config_file.write_text(config_content)

    with patch.dict(os.environ, {"TIDY3D_BASE_DIR": str(tmp_path), "TIDY3D_ENV": "prod"}):
        env = Environment()

    # Should use prod, not custom config
    assert env.current.name == "prod"
    assert "https://tidy3d-api.simulation.cloud" in env.current.web_api_endpoint


def test_environment_defaults_to_prod(temp_config_dir):
    """Test Environment defaults to prod when no config or env var."""
    tmp_path, config_file = temp_config_dir

    if config_file.exists():
        config_file.unlink()

    with patch.dict(os.environ, {"TIDY3D_BASE_DIR": str(tmp_path)}, clear=False):
        if "TIDY3D_ENV" in os.environ:
            del os.environ["TIDY3D_ENV"]
        env = Environment()

    assert env.current.name == "prod"


# Backward Compatibility Tests


def test_configure_command_unchanged(runner):
    """Test that configure command help works."""
    result = runner.invoke(tidy3d_cli, ["configure", "--help"])

    assert result.exit_code == 0
    assert "--apikey" in result.output


def test_existing_config_without_nexus_fields(temp_config_dir):
    """Test that existing config files without nexus fields work normally."""
    tmp_path, config_file = temp_config_dir
    config_file.write_text('apikey = "test-key"\n')

    with patch.dict(os.environ, {"TIDY3D_BASE_DIR": str(tmp_path)}):
        env = Environment()

    # Should default to prod since no custom endpoints
    assert env.current.name == "prod"
