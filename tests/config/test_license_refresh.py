"""Tests for configure-driven local license refresh behavior."""

from __future__ import annotations

import pytest

_REFRESH_MESSAGE = (
    "License cache refreshed (1 cached file(s) removed; "
    "next local license check will fetch current entitlements)."
)
_VALIDATION_ERROR_CASES = ("endpoint_pair", "invalid_key", "validation_exception")
_VALIDATION_REASON_PATTERNS = {
    "endpoint_pair": r"Both --api-endpoint",
    "invalid_key": r"API key validation failed",
    "validation_exception": r"offline",
}
_VALIDATION_OUTPUT_MARKERS = {
    "endpoint_pair": "Both --api-endpoint and --website-endpoint must be provided together",
    "invalid_key": "API key validation failed",
    "validation_exception": "offline",
}


def _get_cli_config_module():
    import tidy3d.web.cli.config as cli_config

    return cli_config


def _mock_apikey_validation(monkeypatch, *, status_code: int = 200):
    from unittest.mock import Mock

    mock_response = Mock()
    mock_response.status_code = status_code
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: mock_response)
    return mock_response


def _write_auth_file(config_dir, name: str = "td-stale.auth"):
    auth_file = config_dir / name
    auth_file.write_text("stale", encoding="utf-8")
    return auth_file


def _fail_license_refresh(config_dir):
    raise OSError("locked")


def _validation_error_case(case, monkeypatch):
    import requests

    if case == "endpoint_pair":
        return {"apikey": None, "api_endpoint": "http://custom:5000/api"}, [
            "--api-endpoint",
            "http://custom:5000/api",
        ]
    if case == "invalid_key":
        _mock_apikey_validation(monkeypatch, status_code=401)
        return {"apikey": "invalid-key"}, ["--apikey", "invalid-key"]
    if case == "validation_exception":

        def raise_connection_error(*args, **kwargs):
            raise requests.ConnectionError("offline")

        monkeypatch.setattr("requests.get", raise_connection_error)
        return {"apikey": "new-key"}, ["--apikey", "new-key"]
    raise AssertionError(f"Unknown validation error case: {case}")


def test_configure_fn_refresh_licenses_only(cli_config_manager, monkeypatch, capsys):
    """Test configure_fn clears the license cache without prompting for an API key."""
    import tidy3d.web.cli.config as cli_config
    from tidy3d.web.cli.config import configure_fn
    from tidy3d.web.license import LICENSE_CACHE_GENERATION_FILE

    config_dir = cli_config_manager.config_dir
    auth_file = _write_auth_file(config_dir)

    configure_fn(apikey=None, refresh_licenses=True)

    captured = capsys.readouterr()
    assert _REFRESH_MESSAGE in captured.out
    assert not auth_file.exists()
    assert (config_dir / LICENSE_CACHE_GENERATION_FILE).exists()

    monkeypatch.setattr(cli_config, "refresh_license_state", _fail_license_refresh)

    with pytest.raises(RuntimeError, match="License cache refresh failed: locked"):
        configure_fn(apikey=None, refresh_licenses=True)


def test_configure_fn_restore_defaults_honors_explicit_refresh_licenses(cli_config_manager, capsys):
    """Test restore_defaults does not skip an explicitly requested license refresh."""
    from tidy3d.web.cli.config import configure_fn
    from tidy3d.web.license import LICENSE_CACHE_GENERATION_FILE

    config_dir = cli_config_manager.config_dir
    auth_file = _write_auth_file(config_dir)

    configure_fn(apikey=None, restore_defaults=True, refresh_licenses=True)

    captured = capsys.readouterr()
    assert "Successfully restored production defaults." in captured.out
    assert _REFRESH_MESSAGE in captured.out
    assert not auth_file.exists()
    assert (config_dir / LICENSE_CACHE_GENERATION_FILE).exists()


def test_configure_fn_refreshes_license_cache_after_apikey_change(
    cli_config_manager, monkeypatch, capsys
):
    """Test API key changes clear cached license auth state."""
    from tidy3d.web.cli.config import configure_fn
    from tidy3d.web.license import LICENSE_CACHE_GENERATION_FILE

    config_dir = cli_config_manager.config_dir
    _mock_apikey_validation(monkeypatch)
    auth_file = _write_auth_file(config_dir)

    configure_fn(apikey="test-api-key")

    captured = capsys.readouterr()
    assert "Configuration saved successfully." in captured.out
    assert _REFRESH_MESSAGE in captured.out
    assert not auth_file.exists()
    marker_file = config_dir / LICENSE_CACHE_GENERATION_FILE
    assert marker_file.exists()
    generation = marker_file.read_text(encoding="utf-8")

    auth_file = _write_auth_file(config_dir, "td-same-apikey.auth")

    configure_fn(apikey="test-api-key")

    captured = capsys.readouterr()
    assert "Configuration saved successfully." in captured.out
    assert "License cache refreshed" not in captured.out
    assert auth_file.read_text(encoding="utf-8") == "stale"
    assert marker_file.read_text(encoding="utf-8") == generation


def test_configure_fn_refreshes_license_cache_for_persisted_auth_change_under_env_override(
    cli_config_manager, monkeypatch, capsys
):
    """Test environment overrides do not hide persisted auth config changes."""
    from tidy3d.web.cli.config import configure_fn
    from tidy3d.web.license import LICENSE_CACHE_GENERATION_FILE

    monkeypatch.setenv("SIMCLOUD_APIKEY", "env-api-key")
    monkeypatch.setenv("TIDY3D_WEB__API_ENDPOINT", "http://env.example.com/tidy3d-api")
    cli_config_manager.switch_profile(cli_config_manager.profile)
    config_dir = cli_config_manager.config_dir
    _mock_apikey_validation(monkeypatch)

    auth_file = _write_auth_file(config_dir, "td-hidden-apikey-change.auth")

    configure_fn(apikey="persisted-api-key")

    captured = capsys.readouterr()
    assert "License cache refreshed (1 cached file(s) removed;" in captured.out
    assert not auth_file.exists()
    marker_file = config_dir / LICENSE_CACHE_GENERATION_FILE
    assert marker_file.exists()
    generation = marker_file.read_text(encoding="utf-8")

    auth_file = _write_auth_file(config_dir, "td-hidden-endpoint-change.auth")

    configure_fn(
        apikey=None,
        api_endpoint="http://custom.example.com/tidy3d-api",
        website_endpoint="http://custom.example.com/tidy3d",
    )

    captured = capsys.readouterr()
    assert "License cache refreshed (1 cached file(s) removed;" in captured.out
    assert not auth_file.exists()
    assert marker_file.read_text(encoding="utf-8") != generation


def test_configure_fn_refreshes_license_cache_for_profile_auth_context_change(
    cli_config_manager, monkeypatch, capsys
):
    """Test profile changes refresh license auth even when key and endpoint match."""
    from tidy3d.web.cli.config import configure_fn
    from tidy3d.web.license import LICENSE_CACHE_GENERATION_FILE

    api_endpoint = "http://custom.example.com/tidy3d-api"
    website_endpoint = "http://custom.example.com/tidy3d"
    config_dir = cli_config_manager.config_dir
    cli_config_manager.update_section(
        "web",
        apikey="test-api-key",
        api_endpoint=api_endpoint,
        website_endpoint=website_endpoint,
    )
    cli_config_manager.save()
    _mock_apikey_validation(monkeypatch)
    auth_file = _write_auth_file(config_dir, "td-default-profile.auth")

    configure_fn(apikey=None, api_endpoint=api_endpoint, website_endpoint=website_endpoint)

    captured = capsys.readouterr()
    assert "License cache refreshed (1 cached file(s) removed;" in captured.out
    assert not auth_file.exists()
    assert (config_dir / LICENSE_CACHE_GENERATION_FILE).exists()


@pytest.mark.parametrize("case", _VALIDATION_ERROR_CASES)
def test_configure_fn_refresh_licenses_does_not_run_on_validation_errors(
    cli_config_manager, monkeypatch, capsys, case
):
    """Test explicit refresh fails clearly when configuration updates are rejected."""
    from tidy3d.web.cli.config import configure_fn
    from tidy3d.web.license import LICENSE_CACHE_GENERATION_FILE

    config_dir = cli_config_manager.config_dir
    configure_kwargs, _ = _validation_error_case(case, monkeypatch)
    auth_file = _write_auth_file(config_dir, f"td-{case}.auth")

    with pytest.raises(
        RuntimeError,
        match=(
            "Configuration update failed: "
            f"{_VALIDATION_REASON_PATTERNS[case]}.*license cache was not refreshed"
        ),
    ):
        configure_fn(refresh_licenses=True, **configure_kwargs)

    captured = capsys.readouterr()
    if case != "validation_exception":
        assert _VALIDATION_OUTPUT_MARKERS[case] in captured.out
    assert "License cache refreshed" not in captured.out
    assert auth_file.read_text(encoding="utf-8") == "stale"
    assert not (config_dir / LICENSE_CACHE_GENERATION_FILE).exists()


@pytest.mark.parametrize(
    ("case", "expected_exit_code"),
    [("endpoint_pair", 0), ("invalid_key", 0), ("validation_exception", 1)],
)
def test_configure_cli_validation_errors_without_refresh(
    cli_config_manager, monkeypatch, case, expected_exit_code
):
    """Test no-refresh validation failures keep their existing CLI behavior."""
    from click.testing import CliRunner

    from tidy3d.web.cli.config import configure

    _ = cli_config_manager
    _, args = _validation_error_case(case, monkeypatch)

    result = CliRunner().invoke(configure, args)

    assert result.exit_code == expected_exit_code
    assert _VALIDATION_OUTPUT_MARKERS[case] in result.output
    if case == "validation_exception":
        assert "Configuration update failed" in result.output
    else:
        assert "Configuration update failed." not in result.output
    assert "license cache was refreshed" not in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("case", _VALIDATION_ERROR_CASES)
def test_configure_cli_refresh_licenses_does_not_run_on_validation_errors(
    cli_config_manager, monkeypatch, case
):
    """Test explicit CLI refresh fails clearly when configuration updates are rejected."""
    from click.testing import CliRunner

    from tidy3d.web.cli.config import configure

    config_dir = cli_config_manager.config_dir
    cli_config = _get_cli_config_module()
    _, args = _validation_error_case(case, monkeypatch)
    auth_file = _write_auth_file(config_dir, f"td-{case}.auth")
    monkeypatch.setattr(cli_config, "refresh_license_state", _fail_license_refresh)

    result = CliRunner().invoke(configure, [*args, "--refresh-licenses"])

    assert result.exit_code == 1
    assert _VALIDATION_OUTPUT_MARKERS[case] in result.output
    assert "Configuration update failed" in result.output
    assert "License cache refreshed" not in result.output
    assert "license cache refresh failed" not in result.output
    assert "license cache was not refreshed" in result.output
    assert "license cache was refreshed" not in result.output
    assert auth_file.read_text(encoding="utf-8") == "stale"
    assert "Traceback" not in result.output


def test_configure_cli_reports_automatic_refresh_failure_after_save(
    cli_config_manager, monkeypatch
):
    """Test automatic license refresh failures are reported after saving config."""
    from click.testing import CliRunner

    from tidy3d.web.cli.config import configure

    cli_config = _get_cli_config_module()
    config_dir = cli_config_manager.config_dir
    _mock_apikey_validation(monkeypatch)
    monkeypatch.setattr(cli_config, "refresh_license_state", _fail_license_refresh)

    result = CliRunner().invoke(configure, ["--apikey", "test-api-key"])

    assert result.exit_code == 1
    assert "Configuration saved successfully." in result.output
    assert "Configuration saved, but license cache refresh failed: locked" in result.output
    assert "Traceback" not in result.output
    assert "test-api-key" in (config_dir / "config.toml").read_text(encoding="utf-8")


def test_configure_fn_reports_automatic_refresh_failure_after_save(cli_config_manager, monkeypatch):
    """Test Python configure reports post-save license refresh failures with context."""
    from tidy3d.web.cli.config import configure_fn

    cli_config = _get_cli_config_module()
    config_dir = cli_config_manager.config_dir
    _mock_apikey_validation(monkeypatch)
    monkeypatch.setattr(cli_config, "refresh_license_state", _fail_license_refresh)

    with pytest.raises(
        RuntimeError, match="Configuration saved, but license cache refresh failed: locked"
    ) as exc_info:
        configure_fn(apikey="test-api-key")

    assert isinstance(exc_info.value.__cause__, OSError)
    assert "test-api-key" in (config_dir / "config.toml").read_text(encoding="utf-8")


def test_configure_fn_does_not_refresh_license_cache_for_non_auth_update(
    cli_config_manager, monkeypatch, capsys
):
    """Test non-auth web configuration updates leave cached license auth state alone."""
    from tidy3d.web.cli.config import configure_fn
    from tidy3d.web.license import LICENSE_CACHE_GENERATION_FILE

    config_dir = cli_config_manager.config_dir
    _mock_apikey_validation(monkeypatch)

    configure_fn(apikey="test-api-key", nexus_url="http://localhost")
    capsys.readouterr()

    auth_file = _write_auth_file(config_dir)
    marker = config_dir / LICENSE_CACHE_GENERATION_FILE
    generation = marker.read_text(encoding="utf-8")

    configure_fn(apikey=None, ssl_verify=True)

    captured = capsys.readouterr()
    assert "License cache refreshed" not in captured.out
    assert auth_file.read_text(encoding="utf-8") == "stale"
    assert marker.read_text(encoding="utf-8") == generation
    assert cli_config_manager.web.ssl_verify is True
