from __future__ import annotations

import json

from click.testing import CliRunner

from tidy3d.web import diagnostics
from tidy3d.web.cli.app import tidy3d_cli


def _diagnostic_report() -> diagnostics.ConnectionDiagnosticReport:
    """Return a small deterministic diagnostics report."""

    return diagnostics.ConnectionDiagnosticReport(
        generated_at="2026-06-18T00:00:00+00:00",
        tidy3d_version="test-version",
        python_version="3.13.0",
        platform="test-platform",
        api_endpoint_host="api.example.com",
        api_key_configured=True,
        results=(
            diagnostics.ConnectionDiagnosticResult(
                name="api_latency",
                status="pass",
                target_host="api.example.com",
                samples=(diagnostics.ConnectionDiagnosticSample(seconds=0.1),),
            ),
        ),
    )


def test_tidy3d_root_command_names_are_unique():
    runner = CliRunner()
    result = runner.invoke(tidy3d_cli, ["--help"])
    assert result.exit_code == 0, result.output

    command_names = list(tidy3d_cli.commands.keys())
    assert len(command_names) == len(set(command_names))
    assert "config" in tidy3d_cli.commands
    assert {"configure", "convert", "develop", "diagnose-connection", "mcp"}.issubset(
        set(command_names)
    )
    assert "migrate" not in tidy3d_cli.commands


def test_config_group_commands_are_namespaced():
    config_group = tidy3d_cli.commands["config"]

    reset_cmd = config_group.commands["reset"]

    assert reset_cmd.name == "config-reset"
    assert "config-reset" not in tidy3d_cli.commands


def test_diagnose_connection_command_prints_support_text(monkeypatch):
    calls = {}

    def fake_diagnose_connection(*, api_samples, timeout, verbose):
        calls.update(api_samples=api_samples, timeout=timeout, verbose=verbose)
        return _diagnostic_report()

    monkeypatch.setattr(
        "tidy3d.web.cli.diagnostics.diagnose_connection",
        fake_diagnose_connection,
    )

    result = CliRunner().invoke(
        tidy3d_cli,
        ["diagnose-connection", "--api-samples", "2", "--timeout", "3.5"],
    )

    assert result.exit_code == 0, result.output
    assert calls == {"api_samples": 2, "timeout": 3.5, "verbose": False}
    assert "Tidy3D connection diagnostics" in result.output
    assert "api_latency: pass" in result.output


def test_diagnose_connection_command_can_print_json(monkeypatch):
    monkeypatch.setattr(
        "tidy3d.web.cli.diagnostics.diagnose_connection",
        lambda **_: _diagnostic_report(),
    )

    result = CliRunner().invoke(tidy3d_cli, ["diagnose-connection", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["tidy3d_version"] == "test-version"
    assert payload["results"][0]["name"] == "api_latency"
