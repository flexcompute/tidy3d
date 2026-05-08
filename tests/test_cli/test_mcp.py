"""Tests for Tidy3D MCP CLI commands."""

from __future__ import annotations

from click.testing import CliRunner

from tidy3d.web.cli.app import tidy3d_cli
from tidy3d.web.mcp import server as mcp_server


def test_mcp_command_launches_server_with_cli_options(monkeypatch):
    calls = []

    def fake_run_mcp_server(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(mcp_server, "run_mcp_server", fake_run_mcp_server)

    result = CliRunner().invoke(
        tidy3d_cli,
        [
            "mcp",
            "--viewer-bridge",
            ":5123",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [{"viewer_bridge": ":5123"}]


def test_mcp_command_reports_server_startup_errors(monkeypatch):
    def fake_run_mcp_server(**_kwargs):
        raise RuntimeError("API key required")

    monkeypatch.setattr(mcp_server, "run_mcp_server", fake_run_mcp_server)

    result = CliRunner().invoke(tidy3d_cli, ["mcp"])

    assert result.exit_code != 0
    assert "API key required" in result.output
