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


def _environment_report() -> diagnostics.EnvironmentDiagnosticReport:
    """Return a small deterministic environment report."""

    return diagnostics.EnvironmentDiagnosticReport(
        generated_at="2026-06-18T00:00:00+00:00",
        tidy3d_version="test-version",
        python_version="3.13.0",
        python_full_version="3.13.0 (test)",
        python_executable="/opt/test/python",
        platform="test-platform",
        machine="x86_64",
        processor="test-cpu",
        in_virtualenv=True,
        in_notebook=False,
        flexcompute_packages=(
            diagnostics.EnvironmentPackage(name="tidy3d", version="test-version"),
        ),
        installed_packages=(diagnostics.EnvironmentPackage(name="numpy", version="2.0.0"),),
    )


def test_tidy3d_root_command_names_are_unique():
    runner = CliRunner()
    result = runner.invoke(tidy3d_cli, ["--help"])
    assert result.exit_code == 0, result.output

    command_names = list(tidy3d_cli.commands.keys())
    assert len(command_names) == len(set(command_names))
    assert "config" in tidy3d_cli.commands
    assert {
        "configure",
        "convert",
        "develop",
        "troubleshoot",
        "mcp",
    }.issubset(set(command_names))
    assert "migrate" not in tidy3d_cli.commands
    # The old top-level `diagnose-connection` command was removed in favor of
    # `tidy3d troubleshoot connection`.
    assert "diagnose-connection" not in tidy3d_cli.commands


def test_troubleshoot_group_has_expected_subcommands():
    group = tidy3d_cli.commands["troubleshoot"]
    assert set(group.commands.keys()) == {"connection", "environment", "report"}


def test_config_group_commands_are_namespaced():
    config_group = tidy3d_cli.commands["config"]

    reset_cmd = config_group.commands["reset"]

    assert reset_cmd.name == "config-reset"
    assert "config-reset" not in tidy3d_cli.commands


def test_troubleshoot_connection_subcommand_prints_support_text(monkeypatch):
    calls = {}

    def fake_diagnose_connection(*, api_samples, timeout, verbose, include_private_network_details):
        calls.update(
            api_samples=api_samples,
            timeout=timeout,
            verbose=verbose,
            include_private_network_details=include_private_network_details,
        )
        return _diagnostic_report()

    monkeypatch.setattr(
        "tidy3d.web.cli.diagnostics.diagnose_connection",
        fake_diagnose_connection,
    )

    result = CliRunner().invoke(
        tidy3d_cli,
        ["troubleshoot", "connection", "--api-samples", "2", "--timeout", "3.5"],
    )

    assert result.exit_code == 0, result.output
    assert calls == {
        "api_samples": 2,
        "timeout": 3.5,
        "verbose": False,
        "include_private_network_details": False,
    }
    assert "Tidy3D connection diagnostics" in result.output
    assert "api_latency: pass" in result.output


def test_troubleshoot_connection_subcommand_can_include_private_network_details(monkeypatch):
    calls = {}

    def fake_diagnose_connection(*, api_samples, timeout, verbose, include_private_network_details):
        calls.update(include_private_network_details=include_private_network_details)
        return _diagnostic_report()

    monkeypatch.setattr(
        "tidy3d.web.cli.diagnostics.diagnose_connection",
        fake_diagnose_connection,
    )

    result = CliRunner().invoke(
        tidy3d_cli,
        ["troubleshoot", "connection", "--private-network-details"],
    )

    assert result.exit_code == 0, result.output
    assert calls == {"include_private_network_details": True}


def test_troubleshoot_connection_subcommand_can_print_json(monkeypatch):
    monkeypatch.setattr(
        "tidy3d.web.cli.diagnostics.diagnose_connection",
        lambda **_: _diagnostic_report(),
    )

    result = CliRunner().invoke(tidy3d_cli, ["troubleshoot", "connection", "--json"])

    assert result.exit_code == 0, result.output
    # ``result.stdout`` is stdout only; ``result.output`` merges the Rich stderr spinner.
    payload = json.loads(result.stdout)
    assert payload["tidy3d_version"] == "test-version"
    assert payload["results"][0]["name"] == "api_latency"


def test_troubleshoot_environment_command_prints_support_text(monkeypatch):
    monkeypatch.setattr(
        "tidy3d.web.cli.diagnostics.diagnose_environment",
        lambda **_: _environment_report(),
    )

    result = CliRunner().invoke(tidy3d_cli, ["troubleshoot", "environment"])

    assert result.exit_code == 0, result.output
    assert "Tidy3D environment" in result.output
    assert "tidy3d_version: test-version" in result.output
    assert "Flexcompute packages:" in result.output
    assert "Installed packages (pip freeze):" in result.output
    assert "numpy==2.0.0" in result.output


def test_troubleshoot_environment_command_json(monkeypatch):
    monkeypatch.setattr(
        "tidy3d.web.cli.diagnostics.diagnose_environment",
        lambda **_: _environment_report(),
    )

    result = CliRunner().invoke(tidy3d_cli, ["troubleshoot", "environment", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["tidy3d_version"] == "test-version"
    assert payload["machine"] == "x86_64"


def _patch_report_probes(monkeypatch):
    """Stub the CLI-level probes so ``troubleshoot report`` tests never hit the network."""

    monkeypatch.setattr(
        "tidy3d.web.cli.diagnostics.diagnose_environment",
        lambda **_: _environment_report(),
    )
    monkeypatch.setattr(
        "tidy3d.web.cli.diagnostics.diagnose_connection",
        lambda **_: _diagnostic_report(),
    )


def test_troubleshoot_report_command_non_interactive(monkeypatch):
    _patch_report_probes(monkeypatch)

    result = CliRunner().invoke(
        tidy3d_cli,
        ["troubleshoot", "report", "--task-id", "abc-123", "--non-interactive"],
    )

    assert result.exit_code == 0, result.output
    stdout = result.stdout
    assert "Tidy3D support report" in stdout
    # The seven-question template is always rendered; task_id lands in question 5.
    assert "Tidy3D Issue Report" in stdout
    assert "5. Task ID or task link" in stdout
    assert "   abc-123" in stdout
    assert "Tidy3D environment" in stdout
    # --no-connection was not passed, so the connection block appears.
    assert "Tidy3D connection diagnostics" in stdout


def test_troubleshoot_report_command_prompts_go_to_stderr(monkeypatch):
    """Interactive prompts must go to stderr so ``troubleshoot report > file`` still shows them."""

    _patch_report_probes(monkeypatch)
    # Provide answers to the five prompts in SUPPORT_REPORT_PROMPTS order; blank Q2 skipped.
    answers = "boom\n\nevery time\nPython API\ngist://abc\n"
    result = CliRunner().invoke(
        tidy3d_cli,
        ["troubleshoot", "report", "--no-connection"],
        input=answers,
    )

    assert result.exit_code == 0, result.output
    # Prompt text and the intro banner must be on stderr, not stdout, so `troubleshoot
    # report > file.txt` still shows the questions on the terminal.
    assert "Answer the following" in result.stderr
    assert "Brief description of the issue:" in result.stderr
    assert "Answer the following" not in result.stdout
    # In a real terminal, user keystrokes echo via the TTY; Click's CliRunner does not
    # replicate that path, so we only verify that the questions themselves stay on stderr.
    stdout = result.stdout
    assert "1. Brief description of the issue:" in stdout
    assert "   boom" in stdout
    assert "3. Is it reproducible?" in stdout
    assert "   every time" in stdout
    assert "   tidy3d test-version - Python API" in stdout
    # Q6 (steps_to_reproduce) is only fillable via --traceback-file, so the interactive
    # path leaves it unfilled.
    assert "6. Steps to reproduce" in stdout
    assert "7. Could you share a relevant script or model?" in stdout
    assert "   gist://abc" in stdout


def test_troubleshoot_report_command_traceback_file(monkeypatch, tmp_path):
    """--traceback-file populates the Steps-to-reproduce answer with the file contents."""

    _patch_report_probes(monkeypatch)
    traceback_path = tmp_path / "tb.txt"
    traceback_path.write_text(
        "Traceback (most recent call last):\n  File 'x.py', line 3\n    boom\n"
    )

    result = CliRunner().invoke(
        tidy3d_cli,
        [
            "troubleshoot",
            "report",
            "--non-interactive",
            "--no-connection",
            "--traceback-file",
            str(traceback_path),
        ],
    )

    assert result.exit_code == 0, result.output
    stdout = result.stdout
    assert "6. Steps to reproduce" in stdout
    # Multi-line traceback is rendered under Q6 with the template's 3-space indent.
    assert "   Traceback (most recent call last):" in stdout
    assert "     File 'x.py', line 3" in stdout


def test_troubleshoot_report_command_json(monkeypatch):
    _patch_report_probes(monkeypatch)

    result = CliRunner().invoke(
        tidy3d_cli,
        ["troubleshoot", "report", "--non-interactive", "--no-connection", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["environment"]["tidy3d_version"] == "test-version"


def test_troubleshoot_report_command_writes_to_path(monkeypatch, tmp_path):
    """--output <path> writes the report to a file instead of stdout."""

    _patch_report_probes(monkeypatch)
    out = tmp_path / "report.md"
    result = CliRunner().invoke(
        tidy3d_cli,
        [
            "troubleshoot",
            "report",
            "--non-interactive",
            "--no-connection",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    # stdout must be empty when --output is used; the file has the report.
    assert result.stdout.strip() == ""
    written = out.read_text()
    assert "Tidy3D support report" in written
    assert "Tidy3D environment" in written
    # -o alias must accept the same argument.
    out2 = tmp_path / "report2.md"
    result2 = CliRunner().invoke(
        tidy3d_cli,
        ["troubleshoot", "report", "--non-interactive", "--no-connection", "-o", str(out2)],
    )
    assert result2.exit_code == 0, result2.output
    assert out2.read_text().startswith("Tidy3D support report")


def test_troubleshoot_report_command_output_write_is_atomic(monkeypatch, tmp_path):
    """A crash mid-write must leave no partial file at the target path (atomic replace)."""

    _patch_report_probes(monkeypatch)
    target = tmp_path / "report.md"
    # Pre-populate the target with known content; the atomic replace must swap it whole.
    target.write_text("stale content")

    result = CliRunner().invoke(
        tidy3d_cli,
        [
            "troubleshoot",
            "report",
            "--non-interactive",
            "--no-connection",
            "--output",
            str(target),
        ],
    )

    assert result.exit_code == 0, result.output
    body = target.read_text()
    assert "stale content" not in body
    assert body.startswith("Tidy3D support report")
    # No leftover temp files in the target directory.
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_troubleshoot_report_command_traceback_file_size_capped(monkeypatch, tmp_path):
    """A giant traceback file must be truncated in-place with a marker, not read whole."""

    from tidy3d.web.cli import diagnostics as cli_diagnostics

    _patch_report_probes(monkeypatch)
    monkeypatch.setattr(cli_diagnostics, "_MAX_TRACEBACK_BYTES", 64)

    traceback_path = tmp_path / "tb.txt"
    traceback_path.write_text("x" * 200)

    result = CliRunner().invoke(
        tidy3d_cli,
        [
            "troubleshoot",
            "report",
            "--non-interactive",
            "--no-connection",
            "--traceback-file",
            str(traceback_path),
        ],
    )
    assert result.exit_code == 0, result.output
    stdout = result.stdout
    assert "6. Steps to reproduce" in stdout
    assert "[truncated: file is 200 bytes, kept first 64]" in stdout


def test_troubleshoot_report_command_traceback_file_non_utf8(monkeypatch, tmp_path):
    """A non-UTF-8 traceback (Windows console dumps in cp1252) must not crash the CLI."""

    _patch_report_probes(monkeypatch)
    traceback_path = tmp_path / "tb.txt"
    # 0xff is not valid UTF-8; open('rb') + decode(errors='replace') keeps the CLI up.
    traceback_path.write_bytes(b"non-utf8 byte: \xff and text.")

    result = CliRunner().invoke(
        tidy3d_cli,
        [
            "troubleshoot",
            "report",
            "--non-interactive",
            "--no-connection",
            "--traceback-file",
            str(traceback_path),
        ],
    )

    assert result.exit_code == 0, result.output
    stdout = result.stdout
    assert "6. Steps to reproduce" in stdout
    # The invalid byte is replaced with the Unicode replacement char; the rest survives.
    assert "non-utf8 byte:" in stdout
    assert "and text." in stdout


def test_troubleshoot_report_command_private_details_propagate_to_both_probes(monkeypatch):
    """--private-network-details must forward include_private_network_details=True to
    both diagnose_environment and diagnose_connection, not just one of them."""

    calls: dict[str, bool | None] = {"env": None, "conn": None}

    def fake_env(**kwargs):
        calls["env"] = kwargs.get("include_private_network_details")
        return _environment_report()

    def fake_conn(**kwargs):
        calls["conn"] = kwargs.get("include_private_network_details")
        return _diagnostic_report()

    monkeypatch.setattr("tidy3d.web.cli.diagnostics.diagnose_environment", fake_env)
    monkeypatch.setattr("tidy3d.web.cli.diagnostics.diagnose_connection", fake_conn)

    result = CliRunner().invoke(
        tidy3d_cli,
        [
            "troubleshoot",
            "report",
            "--non-interactive",
            "--private-network-details",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == {"env": True, "conn": True}


def test_troubleshoot_report_command_verbose_lists_probes(monkeypatch):
    """--verbose prints the individual probes being run to stderr before each phase."""

    _patch_report_probes(monkeypatch)
    result = CliRunner().invoke(
        tidy3d_cli,
        ["troubleshoot", "report", "--non-interactive", "--verbose"],
    )
    assert result.exit_code == 0, result.output
    stderr = result.stderr
    assert "importlib.metadata scan" in stderr
    assert "DNS resolve" in stderr
    assert "GET /health" in stderr


def test_troubleshoot_report_command_private_marks_bundle(monkeypatch):
    """--private-network-details must mark the bundle as private with an inline warning."""

    _patch_report_probes(monkeypatch)
    result = CliRunner().invoke(
        tidy3d_cli,
        [
            "troubleshoot",
            "report",
            "--non-interactive",
            "--no-connection",
            "--private-network-details",
        ],
    )

    assert result.exit_code == 0, result.output
    stdout = result.stdout
    assert "privacy_mode: private" in stdout
    assert diagnostics.PRIVATE_DETAILS_WARNING in stdout
