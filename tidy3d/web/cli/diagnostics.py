"""Troubleshoot CLI commands: `tidy3d troubleshoot {connection,environment,report}`."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click
from rich.console import Console

from tidy3d.web.diagnostics import (
    DEFAULT_API_SAMPLES,
    DEFAULT_DOWNLOAD_BYTES,
    DEFAULT_TIMEOUT,
    SUPPORT_REPORT_PROMPTS,
    compose_support_report,
    diagnose_connection,
    diagnose_environment,
)

# Cap the inlined traceback content so an accidentally-piped log file cannot bloat the
# report. Anything above this is truncated with an explicit marker.
_MAX_TRACEBACK_BYTES = 1 * 1024 * 1024

# Bound TypeVar so every option-decorator helper preserves the Click command's type.
_F = TypeVar("_F", bound=Callable[..., Any])


def _status_console() -> Console:
    """Return a Rich console that emits progress to stderr.

    Rich degrades to plain-line output when stderr is not a TTY (piped, CI), so tests and
    ``tidy3d troubleshoot report --json > out.json`` both stay clean.
    """

    return Console(stderr=True)


def _api_samples_option(fn: _F) -> _F:
    return click.option(
        "--api-samples",
        default=DEFAULT_API_SAMPLES,
        show_default=True,
        type=click.IntRange(min=1),
        help="Number of API latency samples to collect.",
    )(fn)


def _timeout_option(fn: _F) -> _F:
    return click.option(
        "--timeout",
        default=DEFAULT_TIMEOUT,
        show_default=True,
        type=click.FloatRange(min=0, min_open=True),
        help="Per-request timeout in seconds.",
    )(fn)


def _json_option(fn: _F) -> _F:
    return click.option(
        "--json",
        "json_output",
        is_flag=True,
        help="Print structured JSON instead of the support text summary.",
    )(fn)


def _private_network_details_option(fn: _F) -> _F:
    return click.option(
        "--private-network-details",
        is_flag=True,
        help=(
            "Include private network details for the user's IT administrators. "
            "Do not share this output outside the institution."
        ),
    )(fn)


def _output_option(fn: _F) -> _F:
    return click.option(
        "--output",
        "-o",
        "output_path",
        type=click.Path(dir_okay=False, writable=True),
        default=None,
        help="Write the report to this path instead of stdout (e.g. --output report.md).",
    )(fn)


def _verbose_option(fn: _F) -> _F:
    return click.option(
        "--verbose",
        "verbose",
        is_flag=True,
        help="Print the individual probes being run to stderr before each phase.",
    )(fn)


def _emit_output(output: str, output_path: str | None, console: Console) -> None:
    """Write ``output`` either to ``output_path`` (with a stderr confirmation) or stdout.

    The file write is atomic: content is streamed to a same-directory temp file and
    renamed into place, so a full or read-only filesystem cannot leave a partial file
    at ``output_path``.
    """

    if output_path is None:
        click.echo(output)
        return
    body = output if output.endswith("\n") else output + "\n"
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = handle.name
        handle.write(body)
    os.replace(temp_path, output_path)
    console.log(f"wrote report to [cyan]{output_path}[/cyan]")


def _read_traceback_file(path: str) -> str:
    """Read ``path`` for the Q6 traceback field with size + encoding hardening."""

    file_size = os.path.getsize(path)
    with open(path, "rb") as handle:
        raw = handle.read(_MAX_TRACEBACK_BYTES)
    truncated = file_size > _MAX_TRACEBACK_BYTES
    text = raw.decode("utf-8", errors="replace").rstrip()
    if truncated:
        text = (
            f"{text}\n... [truncated: file is {file_size} bytes, kept first {_MAX_TRACEBACK_BYTES}]"
        )
    return text


@click.group(name="troubleshoot")
def troubleshoot_group() -> None:
    """Collect diagnostics and support reports for issue triage."""


@troubleshoot_group.command(name="connection")
@_api_samples_option
@_timeout_option
@_json_option
@_private_network_details_option
@_output_option
@_verbose_option
def troubleshoot_connection_subcommand(
    api_samples: int,
    timeout: float,
    json_output: bool,
    private_network_details: bool,
    output_path: str | None,
    verbose: bool,
) -> None:
    """Run connection diagnostics (DNS, TCP, TLS, latency, auth, download throughput)."""

    console = _status_console()
    if verbose:
        console.log(
            "[dim]probes: DNS resolve, TCP connect, TLS handshake, "
            f"GET /health x{api_samples}, GET /projects (auth), "
            f"{DEFAULT_DOWNLOAD_BYTES // (1024 * 1024)} MiB range download[/dim]"
        )
    with console.status(
        f"Testing connection to Tidy3D API ({api_samples} latency samples, "
        f"{DEFAULT_DOWNLOAD_BYTES // (1024 * 1024)} MiB download probe)...",
        spinner="dots",
    ):
        report = diagnose_connection(
            api_samples=api_samples,
            timeout=timeout,
            verbose=False,
            include_private_network_details=private_network_details,
        )
    console.log("[green]connection diagnostics complete[/green]")
    output = report.model_dump_json(indent=2) if json_output else report.support_text()
    _emit_output(output, output_path, console)


@troubleshoot_group.command(name="environment")
@_json_option
@_private_network_details_option
@_output_option
@_verbose_option
def troubleshoot_environment_subcommand(
    json_output: bool,
    private_network_details: bool,
    output_path: str | None,
    verbose: bool,
) -> None:
    """Collect local environment info: Python, platform, key packages, config."""

    console = _status_console()
    if verbose:
        console.log(
            "[dim]probes: importlib.metadata scan, configured endpoint + redacted env vars[/dim]"
        )
    with console.status(
        "Probing local environment (Python, platform, packages, config)...",
        spinner="dots",
    ):
        report = diagnose_environment(
            verbose=False,
            include_private_network_details=private_network_details,
        )
    console.log("[green]environment snapshot complete[/green]")
    output = report.model_dump_json(indent=2) if json_output else report.support_text()
    _emit_output(output, output_path, console)


@troubleshoot_group.command(name="report")
@click.option(
    "--task-id",
    "task_id",
    type=str,
    default=None,
    help="Task ID the user is reporting on (optional).",
)
@click.option(
    "--no-connection",
    "no_connection",
    is_flag=True,
    help="Skip connection diagnostics (offline / no network).",
)
@click.option(
    "--traceback-file",
    "traceback_file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=None,
    help="Path to a file whose contents fill the 'steps to reproduce / traceback' field.",
)
@_api_samples_option
@_timeout_option
@_private_network_details_option
@_json_option
@_output_option
@_verbose_option
@click.option(
    "--non-interactive",
    is_flag=True,
    help="Do not prompt for narrative fields; emit report from probes only.",
)
def troubleshoot_report_subcommand(
    task_id: str | None,
    no_connection: bool,
    traceback_file: str | None,
    api_samples: int,
    timeout: float,
    private_network_details: bool,
    json_output: bool,
    output_path: str | None,
    verbose: bool,
    non_interactive: bool,
) -> None:
    """Emit a copy-paste support report mapped to the Tidy3D issue-report template."""

    narrative: dict[str, str] = {}
    if not non_interactive:
        click.echo(
            "Answer the following (press Enter to skip). "
            "Answers are only included in the printed output.",
            err=True,
        )
        for key, prompt in SUPPORT_REPORT_PROMPTS:
            # Prompt to stderr so that `troubleshoot report > file.txt` still shows the
            # questions to the user and doesn't smuggle prompt lines into the report body.
            answer = click.prompt(prompt, default="", show_default=False, err=True)
            if answer:
                narrative[key] = answer
        click.echo("", err=True)  # visual separator between the Q&A block and the report

    if traceback_file is not None:
        traceback_text = _read_traceback_file(traceback_file)
        if traceback_text:
            narrative["steps_to_reproduce"] = traceback_text

    console = _status_console()

    # Phase 1: local environment probe (fast, offline).
    if verbose:
        console.log(
            "[dim]probes: importlib.metadata scan, configured endpoint + redacted env vars[/dim]"
        )
    with console.status(
        "Probing local environment (Python, platform, packages, config)...",
        spinner="dots",
    ):
        environment = diagnose_environment(
            verbose=False,
            include_private_network_details=private_network_details,
        )
    console.log("[green]environment snapshot complete[/green]")

    # Phase 2: connection probe (network I/O; can take several seconds).
    connection = None
    if no_connection:
        console.log("[dim]connection diagnostics skipped (--no-connection)[/dim]")
    else:
        if verbose:
            console.log(
                "[dim]probes: DNS resolve, TCP connect, TLS handshake, "
                f"GET /health x{api_samples}, GET /projects (auth), "
                f"{DEFAULT_DOWNLOAD_BYTES // (1024 * 1024)} MiB range download[/dim]"
            )
        with console.status(
            f"Running connection diagnostics ({api_samples} latency samples, "
            f"{DEFAULT_DOWNLOAD_BYTES // (1024 * 1024)} MiB download probe)...",
            spinner="dots",
        ):
            connection = diagnose_connection(
                api_samples=api_samples,
                timeout=timeout,
                verbose=False,
                include_private_network_details=private_network_details,
            )
        console.log("[green]connection diagnostics complete[/green]")

    # Phase 3: assemble the paste-ready bundle via the shared composition helper so this
    # path and `diagnose_report()` cannot drift on privacy handling.
    with console.status("Assembling support report...", spinner="dots"):
        report = compose_support_report(
            environment=environment,
            connection=connection,
            task_id=task_id,
            narrative=narrative,
            include_private_network_details=private_network_details,
        )
    console.log("[green]report ready[/green]")

    output = report.model_dump_json(indent=2) if json_output else report.support_text()
    _emit_output(output, output_path, console)
