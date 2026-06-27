"""Connection diagnostic CLI commands."""

from __future__ import annotations

import click

from tidy3d.web.diagnostics import (
    DEFAULT_API_SAMPLES,
    DEFAULT_TIMEOUT,
    diagnose_connection,
)


@click.command(name="diagnose-connection")
@click.option(
    "--api-samples",
    default=DEFAULT_API_SAMPLES,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of API latency samples to collect.",
)
@click.option(
    "--timeout",
    default=DEFAULT_TIMEOUT,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
    help="Per-request timeout in seconds.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Print structured JSON instead of the support text summary.",
)
def diagnose_connection_command(
    api_samples: int,
    timeout: float,
    json_output: bool,
) -> None:
    """Run connection diagnostics for support debugging."""

    report = diagnose_connection(
        api_samples=api_samples,
        timeout=timeout,
        verbose=False,
    )
    output = report.model_dump_json(indent=2) if json_output else report.support_text()
    click.echo(output)
