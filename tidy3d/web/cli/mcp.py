"""MCP commands for the Tidy3D CLI."""

from __future__ import annotations

import click

__all__ = ["mcp_command"]


@click.command(name="mcp")
@click.option("--viewer-bridge", default=None, help="Existing viewer bridge URL.")
def mcp_command(
    viewer_bridge: str | None,
) -> None:
    """Launch the Tidy3D MCP server."""

    from tidy3d.web.mcp.server import run_mcp_server

    try:
        run_mcp_server(viewer_bridge=viewer_bridge)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
