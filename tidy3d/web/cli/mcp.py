"""MCP commands for the Tidy3D CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["mcp_command"]


class _McpArgumentForwardingCommand(click.Command):
    """Preserve every token after ``tidy3d mcp`` for the MCP runtime."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        ctx.params["mcp_args"] = args
        return []


def _load_tidy3d_mcp_main() -> Callable[[list[str] | None], None]:
    try:
        from tidy3d_mcp.server import main as tidy3d_mcp_main
    except ImportError as exc:
        raise click.ClickException(
            "The Tidy3D MCP runtime could not be imported. "
            f"Install or reinstall 'tidy3d-mcp'. Original error: {exc}"
        ) from exc

    return tidy3d_mcp_main


def run_mcp(mcp_args: list[str]) -> None:
    """Delegate directly to the independent runtime."""

    tidy3d_mcp_main = _load_tidy3d_mcp_main()
    tidy3d_mcp_main(mcp_args)


def run_mcp_from_console_script(mcp_args: list[str]) -> None:
    """Delegate from the console script with facade-only error rendering."""

    try:
        tidy3d_mcp_main = _load_tidy3d_mcp_main()
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from exc

    tidy3d_mcp_main(mcp_args)


@click.command(name="mcp", cls=_McpArgumentForwardingCommand, add_help_option=False)
def mcp_command(mcp_args: list[str]) -> None:
    """Launch the independently released Tidy3D MCP runtime."""

    run_mcp(mcp_args)
