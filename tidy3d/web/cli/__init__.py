"""
tidy3d command line tool.
"""

from __future__ import annotations

import sys

from .app import tidy3d_cli
from .mcp import run_mcp_from_console_script


def main() -> None:
    """Run the Tidy3D CLI, bypassing Click for transparent MCP delegation."""

    if sys.argv[1:2] == ["mcp"]:
        run_mcp_from_console_script(sys.argv[2:])
        return

    tidy3d_cli()


__all__ = ["main", "tidy3d_cli"]
