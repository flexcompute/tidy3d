"""
Commandline interface for tidy3d.
"""

from __future__ import annotations

import os
from typing import Any

import click

from tidy3d.web.cli.cache import cache_group
from tidy3d.web.cli.config import config_group, configure
from tidy3d.web.cli.constants import TIDY3D_DIR
from tidy3d.web.cli.diagnostics import troubleshoot_group
from tidy3d.web.cli.mcp import mcp_command

from .develop.index import develop

# Prevent race condition on threads
os.makedirs(TIDY3D_DIR, exist_ok=True)


@click.group()
def tidy3d_cli() -> None:
    """
    Tidy3d command line tool.
    """


@click.command()
@click.argument("lsf_file")
@click.argument("new_file")
def convert(lsf_file: Any, new_file: Any) -> None:
    """Click command to convert .lsf project into Tidy3D .py file"""
    raise ValueError(
        "The converter feature is deprecated. "
        "To use this feature, please use the external tool at "
        "'https://github.com/hirako22/Lumerical-to-Tidy3D-Converter'."
    )


tidy3d_cli.add_command(configure)
tidy3d_cli.add_command(convert)
tidy3d_cli.add_command(troubleshoot_group)
tidy3d_cli.add_command(develop)
tidy3d_cli.add_command(mcp_command)
tidy3d_cli.add_command(config_group, name="config")
tidy3d_cli.add_command(cache_group)
