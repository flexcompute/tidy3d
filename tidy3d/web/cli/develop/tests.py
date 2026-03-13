"""
This module contains the CLI commands for testing the tidy3d package. This includes testing the base package and the
notebooks in order to achieve reproducibility between hardwares.
"""

from __future__ import annotations

import click

from .index import develop
from .install import install_in_uv
from .utils import echo_and_run_subprocess

__all__ = [
    "test_in_environment_command",
    "test_options",
]


def test_options(options: list) -> None:
    """
    Inclusive rather than exclusive tests in a given set of environments.

    Parameters
    ----------
    options : list
        A list of options for which tests to run. Options are 'base' and 'notebooks'.
    """
    if "base" in options:
        echo_and_run_subprocess(["uv", "run", "--frozen", "pytest", "-rA", "tests"])
    if "notebooks" in options:
        echo_and_run_subprocess(
            ["uv", "run", "--frozen", "pytest", "-rA", "tests/full_test_notebooks.py"]
        )


@click.option(
    "--types",
    default=["base"],
    help="Types of tests to run. Defaults to 'base'. Other options",
    type=list,
)
@click.option(
    "--env",
    default="dev",
    help="Dependency extra to install. Defaults to 'dev'.",
    type=str,
)
@develop.command(name="test-in-envrionment", help="Installs the specified uv environment and tests")
def test_in_environment_command(types: list, env: str = "dev") -> None:
    """
    Installs a uv environment specified by the extra definition in pyproject.toml and runs tests with pytest and
    any additional arguments. Requires uv to be installed and configured.

    If the environment is already installed, it will be reinstalled to ensure the latest version of a reproducible
    envrionment is used.

    Parameters
    ----------
    types : list
        A list of options for which tests to run.
    env : str
        The extra set to install. Defaults to 'dev'. See pyproject.toml.
    """
    install_in_uv(env)
    test_options(types)
