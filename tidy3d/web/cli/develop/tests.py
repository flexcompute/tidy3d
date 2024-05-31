"""
This module contains the CLI commands for testing the tidy3d package. This includes testing the base package and the
notebooks in order to achieve reproducibility between hardwares.
"""

import click

from .index import develop
from .install import install_in_poetry
from .utils import echo_and_run_subprocess

__all__ = [
    "test_options",
    "test_in_environment_command",
]


def test_options(options: list):
    """
    Inclusive rather than exclusive tests in a given set of environments.

    Parameters
    ----------
    options : list
        A list of options for which tests to run. Options are 'base' and 'notebooks'.
    """
    if "base" in options:
        echo_and_run_subprocess(["poetry", "run", "pytest", "-rA", "tests"])
    if "notebooks" in options:
        echo_and_run_subprocess(["poetry", "run", "pytest", "-rA", "tests/full_test_notebooks.py"])


@click.option(
    "--types",
    default=["base"],
    help="Types of tests to run. Defaults to 'base'. Other options",
    type=list,
)
@click.option(
    "--env",
    default="dev",
    help="Poetry environment to install. Defaults to 'dev'.",
    type=str,
)
@develop.command(
    name="test-in-envrionment", help="Installs the specified poetry environment and tests"
)
def test_in_environment_command(types: list, env: str = "dev"):
    """
    Installs a poetry environment specified by the extra definition in pyproject.toml and runs tests with pytest and
    any additional arguments. Requires a poetry installation so make sure to verify the installation previously.

    If the environment is already installed, it will be reinstalled to ensure the latest version of a reproducible
    envrionment is used.

    Parameters
    ----------
    types : list
        A list of options for which tests to run.
    env : str
        The name of the poetry environment to install. Defaults to 'dev'. See pyproject.toml
    """
    install_in_poetry(env)
    test_options(types)
