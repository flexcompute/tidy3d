"""
This module contains the CLI commands for testing the tidy3d package. This includes testing the base package and the
notebooks in order to achieve reproducibility between hardwares.
"""

import click
from typing import Literal
from .utils import echo_and_run_subprocess
from .index import develop
from .install import install_in_poetry

__all__ = [
    "test_options",
    "test_in_environment_command",
]

section_test_options = Literal["base", "notebooks"]
option_test_types = Literal[None, "execute_all"]


def test_options(section: section_test_options, options: option_test_types = None):
    """
    Inclusive rather than exclusive tests in a given set of environments.

    Parameters
    ----------
    section : Literal["base", "notebooks"]
        The section of tests to run.

    options : Literal[None, "execute_all"]
        A list of options for which tests to run. Defaults to None.

    Raises
    ------
    ValueError
        If an invalid option is passed for the section.
    """
    if "base" == section:
        echo_and_run_subprocess(["poetry", "run", "pytest", "-rA", "tests"])
    if "notebooks" == section:
        if "execute_all" in options:
            echo_and_run_subprocess(
                ["poetry", "run", "pytest", "-rA", "tests/docs/notebooks/execute_all_notebooks.py"]
            )
        else:
            raise ValueError(
                "Invalid option for 'notebooks' section. Valid options are 'execute_all'."
            )


@develop.command(name="test", help="Installs the specified poetry environment and tests")
@click.option(
    "-s",
    "--section",
    default="base",
    help="Types of tests to run. Defaults to 'base'. Other options",
    type=str,
)
@click.option(
    "-o",
    "--options",
    default=None,
    help="Options to configure the section tests, a list of options. Defaults to None.",
    type=option_test_types,
)
@click.option(
    "--env",
    default="dev",
    help="Poetry environment to install. Defaults to 'dev'.",
    type=str,
)
def test_in_environment_command(
    section: section_test_options, options: option_test_types, env: str
):
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
    install_in_poetry(env=env)
    test_options(section=section, options=options)
