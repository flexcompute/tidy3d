"""Console script subcommand for tidy3d."""

import click

__all__ = [
    "develop",
]


@click.group(name="develop")
def develop():
    """
    Development related command group in the CLI.

    This command group includes several subcommands for various development tasks such as
    verifying and setting up the development environment, building documentation, testing, and more.
    """
    pass
