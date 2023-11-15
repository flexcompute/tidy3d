"""Console script for tidy3d."""
import sys
import click


@click.group(name="develop")
def develop():
    """Development related tasks."""
    pass


@develop.command(name="verify-dev-environment", help="Verifies the development environment.")
def verify_development_environment(args=None):
    """Verifies that the environment in which this is run conforms to the provided poetry.lock with all the development options included."""
    click.echo("Replace this message by putting your code into " "tidy3d.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


@develop.command(name="verify-docs-configuration", help="Verifies the documentation configuration.")
def verify_documentation_configuration(args=None):
    """Verifies that the environment in which this is run conforms to the provided poetry.lock with all the development options included."""
    click.echo("Replace this message by putting your code into " "tidy3d.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


@develop.command(name="build-docs", help="Verifies the documentation configuration.")
def build_documentation(args=None):
    """Verifies and builds the documentation."""
    click.echo("Replace this message by putting your code into " "tidy3d.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0
