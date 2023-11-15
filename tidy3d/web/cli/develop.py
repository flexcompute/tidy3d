"""Console script for tidy3d."""
import sys
import click


@click.group(name="develop")
def develop():
    """Development related commands."""
    pass


@develop.command(name="configure-dev-environment", help="Verifies the development environment.")
def configure_development_environment(args=None):
    """Configure development environment."""
    # Makes sure the package has installed all the development dependencies.
    click.echo("Replace this message by putting your code into " "tidy3d.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


@develop.command(name="verify-dev-environment", help="Verifies the development environment.")
def verify_development_environment(args=None):
    # Does all the docs verifications
    # Checks all the other development dependencies are properly installed
    """Verifies that the environment in which this is run conforms to the provided poetry.lock with all the development options included."""
    click.echo("Replace this message by putting your code into " "tidy3d.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


@develop.command(
    name="configure-docs-environment", help="Verifies the documentation configuration."
)
def configure_documentation_environment(args=None):
    # Sets up the notebook submodule and updates it.
    # Installs pandoc if required.
    # Runs the documentation verificaiton script
    """Verifies that the environment in which this is run conforms to the provided poetry.lock with all the development options included."""
    click.echo("Replace this message by putting your code into " "tidy3d.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


@develop.command(name="verify-docs-environment", help="Verifies the documentation configuration.")
def verify_documentation_configuration(args=None):
    # Checks pandoc version
    # Checks documentation depednencies match
    # Checks submodule is updated
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
