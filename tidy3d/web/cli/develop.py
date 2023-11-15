"""Console script for tidy3d."""
import click
import subprocess


def verify_pipx_is_installed():
    try:
        # Running 'pipx --version' command
        result = subprocess.run(["pipx", "--version"], capture_output=True, text=True, check=True)

        # If the command was successful, it means pipx is installed
        if result.returncode == 0:
            print("pipx is installed: " + result.stdout)
            return True
    except:
        # This exception is raised if the command returned a non-zero exit status
        print("pipx is not installed or not found in the system PATH.")
        return False


def verify_poetry_is_installed():
    try:
        # Running 'poetry --version' command
        result = subprocess.run(["poetry", "--version"], capture_output=True, text=True, check=True)
        # If the command was successful, we'll get the version info
        if result.returncode == 0:
            print("Poetry is installed: " + result.stdout)
            return True
    except:
        # This exception is raised if the command returned a non-zero exit status
        raise OSError("Poetry is not installed or not found in the system PATH.")


@click.group(name="develop")
def develop():
    """Development related commands."""
    pass


@develop.command(name="verify-dev-environment", help="Verifies the development environment.")
def verify_development_environment(args=None):
    # Does all the docs verifications
    # Checks all the other development dependencies are properly installed
    """Verifies that the environment in which this is run conforms to the provided poetry.lock with all the development options included."""
    # Verify pipx is installed
    verify_pipx_is_installed()
    # Verify poetry is installed
    verify_poetry_is_installed()
    # Dry run the poetry install to understand the configuration
    subprocess.run(["poetry", "install", "-E dev", "--dry-run"])
    print(
        "`poetry install -E dev` dry run on the `poetry.lock` complete.\nManually verify packages are properly installed."
    )
    return 0


@develop.command(name="configure-dev-environment", help="Verifies the development environment.")
def configure_development_environment(args=None):
    """Configure development environment."""
    # Verify and install pipx if required
    try:
        verify_pipx_is_installed()
    except:
        subprocess.run(["python -m pip install --user pipx"])
        subprocess.run(["python -m -m pipx ensurepath"])

    # Verify and install poetry if required
    try:
        verify_poetry_is_installed()
    except:
        subprocess.run(["pipx install poetry"])

    # Makes sure the package has installed all the development dependencies.
    subprocess.run(["poetry", "install", "-E dev"])
    return 0


@develop.command(name="build-docs", help="Verifies the documentation configuration.")
def build_documentation(args=None):
    """Verifies and builds the documentation."""
    click.echo("Replace this message by putting your code into " "tidy3d.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0
