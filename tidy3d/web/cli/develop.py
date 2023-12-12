"""Console script for tidy3d."""
import click
import subprocess
import re


def verify_pandoc_is_installed_and_version_less_than_3():
    # TODO ADD DOCSTRINGS AND DOCUMENTATION
    try:
        # Running 'pandoc --version' command
        result = subprocess.run(["pandoc", "--version"], capture_output=True, text=True, check=True)

        # Extracting the version number from the output
        version_match = re.search(r"pandoc\s+(\d+\.\d+.\d+)", result.stdout)
        if version_match:
            version = version_match.group(1)
            major_version = int(version.split(".")[0])

            if major_version < 3:
                print(f"Pandoc is installed with version {version}, which is less than 3.")
                return True
            else:
                print(f"Pandoc version {version} is installed, but it is not less than 3.")
                return False
        else:
            print("Pandoc version number could not be determined.")
            return False

    except subprocess.CalledProcessError:
        # This exception is raised if the command returned a non-zero exit status
        print("Pandoc is not installed or not found in the system PATH.")
        return False


def verify_pipx_is_installed():
    try:
        # Running 'pipx --version' command
        result = subprocess.run(["pipx", "--version"], capture_output=True, text=True, check=True)

        # If the command was successful, it means pipx is installed
        if result.returncode == 0:
            print("pipx is installed: " + result.stdout)
            return True
    except subprocess.CalledProcessError:
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
    except subprocess.CalledProcessError:
        # This exception is raised if the command returned a non-zero exit status
        raise OSError("Poetry is not installed or not found in the system PATH.")


def verify_sphinx_is_installed():
    # TODO: Not working don't know why?
    try:
        # Running 'poetry --version' command
        subprocess.run(["poetry", "env", "use", "python"])
        result = subprocess.run(
            ["poetry", "run", "python -m", "sphinx --version"],
            capture_output=True,
            text=True,
            check=True,
        )
        # If the command was successful, we'll get the version info
        if result.returncode == 0:
            print("sphinx is installed: " + result.stdout)
            return True
    except subprocess.CalledProcessError:
        # This exception is raised if the command returned a non-zero exit status
        raise OSError("sphinx is not installed or not found in the poetry environment.")


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
    # Verify pandoc is installed
    verify_pandoc_is_installed_and_version_less_than_3()
    # Dry run the poetry install to understand the configuration
    subprocess.run(["poetry", "env", "use", "python"])
    subprocess.run(["poetry", "install", "-E", "dev", "--dry-run"])
    print(
        "`poetry install -E dev` dry run on the `poetry.lock` complete.\nManually verify packages are properly installed."
    )
    return 0


def configure_notebook_submodule(args=None):
    # TODO cd to local installation environment
    subprocess.run(["git", "submodule", "init"])
    subprocess.run(["git", "submodule", "update", "--remote"])
    print("Notebook submodule updated from remote.")
    return 0


@develop.command(
    name="configure-dev-environment",
    help="Installs and configures the full required development environment.",
)
def configure_development_environment(args=None):
    """Configure development environment."""
    # Verify and install pipx if required
    try:
        verify_pipx_is_installed()
    except:  # NOQA: E722
        subprocess.run(["python", "-m", "pip", "install", "--user", "pipx"])
        subprocess.run(["python", "-m", "pipx", "ensurepath"])

    # Verify and install poetry if required
    try:
        verify_poetry_is_installed()
    except:  # NOQA: E722
        subprocess.run(["pipx", "install", "poetry"])

    # Verify pandoc is installed
    try:
        verify_pandoc_is_installed_and_version_less_than_3()
    except:  # NOQA: E722
        raise OSError(
            "Please install pandoc < 3 depending on your platform: https://pandoc.org/installing.html . Then run this command again."
        )

    # Makes sure the package has installed all the development dependencies.
    subprocess.run(["poetry", "install", "-E", "dev"])

    # # Verify the sphinx installation.
    # try:
    #     verify_sphinx_is_installed()
    # except:
    #     print("There has been an error on your poetry sphinx installation.\nManually run `poetry run python -m sphinx --version` to verify.")

    # Configure notebook submodule
    try:
        configure_notebook_submodule()
    except:  # NOQA: E722
        print("Notebook submodule not configured.")

    return 0


@develop.command(
    name="commit", help="Adds and commits the state of the repository and its submodule."
)
@click.argument("message")
@click.option("--submodule-path", default="./docs/notebooks", help="Path to the submodule.")
def commit(message, submodule_path):
    """
    Commit changes in both a Git repository and its submodule.
    TODO sort out tidy3d installation directory defined path

    Args
        commit_message: The commit message to use for both commits.
        submodule_path: The relative path to the submodule.
    """

    def commit_repository(repository_path, commit_message):
        """
        Commit changes in the specified Git repository.

        Args:
            repo_path: Path to the repository.
            message: Commit message.
        """
        subprocess.check_call(["git", "-C", repository_path, "add", "."])
        subprocess.check_call(
            ["git", "-C", repository_path, "commit", "--no-verify", "-am", commit_message]
        )

    # TODO fix errors when commiting between the two repos.
    # Commit to the submodule
    commit_repository(submodule_path, message)
    # Commit to the main repository
    commit_repository(".", message)
    return 0


@develop.command(name="build-docs", help="Builds the sphinx documentation.")
def build_documentation(args=None):
    """Verifies and builds the documentation."""
    # Runs the documentation build from the poetry environment
    # TODO cd to local path
    # TODO update generic path management.
    subprocess.run(["poetry", "run", "python", "-m", "sphinx", "docs/", "_docs/"])
    return 0


@develop.command(name="build-docs-pdf", help="Builds the sphinx documentation pdf.")
def build_documentation_pdf(args=None):
    """Verifies and builds the documentation."""
    # Runs the documentation build from the poetry environment
    # TODO cd to local path
    # TODO update generic path management.
    subprocess.run(["poetry", "run", "python", "-m", "sphinx", "-M", "latexpdf", "docs/", "_pdf/"])
    return 0
