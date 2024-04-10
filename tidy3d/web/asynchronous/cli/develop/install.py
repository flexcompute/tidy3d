"""
This module contains the implementation of the `tidy3d develop` installation commands. These commands are used to
install and configure the development environment for tidy3d. The commands are implemented using the Click library and
are available as CLI commands when tidy3d is installed.
"""

import platform
import re
import subprocess

import click

from .index import develop
from .utils import echo_and_check_subprocess, echo_and_run_subprocess, get_install_directory

__all__ = [
    "activate_correct_poetry_python",
    "configure_submodules",
    "verify_pandoc_is_installed_and_version_less_than_3",
    "verify_pipx_is_installed",
    "verify_poetry_is_installed",
    "verify_sphinx_is_installed",
    "get_install_directory_command",
    "install_development_environment",
    "install_in_poetry",
    "uninstall_development_environment",
    "update_submodules_remote",
    "verify_development_environment",
]


def activate_correct_poetry_python():
    """
    Activate the correct Python environment for Poetry based on the operating system.
    """
    if platform.system() == "Windows":
        echo_and_run_subprocess(["poetry", "env", "use", "python"])
    elif platform.system() == "Darwin":
        echo_and_run_subprocess(["poetry", "env", "use", "python"])
    elif platform.system() == "Linux":
        try:
            echo_and_run_subprocess(["poetry", "env", "use", "python"])
        except subprocess.CalledProcessError:
            echo_and_run_subprocess(["poetry", "env", "use", "python"])
        except:  # NOQA: E722
            print("Do you have a python available in your terminal?")


def configure_submodules(args=None):
    """
    Initialize and update the notebook submodule.

    This command configures the notebook submodule by initializing it and updating it from the remote repository.

    Parameters
    ----------
    args : optional
        Additional arguments for the configuration process.
    """
    echo_and_run_subprocess(["git", "submodule", "init"])
    echo_and_run_subprocess(["git", "submodule", "update", "--remote"])
    print("Submodules updated from remote.")
    return 0


def verify_pandoc_is_installed_and_version_less_than_3():
    """
    Check if Pandoc is installed and its version is less than 3.

    Returns
    -------
    bool
        True if Pandoc is installed and its version is less than 3, False otherwise.
    """
    try:
        # Running 'pandoc --version' command
        result = echo_and_run_subprocess(
            ["pandoc", "--version"], capture_output=True, text=True, check=True
        )

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
    """
    Verify if pipx is installed on the system.

    Returns
    -------
    bool
        True if pipx is installed, False otherwise.
    """
    try:
        # Running 'pipx --version' command
        result = echo_and_run_subprocess(
            ["pipx", "--version"], capture_output=True, text=True, check=True
        )

        # If the command was successful, it means pipx is installed
        if result.returncode == 0:
            print("pipx is installed: " + result.stdout)
            return True
    except subprocess.CalledProcessError:
        # This exception is raised if the command returned a non-zero exit status
        print("pipx is not installed or not found in the system PATH.")
        return False


def verify_poetry_is_installed():
    """
    Check if Poetry is installed on the system.

    Returns
    -------
    bool
        True if Poetry is installed, False otherwise.

    Raises
    ------
    OSError
        If Poetry is not installed or not found in the system PATH.
    """
    try:
        # Running 'poetry --version' command
        result = echo_and_run_subprocess(
            ["poetry", "--version"], capture_output=True, text=True, check=True
        )
        # If the command was successful, we'll get the version info
        if result.returncode == 0:
            print("Poetry is installed: " + result.stdout)
            return True
    except subprocess.CalledProcessError:
        # This exception is raised if the command returned a non-zero exit status
        raise OSError("Poetry is not installed or not found in the system PATH.")


def verify_sphinx_is_installed():
    """
    Verify if Sphinx is installed in the poetry environment.

    Raises
    ------
    OSError
        If Sphinx is not installed or not found in the poetry environment.
    """
    try:
        # Running 'poetry --version' command
        activate_correct_poetry_python()
        result = echo_and_run_subprocess(
            ["poetry", "run", "python -m", "sphinx --version"],
        )
        # If the command was successful, we'll get the version info
        print("sphinx is installed: " + result.stdout)
    except subprocess.CalledProcessError:
        # This exception is raised if the command returned a non-zero exit status
        raise OSError("sphinx is not installed or not found in the poetry environment.")


@develop.command(name="get-install-directory", help="Gets the TIDY3D base directory.")
def get_install_directory_command():
    """
    Get the tidy3d installation directory.

    This command prints the absolute path of the installation directory of the tidy3d module.
    """
    print(get_install_directory())
    return 0


@develop.command(
    name="install-dev-environment",
    help="Installs and configures the full required development environment.",
)
def install_development_environment(args=None):
    """Install and configure the full required development environment.

    This command automates the installation of development tools like pipx, poetry, and pandoc, and sets up
    the development environment according to 'The Detailed Lane' instructions. It is dependent on the
    operating system and may require user interaction for certain steps.

    Parameters
    ----------
    args : optional
        Additional arguments for the installation process.
    """
    # Verify and install pipx if required
    try:
        verify_pipx_is_installed()
    except:  # NOQA: E722
        if platform.system() == "Windows":
            echo_and_check_subprocess(["scoop", "install", "pipx"])
            echo_and_check_subprocess(["pipx", "ensurepath"])
        elif platform.system() == "Darwin":
            echo_and_check_subprocess(["brew", "install", "pipx"])
            echo_and_check_subprocess(["pipx", "ensurepath"])
        elif platform.system() == "Linux":
            echo_and_check_subprocess(["python3", "-m", "pip", "install", "--user", "pipx"])
            echo_and_check_subprocess(["python3", "-m", "pipx", "ensurepath"])
        else:
            raise OSError(
                "Unsupported operating system installation flow. Verify the subprocess commands in "
                "tidy3d develop are compatible with your operating system."
            )

    # Verify and install poetry if required
    try:
        verify_poetry_is_installed()
    except:  # NOQA: E722
        if platform.system() == "Windows":
            echo_and_check_subprocess(["pipx", "install", "poetry"])
        elif platform.system() == "Darwin":
            echo_and_check_subprocess(["pipx", "install", "poetry"])
        elif platform.system() == "Linux":
            echo_and_check_subprocess(["python3", "-m", "pipx", "install", "poetry"])
        else:
            raise OSError(
                "Unsupported operating system installation flow. Verify the subprocess commands in "
                "tidy3d develop are compatible with your operating system."
            )

    # Verify pandoc is installed
    try:
        verify_pandoc_is_installed_and_version_less_than_3()
    except:  # NOQA: E722
        raise OSError(
            "Please install pandoc < 3 depending on your platform: https://pandoc.org/installing.html . Then run this "
            "command again. You can also follow our detailed instructions under the development guide."
        )

    # Makes sure that poetry uses the python environment active on the terminal.

    activate_correct_poetry_python()
    # Makes sure the package has installed all the development dependencies.
    echo_and_check_subprocess(["poetry", "install", "-E", "dev"])
    echo_and_check_subprocess(["poetry", "run", "pre-commit", "install"])

    # Configure notebook submodule
    try:
        configure_submodules()
    except:  # NOQA: E722
        print("Notebook submodule not configured.")

    return 0


@click.option(
    "--env",
    default="dev",
    help="Poetry environment to install. Defaults to 'dev'.",
    type=str,
)
@develop.command(
    name="install-in-poetry", help="Just installs the tidy3d development package in poetry."
)
def install_in_poetry(env: str = "dev"):
    """
    Install the tidy3d development package in the poetry environment with the specified extra option, by default 'dev'.

    This command ensures that the tidy3d package along with its development dependencies is installed in the current
    poetry environment.

    Parameters
    ----------
    env : str
        The extra option to pass to poetry for installation. Defaults to 'dev'.
    """
    # Runs the documentation build from the poetry environment
    activate_correct_poetry_python()
    echo_and_run_subprocess(["poetry", "install", "-E", env])
    return 0


@develop.command(
    name="uninstall-dev-environment", help="Uninstalls the tools installed by this CLI helper."
)
def uninstall_development_environment(args=None):
    """
    Uninstall the development environment and the tools installed by this CLI.

    This command provides a clean-up mechanism to remove development tools like poetry, pipx, and pandoc
    that were installed using this CLI. User confirmation is required before uninstallation.

    Parameters
    ----------
    args : optional
        Additional arguments for the uninstallation process.
    """
    answer = input(
        "This function will uninstall poetry, pipx and request you to uninstall pandoc. Are you sure you want to continue?"
    )
    if answer.lower() in ["y", "yes"]:
        pass
    elif answer.lower() in ["n", "no"]:
        exit("Nothing has been uninstalled.")
    else:
        exit("Nothing has been uninstalled.")

    # Verify and uninstall poetry if required
    if verify_poetry_is_installed():
        if platform.system() == "Windows":
            echo_and_run_subprocess(["pipx", "uninstall", "poetry"])
        elif platform.system() == "Darwin":
            echo_and_run_subprocess(["brew", "uninstall", "poetry"])
            echo_and_run_subprocess(["pipx", "uninstall", "poetry"])
        elif platform.system() == "Linux":
            echo_and_run_subprocess(["python3", "-m", "pipx", "uninstall", "poetry"])
        else:
            raise OSError(
                "Unsupported operating system installation flow. Verify the subprocess commands in "
                "tidy3d develop are compatible with your operating system."
            )
    else:  # NOQA: E722
        print("poetry is not found on the PATH. It is already uninstalled from PATH.")

    # Verify and install pipx if required
    if verify_pipx_is_installed():
        if platform.system() == "Windows":
            echo_and_run_subprocess(["python", "-m", "pip", "uninstall", "-y", "pipx"])
            # TODO what's the deal here?
        elif platform.system() == "Darwin":
            echo_and_run_subprocess(["brew", "uninstall", "pipx"])
            echo_and_run_subprocess(["python", "-m", "pip", "uninstall", "-y", "pipx"])
            echo_and_run_subprocess(["rm", "-rf", "~/.local/pipx"])
        elif platform.system() == "Linux":
            echo_and_run_subprocess(["python3", "-m", "pip", "uninstall", "-y", "pipx"])
            echo_and_run_subprocess(["rm", "-rf", "~/.local/pipx"])
        else:
            raise OSError(
                "Unsupported operating system installation flow. Verify the subprocess commands in "
                "tidy3d develop are compatible with your operating system."
            )
    else:
        print("pipx is not found on the PATH. It is already uninstalled from PATH.")

    # Verify pandoc is installed
    if verify_pandoc_is_installed_and_version_less_than_3():
        raise OSError(
            "Please uninstall pandoc < 3 depending on your platform: https://pandoc.org/installing.html . Then run this "
            "command again. You can also follow our detailed instructions under the development guide."
        )
    else:  # NOQA: E722
        print("pandoc is not found on the PATH. It is already uninstalled from PATH.")

    return 0


@develop.command(name="update-submodules", help="Updates notebooks and FAQ submodule from remote")
def update_submodules_remote(args=None):
    """
    Update the notebooks submodule from the remote repository.

    This command updates the notebook submodule, ensuring it is synchronized with the latest version from the remote repository.

    Parameters
    ----------
    args : optional
        Additional arguments for the update process.
    """
    # Runs the documentation build from the poetry environment
    echo_and_check_subprocess(["git", "submodule", "update", "--remote"])
    return 0


@develop.command(name="verify-dev-environment", help="Verifies the development environment.")
def verify_development_environment(args=None):
    """
    Verify that the current development environment conforms to the specified requirements.

    This command checks various development dependencies like pipx, poetry, and pandoc, and ensures
    they are properly installed and configured. It also performs a dry run of poetry installation to check
    package configurations.

    Parameters
    ----------
    args : optional
        Additional arguments for the verification process.
    """
    # Does all the docs verifications
    # Checks all the other development dependencies are properly installed
    # Verify pipx is installed
    verify_pipx_is_installed()
    # Verify poetry is installed
    verify_poetry_is_installed()
    # Verify pandoc is installed
    verify_pandoc_is_installed_and_version_less_than_3()
    # Dry run the poetry install to understand the configuration
    activate_correct_poetry_python()
    echo_and_check_subprocess(["poetry", "install", "-E", "dev", "--dry-run"])
    print(
        "'poetry install -E dev' dry run on the 'poetry.lock' complete.\nManually verify packages are properly installed."
    )
    return 0
