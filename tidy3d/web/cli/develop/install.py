"""
This module contains the implementation of the `tidy3d develop` installation commands. These commands are used to
install and configure the development environment for tidy3d. The commands are implemented using the Click library and
are available as CLI commands when tidy3d is installed.
"""

from __future__ import annotations

import platform
import re
import subprocess
from typing import Any

import click

from .index import develop
from .utils import echo_and_check_subprocess, echo_and_run_subprocess, get_install_directory

__all__ = [
    "activate_uv_python",
    "configure_submodules",
    "get_install_directory_command",
    "install_development_environment",
    "install_in_uv",
    "uninstall_development_environment",
    "update_submodules_remote",
    "verify_development_environment",
    "verify_pandoc_is_installed_and_version_less_than_3",
    "verify_pipx_is_installed",
    "verify_sphinx_is_installed",
    "verify_uv_is_installed",
]


def activate_uv_python() -> None:
    """Ensure uv is available from the current shell."""
    echo_and_check_subprocess(["uv", "--version"])


def configure_submodules(args: Any = None) -> None:
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


def verify_pandoc_is_installed_and_version_less_than_3() -> bool:
    """
    Check if Pandoc is installed and its version can be determined.

    Returns
    -------
    bool
        True if Pandoc is installed and its version can be determined, False otherwise.
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
            print(f"Pandoc is installed with version {version}.")
            return True
        print("Pandoc version number could not be determined.")
        return False

    except (subprocess.CalledProcessError, FileNotFoundError):
        # This exception is raised if the command returned a non-zero exit status
        print("Pandoc is not installed or not found in the system PATH.")
        return False


def verify_pipx_is_installed() -> bool | None:
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
        print("pipx is installed: " + result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # This exception is raised if the command returned a non-zero exit status
        print("pipx is not installed or not found in the system PATH.")
        return False


def verify_uv_is_installed() -> bool:
    """
    Check if uv is installed on the system.

    Returns
    -------
    bool
        True if uv is installed, raises `OSError` otherwise.

    Raises
    ------
    OSError
        If uv is not installed or not found in the system PATH.
    """
    try:
        # Running 'uv --version' command
        result = echo_and_run_subprocess(
            ["uv", "--version"], capture_output=True, text=True, check=True
        )
        # If the command was successful, we'll get the version info
        print("uv is installed: " + result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        # This exception is raised if the command returned a non-zero exit status
        raise OSError("uv is not installed or not found in the system PATH.") from exc


def verify_sphinx_is_installed() -> None:
    """
    Verify if Sphinx is installed in the uv environment.

    Raises
    ------
    OSError
        If Sphinx is not installed or not found in the uv environment.
    """
    try:
        activate_uv_python()
        result = echo_and_run_subprocess(
            ["uv", "run", "--frozen", "python", "-m", "sphinx", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        # If the command was successful, we'll get the version info
        print("sphinx is installed: " + result.stdout)
    except subprocess.CalledProcessError as exc:
        # This exception is raised if the command returned a non-zero exit status
        raise OSError("sphinx is not installed or not found in the uv environment.") from exc


@develop.command(name="get-install-directory", help="Gets the TIDY3D base directory.")
def get_install_directory_command() -> int:
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
def install_development_environment(args: Any = None) -> None:
    """Install and configure the full required development environment.

    This command automates the installation of development tools like pipx, uv, and pandoc, and sets up
    the development environment according to 'The Detailed Lane' instructions. It is dependent on the
    operating system and may require user interaction for certain steps.

    Parameters
    ----------
    args : optional
        Additional arguments for the installation process.
    """
    # Verify and install pipx if required.
    if not verify_pipx_is_installed():
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

    # Verify and install uv if required.
    try:
        verify_uv_is_installed()
    except OSError as exc:
        if platform.system() == "Windows" or platform.system() == "Darwin":
            echo_and_check_subprocess(["pipx", "install", "uv"])
        elif platform.system() == "Linux":
            echo_and_check_subprocess(["python3", "-m", "pipx", "install", "uv"])
        else:
            raise OSError(
                "Unsupported operating system installation flow. Verify the subprocess commands in "
                "tidy3d develop are compatible with your operating system."
            ) from exc

    # Verify pandoc is installed.
    if not verify_pandoc_is_installed_and_version_less_than_3():
        raise OSError(
            "Please install pandoc depending on your platform: https://pandoc.org/installing.html . Then run this "
            "command again. You can also follow our detailed instructions under the development guide."
        )

    # Install all development dependencies from the lockfile.
    activate_uv_python()
    echo_and_check_subprocess(["uv", "sync", "--frozen", "--extra", "dev"])
    echo_and_check_subprocess(["uv", "run", "--frozen", "pre-commit", "install"])

    # Configure notebook submodule
    try:
        configure_submodules()
    except:  # NOQA: E722
        print("Notebook submodule not configured.")

    return 0


@click.option(
    "--env",
    default="dev",
    help="Extra set to install. Defaults to 'dev'.",
    type=str,
)
@develop.command(
    name="install-in-uv", help="Installs tidy3d in the uv-managed development environment."
)
def install_in_uv(env: str = "dev") -> int:
    """
    Install the tidy3d development package in the uv environment with the specified extra option, by default 'dev'.

    This command ensures that the tidy3d package along with its development dependencies is installed in the current
    uv environment.

    Parameters
    ----------
    env : str
        The extra option to pass to uv for installation. Defaults to 'dev'.
    """
    activate_uv_python()
    echo_and_run_subprocess(["uv", "sync", "--frozen", "--extra", env])
    return 0


@develop.command(
    name="uninstall-dev-environment", help="Uninstalls the tools installed by this CLI helper."
)
def uninstall_development_environment(args: Any = None) -> int:
    """
    Uninstall the development environment and the tools installed by this CLI.

    This command provides a clean-up mechanism to remove development tools like uv, pipx, and pandoc
    that were installed using this CLI. User confirmation is required before uninstallation.

    Parameters
    ----------
    args : optional
        Additional arguments for the uninstallation process.
    """
    answer = input(
        "This function will uninstall uv, pipx and request you to uninstall pandoc. Are you sure you want to continue?"
    )
    if answer.lower() in ["y", "yes"]:
        pass
    elif answer.lower() in ["n", "no"]:
        exit("Nothing has been uninstalled.")
    else:
        exit("Nothing has been uninstalled.")

    # Verify and uninstall uv if required.
    if verify_uv_is_installed():
        if platform.system() == "Windows":
            echo_and_run_subprocess(["pipx", "uninstall", "uv"])
        elif platform.system() == "Darwin":
            echo_and_run_subprocess(["brew", "uninstall", "uv"])
            echo_and_run_subprocess(["pipx", "uninstall", "uv"])
        elif platform.system() == "Linux":
            echo_and_run_subprocess(["python3", "-m", "pipx", "uninstall", "uv"])
        else:
            raise OSError(
                "Unsupported operating system installation flow. Verify the subprocess commands in "
                "tidy3d develop are compatible with your operating system."
            )
    else:
        print("uv is not found on the PATH. It is already uninstalled from PATH.")

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
            "Please uninstall pandoc depending on your platform: https://pandoc.org/installing.html . Then run this "
            "command again. You can also follow our detailed instructions under the development guide."
        )
    print("pandoc is not found on the PATH. It is already uninstalled from PATH.")

    return 0


@develop.command(name="update-submodules", help="Updates notebooks and FAQ submodule from remote")
def update_submodules_remote(args: Any = None) -> int:
    """
    Update the notebooks submodule from the remote repository.

    This command updates the notebook submodule, ensuring it is synchronized with the latest version from the remote repository.

    Parameters
    ----------
    args : optional
        Additional arguments for the update process.
    """
    # Updates submodules in the current repository.
    echo_and_check_subprocess(["git", "submodule", "update", "--remote"])
    return 0


@develop.command(name="verify-dev-environment", help="Verifies the development environment.")
def verify_development_environment(args: Any = None) -> int:
    """
    Verify that the current development environment conforms to the specified requirements.

    This command checks various development dependencies like pipx, uv, and pandoc, and ensures
    they are properly installed and configured. It also performs a dry run of uv sync to check
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
    # Verify uv is installed.
    verify_uv_is_installed()
    # Verify pandoc is installed
    verify_pandoc_is_installed_and_version_less_than_3()
    # Dry run uv sync to verify lockfile compatibility.
    activate_uv_python()
    echo_and_check_subprocess(["uv", "sync", "--frozen", "--extra", "dev", "--dry-run"])
    print(
        "'uv sync --frozen --extra dev' dry run on the 'uv.lock' complete.\n"
        "Manually verify packages are properly installed."
    )
    return 0
