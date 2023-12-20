"""Console script for tidy3d."""
import click
import pathlib
import platform
import subprocess
import re
import tidy3d


def get_install_directory():
    return pathlib.Path(tidy3d.__file__).parent.parent.absolute()


def echo_and_run_subprocess(command: list, **kwargs):
    concatenated_command = " ".join(command)
    print("Running: " + concatenated_command)
    return subprocess.run(command, cwd=get_install_directory(), **kwargs)


def echo_and_check_subprocess(command: list, **kwargs):
    concatenated_command = " ".join(command)
    print("Running: " + concatenated_command)
    return subprocess.check_call(command, cwd=get_install_directory(), **kwargs)


def verify_pandoc_is_installed_and_version_less_than_3():
    # TODO ADD DOCSTRINGS AND DOCUMENTATION
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
    # TODO: Not working don't know why?
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


def activate_correct_poetry_python():
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


@click.group(name="develop")
def develop():
    """Development related commands."""
    pass


@develop.command(name="get-install-directory", help="Gets the TIDY3D base directory.")
def get_directory():
    """Gets the tidy3d installation directory."""
    print(get_install_directory())
    return 0


@develop.command(name="verify-dev-environment", help="Verifies the development environment.")
def verify_development_environment(args=None):
    """Verifies that the environment in which this is run conforms to the provided poetry.lock with all the
    development options included."""
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
        "`poetry install -E dev` dry run on the `poetry.lock` complete.\nManually verify packages are properly installed."
    )
    return 0


def configure_notebook_submodule(args=None):
    echo_and_run_subprocess(["git", "submodule", "init"])
    echo_and_run_subprocess(["git", "submodule", "update", "--remote"])
    print("Notebook submodule updated from remote.")
    return 0


@develop.command(
    name="install-dev-environment",
    help="Installs and configures the full required development environment.",
)
def install_development_environment(args=None):
    """Install development environment.

    Notes
    -----

        Note that this is just a automatic script implementation of the `The Detailed Lane
        <../../development/index.html#the-detailed-lane>`_ instructions. It is intended to help you and raise warnings
        with suggestions of how to fix an environment setup issue. You do not have to use this helper function and can
        just follow the instructions in `The Detailed Lane
        <../../development/index.html#the-detailed-lane>`_.

        The way this command works is dependent on the operating system you are running. There are some prerequisites for
        each platform, but the command line tool will help you identify and install the tools it requires. You should rerun
        the command after you have installed any prerequisite as it will just progress with the rest of the tools
        installation. If you already have the tool installed, it will verify that it conforms to the supported versions.

        This command will first check if you already have installed the development requirements, and if not, it will run the
        installation scripts for pipx, poetry, and ask you to install the required version of pandoc. It will also install
        the development requirements and tidy3d package in a specific poetry environment.
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

    # Configure notebook submodule
    try:
        configure_notebook_submodule()
    except:  # NOQA: E722
        print("Notebook submodule not configured.")

    return 0


@develop.command(
    name="uninstall-dev-environment", help="Uninstalls the tools installed by this CLI helper."
)
def uninstall_development_environment(args=None):
    """Uninstall development environment.

    Notes
    -----

        Provides a clean development environment, and uninstalls the tools installed by this CLI.

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

    # TODO fix errors when committing between the two repos.
    # Commit to the submodule
    commit_repository(submodule_path, message)
    # Commit to the main repository
    commit_repository(".", message)
    return 0


@develop.command(name="build-docs", help="Builds the sphinx documentation.")
def build_documentation(args=None):
    """Verifies and builds the documentation."""
    # Runs the documentation build from the poetry environment
    echo_and_check_subprocess(["poetry", "run", "python", "-m", "sphinx", "docs/", "_docs/"])
    return 0


@develop.command(name="build-docs-pdf", help="Builds the sphinx documentation pdf.")
def build_documentation_pdf(args=None):
    """Verifies and builds the documentation."""
    # Runs the documentation build from the poetry environment
    echo_and_run_subprocess(
        ["poetry", "run", "python", "-m", "sphinx", "-M", "latexpdf", "docs/", "_pdf/"]
    )
    return 0


@develop.command(
    name="build-docs-remote-notebooks", help="Updates notebooks submodule and builds documentation."
)
def build_documentation_from_remote_notebooks(args=None):
    """Updates notebooks submodule and builds documentation. Verifies and builds the documentation."""
    # Runs the documentation build from the poetry environment
    echo_and_check_subprocess(["git", "submodule", "update", "--remote"])
    print("Notebook submodule updated from remote.")
    echo_and_check_subprocess(["poetry", "run", "python", "-m", "sphinx", "docs/", "_docs/"])
    return 0


@develop.command(name="test-base", help="Tests the tidy3d base package.")
def test_base_tidy3d(args=None):
    echo_and_run_subprocess(["poetry", "run", "pytest", "-rA", "tests"])
    return 0


@develop.command(name="test-notebooks", help="Tests the tidy3d notebooks.")
def test_notebooks_tidy3d(args=None):
    """Tests the tidy3d notebooks."""
    echo_and_run_subprocess(["poetry", "run", "pytest", "-rA", "tests/full_test_notebooks.py"])
    return 0


@develop.command(
    name="install-in-poetry", help="Just installs the tidy3d development package in poetry."
)
def install_in_poetry(args=None):
    """Requires pipx and poetry installation to work."""
    # Runs the documentation build from the poetry environment
    # TODO update generic path management.
    activate_correct_poetry_python()
    echo_and_run_subprocess(["poetry", "install", "-E", "dev"])
    return 0


@develop.command(name="update-notebooks", help="Updates notebooks submodule from remote")
def update_notebooks_remote(args=None):
    """Updates notebooks submodule."""
    # Runs the documentation build from the poetry environment
    echo_and_check_subprocess(["git", "submodule", "update", "--remote"])
    return 0
