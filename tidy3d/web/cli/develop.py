"""Console script for tidy3d."""
import click
import json
import pathlib
import platform
import subprocess
import re
import os
import tidy3d
from typing import Optional


def get_install_directory():
    """
    Retrieve the installation directory of the tidy3d module.

    Returns
    -------
    pathlib.Path
        The absolute path of the parent directory of the tidy3d module.
    """
    return pathlib.Path(tidy3d.__file__).parent.parent.absolute()


def echo_and_run_subprocess(command: list, **kwargs):
    """
    Print and execute a subprocess command.

    Parameters
    ----------
    command : list
        A list of command line arguments to be executed.
    **kwargs : dict
        Additional keyword arguments to pass to subprocess.run.

    Returns
    -------
    subprocess.CompletedProcess
        The result of the subprocess execution.
    """
    concatenated_command = " ".join(command)
    print("Running: " + concatenated_command)
    return subprocess.run(command, cwd=get_install_directory(), **kwargs)


def echo_and_check_subprocess(command: list, **kwargs):
    """
    Print and execute a subprocess command, ensuring it completes successfully.

    Parameters
    ----------
    command : list
        A list of command line arguments to be executed.
    **kwargs : dict
        Additional keyword arguments to pass to subprocess.check_call.

    Returns
    -------
    int
        The return code of the subprocess execution.
    """
    concatenated_command = " ".join(command)
    print("Running: " + concatenated_command)
    return subprocess.check_call(command, cwd=get_install_directory(), **kwargs)


def replace_in_files(
    directory: str,
    json_file_path: str,
    selected_version: str,
    dry_run=False,
):
    """
    Recursively finds and replaces strings in files within a directory based on a given dictionary loaded from a JSON
    file. The JSON file also includes a version selector. The function will print the file line and prompt for
    confirmation before replacing each string.

    Example JSON file:

    {
      "0.18.0": {
        "tidy3d.someuniquestringa": "tidy3d.someuniquestring2",
        "tidy3d.someuniquestringb": "tidy3d.someuniquestring2",
        "tidy3d.someuniquestringc": "tidy3d.someuniquestring2"
      }
    }

    Args:
    - directory (str): The directory path to search for files.
    - json_file_path (str): The path to the JSON file containing replacement instructions.
    - selected_version (str): The version to select from the JSON file.
    - dry_run (bool): If True, the function will not modify any files, but will print the changes that would be made.
    """
    allowed_extensions = (".py", ".rst", ".md", ".txt")

    # Load data from the JSON file
    with open(json_file_path, encoding="utf-8") as json_file:
        data = json.load(json_file)
        replace_dict = data.get(str(selected_version), {})

    for root, dirs, files in os.walk(directory):
        # Exclude directories that start with a period ('.')
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()  # Get the file extension
            if not file.startswith("."):
                # Check if the file has an allowed extension
                if file_extension in allowed_extensions:
                    # Read file content and process each line
                    with open(file_path, encoding="utf-8") as f:
                        try:
                            lines = f.readlines()

                            for i, line in enumerate(lines):
                                for find_str, replace_str in replace_dict.items():
                                    if find_str in line:
                                        print(f"File: {file_path} --- Line {i + 1}")
                                        print(f"Original: {line.strip()}")
                                        confirmation = input(
                                            f"Replace '{find_str}' with '{replace_str}' in this line? (y/n): "
                                        )
                                        if confirmation.lower() == "y":
                                            lines[i] = line.replace(find_str, replace_str)
                                            if not dry_run:
                                                print(f"Modified: {lines[i].strip()}")
                                            else:
                                                print(
                                                    f"Not modified because of dry run: {line.strip()}"
                                                )

                            # Write the modified content back to the file if not in dry run mode
                            if not dry_run:
                                with open(file_path, "w", encoding="utf-8") as f:
                                    f.writelines(lines)

                        except:  # NOQA: E722
                            pass


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


@click.group(name="develop")
def develop():
    """
    Development related command group in the CLI.

    This command group includes several subcommands for various development tasks such as
    verifying and setting up the development environment, building documentation, testing, and more.
    """
    pass


@develop.command(name="get-install-directory", help="Gets the TIDY3D base directory.")
def get_directory():
    """
    Get the tidy3d installation directory.

    This command prints the absolute path of the installation directory of the tidy3d module.
    """
    print(get_install_directory())
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
        "`poetry install -E dev` dry run on the `poetry.lock` complete.\nManually verify packages are properly installed."
    )
    return 0


def configure_notebook_submodule(args=None):
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
    print("Notebook submodule updated from remote.")
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
        configure_notebook_submodule()
    except:  # NOQA: E722
        print("Notebook submodule not configured.")

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


@develop.command(
    name="commit", help="Adds and commits the state of the repository and its submodule."
)
@click.argument("message", type=str)  # Specify the type as str for the 'message' argument
@click.option(
    "--submodule-path",
    default="./docs/notebooks",
    help="Path to the submodule.",
    type=str,  # Specify the type as str for the 'submodule-path' option
)
def commit(message: str, submodule_path: str):
    """
    Add and commit changes in both the Git repository and its submodule.

    This command performs git commit operations on both the main repository and the specified submodule
    using the provided commit message.

    Parameters
    ----------
    message : str
        The commit message to use for both commits.
    submodule_path : str
        The relative path to the submodule.
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
    """
    Build the Sphinx documentation.

    This command triggers the Sphinx documentation build process in the current poetry environment.

    Parameters
    ----------
    args : optional
        Additional arguments for the documentation build process.
    """
    # Runs the documentation build from the poetry environment
    echo_and_check_subprocess(["poetry", "run", "python", "-m", "sphinx", "docs/", "_docs/"])
    return 0


@develop.command(name="build-docs-pdf", help="Builds the sphinx documentation pdf.")
def build_documentation_pdf(args=None):
    """
    Build the Sphinx documentation in PDF format.

    This command initiates the process to build the Sphinx documentation and generates a PDF output.

    Parameters
    ----------
    args : optional
        Additional arguments for the PDF documentation build process.
    """
    # Runs the documentation build from the poetry environment
    echo_and_run_subprocess(
        ["poetry", "run", "python", "-m", "sphinx", "-M", "latexpdf", "docs/", "_pdf/"]
    )
    return 0


@develop.command(
    name="build-docs-remote-notebooks", help="Updates notebooks submodule and builds documentation."
)
@click.option(
    "-nb",
    "--notebook-branch",
    default="./docs/notebooks",
    help="The remote branch from tidy3d-notebooks.",
)
def build_documentation_from_remote_notebooks(args=None):
    """
    Update the notebooks submodule and build documentation.

    This command updates the notebook submodule from the remote repository and then builds the Sphinx documentation.

    Parameters
    ----------
    args : optional
        Additional arguments for the process of updating notebooks and building documentation.
    """
    # Runs the documentation build from the poetry environment
    echo_and_check_subprocess(["git", "submodule", "update", "--remote"])

    print("Notebook submodule updated from remote.")
    echo_and_check_subprocess(["poetry", "run", "python", "-m", "sphinx", "docs/", "_docs/"])
    return 0


@develop.command(name="test-base", help="Tests the tidy3d base package.")
def test_base_tidy3d(args=None):
    """
    Test the tidy3d base package.

    This command runs tests on the tidy3d base package using pytest within the current poetry environment.

    Parameters
    ----------
    args : optional
        Additional arguments for the testing process.
    """
    echo_and_run_subprocess(["poetry", "run", "pytest", "-rA", "tests"])
    return 0


@develop.command(name="test-notebooks", help="Tests the tidy3d notebooks.")
def test_notebooks_tidy3d(args=None):
    """
    Test the tidy3d notebooks.

    This command runs tests specifically for the tidy3d notebooks using pytest in the poetry environment.

    Parameters
    ----------
    args : optional
        Additional arguments for the notebook testing process.
    """
    echo_and_run_subprocess(["poetry", "run", "pytest", "-rA", "tests/full_test_notebooks.py"])
    return 0


@develop.command(
    name="install-in-poetry", help="Just installs the tidy3d development package in poetry."
)
def install_in_poetry(args=None):
    """
    Install the tidy3d development package in the poetry environment.

    This command ensures that the tidy3d package along with its development dependencies is installed in the current poetry environment.

    Parameters
    ----------
    args : optional
        Additional arguments for the installation process.
    """
    # Runs the documentation build from the poetry environment
    activate_correct_poetry_python()
    echo_and_run_subprocess(["poetry", "install", "-E", "dev"])
    return 0


@develop.command(name="update-notebooks", help="Updates notebooks submodule from remote")
def update_notebooks_remote(args=None):
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


@develop.command(
    name="replace-in-files",
    help="Recursively find and replace strings in files based on a JSON configuration.",
)
@click.option(
    "--directory",
    "-d",
    type=click.Path(exists=True, file_okay=False, readable=True),
    default=".",
    help="Directory to process (default is current directory)",
)
@click.option(
    "--json-dictionary",
    "-j",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="JSON that contains the docstring version update files.",
)
@click.option(
    "--selected-version",
    "-v",
    type=str,
    help="Version to select from the JSON file",
)
@click.option(
    "--dry-run",
    type=bool,
    default=False,
    help="Dry run the replace in files command.",
)
def replace_in_files_command(
    directory: str, json_dictionary: Optional[str], selected_version: Optional[str], dry_run: bool
):
    """
    Recursively finds and replaces strings in files within a directory based on a given dictionary loaded from a JSON
    file. The JSON file also includes a version selector. The function will print the file line and prompt for
    confirmation before replacing each string.

    Example JSON file:

    {
      "0.18.0": {
        "tidy3d.someuniquestringa": "tidy3d.someuniquestring2",
        "tidy3d.someuniquestringb": "tidy3d.someuniquestring2",
        "tidy3d.someuniquestringc": "tidy3d.someuniquestring2"
      }
    }

    Usage:

        poetry run tidy3d develop replace-in-files -d ./ -j ./docs/versions/test_replace_in_files.json -v 0.18.0 --dry-run True
        poetry run tidy3d develop replace-in-files --directory ./ --json-dictionary ./docs/versions/test_replace_in_files.json --selected-version 0.18.0 --dry-run True

    Args:
    - directory (str): The directory path to search for files.
    - json_file_path (str): The path to the JSON file containing replacement instructions.
    - selected_version (str): The version to select from the JSON file.
    """
    if directory is None:
        directory = get_install_directory()

    replace_in_files(directory, json_dictionary, selected_version, dry_run)
    return 0


@develop.command(
    name="convert-all-markdown-to-rst",
    help="Recursively find all markdown files and convert them to rst files that can be included in the sphinx "
    "documentation",
)
@click.option(
    "--directory",
    "-d",
    type=click.Path(exists=True, file_okay=False, readable=True),
    default=".",
    help="Directory to process (default is current directory)",
)
def convert_markdown_to_rst(directory: str):
    """
    This script converts all Markdown files in a given DIRECTORY to reStructuredText format.
    """
    if directory is None:
        directory = get_install_directory()

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                md_file = os.path.join(root, file)
                rst_file = os.path.splitext(md_file)[0] + ".rst"

                # Confirmation for each file
                if not click.confirm(f"Convert {md_file} to RST format?", default=True):
                    click.echo(f"Skipped {md_file}")
                    continue

                try:
                    # Convert using Pandoc
                    echo_and_check_subprocess(["pandoc", "-s", md_file, "-o", rst_file])
                    click.echo(f"Converted {md_file} to {rst_file}")
                except subprocess.CalledProcessError as e:
                    click.echo(f"Error converting {md_file}: {e}", err=True)
