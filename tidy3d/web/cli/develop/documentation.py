"""
These are common operations CLI tools for the documentation build process.
These functions are used to build the documentation. They are called from the CLI using the following command:

    poetry run tidy3d develop build-docs

The functions are also used to update the notebooks submodule and build the documentation using the following command:

    poetry run tidy3d develop build-docs-remote-notebooks

The functions are also used to convert all Markdown files to RST format using the following command:

    poetry run tidy3d develop convert-all-markdown-to-rst
"""

import json
import os
from typing import Optional

import click

from .index import develop
from .utils import echo_and_check_subprocess, get_install_directory

__all__ = [
    "build_documentation",
    # "build_documentation_pdf",
    "build_documentation_from_remote_notebooks",
    "commit",
    # "convert_all_markdown_to_rst_command",
    "replace_in_files_command",
]


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
    exceptions = []

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

                        except Exception as e:  # Catch any exception
                            exceptions.append(e)

    # At the end of the file/script:
    if exceptions:
        for ex in exceptions:
            # Handle or print the exceptions as needed
            print(f"An error occurred: {ex}")


@develop.command(
    name="commit",
    help="Adds and commits the state of the repository and its notebook & faq submodule.",
)
@click.argument("message", type=str)  # Specify the type as str for the 'message' argument
@click.option(
    "--submodule-path",
    default=str(get_install_directory() / "docs" / "notebooks"),
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

        echo_and_check_subprocess(["git", "-C", repository_path, "add", "."])
        echo_and_check_subprocess(
            ["git", "-C", repository_path, "commit", "--no-verify", "-am", commit_message]
        )

    # TODO fix errors when committing between the two repos.
    # Commit to the submodule
    commit_repository(submodule_path, message)
    # Commit to the main repository
    commit_repository("..", message)
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
    echo_and_check_subprocess(
        ["poetry", "run", "python", "-m", "sphinx", "-j", "auto", "docs/", "_docs/"]
    )
    return 0


# TODO: Fix the PDF build process
# @develop.command(name="build-docs-pdf", help="Builds the sphinx documentation pdf.")
# def build_documentation_pdf(args=None):
#     """
#     Build the Sphinx documentation in PDF format.
#
#     This command initiates the process to build the Sphinx documentation and generates a PDF output.
#
#     Parameters
#     ----------
#     args : optional
#         Additional arguments for the PDF documentation build process.
#     """
#     # Runs the documentation build from the poetry environment
#     echo_and_run_subprocess(
#         ["poetry", "run", "python", "-m", "sphinx", "-M", "latexpdf", "docs/", "_pdf/"]
#     )
#     return 0


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


# TODO decide if this is useful in any form.
# @develop.command(
#     name="convert-all-markdown-to-rst",
#     help="Recursively find all markdown files and convert them to rst files that can be included in the sphinx "
#     "documentation",
# )
# @click.option(
#     "--directory",
#     "-d",
#     type=click.Path(exists=True, file_okay=False, readable=True),
#     default=".",
#     help="Directory to process (default is current directory)",
# )
# def convert_all_markdown_to_rst_command(directory: str):
#     """
#     This script converts all Markdown files in a given DIRECTORY to reStructuredText format.
#     """
#     if directory is None:
#         directory = get_install_directory()
#
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.endswith(".md"):
#                 md_file = os.path.join(root, file)
#                 rst_file = os.path.splitext(md_file)[0] + ".rst"
#
#                 # Confirmation for each file
#                 if not click.confirm(f"Convert {md_file} to RST format?", default=True):
#                     click.echo(f"Skipped {md_file}")
#                     continue
#
#                 try:
#                     # Convert using Pandoc
#                     echo_and_check_subprocess(["pandoc", "-s", md_file, "-o", rst_file])
#                     click.echo(f"Converted {md_file} to {rst_file}")
#                 except subprocess.CalledProcessError as e:
#                     click.echo(f"Error converting {md_file}: {e}", err=True)


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
    default=True,
    help="Dry run the replace in files command.",
)
@develop.command(
    name="replace-in-files",
    help="Recursively find and replace strings in files based on a JSON configuration.",
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
    # Raise helpful errors on missing arguments using the Click API
    if json_dictionary is None:
        raise click.BadParameter("JSON dictionary -j is required.")

    if selected_version is None:
        raise click.BadParameter("Selected version -v is required.")

    if directory is None:
        directory = get_install_directory()

    replace_in_files(directory, json_dictionary, selected_version, dry_run)
    return 0
