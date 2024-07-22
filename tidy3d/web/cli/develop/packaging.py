"""
This module contains the functions and commands used to benchmark the timing of various operations in the codebase.

The idea of this functionality is to be able to track the performance of various operations over time and normalise
it for different hardware. For example, say we just need to use a certain section of the codebase, we can use this
functionality to extract the timing performance of that specific operation and compare it to previous usages.
"""

import pathlib
import subprocess
from pathlib import Path

import click

from .index import develop
from .utils import echo_and_check_subprocess

__all__ = [
    "benchmark_timing_operations",
    "benchmark_timing_operations_command",
]

output_timing_log = ["python", "-X", "importtime", "-c", "import tidy3d"]

# Runs the import 100 times.
average_test_import = [
    "python",
    str(Path("scripts", "benchmark_import.py")),
]

timing_commands = {
    "output_timing_log": output_timing_log,
    "average_test_import": average_test_import,
}


def benchmark_timing_operations(
    timing_command: str, in_poetry_environment: bool = True, output_file: str = "import.log"
):
    """
    This function is used to time and benchmark the timing performance of various operations in the codebase. The
    idea of this functionality is to be able to track the performance of various operations over time and normalise
    it for different hardware. For example, say we just need to use a certain section of the codebase, we can use
    this functionality to extract the timing performance of that specific operation and compare it to previous usages.

    Note that this is run within the top level of the tidy3d package. We can write specific files with specific
    operations in the `tests` section and benchmark them properly using this. This function does not require poetry
    and can be run anywhere where a tidy3d installation is already implemented. The output file has an extension.
    """
    timing_command_list = list()
    if output_file is None:
        output_file = timing_command.split("_")[1:] + ".log"

    output_file_path = pathlib.Path(output_file)

    try:
        output_file_write = open(output_file_path, "w+")
    except FileNotFoundError:
        raise FileNotFoundError(
            "The output file path "
            + str(output_file_path)
            + " does not exist and cannot be created."
        )

    if in_poetry_environment:
        timing_command_list += ["poetry", "run"]

    try:
        timing_command_list = timing_commands[timing_command].copy()
    except KeyError:
        # This has to do with choosing a timing command not available in the dictionary
        raise KeyError(
            f"Make sure the selected timing command {timing_command}"
            + "corresponds to an existing command."
        )

    echo_and_check_subprocess(
        command=timing_command_list, stdout=output_file_write, stderr=subprocess.STDOUT
    )


@click.option(
    "-c",
    "--timing-command",
    type=str,
    help="Choose between any of the existing timing commands.",
)
@click.option(
    "--in-poetry-environment",
    default=True,
    type=bool,
    is_flag=True,
    help="Runs in poetry environment if " "True.",
)
@click.option(
    "-o",
    "--output-file",
    default="import.log",
    type=str,
    help="Output file name. Defaults to 'import.log'.'",
)
@develop.command(
    name="benchmark-timing-operations", help="Benchmarks the timing of various operations."
)
def benchmark_timing_operations_command(
    timing_command: str, in_poetry_environment: bool = True, output_file: str = "import.log"
):
    # If timing_command is inputted without a value, raise a warning and print out the existing key options from the
    # timing command dictionary.
    if timing_command is None:
        raise ValueError(
            "Please input a timing command with -c . The existing timing commands are: "
            + str(list(timing_commands.keys()))
        )

    benchmark_timing_operations(
        timing_command=timing_command,
        in_poetry_environment=in_poetry_environment,
        output_file=output_file,
    )
