"""
Utility functions for the tidy3d develop CLI.
"""

import pathlib
import subprocess

import tidy3d

__all__ = [
    "get_install_directory",
    "echo_and_run_subprocess",
    "echo_and_check_subprocess",
]


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


def echo_and_check_subprocess(command: list, *args, **kwargs):
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
    return subprocess.check_call(command, *args, **kwargs, cwd=get_install_directory())
