"""
This module contains development functionality for managing the notebooks examples.
"""
from .utils import echo_and_check_subprocess
from .index import develop

__all__ = [
    "run_jupyterlab",
]


@develop.command(name="jupyterlab", help="Runs jupyter lab in a poetry environment.")
def run_jupyterlab(args=None):
    """
    Runs jupyter lab in a poetry environment.

    Parameters
    ----------
    args : list
        A list of command line arguments to be passed to jupyter lab.

    """
    if args is None:
        args = []

    echo_and_check_subprocess(["poetry", "run", "jupyter", "lab", *args])
    return 0
