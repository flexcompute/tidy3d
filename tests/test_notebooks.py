import pytest
import nbconvert
import nbformat
from nbconvert.preprocessors import CellExecutionError

import os
import sys

sys.path.append("./")

# load the notebook executor
from nbconvert.preprocessors import ExecutePreprocessor

ep = ExecutePreprocessor(timeout=1000)

# get all notebook files
notebook_directory = "notebooks/"
notebook_filenames = [
    notebook_directory + f
    for f in os.listdir(notebook_directory)
    if ".ipynb" in f and f != ".ipynb_checkpoints"
]

# if you want to run only some notebooks, put here, if empty, run all
run_only = ("Fitting", "ModeSolver", "StartHere", "VizData", "VizGeometry")
if len(run_only):
    notebook_filenames = [notebook_directory + base + ".ipynb" for base in run_only]


@pytest.mark.parametrize("fname", notebook_filenames)
def test_notebooks(fname):
    # loop through notebooks in notebook_filenames and test each of them separately
    _run_notebook(fname)


def _run_notebook(notebook_fname):

    # open the notebook
    with open(notebook_fname) as f:
        nb = nbformat.read(f, as_version=4)

        # try running the notebook
        try:
            # run from the `notebooks/` directory
            out = ep.preprocess(nb, {"metadata": {"path": "./notebooks"}})

        # if there is an error, print message and fail test
        except CellExecutionError as e:
            out = None
            msg = 'Error executing the notebook "%s".\n\n' % notebook_fname
            msg += 'See notebook "%s" for the traceback.' % notebook_fname
            print(msg)
            raise

        # write the executed notebook to file
        finally:
            with open(notebook_fname, mode="w", encoding="utf-8") as f:
                nbformat.write(nb, f)

        # can we get notebook's local variables and do more individual tests?
