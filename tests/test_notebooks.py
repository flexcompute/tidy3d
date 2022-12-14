import os
import sys

import pytest

# note: these libraries throw Deprecation warnings in python 3.9, so they are ignored in pytest.ini
import nbconvert
import nbformat
from nbconvert.preprocessors import CellExecutionError
from nbconvert.preprocessors import ExecutePreprocessor

sys.path.append("tidy3d")

ep = ExecutePreprocessor(timeout=1000, kernel_name="python3")

# get all notebook files
NOTEBOOK_DIR = "docs/source/notebooks/"
notebook_filenames = [
    NOTEBOOK_DIR + f
    for f in os.listdir(NOTEBOOK_DIR)
    if ".ipynb" in f and f != ".ipynb_checkpoints"
]

# if you want to run only some notebooks, put here, if empty, run all
run_only = []
# run_only = ["Metalens"]
if len(run_only):
    notebook_filenames = [NOTEBOOK_DIR + base + ".ipynb" for base in run_only]

# for name in notebook_filenames:
#     print(f"'{name}'")
""" 
as of 12/10/22 (output of ls -a *.ipynb | cat)
Adjoint.ipynb
AutoGrid.ipynb
Bandstructure.ipynb
BiosensorGrating.ipynb
BoundaryConditions.ipynb
BraggGratings.ipynb
CustomFieldSource.ipynb
DielectricMetasurfaceAbsorber.ipynb
Dispersion.ipynb
DistributedBraggReflectorCavity.ipynb
EulerWaveguideBend.ipynb
FieldProjections.ipynb
Fitting.ipynb
GDS_import.ipynb
GratingCoupler.ipynb
GratingEfficiency.ipynb
HighQ_Ge.ipynb
HighQ_Si.ipynb
Metalens.ipynb
MMI_1x4.ipynb
Modal_sources_monitors.ipynb
Modes_bent_angled.ipynb
ModeSolver.ipynb
Near2FarSphereRCS.ipynb
OptimizedL3.ipynb
ParameterScan.ipynb
PlasmonicNanoparticle.ipynb
PlasmonicYagiUdaNanoantenna.ipynb
PolarizationSplitterRotator.ipynb
RingResonator.ipynb
Simulation.ipynb
SMatrix.ipynb
StartHere.ipynb
VizData.ipynb
VizSimulation.ipynb
WaveguideCrossing.ipynb
WebAPI.ipynb
WhatsNew.ipynb
YJunction.ipynb
ZonePlateFieldProjection.ipynb
"""


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
            out = ep.preprocess(nb, {"metadata": {"path": f"{NOTEBOOK_DIR}"}})

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
