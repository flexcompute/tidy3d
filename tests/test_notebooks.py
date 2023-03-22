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
notebook_filenames_all = [
    NOTEBOOK_DIR + f
    for f in os.listdir(NOTEBOOK_DIR)
    if ".ipynb" in f and f != ".ipynb_checkpoints"
]

# for i, path in enumerate(notebook_filenames):
#     notebook_base = path.split('/')[-1]
#     print(f"notebook[{i}] = '{notebook_base}'")


# if you want to run only some notebooks, put here, if empty, run all
run_only = [
    # "WaveguideSizeConverter"
]
skip = [
]

# if any run only supplied, only add those
if len(run_only):
    notebook_filenames_all = [NOTEBOOK_DIR + base + ".ipynb" for base in run_only]

# filter out the skip notebooks
notebook_filenames = []
for fname in notebook_filenames_all:
    if not any((skip_fname in fname for skip_fname in skip)):
        notebook_filenames.append(fname)

""" 
as of Feb 2 2023
notebook[0] = 'Metalens.ipynb'
notebook[1] = 'GDS_import.ipynb'
notebook[2] = 'HighQ_Si.ipynb'
notebook[3] = 'ModeSolver.ipynb'
notebook[4] = 'WhatsNew.ipynb'
notebook[5] = 'BiosensorGrating.ipynb'
notebook[6] = 'FieldProjections.ipynb'
notebook[7] = 'AdjointPlugin_3_InverseDesign.ipynb'
notebook[8] = 'WebAPI.ipynb'
notebook[9] = '8ChannelDemultiplexer.ipynb'
notebook[10] = 'DielectricMetasurfaceAbsorber.ipynb'
notebook[11] = 'ParameterScan.ipynb'
notebook[12] = 'MMI_1x4.ipynb'
notebook[13] = 'Fitting.ipynb'
notebook[14] = 'OptimizedL3.ipynb'
notebook[15] = 'ZonePlateFieldProjection.ipynb'
notebook[16] = 'StartHere.ipynb'
notebook[17] = 'GratingCoupler.ipynb'
notebook[18] = 'WaveguideCrossing.ipynb'
notebook[19] = 'GradientMetasurfaceReflector.ipynb'
notebook[20] = 'GratingEfficiency.ipynb'
notebook[21] = 'AdjointPlugin_2_GradientChecking.ipynb'
notebook[22] = 'YJunction.ipynb'
notebook[23] = 'VizData.ipynb'
notebook[24] = 'Modal_sources_monitors.ipynb'
notebook[25] = 'Modes_bent_angled.ipynb'
notebook[26] = 'BraggGratings.ipynb'
notebook[27] = 'Dispersion.ipynb'
notebook[28] = 'AutoGrid.ipynb'
notebook[29] = 'DistributedBraggReflectorCavity.ipynb'
notebook[30] = 'PlasmonicNanoparticle.ipynb'
notebook[31] = 'BoundaryConditions.ipynb'
notebook[32] = 'WaveguideSizeConverter.ipynb'
notebook[33] = 'PolarizationSplitterRotator.ipynb'
notebook[34] = 'Bandstructure.ipynb'
notebook[35] = 'CustomFieldSource.ipynb'
notebook[36] = 'SMatrix.ipynb'
notebook[37] = 'EulerWaveguideBend.ipynb'
notebook[38] = 'PlasmonicYagiUdaNanoantenna.ipynb'
notebook[39] = 'AdjointPlugin_1_Intro.ipynb'
notebook[40] = 'RingResonator.ipynb'
notebook[41] = 'THzDemultiplexerFilter.ipynb'
notebook[42] = 'Near2FarSphereRCS.ipynb'
notebook[43] = 'Simulation.ipynb'
notebook[44] = 'VizSimulation.ipynb'
notebook[45] = 'CustomMediumTutorial.ipynb'
notebook[46] = 'HighQ_Ge.ipynb'
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
