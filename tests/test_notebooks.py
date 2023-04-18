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


# uncomment to print notebooks
# for i, path in enumerate(notebook_filenames_all):
#     notebook_base = path.split('/')[-1]
#     print(f"notebook[{i}] = '{notebook_base}'")


# if you want to run only some notebooks, put here, if empty, run all
run_only = [
    # "Metalens",
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
as of April 18 2023
notebook[0] = 'Metalens.ipynb'
notebook[1] = 'Self_intersecting_polyslab.ipynb'
notebook[2] = 'STLImport.ipynb'
notebook[3] = 'GDS_import.ipynb'
notebook[4] = 'HighQ_Si.ipynb'
notebook[5] = 'ModeSolver.ipynb'
notebook[6] = 'BiosensorGrating.ipynb'
notebook[7] = 'FieldProjections.ipynb'
notebook[8] = 'AdjointPlugin_3_InverseDesign.ipynb'
notebook[9] = 'WebAPI.ipynb'
notebook[10] = '8ChannelDemultiplexer.ipynb'
notebook[11] = 'TFSF.ipynb'
notebook[12] = 'AdjointPlugin_4_MultiObjective.ipynb'
notebook[13] = 'DielectricMetasurfaceAbsorber.ipynb'
notebook[14] = 'ParameterScan.ipynb'
notebook[15] = 'MMI_1x4.ipynb'
notebook[16] = 'OpticalLuneburgLens.ipynb'
notebook[17] = 'Fitting.ipynb'
notebook[18] = 'OptimizedL3.ipynb'
notebook[19] = 'ZonePlateFieldProjection.ipynb'
notebook[20] = 'StartHere.ipynb'
notebook[21] = 'WaveguidePluginDemonstration.ipynb'
notebook[22] = 'GratingCoupler.ipynb'
notebook[23] = 'WaveguideCrossing.ipynb'
notebook[24] = 'GradientMetasurfaceReflector.ipynb'
notebook[25] = 'GratingEfficiency.ipynb'
notebook[26] = 'AdjointPlugin_2_GradientChecking.ipynb'
notebook[27] = 'YJunction.ipynb'
notebook[28] = 'VizData.ipynb'
notebook[29] = 'Modal_sources_monitors.ipynb'
notebook[30] = 'Modes_bent_angled.ipynb'
notebook[31] = 'BraggGratings.ipynb'
notebook[32] = 'Dispersion.ipynb'
notebook[33] = 'EdgeCoupler.ipynb'
notebook[34] = 'AutoGrid.ipynb'
notebook[35] = 'DistributedBraggReflectorCavity.ipynb'
notebook[36] = 'PlasmonicNanoparticle.ipynb'
notebook[37] = 'MicrowaveFrequencySelectiveSurface.ipynb'
notebook[38] = 'BoundaryConditions.ipynb'
notebook[39] = 'WaveguideSizeConverter.ipynb'
notebook[40] = 'PolarizationSplitterRotator.ipynb'
notebook[41] = 'PhotonicCrystalWaveguidePolarizationFilter.ipynb'
notebook[42] = 'Bandstructure.ipynb'
notebook[43] = 'CustomFieldSource.ipynb'
notebook[44] = 'SMatrix.ipynb'
notebook[45] = '90OpticalHybrid.ipynb'
notebook[46] = 'Primer.ipynb'
notebook[47] = 'EulerWaveguideBend.ipynb'
notebook[48] = 'PlasmonicYagiUdaNanoantenna.ipynb'
notebook[49] = 'GrapheneMetamaterial.ipynb'
notebook[50] = 'AdjointPlugin_1_Intro.ipynb'
notebook[51] = 'RingResonator.ipynb'
notebook[52] = 'THzDemultiplexerFilter.ipynb'
notebook[53] = 'Near2FarSphereRCS.ipynb'
notebook[54] = 'Simulation.ipynb'
notebook[55] = 'VizSimulation.ipynb'
notebook[56] = 'CustomMediumTutorial.ipynb'
notebook[57] = 'HighQ_Ge.ipynb'
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
