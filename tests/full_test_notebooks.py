import os
import sys

import pytest

# note: these libraries throw Deprecation warnings in python 3.9, so they are ignored in pytest.ini
import nbformat
from nbconvert.preprocessors import CellExecutionError
from nbconvert.preprocessors import ExecutePreprocessor

sys.path.append("tidy3d")

ep = ExecutePreprocessor(timeout=3000, kernel_name="python3")

# get all notebook files
NOTEBOOK_DIR = "docs/notebooks/"
notebook_filenames_all = [
    NOTEBOOK_DIR + f
    for f in os.listdir(NOTEBOOK_DIR)
    if ".ipynb" in f and f != ".ipynb_checkpoints"
]

# sort alphabetically
notebook_filenames_all.sort()

# uncomment to print notebooks in a way that's useful for `run_only` and `skip` below
for _, path in enumerate(notebook_filenames_all):
    notebook_base = path.split("/")[-1]
    print(f"'{notebook_base[:-6]}',")


# if you want to run only some notebooks, put here, if empty, run all
run_only = []

skip = [
    "8ChannelDemultiplexer",
    "90BendPolarizationSplitterRotator",
    "AdjointPlugin5BoundaryGradients",
    "AdjointPlugin6GratingCoupler",
    "AdjointPlugin7Metalens",
    "AdjointPlugin8WaveguideBend",
    "AdjointPlugin9WDM",
    "AdjointPlugin11CircuitMZI",
]

# if any run only supplied, only add those
if len(run_only):
    notebook_filenames_all = [NOTEBOOK_DIR + base + ".ipynb" for base in run_only]

# filter out the skip notebooks
notebook_filenames = []
for fname in notebook_filenames_all:
    if not any(skip_fname in fname for skip_fname in skip):
        notebook_filenames.append(fname)

"""
as of Nov 10 2023
'8ChannelDemultiplexer',
'90BendPolarizationSplitterRotator',
'90OpticalHybrid',
'AdiabaticCouplerLN',
'AdjointPlugin10YBranchLevelSet',
'AdjointPlugin11CircuitMZI',
'AdjointPlugin1Intro',
'AdjointPlugin2GradientChecking',
'AdjointPlugin3InverseDesign',
'AdjointPlugin4MultiObjective',
'AdjointPlugin5BoundaryGradients',
'AdjointPlugin6GratingCoupler',
'AdjointPlugin7Metalens',
'AdjointPlugin8WaveguideBend',
'AdjointPlugin9WDM',
'AndersonLocalization',
'AnimationTutorial',
'AutoGrid',
'Bandstructure',
'BilevelPSR',
'BiosensorGrating',
'BistablePCCavity',
'BoundaryConditions',
'BraggGratings',
'BroadbandDirectionalCoupler',
'BullseyeCavityPSO',
'CavityFOM',
'CreatingGeometryUsingTrimesh',
'CustomFieldSource',
'CustomMediumTutorial',
'DielectricMetasurfaceAbsorber',
'Dispersion',
'DistributedBraggReflectorCavity',
'DivergedFDTDSimulation',
'EdgeCoupler',
'EulerWaveguideBend',
'FieldProjections',
'Fitting',
'FocusedApodGC',
'FresnelLens',
'FullyAnisotropic',
'GDSImport',
'GradientMetasurfaceReflector',
'GrapheneMetamaterial',
'GratingCoupler',
'GratingEfficiency',
'Gyrotropic',
'HeatSolver',
'HighQGe',
'HighQSi',
'MMI1x4',
'MetalHeaterPhaseShifter',
'Metalens',
'MicrowaveFrequencySelectiveSurface',
'MoS2Waveguide',
'ModalSourcesMonitors',
'ModeSolver',
'ModesBentAngled',
'NanostructuredBoronNitride',
'Near2FarSphereRCS',
'NonHermitianMetagratings',
'OpticalLuneburgLens',
'OptimizedL3',
'PICComponents',
'ParameterScan',
'ParticleSwarmOptimizedPBS',
'PhotonicCrystalWaveguidePolarizationFilter',
'PhotonicCrystalsComponents',
'PlasmonicNanoparticle',
'PlasmonicWaveguideCO2Sensor',
'PlasmonicYagiUdaNanoantenna',
'PolarizationSplitterRotator',
'Primer',
'RadiativeCoolingGlass',
'ResonanceFinder',
'RingResonator',
'SMatrix',
'STLImport',
'SWGBroadbandPolarizer',
'SelfIntersectingPolyslab',
'Simulation',
'StartHere',
'StripToSlotConverters',
'Symmetry',
'TFSF',
'THzDemultiplexerFilter',
'ThermallyTunedRingResonator',
'TimeModulationTutorial',
'VizData',
'VizSimulation',
'WaveguideCrossing',
'WaveguidePluginDemonstration',
'WaveguideSizeConverter',
'WaveguideToRingCoupling',
'WebAPI',
'YJunction',
'ZeroCrossTalkTE',
'ZonePlateFieldProjection',
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
        except CellExecutionError:
            # if there is an error, print message and fail test
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
