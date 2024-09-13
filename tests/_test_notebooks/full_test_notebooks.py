import os
import sys

# note: these libraries throw Deprecation warnings in python 3.9, so they are ignored in pytest.ini
import nbformat
import pytest
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

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
    # WIP
    "Autograd10YBranchLevelSet",
    "Autograd13Metasurface",
    # long time (excluding most adjoint)
    "8ChannelDemultiplexer",
    "90BendPolarizationSplitterRotator",
    "BullseyeCavityPSO",
    "FocusedApodGC",
    "GeneticAlgorithmReflector",
    "ParticleSwarmOptimizedPBS",
    # hang by default
    "AdjointPlugin14PreFab.ipynb",
    "WaveguideBendSimulator",
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
as of Sept 04 2024
'8ChannelDemultiplexer',
'90BendPolarizationSplitterRotator',
'90OpticalHybrid',
'AdiabaticCouplerLN',
'AdjointPlugin0Quickstart',
'AdjointPlugin10YBranchLevelSet',
'AdjointPlugin11CircuitMZI',
'AdjointPlugin12LightExtractor',
'AdjointPlugin13Metasurface',
'AdjointPlugin14PreFab',
'AdjointPlugin1Intro',
'AdjointPlugin2GradientChecking',
'AdjointPlugin3InverseDesign',
'AdjointPlugin4MultiObjective',
'AdjointPlugin5BoundaryGradients',
'AdjointPlugin6GratingCoupler',
'AdjointPlugin7Metalens',
'AdjointPlugin8WaveguideBend',
'AdjointPlugin9WDM',
'AllDielectricStructuralColor',
'AndersonLocalization',
'AnimationTutorial',
'AntiResonantHollowCoreFiber',
'AutoGrid',
'Autograd0Quickstart',
'Autograd15Antenna',
'Autograd16BilayerCoupler',
'Autograd17BandPassFilter',
'Autograd1Intro',
'Autograd2GradientChecking',
'Autograd3InverseDesign',
'Autograd4MultiObjective',
'Autograd5BoundaryGradients',
'Autograd6GratingCoupler',
'Autograd7Metalens',
'Autograd8WaveguideBend',
'Autograd9WDM',
'Bandstructure',
'BatchModeSolver',
'BilayerSiNEdgeCoupler',
'BilevelPSR',
'BiosensorGrating',
'BistablePCCavity',
'BoundaryConditions',
'BraggGratings',
'BroadbandDirectionalCoupler',
'BullseyeCavityPSO',
'CMOSRGBSensor',
'CavityFOM',
'CharacteristicImpedanceCalculator',
'CircularlyPolarizedPatchAntenna',
'CoupledLineBandpassFilter',
'CreatingGeometryUsingTrimesh',
'CustomFieldSource',
'CustomMediumTutorial',
'Design',
'DielectricMetasurfaceAbsorber',
'Dispersion',
'DistributedBraggReflectorCavity',
'DivergedFDTDSimulation',
'EMESolver',
'EdgeCoupler',
'EffectiveIndexApproximation',
'EulerWaveguideBend',
'FieldProjections',
'Fitting',
'FocusedApodGC',
'FreeFormCoupler',
'FresnelLens',
'FullyAnisotropic',
'GDSExport',
'GDSImport',
'GeneticAlgorithmReflector',
'GeometryTransformations',
'GradientMetasurfaceReflector',
'GrapheneMetamaterial',
'GratingCoupler',
'GratingEfficiency',
'Gyrotropic',
'HeatSolver',
'HighQGe',
'HighQSi',
'IntegratedVivaldiAntenna',
'InverseDesign',
'LNOIPolarizationSplitterRotator',
'MIMResonator',
'MMI1x4',
'MachZehnderModulator',
'MetalHeaterPhaseShifter',
'MetalOxideSunscreen',
'Metalens',
'MicrowaveFrequencySelectiveSurface',
'MidIRMetalens',
'MoS2Waveguide',
'ModalSourcesMonitors',
'ModeSolver',
'ModesBentAngled',
'MultiplexingMMI',
'NanobeamCavity',
'NanostructuredBoronNitride',
'Near2FarSphereRCS',
'NonHermitianMetagratings',
'OpticalLuneburgLens',
'OpticalSwitchDBS',
'OptimizedL3',
'PICComponents',
'ParameterScan',
'ParticleSwarmOptimizedPBS',
'PhotonicCrystalWaveguidePolarizationFilter',
'PhotonicCrystalsComponents',
'PlasmonicNanoparticle',
'PlasmonicNanorodArray',
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
'SbendCMAES',
'ScaleInvariantWaveguide',
'SelfIntersectingPolyslab',
'Simulation',
'StartHere',
'StripToSlotConverters',
'Symmetry',
'TFSF',
'THzDemultiplexerFilter',
'TaperedWgDispersion',
'ThermallyTunedRingResonator',
'ThermoOpticDopedModulator',
'TimeModulationTutorial',
'TunableChiralMetasurface',
'UnstructuredData',
'VizData',
'VizSimulation',
'VortexMetasurface',
'WaveguideBendSimulator',
'WaveguideCrossing',
'WaveguideGratingAntenna',
'WaveguidePluginDemonstration',
'WaveguideSizeConverter',
'WaveguideToRingCoupling',
'WebAPI',
'XarrayTutorial',
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
            ep.preprocess(nb, {"metadata": {"path": f"{NOTEBOOK_DIR}"}})
        except CellExecutionError:
            # if there is an error, print message and fail test
            msg = f'Error executing the notebook "{notebook_fname}".\n\n'
            msg += f'See notebook "{notebook_fname}" for the traceback.'
            print(msg)
            raise

        # write the executed notebook to file
        finally:
            with open(notebook_fname, mode="w", encoding="utf-8") as f:
                nbformat.write(nb, f)

        # can we get notebook's local variables and do more individual tests?
