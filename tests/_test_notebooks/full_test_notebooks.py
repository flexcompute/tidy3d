from __future__ import annotations

import os
import sys

# note: these libraries throw Deprecation warnings in python 3.9, so they are ignored in pytest.ini
import nbformat
import pytest
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

sys.path.append("tidy3d")

ep = ExecutePreprocessor(timeout=3000, kernel_name="python3")

# Optionally read a list of notebooks that have already been ran
notebooks_ran = ""
# with open("notebooks_run_2.8.0rc1.txt") as fhandle:
#     notebooks_ran = fhandle.read()

# get all notebook files
NOTEBOOK_DIR = "docs/notebooks/"
notebook_filenames_all = [
    NOTEBOOK_DIR + f
    for f in os.listdir(NOTEBOOK_DIR)
    if ".ipynb" in f and f != ".ipynb_checkpoints" and f.split("/")[-1] not in notebooks_ran
]

# sort alphabetically
notebook_filenames_all.sort()

# uncomment to print notebooks in a way that's useful for `run_only` and `skip` below
for _, path in enumerate(notebook_filenames_all):
    notebook_base = path.split("/")[-1]
    print(f"'{notebook_base[:-6]}',")


# if you want to run only some notebooks, put here, if empty, run all
run_only = [
    "Autograd0Quickstart",
    "Autograd10YBranchLevelSet",
    "BoundaryConditions",
    "BroadbandPlaneWaveWithConstantObliqueIncidentAngle",
    "ChargeSolver",
    "EMESolver",
    "FieldProjections",
    "HeatSolver",
    "ModeSolver",
    "ThermallyTunedRingResonator",
    "ThermoOpticDopedModulator",
    "VortexMetasurface",
    "MetalHeaterPhaseShifter",
    "TFSF",
]

skip = [
    # long time (excluding most adjoint)
    "8ChannelDemultiplexer",
    "90BendPolarizationSplitterRotator",
    "BullseyeCavityPSO",
    "FocusedApodGC",
    "GeneticAlgorithmReflector",
    "ParticleSwarmOptimizedPBS",
    "DirectionalCouplerSurrogate.ipynb",
    # hang or error by default
    "AdvancedGaussianSources.ipynb",
    "AdjointPlugin14PreFab.ipynb",
    "Autograd23FabricationAwareInvdes",
    "KLayoutPlugin_DRCQuickstart.ipynb",
    "SimpleModeSolverGUI",
    "WaveguideBendSimulator",
    "GUIDataTutorial.ipynb",
]

# if any run only supplied, only add those
if run_only:
    notebook_filenames_all = [NOTEBOOK_DIR + base + ".ipynb" for base in run_only]

# filter out the skip notebooks
notebook_filenames = []
for fname in notebook_filenames_all:
    if not any(skip_fname in fname for skip_fname in skip):
        notebook_filenames.append(fname)

"""
as of Nov 23 2025
'8ChannelDemultiplexer',
'90BendPolarizationSplitterRotator',
'90OpticalHybrid',
'AbsorbingBoundaryReflection',
'AdiabaticCouplerLN',
'AdvancedGaussianSources',
'AllDielectricStructuralColor',
'AndersonLocalization',
'AnimationTutorial',
'AnisotropicMetamaterialBroadbandPBS',
'AntennaCharacteristics',
'AntiResonantHollowCoreFiber',
'AutoGrid',
'Autograd0Overview',
'Autograd0Quickstart',
'Autograd0QuickstartII',
'Autograd10YBranchLevelSet',
'Autograd12LightExtractor',
'Autograd13Metasurface',
'Autograd15Antenna',
'Autograd16BilayerCoupler',
'Autograd17BandPassFilter',
'Autograd18TopologyBend',
'Autograd19ApodizedCoupler',
'Autograd1Intro',
'Autograd20MetalensWaveguideTaper',
'Autograd21GaPLightExtractor',
'Autograd22PhotonicCrystal',
'Autograd23FabricationAwareInvdes',
'Autograd24DigitalSplitter',
'Autograd26Smatrix',
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
'BayesianOptimizationYJunction',
'BilayerSiNEdgeCoupler',
'BilevelPSR',
'BiosensorGrating',
'BistablePCCavity',
'BlueMicroLED',
'BoundaryConditions',
'BraggGratings',
'BroadbandDirectionalCoupler',
'BroadbandPlaneWaveWithConstantObliqueIncidentAngle',
'BullseyeCavityPSO',
'CMOSRGBSensor',
'CPWRFPhotonics1',
'CPWRFPhotonics2',
'CavityFOM',
'CharacteristicImpedanceCalculator',
'ChargeSolver',
'CircularlyPolarizedPatchAntenna',
'CoupledLineBandpassFilter',
'CreatingGeometryUsingTrimesh',
'CustomFieldSource',
'CustomMediumTutorial',
'Design',
'DielectricMetasurfaceAbsorber',
'DifferentialStripline',
'DirectionalCoupler',
'DirectionalCouplerSurrogate',
'DirectionalScatteringNanodisks',
'DisorderedPlasmonicColor',
'Dispersion',
'DistributedBraggReflectorCavity',
'DivergedFDTDSimulation',
'EMEBends',
'EMESolver',
'EdgeCoupler',
'EdgeFeedPatchAntennaBenchmark',
'EffectiveIndexApproximation',
'EulerWaveguideBend',
'FieldProjections',
'Fitting',
'FocusedApodGC',
'FreeFormCoupler',
'FresnelLens',
'FullyAnisotropic',
'GDSCreation',
'GDSExport',
'GDSImport',
'GUIDataTutorial',
'GeneticAlgorithmReflector',
'GeometryTransformations',
'GradientMetasurfaceReflector',
'GrapheneMetamaterial',
'GratingCoupler',
'GratingEfficiency',
'GroundedCPWViaFence',
'GroupDelayCalculation',
'Gyrotropic',
'HeatDissipationSOI',
'HeatSolver',
'HighQGe',
'HighQSi',
'HybridMicrostripCPWBandpassFilter',
'IntegratedVivaldiAntenna',
'InverseDesign',
'KLayoutPlugin_DRCQuickstart',
'KerrSidebands',
'LNOIPolarizationSplitterRotator',
'LayerRefinement',
'LinearLumpedElements',
'LowContrastWaveguide',
'MIMResonator',
'MMI1x4',
'MMIMeepBenchmark',
'MachZehnderModulator',
'MetalHeaterPhaseShifter',
'MetalOxideSunscreen',
'Metalens',
'MetasurfaceBIC',
'MicroringRFElectrode',
'MicrowaveFrequencySelectiveSurface',
'MidIRMetalens',
'MoS2Waveguide',
'ModalSourcesMonitors',
'ModeOverlap',
'ModeSolver',
'ModesBentAngled',
'MultiplexingMMI',
'MultipoleExpansion',
'NanobeamCavity',
'NanostructuredBoronNitride',
'Near2FarSphereRCS',
'NonHermitianMetagratings',
'OpticalLuneburgLens',
'OpticalSwitchDBS',
'OptimizedL3',
'PCMBraggGratingFilter',
'PECSphereRCS',
'PICComponents',
'ParameterScan',
'ParticleSwarmOptimizedPBS',
'PhotonicCrystalWaveguidePolarizationFilter',
'PhotonicCrystalsComponents',
'PlasmonicNanoparticle',
'PlasmonicNanorodArray',
'PlasmonicPhotothermalHeating',
'PlasmonicWaveguideCO2Sensor',
'PlasmonicYagiUdaNanoantenna',
'PolarizationSplitterRotator',
'Primer',
'RadarAbsorbingMetamaterial',
'RadiativeCoolingGlass',
'RadiativeLossesModeSolver',
'ResonanceFinder',
'RingResonator',
'SMAEdgeMount',
'SMatrix',
'STLImport',
'SWGBroadbandPolarizer',
'SWGWaveguideCrossing',
'SbendCMAES',
'ScaleInvariantWaveguide',
'SelfIntersectingPolyslab',
'SiWaveguideTPA',
'SimpleModeSolverGUI',
'Simulation',
'SourceNormalization',
'StartHere',
'StripToSlotConverters',
'Symmetry',
'TFSF',
'THzDemultiplexerFilter',
'TaperedWgDispersion',
'ThermallyTunedRingResonator',
'ThermoOpticDopedModulator',
'ThroughSiliconVia',
'TidyFab0GC',
'TimeModulationTutorial',
'TopoQuantumPhC',
'TunableChiralMetasurface',
'UnstructuredData',
'VizData',
'VizSimulation',
'VortexMetasurface',
'WPDHarmonicSuppression1',
'WPDHarmonicSuppression2',
'WPDHarmonicSuppression3',
'WaveguideBendSimulator',
'WaveguideCrossing',
'WaveguideGratingAntenna',
'WaveguidePluginDemonstration',
'WaveguideSizeConverter',
'WaveguideToRingCoupling',
'WebAPI',
'WidebandBeamSteerableReflectarrayWithPRUC',
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
