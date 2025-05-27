"""Tidy3d package imports"""

from __future__ import annotations

from tidy3d.components.material.multi_physics import MultiPhysicsMedium
from tidy3d.components.material.tcad.charge import (
    ChargeConductorMedium,
    ChargeInsulatorMedium,
    SemiconductorMedium,
)
from tidy3d.components.material.tcad.heat import (
    FluidMedium,
    FluidSpec,
    SolidMedium,
    SolidSpec,
)
from tidy3d.components.microwave.data.monitor_data import (
    AntennaMetricsData,
)
from tidy3d.components.spice.analysis.dc import (
    ChargeToleranceSpec,
    IsothermalSteadyChargeDCAnalysis,
)
from tidy3d.components.spice.sources.dc import DCCurrentSource, DCVoltageSource
from tidy3d.components.spice.sources.types import VoltageSourceType
from tidy3d.components.tcad.analysis.heat_simulation_type import UnsteadyHeatAnalysis, UnsteadySpec
from tidy3d.components.tcad.boundary.specification import (
    HeatBoundarySpec,
    HeatChargeBoundarySpec,
)
from tidy3d.components.tcad.data.sim_data import (
    DeviceCharacteristics,
    HeatChargeSimulationData,
    HeatSimulationData,
)
from tidy3d.components.tcad.data.types import (
    SteadyCapacitanceData,
    SteadyEnergyBandData,
    SteadyFreeCarrierData,
    SteadyPotentialData,
    TemperatureData,
)
from tidy3d.components.tcad.doping import ConstantDoping, GaussianDoping
from tidy3d.components.tcad.generation_recombination import FossumCarrierLifetime
from tidy3d.components.tcad.grid import (
    DistanceUnstructuredGrid,
    GridRefinementLine,
    GridRefinementRegion,
    UniformUnstructuredGrid,
)
from tidy3d.components.tcad.monitors.charge import (
    SteadyCapacitanceMonitor,
    SteadyEnergyBandMonitor,
    SteadyFreeCarrierMonitor,
    SteadyPotentialMonitor,
)
from tidy3d.components.tcad.monitors.heat import (
    TemperatureMonitor,
)
from tidy3d.components.tcad.simulation.heat import HeatSimulation
from tidy3d.components.tcad.simulation.heat_charge import HeatChargeSimulation
from tidy3d.components.tcad.types import (
    AugerRecombination,
    CaugheyThomasMobility,
    ConstantMobilityModel,
    ConvectionBC,
    CurrentBC,
    HeatFluxBC,
    HeatFromElectricSource,
    HeatSource,
    InsulatingBC,
    RadiativeRecombination,
    ShockleyReedHallRecombination,
    SlotboomBandGapNarrowing,
    TemperatureBC,
    UniformHeatSource,
    VoltageBC,
)

from .components.apodization import ApodizationSpec

# boundary placement for other solvers
# boundary placement for other solvers
from .components.bc_placement import (
    MediumMediumInterface,
    SimulationBoundary,
    StructureBoundary,
    StructureSimulationBoundary,
    StructureStructureInterface,
)

# analytic beams
from .components.beam import (
    AstigmaticGaussianBeamProfile,
    GaussianBeamProfile,
    PlaneWaveBeamProfile,
)

# boundary
from .components.boundary import (
    PML,
    Absorber,
    AbsorberParams,
    BlochBoundary,
    Boundary,
    BoundaryEdge,
    BoundaryEdgeType,
    BoundarySpec,
    DefaultAbsorberParameters,
    DefaultPMLParameters,
    DefaultStablePMLParameters,
    PECBoundary,
    Periodic,
    PMCBoundary,
    PMLParams,
    PMLTypes,
    StablePML,
)

# data
from .components.data.data_array import (
    CellDataArray,
    ChargeDataArray,
    DiffractionDataArray,
    EMECoefficientDataArray,
    EMEModeIndexDataArray,
    EMEScalarFieldDataArray,
    EMEScalarModeFieldDataArray,
    EMESMatrixDataArray,
    FieldProjectionAngleDataArray,
    FieldProjectionCartesianDataArray,
    FieldProjectionKSpaceDataArray,
    FluxDataArray,
    FluxTimeDataArray,
    HeatDataArray,
    IndexedDataArray,
    IndexedTimeDataArray,
    IndexedVoltageDataArray,
    ModeAmpsDataArray,
    ModeIndexDataArray,
    PointDataArray,
    ScalarFieldDataArray,
    ScalarFieldTimeDataArray,
    ScalarModeFieldCylindricalDataArray,
    ScalarModeFieldDataArray,
    SpatialDataArray,
    SpatialVoltageDataArray,
    SteadyVoltageDataArray,
)
from .components.data.dataset import (
    FieldDataset,
    FieldTimeDataset,
    ModeSolverDataset,
    PermittivityDataset,
)
from .components.data.monitor_data import (
    AbstractFieldProjectionData,
    DiffractionData,
    DirectivityData,
    FieldData,
    FieldProjectionAngleData,
    FieldProjectionCartesianData,
    FieldProjectionKSpaceData,
    FieldTimeData,
    FluxData,
    FluxTimeData,
    ModeData,
    ModeSolverData,
    PermittivityData,
)
from .components.data.sim_data import DATA_TYPE_MAP, SimulationData
from .components.data.utils import (
    TetrahedralGridDataset,
    TriangularGridDataset,
)
from .components.eme.data.dataset import (
    EMECoefficientDataset,
    EMEFieldDataset,
    EMEModeSolverDataset,
    EMESMatrixDataset,
)
from .components.eme.data.monitor_data import EMECoefficientData, EMEFieldData, EMEModeSolverData
from .components.eme.data.sim_data import EMESimulationData
from .components.eme.grid import (
    EMECompositeGrid,
    EMEExplicitGrid,
    EMEGrid,
    EMEModeSpec,
    EMEUniformGrid,
)
from .components.eme.monitor import (
    EMECoefficientMonitor,
    EMEFieldMonitor,
    EMEModeSolverMonitor,
    EMEMonitor,
)

# EME
from .components.eme.simulation import EMESimulation
from .components.eme.sweep import EMEFreqSweep, EMELengthSweep, EMEModeSweep, EMEPeriodicitySweep

# field projection
from .components.field_projection import FieldProjector

# frequency conversion utilities
from .components.frequencies import frequencies, wavelengths

# geometry
from .components.geometry.base import Box, ClipOperation, Geometry, GeometryGroup, Transformed
from .components.geometry.mesh import TriangleMesh
from .components.geometry.polyslab import PolySlab
from .components.geometry.primitives import Cylinder, Sphere
from .components.grid.corner_finder import CornerFinderSpec
from .components.grid.grid import Coords, Coords1D, FieldGrid, Grid, YeeGrid
from .components.grid.grid_spec import (
    AutoGrid,
    CustomGrid,
    CustomGridBoundaries,
    GridRefinement,
    GridSpec,
    LayerRefinementSpec,
    QuasiUniformGrid,
    UniformGrid,
)

# lumped elements
from .components.lumped_element import (
    AdmittanceNetwork,
    CoaxialLumpedResistor,
    LinearLumpedElement,
    LumpedElement,
    LumpedResistor,
    RectangularLumpedElement,
    RLCNetwork,
)

# medium
# for docs
from .components.medium import (
    PEC,
    PEC2D,
    AbstractMedium,
    AnisotropicMedium,
    CustomAnisotropicMedium,
    CustomDebye,
    CustomDrude,
    CustomLorentz,
    CustomMedium,
    CustomPoleResidue,
    CustomSellmeier,
    Debye,
    Drude,
    FullyAnisotropicMedium,
    HammerstadSurfaceRoughness,
    HuraySurfaceRoughness,
    KerrNonlinearity,
    Lorentz,
    LossyMetalMedium,
    Medium,
    Medium2D,
    NonlinearModel,
    NonlinearSpec,
    NonlinearSusceptibility,
    PECMedium,
    PerturbationMedium,
    PerturbationPoleResidue,
    PoleResidue,
    Sellmeier,
    SurfaceImpedanceFitterParam,
    TwoPhotonAbsorption,
    medium_from_nk,
)
from .components.mode.data.sim_data import ModeSimulationData

# Mode
from .components.mode.simulation import ModeSimulation

# modes
from .components.mode_spec import ModeSpec

# monitors
from .components.monitor import (
    AuxFieldTimeMonitor,
    DiffractionMonitor,
    DirectivityMonitor,
    FieldMonitor,
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionKSpaceMonitor,
    FieldProjectionSurface,
    FieldTimeMonitor,
    FluxMonitor,
    FluxTimeMonitor,
    ModeMonitor,
    ModeSolverMonitor,
    Monitor,
    PermittivityMonitor,
)
from .components.parameter_perturbation import (
    CustomChargePerturbation,
    CustomHeatPerturbation,
    IndexPerturbation,
    LinearChargePerturbation,
    LinearHeatPerturbation,
    NedeljkovicSorefMashanovich,
    ParameterPerturbation,
    PermittivityPerturbation,
)

# run time spec
from .components.run_time_spec import RunTimeSpec

# scene
# scene
from .components.scene import Scene

# simulation
from .components.simulation import Simulation
from .components.source.base import Source
from .components.source.current import (
    CustomCurrentSource,
    PointDipole,
    UniformCurrentSource,
)
from .components.source.field import (
    TFSF,
    AstigmaticGaussianBeam,
    CustomFieldSource,
    FixedAngleSpec,
    FixedInPlaneKSpec,
    GaussianBeam,
    ModeSource,
    PlaneWave,
)

# sources
from .components.source.time import (
    ContinuousWave,
    CustomSourceTime,
    GaussianPulse,
    SourceTime,
)

# structures
from .components.structure import MeshOverrideStructure, Structure

# subpixel
from .components.subpixel_spec import (
    ContourPathAveraging,
    HeuristicPECStaircasing,
    PECConformal,
    PolarizedAveraging,
    Staircasing,
    SubpixelSpec,
    SurfaceImpedance,
    VolumetricAveraging,
)

# time modulation
from .components.time_modulation import (
    ContinuousWaveTimeModulation,
    ModulationSpec,
    SpaceModulation,
    SpaceTimeModulation,
)
from .components.transformation import RotationAroundAxis
from .components.viz import VisualizationSpec, restore_matplotlib_rcparams

# config
from .config import config

# constants imported as `C_0 = td.C_0` or `td.constants.C_0`
from .constants import C_0, EPSILON_0, ETA_0, HBAR, K_B, MU_0, Q_e, inf
from .log import log, set_logging_console, set_logging_file

# material library dict imported as `from tidy3d import material_library`
# get material `mat` and variant `var` as `material_library[mat][var]`
from .material_library.material_library import material_library
from .material_library.parametric_materials import Graphene

# updater
from .updater import Updater

# version
from .version import __version__


def set_logging_level(level: str) -> None:
    """Raise a warning here instead of setting the logging level."""
    raise DeprecationWarning(
        "``set_logging_level`` no longer supported. "
        f"To set the logging level, call ``tidy3d.config.logging_level = {level}``."
    )


log.info(f"Using client version: {__version__}")

Transformed.update_forward_refs()
ClipOperation.update_forward_refs()
GeometryGroup.update_forward_refs()

__all__ = [
    "C_0",
    "DATA_TYPE_MAP",
    "EPSILON_0",
    "ETA_0",
    "HBAR",
    "K_B",
    "MU_0",
    "PEC",
    "PEC2D",
    "PML",
    "TFSF",
    "Absorber",
    "AbsorberParams",
    "AbstractFieldProjectionData",
    "AbstractMedium",
    "AdmittanceNetwork",
    "AnisotropicMedium",
    "AntennaMetricsData",
    "ApodizationSpec",
    "AstigmaticGaussianBeam",
    "AstigmaticGaussianBeamProfile",
    "AugerRecombination",
    "AutoGrid",
    "AuxFieldTimeData",
    "AuxFieldTimeMonitor",
    "BlochBoundary",
    "Boundary",
    "BoundaryEdge",
    "BoundaryEdgeType",
    "BoundarySpec",
    "Box",
    "CaugheyThomasMobility",
    "CellDataArray",
    "ChargeConductorMedium",
    "ChargeDataArray",
    "ChargeInsulatorMedium",
    "ChargeToleranceSpec",
    "ClipOperation",
    "CoaxialLumpedResistor",
    "ConstantDoping",
    "ConstantMobilityModel",
    "ContinuousWave",
    "ContinuousWaveTimeModulation",
    "ContourPathAveraging",
    "ConvectionBC",
    "Coords",
    "Coords1D",
    "CornerFinderSpec",
    "CurrentBC",
    "CustomAnisotropicMedium",
    "CustomChargePerturbation",
    "CustomCurrentSource",
    "CustomDebye",
    "CustomDrude",
    "CustomFieldSource",
    "CustomGrid",
    "CustomGridBoundaries",
    "CustomHeatPerturbation",
    "CustomLorentz",
    "CustomMedium",
    "CustomPoleResidue",
    "CustomSellmeier",
    "CustomSourceTime",
    "Cylinder",
    "DCCurrentSource",
    "DCVoltageSource",
    "Debye",
    "DefaultAbsorberParameters",
    "DefaultPMLParameters",
    "DefaultStablePMLParameters",
    "DeviceCharacteristics",
    "DiffractionData",
    "DiffractionDataArray",
    "DiffractionMonitor",
    "DirectivityData",
    "DirectivityMonitor",
    "DistanceUnstructuredGrid",
    "Drude",
    "EMECoefficientData",
    "EMECoefficientDataArray",
    "EMECoefficientDataset",
    "EMECoefficientMonitor",
    "EMECompositeGrid",
    "EMEExplicitGrid",
    "EMEFieldData",
    "EMEFieldDataset",
    "EMEFieldMonitor",
    "EMEFreqSweep",
    "EMEGrid",
    "EMELengthSweep",
    "EMEModeIndexDataArray",
    "EMEModeSolverData",
    "EMEModeSolverDataset",
    "EMEModeSolverMonitor",
    "EMEModeSpec",
    "EMEModeSweep",
    "EMEMonitor",
    "EMEPeriodicitySweep",
    "EMESMatrixDataArray",
    "EMESMatrixDataset",
    "EMEScalarFieldDataArray",
    "EMEScalarModeFieldDataArray",
    "EMESimulation",
    "EMESimulationData",
    "EMESweepSpec",
    "EMEUniformGrid",
    "FieldData",
    "FieldDataset",
    "FieldGrid",
    "FieldMonitor",
    "FieldProjectionAngleData",
    "FieldProjectionAngleDataArray",
    "FieldProjectionAngleMonitor",
    "FieldProjectionCartesianData",
    "FieldProjectionCartesianDataArray",
    "FieldProjectionCartesianMonitor",
    "FieldProjectionKSpaceData",
    "FieldProjectionKSpaceDataArray",
    "FieldProjectionKSpaceMonitor",
    "FieldProjectionSurface",
    "FieldProjector",
    "FieldTimeData",
    "FieldTimeDataset",
    "FieldTimeMonitor",
    "FixedAngleSpec",
    "FixedInPlaneKSpec",
    "FluidMedium",
    "FluidSpec",
    "FluxData",
    "FluxDataArray",
    "FluxMonitor",
    "FluxTimeData",
    "FluxTimeDataArray",
    "FluxTimeMonitor",
    "FossumCarrierLifetime",
    "FullyAnisotropicMedium",
    "GaussianBeam",
    "GaussianBeamProfile",
    "GaussianDoping",
    "GaussianPulse",
    "Geometry",
    "GeometryGroup",
    "Graphene",
    "Grid",
    "GridRefinement",
    "GridRefinementLine",
    "GridRefinementRegion",
    "GridSpec",
    "HammerstadSurfaceRoughness",
    "HeatBoundarySpec",
    "HeatChargeBoundarySpec",
    "HeatChargeSimulation",
    "HeatChargeSimulationData",
    "HeatDataArray",
    "HeatFluxBC",
    "HeatFromElectricSource",
    "HeatSimulation",
    "HeatSimulationData",
    "HeatSource",
    "HeuristicPECStaircasing",
    "HuraySurfaceRoughness",
    "IndexPerturbation",
    "IndexedDataArray",
    "IndexedTimeDataArray",
    "IndexedVoltageDataArray",
    "InsulatingBC",
    "IsothermalSteadyChargeDCAnalysis",
    "KerrNonlinearity",
    "LayerRefinementSpec",
    "LinearChargePerturbation",
    "LinearHeatPerturbation",
    "LinearLumpedElement",
    "Lorentz",
    "LossyMetalMedium",
    "LumpedElement",
    "LumpedResistor",
    "Medium",
    "Medium2D",
    "MediumMediumInterface",
    "MeshOverrideStructure",
    "ModeAmpsDataArray",
    "ModeData",
    "ModeIndexDataArray",
    "ModeMonitor",
    "ModeSimulation",
    "ModeSimulationData",
    "ModeSolverData",
    "ModeSolverDataset",
    "ModeSolverMonitor",
    "ModeSource",
    "ModeSpec",
    "ModulationSpec",
    "Monitor",
    "MultiPhysicsMedium",
    "NedeljkovicSorefMashanovich",
    "NonlinearModel",
    "NonlinearSpec",
    "NonlinearSusceptibility",
    "PECBoundary",
    "PECConformal",
    "PECMedium",
    "PMCBoundary",
    "PMLParams",
    "PMLTypes",
    "ParameterPerturbation",
    "Periodic",
    "PermittivityData",
    "PermittivityDataset",
    "PermittivityMonitor",
    "PermittivityPerturbation",
    "PerturbationMedium",
    "PerturbationPoleResidue",
    "PlaneWave",
    "PlaneWaveBeamProfile",
    "PointDataArray",
    "PointDipole",
    "PolarizedAveraging",
    "PoleResidue",
    "PolySlab",
    "Q_e",
    "QuasiUniformGrid",
    "RLCNetwork",
    "RadiativeRecombination",
    "RectangularLumpedElement",
    "RotationAroundAxis",
    "RunTimeSpec",
    "ScalarFieldDataArray",
    "ScalarFieldTimeDataArray",
    "ScalarModeFieldCylindricalDataArray",
    "ScalarModeFieldDataArray",
    "Scene",
    "Sellmeier",
    "SemiconductorMedium",
    "ShockleyReedHallRecombination",
    "Simulation",
    "SimulationBoundary",
    "SimulationData",
    "SlotboomBandGapNarrowing",
    "SolidMedium",
    "SolidSpec",
    "Source",
    "SourceTime",
    "SpaceModulation",
    "SpaceTimeModulation",
    "SpatialDataArray",
    "SpatialVoltageDataArray",
    "Sphere",
    "StablePML",
    "Staircasing",
    "SteadyCapacitanceData",
    "SteadyCapacitanceMonitor",
    "SteadyEnergyBandData",
    "SteadyEnergyBandMonitor",
    "SteadyFreeCarrierData",
    "SteadyFreeCarrierMonitor",
    "SteadyPotentialData",
    "SteadyPotentialMonitor",
    "SteadyVoltageDataArray",
    "Structure",
    "StructureBoundary",
    "StructureSimulationBoundary",
    "StructureStructureInterface",
    "SubpixelSpec",
    "SurfaceImpedance",
    "SurfaceImpedanceFitterParam",
    "TemperatureBC",
    "TemperatureData",
    "TemperatureMonitor",
    "TetrahedralGridDataset",
    "Transformed",
    "TriangleMesh",
    "TriangularGridDataset",
    "TwoPhotonAbsorption",
    "UniformCurrentSource",
    "UniformGrid",
    "UniformHeatSource",
    "UniformUnstructuredGrid",
    "UnsteadyHeatAnalysis",
    "UnsteadySpec",
    "Updater",
    "VisualizationSpec",
    "VoltageBC",
    "VoltageSourceType",
    "VolumetricAveraging",
    "YeeGrid",
    "__version__",
    "config",
    "frequencies",
    "inf",
    "log",
    "material_library",
    "medium_from_nk",
    "restore_matplotlib_rcparams",
    "set_logging_console",
    "set_logging_file",
    "wavelengths",
]
