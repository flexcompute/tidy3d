"""Tidy3d package imports"""

# grid
# apodization
from tidy3d.components.apodization import ApodizationSpec

# boundary placement for other solvers
# boundary placement for other solvers
from tidy3d.components.bc_placement import (
    MediumMediumInterface,
    SimulationBoundary,
    StructureBoundary,
    StructureSimulationBoundary,
    StructureStructureInterface,
)

# boundary
from tidy3d.components.boundary import (
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
from tidy3d.components.data.data_array import (
    AxialRatioDataArray,
    CellDataArray,
    ChargeDataArray,
    DCCapacitanceDataArray,
    DCIVCurveDataArray,
    DiffractionDataArray,
    DirectivityDataArray,
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
    ModeAmpsDataArray,
    ModeIndexDataArray,
    PointDataArray,
    ScalarFieldDataArray,
    ScalarFieldTimeDataArray,
    ScalarModeFieldCylindricalDataArray,
    ScalarModeFieldDataArray,
    SpatialDataArray,
)
from tidy3d.components.data.dataset import (
    FieldDataset,
    FieldTimeDataset,
    ModeSolverDataset,
    PermittivityDataset,
    TetrahedralGridDataset,
    TriangularGridDataset,
)
from tidy3d.components.data.monitor_data import (
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
from tidy3d.components.data.sim_data import DATA_TYPE_MAP, SimulationData
from tidy3d.components.eme.data.dataset import (
    EMECoefficientDataset,
    EMEFieldDataset,
    EMEModeSolverDataset,
    EMESMatrixDataset,
)
from tidy3d.components.eme.data.monitor_data import (
    EMECoefficientData,
    EMEFieldData,
    EMEModeSolverData,
)
from tidy3d.components.eme.data.sim_data import EMESimulationData
from tidy3d.components.eme.grid import (
    EMECompositeGrid,
    EMEExplicitGrid,
    EMEGrid,
    EMEModeSpec,
    EMEUniformGrid,
)
from tidy3d.components.eme.monitor import (
    EMECoefficientMonitor,
    EMEFieldMonitor,
    EMEModeSolverMonitor,
    EMEMonitor,
)

# EME
from tidy3d.components.eme.simulation import EMESimulation
from tidy3d.components.eme.sweep import EMEFreqSweep, EMELengthSweep, EMEModeSweep

# field projection
from tidy3d.components.field_projection import FieldProjector

# frequency conversion utilities
from tidy3d.components.frequencies import frequencies, wavelengths

# geometry
from tidy3d.components.geometry.base import Box, ClipOperation, Geometry, GeometryGroup, Transformed
from tidy3d.components.geometry.mesh import TriangleMesh
from tidy3d.components.geometry.polyslab import PolySlab
from tidy3d.components.geometry.primitives import Cylinder, Sphere
from tidy3d.components.grid.grid import Coords, Coords1D, FieldGrid, Grid, YeeGrid
from tidy3d.components.grid.grid_spec import (
    AutoGrid,
    CustomGrid,
    CustomGridBoundaries,
    GridSpec,
    UniformGrid,
)
from tidy3d.components.heat_charge.boundary import (
    ConvectionBC,
    CurrentBC,
    HeatBoundarySpec,
    HeatChargeBoundarySpec,
    HeatFluxBC,
    InsulatingBC,
    TemperatureBC,
    VoltageBC,
)
from tidy3d.components.heat_charge.charge_settings import (
    AugerRecombination,
    CaugheyThomasMobility,
    ChargeToleranceSpec,
    DCSpec,
    RadiativeRecombination,
    ShockleyReedHallRecombination,
    SlotboomNarrowingBandGap,
)
from tidy3d.components.heat_charge.grid import DistanceUnstructuredGrid, UniformUnstructuredGrid
from tidy3d.components.heat_charge.heat.simulation import HeatSimulation
from tidy3d.components.heat_charge.monitor import (
    StaticCapacitanceMonitor,
    StaticChargeCarrierMonitor,
    StaticVoltageMonitor,
    TemperatureMonitor,
)
from tidy3d.components.heat_charge.monitor_data import (
    StaticCapacitanceData,
    StaticFreeCarrierData,
    StaticPotentialData,
    StaticVoltageData,
    TemperatureData,
)
from tidy3d.components.heat_charge.sim_data import HeatChargeSimulationData, HeatSimulationData
from tidy3d.components.heat_charge.simulation import HeatChargeSimulation
from tidy3d.components.heat_charge.source import (
    HeatFromElectricSource,
    HeatSource,
    UniformHeatSource,
)

# heat
# heat
from tidy3d.components.heat_charge_spec import (
    FluidSpec,
    SolidSpec,
)

# lumped elements
from tidy3d.components.lumped_element import (
    AdmittanceNetwork,
    CoaxialLumpedResistor,
    LinearLumpedElement,
    LumpedElement,
    LumpedResistor,
    RectangularLumpedElement,
    RLCNetwork,
)
from tidy3d.components.materials.tcad.charge import (
    AbstractChargeMedium,
    SemiconductorMedium,
)

# medium
# for docs
from tidy3d.components.medium import (
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
    SkinDepthFitterParam,
    TwoPhotonAbsorption,
    medium_from_nk,
)

# modes
from tidy3d.components.mode import ModeSpec

# monitors
from tidy3d.components.monitor import (
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
from tidy3d.components.parameter_perturbation import (
    CustomChargePerturbation,
    CustomHeatPerturbation,
    IndexPerturbation,
    LinearChargePerturbation,
    LinearHeatPerturbation,
    ParameterPerturbation,
    PermittivityPerturbation,
)

# run time spec
from tidy3d.components.run_time_spec import RunTimeSpec

# scene
# scene
from tidy3d.components.scene import Scene

# simulation
from tidy3d.components.simulation import Simulation
from tidy3d.components.source.base import Source
from tidy3d.components.source.current import (
    CustomCurrentSource,
    PointDipole,
    UniformCurrentSource,
)
from tidy3d.components.source.field import (
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
from tidy3d.components.source.time import (
    ContinuousWave,
    CustomSourceTime,
    GaussianPulse,
    SourceTime,
)

# structures
from tidy3d.components.structure import MeshOverrideStructure, Structure

# subpixel
from tidy3d.components.subpixel_spec import (
    HeuristicPECStaircasing,
    PECConformal,
    PolarizedAveraging,
    Staircasing,
    SubpixelSpec,
    SurfaceImpedance,
    VolumetricAveraging,
)

# time modulation
from tidy3d.components.time_modulation import (
    ContinuousWaveTimeModulation,
    ModulationSpec,
    SpaceModulation,
    SpaceTimeModulation,
)
from tidy3d.components.transformation import RotationAroundAxis

# config
from tidy3d.config import config

# constants imported as `C_0 = td.C_0` or `td.constants.C_0`
from tidy3d.constants import C_0, EPSILON_0, ETA_0, HBAR, K_B, MU_0, Q_e, inf
from tidy3d.log import log, set_logging_console, set_logging_file

# material library dict imported as `from tidy3d import material_library`
# get material `mat` and variant `var` as `material_library[mat][var]`
from tidy3d.material_library.material_library import material_library
from tidy3d.material_library.parametric_materials import Graphene

# updater
from tidy3d.updater import Updater

# version
from tidy3d.version import __version__


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
    "Absorber",
    "AbsorberParams",
    "AbstractFieldProjectionData",
    "AbstractMedium",
    "AdmittanceNetwork",
    "AnisotropicMedium",
    "ApodizationSpec",
    "AstigmaticGaussianBeam",
    "AugerRecombination",
    "AutoGrid",
    "AxialRatioDataArray",
    "BlochBoundary",
    "Boundary",
    "BoundaryEdge",
    "BoundaryEdgeType",
    "BoundarySpec",
    "Box",
    "C_0",
    "CaugheyThomasMobility",
    "CellDataArray",
    "ChargeDataArray",
    "ChargeToleranceSpec",
    "ClipOperation",
    "CoaxialLumpedResistor",
    "AbstractChargeMedium",
    "ContinuousWave",
    "ContinuousWaveTimeModulation",
    "ConvectionBC",
    "Coords",
    "Coords1D",
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
    "DATA_TYPE_MAP",
    "DCCapacitanceDataArray",
    "DCIVCurveDataArray",
    "DCSpec",
    "Debye",
    "DefaultAbsorberParameters",
    "DefaultPMLParameters",
    "DefaultStablePMLParameters",
    "DiffractionData",
    "DiffractionDataArray",
    "DiffractionMonitor",
    "DirectivityData",
    "DirectivityDataArray",
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
    "EMESMatrixDataArray",
    "EMESMatrixDataset",
    "EMEScalarFieldDataArray",
    "EMEScalarModeFieldDataArray",
    "EMESimulation",
    "EMESimulationData",
    "EMESweepSpec",
    "EMELengthSweep",
    "EMEModeSweep",
    "EMEFreqSweep",
    "FixedAngleSpec",
    "FixedInPlaneKSpec",
    "EMEUniformGrid",
    "EPSILON_0",
    "ETA_0",
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
    "FluidSpec",
    "FluxData",
    "FluxDataArray",
    "FluxMonitor",
    "FluxTimeData",
    "FluxTimeDataArray",
    "FluxTimeMonitor",
    "FullyAnisotropicMedium",
    "GaussianBeam",
    "GaussianPulse",
    "Geometry",
    "GeometryGroup",
    "Graphene",
    "Grid",
    "GridSpec",
    "HBAR",
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
    "IndexPerturbation",
    "IndexedDataArray",
    "InsulatingBC",
    "AbstractChargeMedium",
    "K_B",
    "KerrNonlinearity",
    "LinearChargePerturbation",
    "LinearHeatPerturbation",
    "LinearLumpedElement",
    "Lorentz",
    "LossyMetalMedium",
    "LumpedElement",
    "LumpedResistor",
    "MU_0",
    "Medium",
    "Medium2D",
    "MediumMediumInterface",
    "MeshOverrideStructure",
    "ModeAmpsDataArray",
    "ModeData",
    "ModeIndexDataArray",
    "ModeMonitor",
    "ModeSolverData",
    "ModeSolverDataset",
    "ModeSolverMonitor",
    "ModeSource",
    "ModeSpec",
    "ModulationSpec",
    "Monitor",
    "NonlinearModel",
    "NonlinearSpec",
    "NonlinearSusceptibility",
    "PEC",
    "PEC2D",
    "PECBoundary",
    "PECConformal",
    "PECMedium",
    "PMCBoundary",
    "PML",
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
    "PointDataArray",
    "PointDipole",
    "PolarizedAveraging",
    "PoleResidue",
    "PolySlab",
    "Q_e",
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
    "SkinDepthFitterParam",
    "SlotboomNarrowingBandGap",
    "SolidSpec",
    "Source",
    "SourceTime",
    "SpaceModulation",
    "SpaceTimeModulation",
    "SpatialDataArray",
    "Sphere",
    "StablePML",
    "Staircasing",
    "StaticCapacitanceData",
    "StaticCapacitanceMonitor",
    "StaticChargeCarrierMonitor",
    "StaticFreeCarrierData",
    "StaticPotentialData",
    "StaticVoltageData",
    "StaticVoltageMonitor",
    "Structure",
    "StructureBoundary",
    "StructureSimulationBoundary",
    "StructureStructureInterface",
    "SubpixelSpec",
    "SurfaceImpedance",
    "TFSF",
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
    "Updater",
    "VoltageBC",
    "VolumetricAveraging",
    "YeeGrid",
    "__version__",
    "config",
    "frequencies",
    "inf",
    "log",
    "material_library",
    "medium_from_nk",
    "set_logging_console",
    "set_logging_file",
    "wavelengths",
]
