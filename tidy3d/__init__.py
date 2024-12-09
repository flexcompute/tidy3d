"""Tidy3d package imports"""

# grid
# apodization
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
    AxialRatioDataArray,
    CellDataArray,
    ChargeDataArray,
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
from .components.data.dataset import (
    FieldDataset,
    FieldTimeDataset,
    ModeSolverDataset,
    PermittivityDataset,
    TetrahedralGridDataset,
    TriangularGridDataset,
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
from .components.eme.sweep import EMEFreqSweep, EMELengthSweep, EMEModeSweep

# field projection
from .components.field_projection import FieldProjector

# frequency conversion utilities
from .components.frequencies import frequencies, wavelengths

# geometry
from .components.geometry.base import Box, ClipOperation, Geometry, GeometryGroup, Transformed
from .components.geometry.mesh import TriangleMesh
from .components.geometry.polyslab import PolySlab
from .components.geometry.primitives import Cylinder, Sphere
from .components.grid.grid import Coords, Coords1D, FieldGrid, Grid, YeeGrid
from .components.grid.grid_spec import (
    AutoGrid,
    CustomGrid,
    CustomGridBoundaries,
    GridSpec,
    UniformGrid,
)
from .components.heat_charge.boundary import (
    ConvectionBC,
    CurrentBC,
    HeatBoundarySpec,
    HeatChargeBoundarySpec,
    HeatFluxBC,
    InsulatingBC,
    TemperatureBC,
    VoltageBC,
)
from .components.heat_charge.grid import DistanceUnstructuredGrid, UniformUnstructuredGrid
from .components.heat_charge.heat.simulation import HeatSimulation
from .components.heat_charge.monitor import TemperatureMonitor, VoltageMonitor
from .components.heat_charge.monitor_data import TemperatureData, VoltageData
from .components.heat_charge.sim_data import HeatChargeSimulationData, HeatSimulationData
from .components.heat_charge.simulation import HeatChargeSimulation
from .components.heat_charge.source import HeatFromElectricSource, HeatSource, UniformHeatSource

# heat
# heat
from .components.heat_charge_spec import ConductorSpec, FluidSpec, InsulatorSpec, SolidSpec

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
from .components.mode import ModeSpec

# monitors
from .components.monitor import (
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
    "Grid",
    "Coords",
    "GridSpec",
    "UniformGrid",
    "CustomGrid",
    "AutoGrid",
    "CustomGridBoundaries",
    "Box",
    "Sphere",
    "Cylinder",
    "PolySlab",
    "GeometryGroup",
    "ClipOperation",
    "Transformed",
    "TriangleMesh",
    "Medium",
    "PoleResidue",
    "AnisotropicMedium",
    "PEC",
    "PECMedium",
    "Medium2D",
    "PEC2D",
    "Sellmeier",
    "Debye",
    "Drude",
    "Lorentz",
    "CustomMedium",
    "CustomPoleResidue",
    "CustomSellmeier",
    "FullyAnisotropicMedium",
    "CustomLorentz",
    "CustomDrude",
    "CustomDebye",
    "CustomAnisotropicMedium",
    "LossyMetalMedium",
    "SkinDepthFitterParam",
    "RotationAroundAxis",
    "PerturbationMedium",
    "PerturbationPoleResidue",
    "ParameterPerturbation",
    "LinearHeatPerturbation",
    "CustomHeatPerturbation",
    "LinearChargePerturbation",
    "CustomChargePerturbation",
    "PermittivityPerturbation",
    "IndexPerturbation",
    "NonlinearSpec",
    "NonlinearModel",
    "NonlinearSusceptibility",
    "TwoPhotonAbsorption",
    "KerrNonlinearity",
    "Structure",
    "MeshOverrideStructure",
    "ModeSpec",
    "ApodizationSpec",
    "GaussianPulse",
    "ContinuousWave",
    "CustomSourceTime",
    "UniformCurrentSource",
    "PlaneWave",
    "ModeSource",
    "PointDipole",
    "GaussianBeam",
    "AstigmaticGaussianBeam",
    "CustomFieldSource",
    "TFSF",
    "CustomCurrentSource",
    "FieldMonitor",
    "FieldTimeMonitor",
    "FluxMonitor",
    "FluxTimeMonitor",
    "ModeMonitor",
    "ModeSolverMonitor",
    "PermittivityMonitor",
    "FieldProjectionAngleMonitor",
    "FieldProjectionCartesianMonitor",
    "FieldProjectionKSpaceMonitor",
    "FieldProjectionSurface",
    "DiffractionMonitor",
    "DirectivityMonitor",
    "RunTimeSpec",
    "Simulation",
    "FieldProjector",
    "ScalarFieldDataArray",
    "ScalarModeFieldDataArray",
    "ScalarModeFieldCylindricalDataArray",
    "ScalarFieldTimeDataArray",
    "SpatialDataArray",
    "ModeAmpsDataArray",
    "ModeIndexDataArray",
    "FluxDataArray",
    "FluxTimeDataArray",
    "FieldProjectionAngleDataArray",
    "FieldProjectionCartesianDataArray",
    "FieldProjectionKSpaceDataArray",
    "DiffractionDataArray",
    "DirectivityDataArray",
    "AxialRatioDataArray",
    "HeatDataArray",
    "ChargeDataArray",
    "FieldDataset",
    "FieldTimeDataset",
    "PermittivityDataset",
    "ModeSolverDataset",
    "FieldData",
    "FieldTimeData",
    "PermittivityData",
    "FluxData",
    "FluxTimeData",
    "ModeData",
    "ModeSolverData",
    "AbstractFieldProjectionData",
    "FieldProjectionAngleData",
    "FieldProjectionCartesianData",
    "FieldProjectionKSpaceData",
    "DiffractionData",
    "DirectivityData",
    "SimulationData",
    "DATA_TYPE_MAP",
    "BoundarySpec",
    "Boundary",
    "BoundaryEdge",
    "BoundaryEdgeType",
    "BlochBoundary",
    "Periodic",
    "PECBoundary",
    "PMCBoundary",
    "PML",
    "StablePML",
    "Absorber",
    "PMLParams",
    "AbsorberParams",
    "PMLTypes",
    "DefaultPMLParameters",
    "DefaultStablePMLParameters",
    "DefaultAbsorberParameters",
    "C_0",
    "ETA_0",
    "HBAR",
    "EPSILON_0",
    "MU_0",
    "Q_e",
    "K_B",
    "inf",
    "frequencies",
    "wavelengths",
    "material_library",
    "Graphene",
    "AbstractMedium",
    "Geometry",
    "Source",
    "SourceTime",
    "Monitor",
    "YeeGrid",
    "FieldGrid",
    "Coords1D",
    "log",
    "set_logging_file",
    "set_logging_console",
    "config",
    "__version__",
    "Updater",
    "AdmittanceNetwork",
    "CoaxialLumpedResistor",
    "LinearLumpedElement",
    "LumpedElement",
    "LumpedResistor",
    "RectangularLumpedElement",
    "RLCNetwork",
    "Scene",
    "StructureStructureInterface",
    "StructureBoundary",
    "MediumMediumInterface",
    "StructureSimulationBoundary",
    "SimulationBoundary",
    "FluidSpec",
    "SolidSpec",
    "ConductorSpec",
    "InsulatorSpec",
    "HeatSimulation",
    "HeatSimulationData",
    "TemperatureBC",
    "ConvectionBC",
    "HeatFluxBC",
    "HeatBoundarySpec",
    "VoltageBC",
    "CurrentBC",
    "InsulatingBC",
    "UniformHeatSource",
    "HeatSource",
    "HeatFromElectricSource",
    "UniformUnstructuredGrid",
    "DistanceUnstructuredGrid",
    "TemperatureData",
    "TemperatureMonitor",
    "HeatChargeSimulation",
    "HeatChargeSimulationData",
    "VoltageData",
    "HeatChargeBoundarySpec",
    "VoltageMonitor",
    "SpaceTimeModulation",
    "SpaceModulation",
    "ContinuousWaveTimeModulation",
    "ModulationSpec",
    "PointDataArray",
    "CellDataArray",
    "IndexedDataArray",
    "TriangularGridDataset",
    "TetrahedralGridDataset",
    "medium_from_nk",
    "SubpixelSpec",
    "Staircasing",
    "VolumetricAveraging",
    "PolarizedAveraging",
    "HeuristicPECStaircasing",
    "PECConformal",
    "SurfaceImpedance",
    "EMESimulation",
    "EMESimulationData",
    "EMEMonitor",
    "EMEModeSolverMonitor",
    "EMEFieldMonitor",
    "EMESMatrixDataArray",
    "EMEFieldDataset",
    "EMECoefficientDataset",
    "EMESMatrixDataset",
    "EMEModeSolverData",
    "EMEFieldData",
    "EMECoefficientData",
    "EMECoefficientMonitor",
    "EMEModeSpec",
    "EMEGrid",
    "EMEUniformGrid",
    "EMECompositeGrid",
    "EMEExplicitGrid",
    "EMEScalarFieldDataArray",
    "EMEScalarModeFieldDataArray",
    "EMEModeIndexDataArray",
    "EMECoefficientDataArray",
    "EMEModeSolverDataset",
    "EMESweepSpec",
    "EMELengthSweep",
    "EMEModeSweep",
    "EMEFreqSweep",
    "FixedAngleSpec",
    "FixedInPlaneKSpec",
]
