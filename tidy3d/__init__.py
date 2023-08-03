""" Tidy3d package imports"""

# grid
# apodization
from .components.apodization import ApodizationSpec

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
    ChargeDataArray,
    DiffractionDataArray,
    FieldProjectionAngleDataArray,
    FieldProjectionCartesianDataArray,
    FieldProjectionKSpaceDataArray,
    FluxDataArray,
    FluxTimeDataArray,
    HeatDataArray,
    ModeAmpsDataArray,
    ModeIndexDataArray,
    ScalarFieldDataArray,
    ScalarFieldTimeDataArray,
    ScalarModeFieldDataArray,
    SpatialDataArray,
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

# field projection
from .components.field_projection import FieldProjector

# geometry
from .components.geometry import (
    Box,
    Cylinder,
    Geometry,
    GeometryGroup,
    PolySlab,
    Sphere,
    TriangleMesh,
)
from .components.grid.grid import Coords, Coords1D, FieldGrid, Grid, YeeGrid
from .components.grid.grid_spec import AutoGrid, CustomGrid, GridSpec, UniformGrid

# medium
# for docs
from .components.medium import (
    PEC,
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
    Lorentz,
    Medium,
    Medium2D,
    PECMedium,
    PerturbationMedium,
    PerturbationPoleResidue,
    PoleResidue,
    Sellmeier,
)

# modes
from .components.mode import ModeSpec

# monitors
from .components.monitor import (
    DiffractionMonitor,
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
    LinearChargePerturbation,
    LinearHeatPerturbation,
    ParameterPerturbation,
)

# simulation
from .components.simulation import Simulation

# sources
from .components.source import (
    TFSF,
    AstigmaticGaussianBeam,
    ContinuousWave,
    CustomCurrentSource,
    CustomFieldSource,
    CustomSourceTime,
    GaussianBeam,
    GaussianPulse,
    ModeSource,
    PlaneWave,
    PointDipole,
    Source,
    SourceTime,
    UniformCurrentSource,
)

# structures
from .components.structure import MeshOverrideStructure, Structure
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

__all__ = [
    "Grid",
    "Coords",
    "GridSpec",
    "UniformGrid",
    "CustomGrid",
    "AutoGrid",
    "Box",
    "Sphere",
    "Cylinder",
    "PolySlab",
    "GeometryGroup",
    "TriangleMesh",
    "Medium",
    "PoleResidue",
    "AnisotropicMedium",
    "PEC",
    "PECMedium",
    "Medium2D",
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
    "RotationAroundAxis",
    "PerturbationMedium",
    "PerturbationPoleResidue",
    "ParameterPerturbation",
    "LinearHeatPerturbation",
    "CustomHeatPerturbation",
    "LinearChargePerturbation",
    "CustomChargePerturbation",
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
    "Simulation",
    "FieldProjector",
    "ScalarFieldDataArray",
    "ScalarModeFieldDataArray",
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
]
