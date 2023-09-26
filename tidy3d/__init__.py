""" Tidy3d package imports"""

# grid
from .components.grid.grid import Grid, Coords
from .components.grid.grid_spec import GridSpec, UniformGrid, CustomGrid, AutoGrid

# geometry
from .components.geometry.base import Box, ClipOperation, GeometryGroup
from .components.geometry.primitives import Sphere, Cylinder
from .components.geometry.mesh import TriangleMesh
from .components.geometry.polyslab import PolySlab

# medium
from .components.medium import Medium, PoleResidue, AnisotropicMedium, PEC, PECMedium, Medium2D
from .components.medium import Sellmeier, Debye, Drude, Lorentz
from .components.medium import CustomMedium, CustomPoleResidue
from .components.medium import CustomSellmeier, FullyAnisotropicMedium
from .components.medium import CustomLorentz, CustomDrude, CustomDebye, CustomAnisotropicMedium
from .components.medium import NonlinearSusceptibility
from .components.transformation import RotationAroundAxis
from .components.medium import PerturbationMedium, PerturbationPoleResidue
from .components.parameter_perturbation import ParameterPerturbation
from .components.parameter_perturbation import LinearHeatPerturbation, CustomHeatPerturbation
from .components.parameter_perturbation import LinearChargePerturbation, CustomChargePerturbation

# structures
from .components.structure import Structure, MeshOverrideStructure

# modes
from .components.mode import ModeSpec

# apodization
from .components.apodization import ApodizationSpec

# sources
from .components.source import GaussianPulse, ContinuousWave, CustomSourceTime
from .components.source import UniformCurrentSource, PlaneWave, ModeSource, PointDipole
from .components.source import GaussianBeam, AstigmaticGaussianBeam
from .components.source import CustomFieldSource, TFSF, CustomCurrentSource

# monitors
from .components.monitor import FieldMonitor, FieldTimeMonitor, FluxMonitor, FluxTimeMonitor
from .components.monitor import ModeMonitor, ModeSolverMonitor, PermittivityMonitor
from .components.monitor import FieldProjectionAngleMonitor, FieldProjectionCartesianMonitor
from .components.monitor import FieldProjectionKSpaceMonitor, FieldProjectionSurface
from .components.monitor import DiffractionMonitor

# simulation
from .components.simulation import Simulation

# field projection

from .components.field_projection import FieldProjector

# data
from .components.data.data_array import ScalarFieldDataArray, ScalarModeFieldDataArray
from .components.data.data_array import ScalarFieldTimeDataArray, SpatialDataArray
from .components.data.data_array import ModeAmpsDataArray, ModeIndexDataArray
from .components.data.data_array import FluxDataArray, FluxTimeDataArray
from .components.data.data_array import FieldProjectionAngleDataArray
from .components.data.data_array import FieldProjectionCartesianDataArray
from .components.data.data_array import FieldProjectionKSpaceDataArray
from .components.data.data_array import DiffractionDataArray
from .components.data.data_array import HeatDataArray, ChargeDataArray
from .components.data.dataset import FieldDataset, FieldTimeDataset
from .components.data.dataset import PermittivityDataset, ModeSolverDataset
from .components.data.monitor_data import FieldData, FieldTimeData, PermittivityData
from .components.data.monitor_data import FluxData, FluxTimeData
from .components.data.monitor_data import ModeData, ModeSolverData
from .components.data.monitor_data import AbstractFieldProjectionData
from .components.data.monitor_data import FieldProjectionAngleData, FieldProjectionCartesianData
from .components.data.monitor_data import FieldProjectionKSpaceData
from .components.data.monitor_data import DiffractionData
from .components.data.sim_data import SimulationData
from .components.data.sim_data import DATA_TYPE_MAP

# boundary
from .components.boundary import BoundarySpec, Boundary, BoundaryEdge, BoundaryEdgeType
from .components.boundary import BlochBoundary, Periodic, PECBoundary, PMCBoundary
from .components.boundary import PML, StablePML, Absorber, PMLParams, AbsorberParams, PMLTypes
from .components.boundary import DefaultPMLParameters, DefaultStablePMLParameters
from .components.boundary import DefaultAbsorberParameters

# constants imported as `C_0 = td.C_0` or `td.constants.C_0`
from .constants import C_0, ETA_0, HBAR, EPSILON_0, MU_0, Q_e, K_B, inf

# material library dict imported as `from tidy3d import material_library`
# get material `mat` and variant `var` as `material_library[mat][var]`
from .material_library.material_library import material_library
from .material_library.parametric_materials import Graphene

# for docs
from .components.medium import AbstractMedium, NonlinearSpec
from .components.geometry.base import Geometry
from .components.source import Source, SourceTime
from .components.monitor import Monitor
from .components.grid.grid import YeeGrid, FieldGrid, Coords1D

from .log import log, set_logging_file, set_logging_console

# config
from .config import config

# version
from .version import __version__

# updater
from .updater import Updater

# scene
from .components.scene import Scene

# boundary placement for other solvers
from .components.bc_placement import StructureStructureInterface, StructureBoundary
from .components.bc_placement import MediumMediumInterface
from .components.bc_placement import StructureSimulationBoundary
from .components.bc_placement import SimulationBoundary

# heat
from .components.heat_spec import FluidSpec, SolidSpec
from .components.heat.simulation import HeatSimulation
from .components.heat.data.sim_data import HeatSimulationData
from .components.heat.data.monitor_data import TemperatureData
from .components.heat.boundary import TemperatureBC, ConvectionBC, HeatFluxBC, HeatBoundarySpec
from .components.heat.source import UniformHeatSource
from .components.heat.monitor import TemperatureMonitor
from .components.heat.grid import UniformUnstructuredGrid, DistanceUnstructuredGrid


def set_logging_level(level: str) -> None:
    """Raise a warning here instead of setting the logging level."""
    raise DeprecationWarning(
        "``set_logging_level`` no longer supported. "
        f"To set the logging level, call ``tidy3d.config.logging_level = {level}``."
    )


log.info(f"Using client version: {__version__}")

ClipOperation.update_forward_refs()
GeometryGroup.update_forward_refs()

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
    "NonlinearSpec",
    "NonlinearSusceptibility",
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
    "Scene",
    "StructureStructureInterface",
    "StructureBoundary",
    "MediumMediumInterface",
    "StructureSimulationBoundary",
    "SimulationBoundary",
    "FluidSpec",
    "SolidSpec",
    "HeatSimulation",
    "HeatSimulationData",
    "TemperatureBC",
    "ConvectionBC",
    "HeatFluxBC",
    "HeatBoundarySpec",
    "UniformHeatSource",
    "UniformUnstructuredGrid",
    "DistanceUnstructuredGrid",
    "TemperatureData",
    "TemperatureMonitor",
]
