""" Tidy3d package imports"""
from concurrent.futures import ProcessPoolExecutor, process

from rich import pretty, traceback

# grid
from .components import Grid, Coords, GridSpec, UniformGrid, CustomGrid, AutoGrid

# geometry
from .components import Box, Sphere, Cylinder, PolySlab, GeometryGroup

# medium
from .components import Medium, PoleResidue, AnisotropicMedium, PEC, PECMedium
from .components import Sellmeier, Debye, Drude, Lorentz

# structures
from .components import Structure

# modes
from .components import ModeSpec

# sources
from .components import GaussianPulse, ContinuousWave
from .components import UniformCurrentSource, PlaneWave, ModeSource, PointDipole
from .components import GaussianBeam, AstigmaticGaussianBeam

# monitors
from .components import FieldMonitor, FieldTimeMonitor, FluxMonitor, FluxTimeMonitor
from .components import ModeMonitor, ModeSolverMonitor, PermittivityMonitor
from .components import Near2FarAngleMonitor, Near2FarCartesianMonitor, Near2FarKSpaceMonitor

# simulation
from .components import Simulation

# data
from .components import ScalarFieldDataArray, ScalarModeFieldDataArray, ScalarFieldTimeDataArray
from .components import ModeAmpsDataArray, ModeIndexDataArray
from .components import FluxDataArray, FluxTimeDataArray
from .components import Near2FarAngleDataArray, Near2FarCartesianDataArray, Near2FarKSpaceDataArray
from .components import FieldData, FieldTimeData, PermittivityData
from .components import FluxData, FluxTimeData
from .components import ModeData, ModeSolverData
from .components import AbstractNear2FarData
from .components import Near2FarAngleData, Near2FarCartesianData, Near2FarKSpaceData
from .components import Near2FarSurface, RadiationVectors
from .components import SimulationData
from .components import DATA_TYPE_MAP

# boundary
from .components import BoundarySpec, Boundary, BoundaryEdge, BoundaryEdgeType
from .components import BlochBoundary, Symmetry, Periodic, PECBoundary, PMCBoundary
from .components import PML, StablePML, Absorber, PMLParams, AbsorberParams, PMLTypes
from .components import DefaultPMLParameters, DefaultStablePMLParameters, DefaultAbsorberParameters

# constants imported as `C_0 = td.C_0` or `td.constants.C_0`
from .constants import C_0, ETA_0, HBAR, EPSILON_0, MU_0, Q_e, inf

# material library dict imported as `from tidy3d import material_library`
# get material `mat` and variant `var` as `material_library[mat][var]`
from .material_library import material_library

# for docs
from .components.medium import AbstractMedium
from .components.geometry import Geometry
from .components.source import Source, SourceTime
from .components.monitor import Monitor
from .components.grid import YeeGrid, FieldGrid, Coords1D

# logging
from .log import log, set_logging_file

# config
from .config import config

# version
from .version import __version__

# updater
from .updater import Updater


def set_logging_level(level: str) -> None:
    """Raise a warning here instead of setting the logging level."""
    raise DeprecationWarning(
        "``set_logging_level`` no longer supported. "
        f"To set the logging level, call ``tidy3d.config.logging_level = {level}``."
    )


# make all stdout and errors pretty
pretty.install()
# traceback.install()
