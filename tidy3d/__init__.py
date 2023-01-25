""" Tidy3d package imports"""
from concurrent.futures import ProcessPoolExecutor, process

from rich import pretty, traceback

# grid
from .components.grid.grid import Grid, Coords
from .components.grid.grid_spec import GridSpec, UniformGrid, CustomGrid, AutoGrid

# geometry
from .components.geometry import Box, Sphere, Cylinder, PolySlab, GeometryGroup

# medium
from .components.medium import Medium, PoleResidue, AnisotropicMedium, PEC, PECMedium
from .components.medium import Sellmeier, Debye, Drude, Lorentz
from .components.medium import CustomMedium

# structures
from .components.structure import Structure, MeshOverrideStructure

# modes
from .components.mode import ModeSpec

# apodization
from .components.apodization import ApodizationSpec

# sources
from .components.source import GaussianPulse, ContinuousWave
from .components.source import UniformCurrentSource, PlaneWave, ModeSource, PointDipole
from .components.source import GaussianBeam, AstigmaticGaussianBeam
from .components.source import CustomFieldSource

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
from .components.data.data_array import ScalarFieldTimeDataArray
from .components.data.data_array import ModeAmpsDataArray, ModeIndexDataArray
from .components.data.data_array import FluxDataArray, FluxTimeDataArray
from .components.data.data_array import FieldProjectionAngleDataArray
from .components.data.data_array import FieldProjectionCartesianDataArray
from .components.data.data_array import FieldProjectionKSpaceDataArray
from .components.data.data_array import DiffractionDataArray
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
from .constants import C_0, ETA_0, HBAR, EPSILON_0, MU_0, Q_e, inf

# material library dict imported as `from tidy3d import material_library`
# get material `mat` and variant `var` as `material_library[mat][var]`
from .material_library.material_library import material_library

# for docs
from .components.medium import AbstractMedium
from .components.geometry import Geometry
from .components.source import Source, SourceTime
from .components.monitor import Monitor
from .components.grid.grid import YeeGrid, FieldGrid, Coords1D

# logging
from .log import log, set_logging_file, WebError

# config
from .config import config

# version
from .version import __version__, PIP_NAME

# updater
from .updater import Updater

# warn if this was uploaded to the beta PyPI, also sets the pyPI repository that we upload to
if "beta" in PIP_NAME:
    log.warning(
        "This version of Tidy3D was pip installed from the 'tidy3d-beta' repository on PyPI. "
        "Future releases will be uploaded to the 'tidy3d' repository. "
        "From now on, please use 'pip install tidy3d' instead."
    )


def set_logging_level(level: str) -> None:
    """Raise a warning here instead of setting the logging level."""
    raise DeprecationWarning(
        "``set_logging_level`` no longer supported. "
        f"To set the logging level, call ``tidy3d.config.logging_level = {level}``."
    )


# make all stdout and errors pretty
# pretty.install()
# traceback.install()
log.info(f"Using client version: {__version__}")
