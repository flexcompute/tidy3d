""" Tidy3d package imports"""
__version__ = "0.0.0"

# import component as `from tidy3d import Simulation`
from .components import PMLLayer
from .components import Box, Sphere, Cylinder, PolySlab
from .components import Structure
from .components import Medium, PoleResidue, Sellmeier, Debye, Lorentz
from .components import nk_to_eps_complex, nk_to_eps_sigma, eps_complex_to_nk
from .components import nk_to_medium, eps_sigma_to_eps_complex

from .components import GaussianPulse
from .components import VolumeSource, PlaneWave, ModeSource, GaussianBeam
from .components import uniform_freq_sampler, uniform_time_sampler
from .components import (
    FieldMonitor,
    FieldTimeMonitor,
    FluxMonitor,
    FluxTimeMonitor,
    ModeMonitor,
    PermittivityMonitor,
)
from .components import Mode

from .components import Simulation

from .components import SimulationData, FieldData, FluxData, ModeData, FieldTimeData, FluxTimeData
from .components import monitor_data_map

from .constants import inf

# plugins imported as `from tidy3d.plugins.dispersion_fitter import *` for now
from . import plugins

# web API imported as `from tidy3d.web import *`
from . import web

# material library dict imported as `from tidy3d import material_library`
# specific material imported as `from tidy3d.material_library import SiC_xxx`
from .material_library import material_library


# import pkg_resources
# try:
#     __version__ = pkg_resources.get_distribution("tidy3d").version
# except ImportError:  # pragma: nocover
#     # Local copy, not installed with setuptools
#     __version__ = "0.0.0"
