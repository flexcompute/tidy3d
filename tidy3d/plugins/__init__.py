"""External plugins that have tidy3d as dependency and add functionality."""

from .dispersion.fit import DispersionFitter
from .dispersion.fit_web import StableDispersionFitter, AdvancedFitterParam
from .mode.mode_solver import ModeSolver
from .near2far.near2far import Near2Far, Near2FarSurface
from .smatrix.smatrix import ComponentModeler, Port
from .plotly.app import SimulationDataApp
from .plotly.simulation import SimulationPlotly
