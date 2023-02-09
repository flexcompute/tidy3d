"""External plugins that have tidy3d as dependency and add functionality."""

# dispersion fitter
from .dispersion.fit import DispersionFitter
from .dispersion.fit_web import StableDispersionFitter, AdvancedFitterParam

# mode solver
from .mode.mode_solver import ModeSolver, ModeSolverData

# scattering matrix
from .smatrix.smatrix import ComponentModeler, Port

# resonance finder
from .resonance.resonance import ResonanceFinder

# adjoint removed because requires jax, import directly through tidy3d.plugins.adjoint
