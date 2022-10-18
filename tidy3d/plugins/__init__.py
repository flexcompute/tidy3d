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

# adjoint
from .adjoint.web import run
from .adjoint.components.geometry import JaxBox, JaxPolySlab
from .adjoint.components.medium import JaxMedium, JaxAnisotropicMedium, JaxCustomMedium
from .adjoint.components.structure import JaxStructure
from .adjoint.components.simulation import JaxSimulation
from .adjoint.components.data.sim_data import JaxSimulationData
from .adjoint.components.data.monitor_data import JaxModeData
from .adjoint.components.data.dataset import JaxPermittivityDataset
from .adjoint.components.data.data_array import JaxDataArray
