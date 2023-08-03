"""Imports for adjoint plugin."""

# import the jax version of tidy3d components
try:
    from .components.data.data_array import JaxDataArray
    from .components.data.dataset import JaxPermittivityDataset
    from .components.data.monitor_data import JaxModeData
    from .components.data.sim_data import JaxSimulationData
    from .components.geometry import JaxBox, JaxGeometryGroup, JaxPolySlab
    from .components.medium import JaxAnisotropicMedium, JaxCustomMedium, JaxMedium
    from .components.simulation import JaxSimulation
    from .components.structure import JaxStructure
except ImportError as e:
    raise ImportError(
        "The 'jax' package is required for adjoint plugin and not installed. "
        "To get the appropriate packages, install tidy3d using '[jax]' option, for example: "
        "$pip install 'tidy3d[jax]'."
    ) from e

try:
    from .web import run, run_async
except ImportError:
    pass

__all__ = [
    "JaxBox",
    "JaxPolySlab",
    "JaxGeometryGroup",
    "JaxMedium",
    "JaxAnisotropicMedium",
    "JaxCustomMedium",
    "JaxStructure",
    "JaxSimulation",
    "JaxSimulationData",
    "JaxModeData",
    "JaxPermittivityDataset",
    "JaxDataArray",
    "run",
    "run_async",
]
