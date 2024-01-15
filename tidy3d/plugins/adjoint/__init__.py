"""Imports for adjoint plugin."""

# import the jax version of tidy3d components
try:
    from .components.geometry import JaxBox, JaxPolySlab, JaxGeometryGroup
    from .components.medium import JaxMedium, JaxAnisotropicMedium, JaxCustomMedium, JaxPoleResidue
    from .components.structure import (
        JaxStructure,
        JaxStructureStaticGeometry,
        JaxStructureStaticMedium,
    )
    from .components.simulation import JaxSimulation
    from .components.data.sim_data import JaxSimulationData
    from .components.data.monitor_data import JaxModeData
    from .components.data.dataset import JaxPermittivityDataset
    from .components.data.data_array import JaxDataArray
except ImportError as e:
    raise ImportError(
        "The 'jax' package is required for adjoint plugin. We were not able to import it. "
        "To get the appropriate packages for your system, install tidy3d using '[jax]' option, "
        "for example: $pip install 'tidy3d[jax]'."
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
    "JaxPoleResidue",
    "JaxStructure",
    "JaxStructureStaticMedium",
    "JaxStructureStaticGeometry",
    "JaxSimulation",
    "JaxSimulationData",
    "JaxModeData",
    "JaxPermittivityDataset",
    "JaxDataArray",
    "run",
    "run_async",
]
