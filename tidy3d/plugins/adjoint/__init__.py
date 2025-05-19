"""Imports for adjoint plugin."""

# import the jax version of tidy3d components
from __future__ import annotations

try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError as e:
    raise ImportError(
        "The 'jax' package is required for adjoint plugin. We were not able to import it. "
        "To get the appropriate packages for your system, install tidy3d using '[jax]' option, "
        "for example: $pip install 'tidy3d[jax]'."
    ) from e

from .components.data.data_array import JaxDataArray
from .components.data.dataset import JaxPermittivityDataset
from .components.data.monitor_data import JaxModeData
from .components.data.sim_data import JaxSimulationData
from .components.geometry import JaxBox, JaxComplexPolySlab, JaxGeometryGroup, JaxPolySlab
from .components.medium import JaxAnisotropicMedium, JaxCustomMedium, JaxMedium
from .components.simulation import JaxSimulation
from .components.structure import (
    JaxStructure,
    JaxStructureStaticGeometry,
    JaxStructureStaticMedium,
)
from .web import run, run_async

__all__ = [
    "JaxAnisotropicMedium",
    "JaxBox",
    "JaxComplexPolySlab",
    "JaxCustomMedium",
    "JaxDataArray",
    "JaxGeometryGroup",
    "JaxMedium",
    "JaxModeData",
    "JaxPermittivityDataset",
    "JaxPolySlab",
    "JaxSimulation",
    "JaxSimulationData",
    "JaxStructure",
    "JaxStructureStaticGeometry",
    "JaxStructureStaticMedium",
    "run",
    "run_async",
]
