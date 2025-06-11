"""Imports for adjoint plugin."""
# ruff: noqa: E402

# import the jax version of tidy3d components
from __future__ import annotations

from textwrap import dedent

from tidy3d.log import log

_DOC_URL = "https://github.com/flexcompute/tidy3d/blob/develop/tidy3d/plugins/autograd/README.md"

_MSG = dedent(
    f"""
    The 'adjoint' plugin (legacy JAX-based adjoint plugin) was deprecated in Tidy3D '2.7.0' and will be disabled as of '2.9.0'.

    Migrate to the native autograd workflow:
        import tidy3d as td
        import autograd.numpy as np
        from autograd import grad

    It uses standard 'td.' objects, has fewer dependencies, and offers a smoother optimization experience.
    Full guide: {_DOC_URL}
    """
).strip()

log.warning(_MSG)

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
