"""Imports from scattering matrix plugin."""

from __future__ import annotations

import warnings

from tidy3d.plugins.smatrix.component_modelers.base import (
    AbstractComponentModeler,
)
from tidy3d.plugins.smatrix.component_modelers.modal import ComponentModeler
from tidy3d.plugins.smatrix.data.data_array import (
    ModalPortDataArray,
)
from tidy3d.plugins.smatrix.data.modal import ComponentModelerData, PortSimulationData
from tidy3d.plugins.smatrix.ports.modal import Port

# Note: do not import run helpers here to avoid importing web at package import time.

# Instantiate on plugin import till we unite with toplevel
warnings.filterwarnings(
    "once",
    message="ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
    category=FutureWarning,
)


__all__ = [
    "AbstractComponentModeler",
    "ComponentModeler",
    "ComponentModelerData",
    "ModalPortDataArray",
    "Port",
    "PortSimulationData",
]
