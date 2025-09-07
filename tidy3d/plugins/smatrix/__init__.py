"""Imports from scattering matrix plugin."""

from __future__ import annotations

import warnings

from tidy3d.plugins.smatrix.component_modelers.base import (
    AbstractComponentModeler,
)
from tidy3d.plugins.smatrix.component_modelers.modal import ModalComponentModeler
from tidy3d.plugins.smatrix.component_modelers.terminal import TerminalComponentModeler
from tidy3d.plugins.smatrix.component_modelers.types import ComponentModelerType
from tidy3d.plugins.smatrix.data.data_array import (
    ModalPortDataArray,
    PortDataArray,
    TerminalPortDataArray,
)
from tidy3d.plugins.smatrix.data.modal import ModalComponentModelerData
from tidy3d.plugins.smatrix.data.terminal import MicrowaveSMatrixData, TerminalComponentModelerData
from tidy3d.plugins.smatrix.data.types import ComponentModelerDataType
from tidy3d.plugins.smatrix.ports.coaxial_lumped import CoaxialLumpedPort
from tidy3d.plugins.smatrix.ports.modal import Port
from tidy3d.plugins.smatrix.ports.rectangular_lumped import LumpedPort
from tidy3d.plugins.smatrix.ports.wave import WavePort

# Instantiate on plugin import till we unite with toplevel
warnings.filterwarnings(
    "once",
    message="ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
    category=FutureWarning,
)

# Legacy type to support previous flows
ComponentModeler = ModalComponentModeler

__all__ = [
    "AbstractComponentModeler",
    "CoaxialLumpedPort",
    "ComponentModeler",
    "ComponentModelerDataType",
    "ComponentModelerType",
    "LumpedPort",
    "MicrowaveSMatrixData",
    "ModalComponentModeler",
    "ModalComponentModelerData",
    "ModalPortDataArray",
    "Port",
    "PortDataArray",
    "TerminalComponentModeler",
    "TerminalComponentModelerData",
    "TerminalPortDataArray",
    "WavePort",
]
