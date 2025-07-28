"""Imports from scattering matrix plugin."""

from __future__ import annotations

import warnings

from tidy3d.plugins.smatrix.component_modelers.base import (
    AbstractComponentModeler,
)
from tidy3d.plugins.smatrix.component_modelers.modal import ComponentModeler
from tidy3d.plugins.smatrix.component_modelers.terminal import TerminalComponentModeler
from tidy3d.plugins.smatrix.data.data_array import (
    ModalPortDataArray,
    PortDataArray,
    TerminalPortDataArray,
)
from tidy3d.plugins.smatrix.data.modal import ComponentModelerData, PortSimulationData
from tidy3d.plugins.smatrix.data.terminal import MicrowaveSMatrixData, TerminalComponentModelerData
from tidy3d.plugins.smatrix.ports.coaxial_lumped import CoaxialLumpedPort
from tidy3d.plugins.smatrix.ports.modal import Port
from tidy3d.plugins.smatrix.ports.rectangular_lumped import LumpedPort
from tidy3d.plugins.smatrix.ports.wave import WavePort
from tidy3d.plugins.smatrix.run import compose_modeler_data, create_batch, run

# Instantiate on plugin import till we unite with toplevel
warnings.filterwarnings(
    "once",
    message="ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
    category=FutureWarning,
)


__all__ = [
    "AbstractComponentModeler",
    "CoaxialLumpedPort",
    "ComponentModeler",
    "ComponentModelerData",
    "ComponentModelerDataLumpedPort",
    "LumpedPort",
    "MicrowaveSMatrixData",
    "ModalPortDataArray",
    "Port",
    "PortDataArray",
    "PortSimulationData",
    "TerminalComponentModeler",
    "TerminalComponentModelerData",
    "TerminalPortDataArray",
    "WavePort",
    "compose_modeler_data",
    "create_batch",
    "run",
]
