"""Imports from scattering matrix plugin."""

from __future__ import annotations

import warnings

from tidy3d.plugins.smatrix.component_modelers.base import (
    AbstractComponentModeler,
)
from tidy3d.plugins.smatrix.component_modelers.modal import ModalComponentModeler
from tidy3d.plugins.smatrix.component_modelers.terminal import (
    DirectivityMonitorSpec,
    ModelerLowFrequencySmoothingSpec,
    TerminalComponentModeler,
)
from tidy3d.plugins.smatrix.component_modelers.types import ComponentModelerType
from tidy3d.plugins.smatrix.data.data_array import (
    ModalPortDataArray,
    PortDataArray,
    TerminalPortDataArray,
)
from tidy3d.plugins.smatrix.data.modal import ModalComponentModelerData
from tidy3d.plugins.smatrix.data.terminal import (
    MicrowaveSMatrixData,
    TerminalComponentModelerData,
)
from tidy3d.plugins.smatrix.data.types import ComponentModelerDataType
from tidy3d.plugins.smatrix.ports.coaxial_lumped import CoaxialLumpedPort
from tidy3d.plugins.smatrix.ports.modal import Port
from tidy3d.plugins.smatrix.ports.rectangular_lumped import LumpedPort
from tidy3d.plugins.smatrix.ports.wave import WavePort

# Instantiate on plugin import till we unite with toplevel
warnings.filterwarnings(
    "once",
    message="RF simulations and functionality will require new license requirements in an upcoming release. All RF-specific classes are now available within the sub-package 'tidy3d.rf'.",
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
    "DirectivityMonitorSpec",
    "LumpedPort",
    "MicrowaveSMatrixData",
    "ModalComponentModeler",
    "ModalComponentModelerData",
    "ModalPortDataArray",
    "ModelerLowFrequencySmoothingSpec",
    "Port",
    "PortDataArray",
    "TerminalComponentModeler",
    "TerminalComponentModelerData",
    "TerminalPortDataArray",
    "WavePort",
]
