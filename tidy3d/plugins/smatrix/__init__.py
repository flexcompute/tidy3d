"""Imports from scattering matrix plugin."""

from __future__ import annotations

import warnings

from tidy3d.em.microwave.component_modelers.modal import (
    AbstractComponentModeler,
    ComponentModeler,
    ModalPortDataArray,
)
from tidy3d.em.microwave.component_modelers.terminal import TerminalComponentModeler
from tidy3d.em.microwave.data.terminal import PortDataArray, TerminalPortDataArray
from tidy3d.em.microwave.ports.coaxial_lumped import CoaxialLumpedPort
from tidy3d.em.microwave.ports.modal import Port
from tidy3d.em.microwave.ports.rectangular_lumped import LumpedPort
from tidy3d.em.microwave.ports.wave import WavePort

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
    "LumpedPort",
    "ModalPortDataArray",
    "Port",
    "PortDataArray",
    "TerminalComponentModeler",
    "TerminalPortDataArray",
    "WavePort",
]
