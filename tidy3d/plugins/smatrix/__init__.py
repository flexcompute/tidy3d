"""Imports from scattering matrix plugin."""

import warnings

from .component_modelers.modal import AbstractComponentModeler, ComponentModeler, ModalPortDataArray
from .component_modelers.terminal import TerminalComponentModeler
from .data.terminal import PortDataArray, TerminalPortDataArray
from .ports.coaxial_lumped import CoaxialLumpedPort
from .ports.modal import Port
from .ports.rectangular_lumped import LumpedPort
from .ports.wave import WavePort

# Instantiate on plugin import till we unite with toplevel
warnings.filterwarnings(
    "once",
    message="ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
    category=FutureWarning,
)


__all__ = [
    "AbstractComponentModeler",
    "ComponentModeler",
    "Port",
    "ModalPortDataArray",
    "TerminalComponentModeler",
    "CoaxialLumpedPort",
    "LumpedPort",
    "WavePort",
    "TerminalPortDataArray",
    "PortDataArray",
]
