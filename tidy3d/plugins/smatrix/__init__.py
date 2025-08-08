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

# --- Temporary deprecation shims for RF API (one minor release) ---
# Re-export RF items from `tidy3d.plugins.rf` and warn once when this package is imported.
_RF_AVAILABLE = False
try:
    from tidy3d.plugins.rf import (
        CoaxialLumpedPort,
        LumpedPort,
        MicrowaveSMatrixData,
        PortDataArray,
        TerminalComponentModeler,
        TerminalComponentModelerData,
        TerminalPortDataArray,
        WavePort,
        ab_to_s,
        check_port_impedance_sign,
        compute_F,
        compute_port_VI,
        compute_power_delivered_by_port,
        compute_power_wave_amplitudes,
        s_to_z,
    )

    _RF_AVAILABLE = True

    warnings.filterwarnings("once", category=DeprecationWarning)
    warnings.warn(
        "RF APIs moved: import from 'tidy3d.plugins.rf' (e.g., 'from tidy3d.plugins.rf import TerminalComponentModeler')\n"
        "This re-export from 'tidy3d.plugins.smatrix' is deprecated and will be removed in a future minor release.",
        DeprecationWarning,
        stacklevel=2,
    )
except Exception:
    _RF_AVAILABLE = False

# Note: do not import run helpers here to avoid importing web at package import time.

# Instantiate on plugin import till we unite with toplevel
warnings.filterwarnings(
    "once",
    message="ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
    category=FutureWarning,
)


if _RF_AVAILABLE:
    __all__ = [
        "AbstractComponentModeler",
        "CoaxialLumpedPort",
        "ComponentModeler",
        "ComponentModelerData",
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
        "ab_to_s",
        "check_port_impedance_sign",
        "compute_F",
        "compute_port_VI",
        "compute_power_delivered_by_port",
        "compute_power_wave_amplitudes",
        "s_to_z",
    ]
else:
    __all__ = [
        "AbstractComponentModeler",
        "ComponentModeler",
        "ComponentModelerData",
        "ModalPortDataArray",
        "Port",
        "PortSimulationData",
    ]
