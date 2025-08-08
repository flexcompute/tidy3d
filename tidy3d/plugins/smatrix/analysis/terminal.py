"""Deprecation shims for RF terminal analysis; moved to `tidy3d.plugins.rf`."""

from __future__ import annotations

import warnings

from tidy3d.components.data.sim_data import SimulationData as _SimulationData
from tidy3d.plugins.rf.analysis.terminal import (
    compute_power_wave_amplitudes_at_each_port as _compute_power_wave_amplitudes_at_each_port,
)
from tidy3d.plugins.rf.analysis.terminal import (
    port_reference_impedances as _port_reference_impedances,
)
from tidy3d.plugins.rf.analysis.terminal import (
    terminal_construct_smatrix as _terminal_construct_smatrix,
)
from tidy3d.plugins.rf.component_modelers.terminal import (
    TerminalComponentModeler as _TerminalComponentModeler,
)
from tidy3d.plugins.rf.data.data_array import (
    PortDataArray as _PortDataArray,
)
from tidy3d.plugins.rf.data.data_array import (
    TerminalPortDataArray as _TerminalPortDataArray,
)

__all__ = [
    "compute_power_wave_amplitudes_at_each_port",
    "port_reference_impedances",
    "terminal_construct_smatrix",
]

warnings.warn(
    "tidy3d.plugins.smatrix.analysis.terminal is deprecated; use tidy3d.plugins.rf.analysis.terminal",
    DeprecationWarning,
    stacklevel=2,
)

TerminalPortDataArray = _TerminalPortDataArray
PortDataArray = _PortDataArray
TerminalComponentModeler = _TerminalComponentModeler
SimulationData = _SimulationData

terminal_construct_smatrix = _terminal_construct_smatrix
port_reference_impedances = _port_reference_impedances
compute_power_wave_amplitudes_at_each_port = _compute_power_wave_amplitudes_at_each_port
