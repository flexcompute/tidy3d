"""RF (terminal) scattering-matrix plugin public API."""

from __future__ import annotations

from tidy3d.plugins.rf.component_modelers.terminal import TerminalComponentModeler
from tidy3d.plugins.rf.data.data_array import PortDataArray, TerminalPortDataArray
from tidy3d.plugins.rf.data.terminal import (
    MicrowaveSMatrixData,
    TerminalComponentModelerData,
)
from tidy3d.plugins.rf.ports.coaxial_lumped import CoaxialLumpedPort
from tidy3d.plugins.rf.ports.rectangular_lumped import LumpedPort
from tidy3d.plugins.rf.ports.wave import WavePort
from tidy3d.plugins.rf.utils import (
    ab_to_s,
    check_port_impedance_sign,
    compute_F,
    compute_port_VI,
    compute_power_delivered_by_port,
    compute_power_wave_amplitudes,
    s_to_z,
)

__all__ = [
    "CoaxialLumpedPort",
    "LumpedPort",
    "MicrowaveSMatrixData",
    "PortDataArray",
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
