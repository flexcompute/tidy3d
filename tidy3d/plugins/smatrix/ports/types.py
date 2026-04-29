from __future__ import annotations

from tidy3d.components.data.data_array import (
    CurrentFreqDataArray,
    CurrentFreqModeDataArray,
    VoltageFreqDataArray,
    VoltageFreqModeDataArray,
)
from tidy3d.plugins.smatrix.ports.coaxial_lumped import CoaxialLumpedPort
from tidy3d.plugins.smatrix.ports.modal import Port
from tidy3d.plugins.smatrix.ports.rectangular_lumped import LumpedPort
from tidy3d.plugins.smatrix.ports.wave import TerminalWavePort, WavePort

LumpedPortType = LumpedPort | CoaxialLumpedPort
WavePortType = WavePort | TerminalWavePort
TerminalPortType = LumpedPortType | WavePortType
PortType = Port | TerminalPortType
PortVoltageType = VoltageFreqDataArray | VoltageFreqModeDataArray
PortCurrentType = CurrentFreqDataArray | CurrentFreqModeDataArray
