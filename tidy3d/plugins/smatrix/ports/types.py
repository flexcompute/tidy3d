from __future__ import annotations

from tidy3d.components.data.data_array import (
    CurrentFreqDataArray,
    CurrentFreqModeDataArray,
    VoltageFreqDataArray,
    VoltageFreqModeDataArray,
)
from tidy3d.plugins.smatrix.ports.coaxial_lumped import CoaxialLumpedPort
from tidy3d.plugins.smatrix.ports.rectangular_lumped import LumpedPort
from tidy3d.plugins.smatrix.ports.wave import WavePort

LumpedPortType = LumpedPort | CoaxialLumpedPort
TerminalPortType = LumpedPortType | WavePort
PortVoltageType = VoltageFreqDataArray | VoltageFreqModeDataArray
PortCurrentType = CurrentFreqDataArray | CurrentFreqModeDataArray
