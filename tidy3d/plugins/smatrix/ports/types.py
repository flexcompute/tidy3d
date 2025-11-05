from __future__ import annotations

from typing import Union

from tidy3d.components.data.data_array import (
    CurrentFreqDataArray,
    CurrentFreqModeDataArray,
    VoltageFreqDataArray,
    VoltageFreqModeDataArray,
)
from tidy3d.plugins.smatrix.ports.coaxial_lumped import CoaxialLumpedPort
from tidy3d.plugins.smatrix.ports.modal import Port
from tidy3d.plugins.smatrix.ports.rectangular_lumped import LumpedPort
from tidy3d.plugins.smatrix.ports.wave import WavePort

LumpedPortType = Union[LumpedPort, CoaxialLumpedPort]
TerminalPortType = Union[LumpedPortType, WavePort]
PortType = Union[Port, TerminalPortType]
PortVoltageType = Union[VoltageFreqDataArray, VoltageFreqModeDataArray]
PortCurrentType = Union[CurrentFreqDataArray, CurrentFreqModeDataArray]
