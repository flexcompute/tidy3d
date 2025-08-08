from __future__ import annotations

from typing import Union

from tidy3d.plugins.rf.ports.coaxial_lumped import CoaxialLumpedPort
from tidy3d.plugins.rf.ports.rectangular_lumped import LumpedPort
from tidy3d.plugins.rf.ports.wave import WavePort

LumpedPortType = Union[LumpedPort, CoaxialLumpedPort]
TerminalPortType = Union[LumpedPortType, WavePort]
PortReferenceType = Union[str, TerminalPortType]
