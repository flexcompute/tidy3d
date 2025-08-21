from __future__ import annotations

import typing

from tidy3d.plugins.smatrix.models.port import CoaxialLumpedPort as _CoaxialLumpedPort
from tidy3d.plugins.smatrix.models.port import LumpedPort as _LumpedPort
from tidy3d.plugins.smatrix.models.port import ModalPort as _ModalPort
from tidy3d.plugins.smatrix.models.port import WavePort as _WavePort

# implement the methods here


class ModalPort(_ModalPort):
    pass


class WavePort(_WavePort):
    pass


class LumpedPort(_LumpedPort):
    pass


class CoaxialLumpedPort(_CoaxialLumpedPort):
    pass


# types
PortType = typing.Union[ModalPort, WavePort, LumpedPort, CoaxialLumpedPort]
ModalPortType = ModalPort
TerminalPortType = typing.Union[WavePort, LumpedPort, CoaxialLumpedPort]
