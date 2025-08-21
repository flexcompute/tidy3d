from __future__ import annotations

import typing
from abc import ABC

import pydantic.v1 as pd

import tidy3d as td
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.plugins.microwave.path_integrals import AbstractAxesRH


class Port(Tidy3dBaseModel, ABC):
    """Specifies a port in the scattering matrix."""

    name: str = pd.Field(
        ...,
        title="Name",
        description="Unique name for the port.",
        min_length=1,
    )


# IMPLEMENT ALL PORTS, schema only


class AbstractTerminalPort(Port, ABC):
    pass


class WavePort(AbstractTerminalPort, td.Box):
    pass


class AbstractLumpedPort(AbstractTerminalPort, ABC):
    pass


class CoaxialLumpedPort(AbstractLumpedPort, AbstractAxesRH):
    pass


class LumpedPort(AbstractLumpedPort, td.Box):
    pass


class ModalPort(Port, td.Box):
    pass


# types

PortType = typing.Union[ModalPort, WavePort, LumpedPort, CoaxialLumpedPort]
ModalPortType = ModalPort
TerminalPortType = typing.Union[WavePort, LumpedPort, CoaxialLumpedPort]
