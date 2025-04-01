"""Class and custom data array for representing a scattering matrix port based on waveguide modes."""

from __future__ import annotations

from pydantic import Field

from tidy3d.components.data.data_array import DataArray
from tidy3d.components.geometry.base import Box
from tidy3d.components.mode_spec import ModeSpec
from tidy3d.components.types import Direction


class ModalPortDataArray(DataArray):
    """Port parameter matrix elements for modal ports.

    Example
    -------
    >>> import numpy as np
    >>> ports_in = ['port1', 'port2']
    >>> ports_out = ['port1', 'port2']
    >>> mode_index_in = [0, 1]
    >>> mode_index_out = [0, 1]
    >>> f = [2e14]
    >>> coords = dict(
    ...     port_in=ports_in,
    ...     port_out=ports_out,
    ...     mode_index_in=mode_index_in,
    ...     mode_index_out=mode_index_out,
    ...     f=f
    ... )
    >>> fd = ModalPortDataArray((1 + 1j) * np.random.random((2, 2, 2, 2, 1)), coords=coords)
    """

    __slots__ = ()
    _dims = ("port_out", "mode_index_out", "port_in", "mode_index_in", "f")
    _data_attrs = {"long_name": "modal port matrix element"}


class Port(Box):
    """Specifies a port in the scattering matrix."""

    direction: Direction = Field(
        title="Direction",
        description="'+' or '-', defining which direction is considered 'input'.",
    )
    mode_spec: ModeSpec = Field(
        default_factory=ModeSpec,
        title="Mode Specification",
        description="Specifies how the mode solver will solve for the modes of the port.",
    )
    name: str = Field(
        title="Name",
        description="Unique name for the port.",
        min_length=1,
    )
