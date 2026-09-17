"""Data arrays associated with modal component-modeler results."""

from __future__ import annotations

from tidy3d._rf_migration import missing_rf_attribute
from tidy3d.components.data.data_array import DataArray


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
    >>> port_data = ModalPortDataArray((1 + 1j) * np.random.random((2, 2, 2, 2, 1)), coords=coords)
    """

    __slots__ = ()
    _dims = ("port_out", "mode_index_out", "port_in", "mode_index_in", "f")
    _data_attrs = {"long_name": "modal port matrix element"}


def __getattr__(name: str) -> None:
    if name in {"PortDataArray", "PortNameDataArray", "TerminalPortDataArray"}:
        missing_rf_attribute(__name__, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
