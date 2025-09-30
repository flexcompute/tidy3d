"""Storing data associated with results from the TerminalComponentModeler"""

from __future__ import annotations

from tidy3d.components.data.data_array import DataArray


class PortDataArray(DataArray):
    """Array of values over dimensions of frequency and port name.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9, 4e9]
    >>> ports = ["port1", "port2"]
    >>> coords = dict(f=f, port=ports)
    >>> data = (1+1j) * np.random.random((3, 2))
    >>> port_data = PortDataArray(data, coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "port")


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


class TerminalPortDataArray(DataArray):
    """Port parameter matrix elements for terminal-based ports.

    Example
    -------
    >>> import numpy as np
    >>> ports_in = ["port1", "port2"]
    >>> ports_out = ["port1", "port2"]
    >>> f = [2e14]
    >>> coords = dict(f=f, port_out=ports_out, port_in=ports_in)
    >>> data = (1+1j) * np.random.random((1, 2, 2))
    >>> port_data = TerminalPortDataArray(data, coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "port_out", "port_in")
    _data_attrs = {"long_name": "terminal-based port matrix element"}


class PortNameDataArray(DataArray):
    """Array of values indexed by port name.

    Example
    -------
    >>> import numpy as np
    >>> port_names = ["port1", "port2"]
    >>> coords = dict(port_name=port_names)
    >>> data = (1 + 1j) * np.random.random((2,))
    >>> port_data = PortNameDataArray(data, coords=coords)
    """

    __slots__ = ()
    _dims = "port_name"
