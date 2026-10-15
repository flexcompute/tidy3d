"""Frequency, time, mode, and terminal data arrays."""

from __future__ import annotations

import numpy as np  # noqa: F401 - used by doctest examples

from .base import DataArray


class FreqDataArray(DataArray):
    """Frequency-domain array.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> fd = FreqDataArray((1+1j) * np.random.random((2,)), coords=dict(f=f))
    """

    __slots__ = ()
    _dims = ("f",)


class ModeDataArray(DataArray):
    """Mode index data array.
    Example
    -------
    >>> mode_index = np.arange(4)
    >>> coords = dict(mode_index=mode_index)
    >>> data = ModeDataArray((1+1j) * np.random.random(4), coords=coords)
    """

    __slots__ = ()
    _dims = ("mode_index",)


class TerminalDataArray(DataArray):
    """Terminal index data array.
    Example
    -------
    >>> terminal_label = ["t0", "t1", "t2", "t3", "t4"]
    >>> coords = dict(terminal_label=terminal_label)
    >>> data = TerminalDataArray((1+1j) * np.random.random(5), coords=coords)
    """

    __slots__ = ()
    _dims = ("terminal_label",)


class FreqModeDataArray(DataArray):
    """Array over frequency and mode index.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> coords = dict(f=f, mode_index=mode_index)
    >>> fd = FreqModeDataArray((1+1j) * np.random.random((2, 5)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "mode_index")


class FreqTerminalDataArray(DataArray):
    """Array over frequency and terminal index.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> terminal_label = ["t0", "t1", "t2", "t3", "t4"]
    >>> coords = dict(f=f, terminal_label=terminal_label)
    >>> fd = FreqTerminalDataArray((1+1j) * np.random.random((2, 5)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "terminal_label")


class FreqModeModeDataArray(DataArray):
    """Array over frequency, mode index, and mode index.
    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index_out = np.arange(5)
    >>> mode_index_in = np.arange(5)
    >>> coords = dict(f=f, mode_index_out=mode_index_out, mode_index_in=mode_index_in)
    >>> fd = FreqModeModeDataArray((1+1j) * np.random.random((2, 5, 5)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "mode_index_out", "mode_index_in")


class FreqTerminalModeDataArray(DataArray):
    """Array over frequency, terminal index, and mode index.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> terminal_label = ["t0", "t1"]
    >>> coords = dict(f=f, terminal_label=terminal_label, mode_index=mode_index)
    >>> fd = FreqTerminalModeDataArray((1+1j) * np.random.random((2, 2, 5)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "terminal_label", "mode_index")


class FreqModeTerminalDataArray(DataArray):
    """Array over frequency, mode index, and terminal index.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> terminal_label = ["t0", "t1"]
    >>> coords = dict(f=f, mode_index=mode_index, terminal_label=terminal_label)
    >>> fd = FreqModeTerminalDataArray((1+1j) * np.random.random((2, 5, 2)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "mode_index", "terminal_label")


class FreqTerminalTerminalDataArray(DataArray):
    """Array over frequency, terminal index, and terminal index.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> terminal_label_out = ["t0", "t1"]
    >>> terminal_label_in = ["t0", "t1"]
    >>> coords = dict(f=f, terminal_label_out=terminal_label_out, terminal_label_in=terminal_label_in)
    >>> fd = FreqTerminalTerminalDataArray((1+1j) * np.random.random((2, 2, 2)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "terminal_label_out", "terminal_label_in")


class TimeDataArray(DataArray):
    """Time-domain array.

    Example
    -------
    >>> t = [0, 1e-12, 2e-12]
    >>> td = TimeDataArray((1+1j) * np.random.random((3,)), coords=dict(t=t))
    """

    __slots__ = ()
    _dims = ("t",)


class MixedModeDataArray(DataArray):
    """Scalar property associated with mode pairs

    Example
    -------
    >>> f = [1e14, 2e14, 3e14]
    >>> mode_index_0 = np.arange(4)
    >>> mode_index_1 = np.arange(2)
    >>> coords = dict(f=f, mode_index_0=mode_index_0, mode_index_1=mode_index_1)
    >>> data = MixedModeDataArray((1+1j) * np.random.random((3, 4, 2)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "mode_index_0", "mode_index_1")
