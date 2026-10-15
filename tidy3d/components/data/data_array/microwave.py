"""Voltage, current, and impedance data arrays and integral result helpers."""

from __future__ import annotations

import numpy as np  # noqa: F401 - used by doctest examples

from tidy3d.constants import (
    AMP,
    OHM,
    VOLT,
)

from .base import DataArray
from .spectral import (
    FreqDataArray,
    FreqModeDataArray,
    FreqModeModeDataArray,
    FreqModeTerminalDataArray,
    FreqTerminalDataArray,
    FreqTerminalModeDataArray,
    FreqTerminalTerminalDataArray,
    ModeDataArray,
    TerminalDataArray,
    TimeDataArray,
)


class VoltageArray(DataArray):
    # Always set __slots__ = () to avoid xarray warnings
    __slots__ = ()
    _data_attrs = {"units": VOLT, "long_name": "voltage"}


class CurrentArray(DataArray):
    # Always set __slots__ = () to avoid xarray warnings
    __slots__ = ()
    _data_attrs = {"units": AMP, "long_name": "current"}


class ImpedanceArray(DataArray):
    # Always set __slots__ = () to avoid xarray warnings
    __slots__ = ()
    _data_attrs = {"units": OHM, "long_name": "impedance"}


# Voltage arrays
class VoltageFreqDataArray(VoltageArray, FreqDataArray):
    """Voltage data array in frequency domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9, 4e9]
    >>> coords = dict(f=f)
    >>> data = np.random.random(3) + 1j * np.random.random(3)
    >>> vfd = VoltageFreqDataArray(data, coords=coords)
    """

    __slots__ = ()


class VoltageTimeDataArray(VoltageArray, TimeDataArray):
    """Voltage data array in time domain.

    Example
    -------
    >>> import numpy as np
    >>> t = [0, 1e-9, 2e-9, 3e-9]
    >>> coords = dict(t=t)
    >>> data = np.sin(2 * np.pi * 1e9 * np.array(t))
    >>> vtd = VoltageTimeDataArray(data, coords=coords)
    """

    __slots__ = ()


class VoltageFreqModeDataArray(VoltageArray, FreqModeDataArray):
    """Voltage data array in frequency-mode domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9]
    >>> mode_index = [0, 1]
    >>> coords = dict(f=f, mode_index=mode_index)
    >>> data = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
    >>> vfmd = VoltageFreqModeDataArray(data, coords=coords)
    """

    __slots__ = ()


class VoltageFreqTerminalDataArray(VoltageArray, FreqTerminalDataArray):
    """Voltage data array in frequency-terminal domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9]
    >>> terminal_label = ["t0", "t1"]
    >>> coords = dict(f=f, terminal_label=terminal_label)
    >>> data = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
    >>> vftd = VoltageFreqTerminalDataArray(data, coords=coords)
    """

    __slots__ = ()


class VoltageFreqTerminalModeDataArray(VoltageArray, FreqTerminalModeDataArray):
    """Voltage transformation matrix data array from modes to terminals in frequency domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9]
    >>> terminal_label = ["t0", "t1"]
    >>> mode_index = [0, 1]
    >>> coords = dict(f=f, terminal_label=terminal_label, mode_index=mode_index)
    >>> data = np.random.random((2, 2, 2)) + 1j * np.random.random((2, 2, 2))
    >>> vtransform = VoltageFreqTerminalModeDataArray(data, coords=coords)
    """

    __slots__ = ()


class VoltageFreqModeTerminalDataArray(VoltageArray, FreqModeTerminalDataArray):
    """Inverse voltage transformation matrix data array from terminals to modes in frequency domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9]
    >>> mode_index = [0, 1]
    >>> terminal_label = ["t0", "t1"]
    >>> coords = dict(f=f, mode_index=mode_index, terminal_label=terminal_label)
    >>> data = np.random.random((2, 2, 2)) + 1j * np.random.random((2, 2, 2))
    >>> vtransform_inv = VoltageFreqModeTerminalDataArray(data, coords=coords)
    """

    __slots__ = ()


# Current arrays
class CurrentFreqDataArray(CurrentArray, FreqDataArray):
    """Current data array in frequency domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9, 4e9]
    >>> coords = dict(f=f)
    >>> data = np.random.random(3) + 1j * np.random.random(3)
    >>> cfd = CurrentFreqDataArray(data, coords=coords)
    """

    __slots__ = ()


class CurrentTimeDataArray(CurrentArray, TimeDataArray):
    """Current data array in time domain.

    Example
    -------
    >>> import numpy as np
    >>> t = [0, 1e-9, 2e-9, 3e-9]
    >>> coords = dict(t=t)
    >>> data = np.cos(2 * np.pi * 1e9 * np.array(t))
    >>> ctd = CurrentTimeDataArray(data, coords=coords)
    """

    __slots__ = ()


class CurrentFreqModeDataArray(CurrentArray, FreqModeDataArray):
    """Current data array in frequency-mode domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9]
    >>> mode_index = [0, 1]
    >>> coords = dict(f=f, mode_index=mode_index)
    >>> data = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
    >>> cfmd = CurrentFreqModeDataArray(data, coords=coords)
    """

    __slots__ = ()


class CurrentFreqTerminalDataArray(CurrentArray, FreqTerminalDataArray):
    """Current data array in frequency-terminal domain.
    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9]
    >>> terminal_label = ["t0", "t1", "t2", "t3", "t4"]
    >>> coords = dict(f=f, terminal_label=terminal_label)
    >>> data = np.random.random((2, 5)) + 1j * np.random.random((2, 5))
    >>> cftd = CurrentFreqTerminalDataArray(data, coords=coords)
    """

    __slots__ = ()


class CurrentFreqTerminalModeDataArray(CurrentArray, FreqTerminalModeDataArray):
    """Current transformation matrix data array from modes to terminals in frequency domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9]
    >>> mode_index = [0, 1]
    >>> terminal_label = ["t0", "t1"]
    >>> coords = dict(f=f, terminal_label=terminal_label, mode_index=mode_index)
    >>> data = np.random.random((2, 2, 2)) + 1j * np.random.random((2, 2, 2))
    >>> itransform = CurrentFreqTerminalModeDataArray(data, coords=coords)
    """

    __slots__ = ()


# Impedance arrays
class ImpedanceModeDataArray(ImpedanceArray, ModeDataArray):
    """Impedance data array in mode index domain.
    Example
    -------
    >>> mode_index = np.arange(4)
    >>> coords = dict(mode_index=mode_index)
    >>> data = ImpedanceModeDataArray((1+1j) * np.random.random(4), coords=coords)
    """

    __slots__ = ()


class ImpedanceTerminalDataArray(ImpedanceArray, TerminalDataArray):
    """Impedance data array in terminal index domain.
    Example
    -------
    >>> terminal_label = ["t0", "t1", "t2", "t3", "t4"]
    >>> coords = dict(terminal_label=terminal_label)
    >>> data = ImpedanceTerminalDataArray((1+1j) * np.random.random(5), coords=coords)
    """

    __slots__ = ()


class ImpedanceFreqDataArray(ImpedanceArray, FreqDataArray):
    """Impedance data array in frequency domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9, 4e9]
    >>> coords = dict(f=f)
    >>> data = 50.0 + 1j * np.random.random(3)
    >>> zfd = ImpedanceFreqDataArray(data, coords=coords)
    """

    __slots__ = ()


class ImpedanceTimeDataArray(ImpedanceArray, TimeDataArray):
    """Impedance data array in time domain.

    Example
    -------
    >>> import numpy as np
    >>> t = [0, 1e-9, 2e-9, 3e-9]
    >>> coords = dict(t=t)
    >>> data = 50.0 * np.ones_like(t)
    >>> ztd = ImpedanceTimeDataArray(data, coords=coords)
    """

    __slots__ = ()


class ImpedanceFreqModeDataArray(ImpedanceArray, FreqModeDataArray):
    """Impedance data array in frequency-mode domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9]
    >>> mode_index = [0, 1]
    >>> coords = dict(f=f, mode_index=mode_index)
    >>> data = 50.0 + 10.0 * np.random.random((2, 2))
    >>> zfmd = ImpedanceFreqModeDataArray(data, coords=coords)
    """

    __slots__ = ()


class ImpedanceFreqModeModeDataArray(ImpedanceArray, FreqModeModeDataArray):
    """Impedance matrix data array between modes in frequency domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9]
    >>> mode_index_out = np.arange(2)
    >>> mode_index_in = np.arange(2)
    >>> coords = dict(f=f, mode_index_out=mode_index_out, mode_index_in=mode_index_in)
    >>> data = ImpedanceFreqModeModeDataArray(50.0 + 10.0 * np.random.random((2, 2, 2)), coords=coords)
    >>> zfmmd = data
    """

    __slots__ = ()


class ImpedanceFreqTerminalTerminalDataArray(ImpedanceArray, FreqTerminalTerminalDataArray):
    """Impedance matrix data array between terminals in frequency domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9]
    >>> terminal_label_out = ["t0", "t1"]
    >>> terminal_label_in = ["t0", "t1"]
    >>> coords = dict(f=f, terminal_label_out=terminal_label_out, terminal_label_in=terminal_label_in)
    >>> data = ImpedanceFreqTerminalTerminalDataArray(50.0 + 10.0 * np.random.random((2, 2, 2)), coords=coords)
    >>> zfttd = data
    """

    __slots__ = ()


def _make_base_result_data_array(result: DataArray) -> IntegralResultType:
    """Helper for creating the proper base result type."""
    cls = FreqDataArray
    if "t" in result.coords:
        cls = TimeDataArray
    if "f" in result.coords and "mode_index" in result.coords:
        cls = FreqModeDataArray
    if (
        "f" in result.coords
        and "terminal_label" in result.coords
        and "mode_index" not in result.coords
    ):
        cls = FreqTerminalDataArray
    if "f" in result.coords and "terminal_label" in result.coords and "mode_index" in result.coords:
        cls = FreqTerminalModeDataArray
    return cls._assign_data_attrs(cls(data=result.data, coords=result.coords))


def _make_voltage_data_array(result: DataArray) -> VoltageIntegralResultType:
    """Helper for creating the proper voltage array type."""
    cls = VoltageFreqDataArray
    if "t" in result.coords:
        cls = VoltageTimeDataArray
    if "f" in result.coords and "mode_index" in result.coords:
        cls = VoltageFreqModeDataArray
    if (
        "f" in result.coords
        and "terminal_label" in result.coords
        and "mode_index" not in result.coords
    ):
        cls = VoltageFreqTerminalDataArray
    if "f" in result.coords and "terminal_label" in result.coords and "mode_index" in result.coords:
        cls = VoltageFreqTerminalModeDataArray
    return cls._assign_data_attrs(cls(data=result.data, coords=result.coords))


def _make_current_data_array(result: DataArray) -> CurrentIntegralResultType:
    """Helper for creating the proper current array type."""
    cls = CurrentFreqDataArray
    if "t" in result.coords:
        cls = CurrentTimeDataArray
    if "f" in result.coords and "mode_index" in result.coords:
        cls = CurrentFreqModeDataArray
    if (
        "f" in result.coords
        and "terminal_label" in result.coords
        and "mode_index" not in result.coords
    ):
        cls = CurrentFreqTerminalDataArray
    if "f" in result.coords and "terminal_label" in result.coords and "mode_index" in result.coords:
        cls = CurrentFreqTerminalModeDataArray
    return cls._assign_data_attrs(cls(data=result.data, coords=result.coords))


def _make_impedance_data_array(result: DataArray) -> ImpedanceResultType:
    """Helper for creating the proper impedance array type."""
    cls = ImpedanceFreqDataArray
    if "t" in result.coords:
        cls = ImpedanceTimeDataArray
    if "f" in result.coords and "mode_index" in result.coords:
        cls = ImpedanceFreqModeDataArray
    if (
        "f" in result.coords
        and "terminal_label_out" in result.coords
        and "terminal_label_in" in result.coords
    ):
        cls = ImpedanceFreqTerminalTerminalDataArray
    return cls._assign_data_attrs(cls(data=result.data, coords=result.coords))


IntegralResultType = (
    FreqDataArray
    | FreqModeDataArray
    | FreqTerminalDataArray
    | FreqTerminalModeDataArray
    | TimeDataArray
)


VoltageIntegralResultType = (
    VoltageFreqDataArray
    | VoltageFreqModeDataArray
    | VoltageFreqTerminalDataArray
    | VoltageTimeDataArray
    | VoltageFreqTerminalModeDataArray
)


CurrentIntegralResultType = (
    CurrentFreqDataArray
    | CurrentFreqModeDataArray
    | CurrentFreqTerminalDataArray
    | CurrentTimeDataArray
    | CurrentFreqTerminalModeDataArray
)


ImpedanceResultType = (
    ImpedanceFreqDataArray
    | ImpedanceFreqModeDataArray
    | ImpedanceTimeDataArray
    | ImpedanceFreqTerminalTerminalDataArray
)
