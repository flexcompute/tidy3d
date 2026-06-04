"""Data arrays for dipole emission study results."""

from __future__ import annotations

from tidy3d.components.data.data_array import (
    DATA_ARRAY_MAP,
    DIPOLE_EMISSION_INTENSITY_UNITS,
    DataArray,
)


class DipoleEmissionStudyDataArray(DataArray):
    """Angular radiation intensity per dipole moment squared.

    Example
    -------
    >>> import numpy as np
    >>> dipole_axis = ["x", "y", "z"]
    >>> polarization = ["p", "s"]
    >>> angle = np.arange(4)
    >>> f = np.linspace(1e14, 2e14, 3)
    >>> coords = dict(dipole_axis=dipole_axis, polarization=polarization, angle=angle, f=f)
    >>> values = np.random.random((len(dipole_axis), len(polarization), len(angle), len(f)))
    >>> data = DipoleEmissionStudyDataArray(values, coords=coords)
    """

    __slots__ = ()
    _dims = ("dipole_axis", "polarization", "angle", "f")
    _data_attrs = {
        "long_name": "angular radiation intensity per dipole moment squared",
        "units": DIPOLE_EMISSION_INTENSITY_UNITS,
    }


class DipoleEmissionStudyPositionDataArray(DataArray):
    """Position-resolved angular radiation intensity per dipole moment squared.

    Example
    -------
    >>> import numpy as np
    >>> index = np.arange(2)
    >>> dipole_axis = ["x", "y", "z"]
    >>> polarization = ["p", "s"]
    >>> angle = np.arange(4)
    >>> f = np.linspace(1e14, 2e14, 3)
    >>> coords = dict(
    ...     index=index, dipole_axis=dipole_axis, polarization=polarization, angle=angle, f=f
    ... )
    >>> values = np.random.random((len(index), len(dipole_axis), len(polarization), len(angle), len(f)))
    >>> data = DipoleEmissionStudyPositionDataArray(values, coords=coords)
    """

    __slots__ = ()
    _dims = ("index", "dipole_axis", "polarization", "angle", "f")
    _data_attrs = {
        "long_name": "position-resolved angular radiation intensity per dipole moment squared",
        "units": DIPOLE_EMISSION_INTENSITY_UNITS,
    }


DATA_ARRAY_MAP.update(
    {
        DipoleEmissionStudyDataArray.__name__: DipoleEmissionStudyDataArray,
        DipoleEmissionStudyPositionDataArray.__name__: DipoleEmissionStudyPositionDataArray,
    }
)
