"""Fields on structured Cartesian or cylindrical grids, plus spherical angle sets."""

from __future__ import annotations

import numpy as np  # noqa: F401 - used by doctest examples
from flexcompute.data.arrays import spatial as _core

from .abstract_spatial import AbstractSpatialDataArray
from .base import DataArray


class SpatialDataArray(AbstractSpatialDataArray, _core.SpatialDataArray):
    """Spatial distribution.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> coords = dict(x=x, y=y, z=z)
    >>> fd = SpatialDataArray((1+1j) * np.random.random((2,3,4)), coords=coords)
    """

    __slots__ = ()


class ScalarFieldDataArray(AbstractSpatialDataArray, _core.ScalarFieldDataArray):
    """Spatial distribution in the frequency-domain.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> fd = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    """

    __slots__ = ()


class ScalarFieldTimeDataArray(AbstractSpatialDataArray, _core.ScalarFieldTimeDataArray):
    """Spatial distribution in the time-domain.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(x=x, y=y, z=z, t=t)
    >>> fd = ScalarFieldTimeDataArray(np.random.random((2,3,4,3)), coords=coords)
    """

    __slots__ = ()


class ScalarModeFieldDataArray(AbstractSpatialDataArray, _core.ScalarModeFieldDataArray):
    """Spatial distribution of a mode in frequency-domain as a function of mode index.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> coords = dict(x=x, y=y, z=z, f=f, mode_index=mode_index)
    >>> fd = ScalarModeFieldDataArray((1+1j) * np.random.random((2,3,4,2,5)), coords=coords)
    """

    __slots__ = ()


class ScalarModeFieldCylindricalDataArray(
    AbstractSpatialDataArray, _core.ScalarModeFieldCylindricalDataArray
):
    """Spatial distribution of a mode in frequency-domain as a function of mode index.

    Example
    -------
    >>> rho = [1,2]
    >>> theta = [2,3,4]
    >>> axial = [3,4,5,6]
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> coords = dict(rho=rho, theta=theta, axial=axial, f=f, mode_index=mode_index)
    >>> fd = ScalarModeFieldCylindricalDataArray((1+1j) * np.random.random((2,3,4,2,5)), coords=coords)
    """

    __slots__ = ()


class ScalarTerminalFieldDataArray(AbstractSpatialDataArray, _core.ScalarTerminalFieldDataArray):
    """Spatial distribution of a terminal field in frequency-domain as a function of terminal index.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> f = [2e14, 3e14]
    >>> terminal_label = ["t0", "t1", "t2"]
    >>> coords = dict(x=x, y=y, z=z, f=f, terminal_label=terminal_label)
    >>> fd = ScalarTerminalFieldDataArray((1+1j) * np.random.random((2,3,4,2,3)), coords=coords)
    """

    __slots__ = ()


class SphericalAngleDataArray(DataArray, _core.SphericalAngleDataArray):
    """Spherical angles as indexed ``(theta, phi)`` pairs.

    The ``spherical_coordinate`` coordinate is ordered as ``("theta", "phi")`` and
    values are in radians.

    Example
    -------
    >>> index = np.arange(2)
    >>> spherical_coordinate = ["theta", "phi"]
    >>> coords = dict(index=index, spherical_coordinate=spherical_coordinate)
    >>> values = np.array([[0.0, 0.0], [0.1, 1.2]])
    >>> data = SphericalAngleDataArray(values, coords=coords)
    """

    __slots__ = ()
