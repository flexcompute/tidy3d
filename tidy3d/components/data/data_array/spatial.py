"""Fields on structured Cartesian or cylindrical grids, plus spherical angle sets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from tidy3d.constants import RADIAN

from .abstract_spatial import AbstractSpatialDataArray
from .base import DataArray

if TYPE_CHECKING:
    from xarray.core.types import Self


class SpatialDataArray(AbstractSpatialDataArray):
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


class ScalarFieldDataArray(AbstractSpatialDataArray):
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
    _dims = ("x", "y", "z", "f")


class ScalarFieldTimeDataArray(AbstractSpatialDataArray):
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
    _dims = ("x", "y", "z", "t")


class ScalarModeFieldDataArray(AbstractSpatialDataArray):
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
    _dims = ("x", "y", "z", "f", "mode_index")


class ScalarModeFieldCylindricalDataArray(AbstractSpatialDataArray):
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
    _dims = ("rho", "theta", "axial", "f", "mode_index")


class ScalarTerminalFieldDataArray(AbstractSpatialDataArray):
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
    _dims = ("x", "y", "z", "f", "terminal_label")


class SphericalAngleDataArray(DataArray):
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
    _dims = ("index", "spherical_coordinate")
    _data_attrs = {
        "long_name": "spherical angles",
        "units": RADIAN,
    }

    def __init__(self, data: Any, *args: Any, **kwargs: Any) -> None:
        skip_canonicalization = kwargs.pop("_skip_bare_canonicalization", False)
        if (
            not skip_canonicalization
            and not args
            and "coords" not in kwargs
            and "dims" not in kwargs
        ):
            shape = np.shape(data)
            if (
                len(shape) == 2
                and shape[1] == 2
                and not isinstance(data, xr.Variable | xr.DataArray)
            ):
                kwargs["dims"] = ("index", "spherical_coordinate")
                kwargs["coords"] = {
                    "index": np.arange(shape[0]),
                    "spherical_coordinate": ["theta", "phi"],
                }
        super().__init__(data, *args, **kwargs)

    @classmethod
    def _constructor_kwargs_for_pydantic(cls) -> dict[str, Any]:
        """Keep model parsing strict while direct user construction is ergonomic.

        Direct ``SphericalAngleDataArray([[theta, phi], ...])`` construction can
        infer the standard dimensions and coordinates. Pydantic validation of
        model fields should not do this inference, because raw lists embedded in
        model payloads would otherwise bypass the usual serialized DataArray
        shape.
        """
        return {"_skip_bare_canonicalization": True}

    @classmethod
    def _validate_dims(cls, val: Self) -> Self:
        """Require explicit ``(theta, phi)`` spherical coordinate ordering."""
        val = super()._validate_dims(val)
        spherical_coords = tuple(str(coord) for coord in val.coords["spherical_coordinate"].values)
        if spherical_coords != ("theta", "phi"):
            raise ValueError(
                "SphericalAngleDataArray requires spherical_coordinate=('theta', 'phi')."
            )
        return val
