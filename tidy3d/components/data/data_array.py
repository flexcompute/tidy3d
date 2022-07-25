"""Storing tidy3d data at it's most fundamental level as xr.DataArray objects"""
from typing import Dict

import xarray as xr
import numpy as np

from ..types import DataObject
from ...constants import HERTZ, SECOND, MICROMETER
from ...log import DataError

# maps the dimension names to their attributes
DIM_ATTRS = {
    "x": {"units": MICROMETER, "long_name": "x position"},
    "y": {"units": MICROMETER, "long_name": "y position"},
    "z": {"units": MICROMETER, "long_name": "z position"},
    "f": {"units": HERTZ, "long_name": "frequency"},
    "t": {"units": SECOND, "long_name": "time"},
    "direction": {"long_name": "propagation direction"},
    "mode_index": {"long_name": "mode index"},
}


class DataArray(xr.DataArray):
    """Subclass of ``xr.DataArray`` that requires __slots__ to match the keys of the coords."""

    # stores an ordered tuple of strings corresponding to the data dimensions
    __slots__ = ()

    # stores a dictionary of attributes corresponding to the data values
    _data_attrs: Dict[str, str] = {}

    def __init__(self, *args, **kwargs):

        # if dimensions are supplied, make sure they are valid
        if "dims" in kwargs:
            dims = kwargs["dims"]
            if dims != self.__slots__:
                raise DataError(
                    f"supplied dims {dims} different from hardcoded slots {self.__slots__}."
                )

        # if dimensions not supplied (and not fastpath, which rejects coords), use __slots__ as dims
        if ("dims" not in kwargs) and (not kwargs.get("fastpath")):
            kwargs["dims"] = self.__slots__

        # set up coords to use attributes by supplying coords as sequence of (dim, data, attrs)
        if "coords" in kwargs:
            coords = [(dim, kwargs["coords"][dim], DIM_ATTRS.get(dim)) for dim in self.__slots__]
            kwargs["coords"] = coords

        # add class level attributes to data values, if not supplied
        if (not kwargs.get("fastpath")) and ("attrs" not in kwargs) and (self._data_attrs):
            kwargs["attrs"] = self._data_attrs

        # dont feed data values if there is no data
        data_values, *other_args = args
        if np.array(data_values).size == 0:
            args = other_args

        # call xarray's initializer to make the data array
        super().__init__(*args, **kwargs)

    @classmethod
    def __get_validators__(cls):
        """Defines which validator function to use for ComplexNumber."""
        yield cls.validate

    @classmethod
    def validate(cls, value):
        """What gets called when you construct a DataArray."""

        # loading from raw dict (usually from file)
        if isinstance(value, dict):
            data = value.get("data")
            coords = value.get("coords")
            coords = {name: np.array(val.get("data")) for name, val in coords.items()}
            return cls(np.array(data), coords=coords)

        return cls(value)

    @classmethod
    def __modify_schema__(cls, field_schema):
        """Sets the schema of ComplexNumber."""
        field_schema.update(DataObject.schema())

    def __eq__(self, other) -> bool:
        """Whether two data array objects are equal."""

        return self.to_dict() == other.to_dict()


class ScalarFieldDataArray(DataArray):
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

    __slots__ = ("x", "y", "z", "f")
    _data_attrs = {"long_name": "field value"}


class ScalarFieldTimeDataArray(DataArray):
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

    __slots__ = ("x", "y", "z", "t")
    _data_attrs = {"long_name": "field value"}


class ScalarModeFieldDataArray(DataArray):
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

    __slots__ = ("x", "y", "z", "f", "mode_index")
    _data_attrs = {"long_name": "field value"}


class FluxDataArray(DataArray):
    """Flux through a surface in the frequency-domain.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> coords = dict(f=f)
    >>> fd = FluxDataArray(np.random.random(2), coords=coords)
    """

    __slots__ = ("f",)
    _data_attrs = {"units": "W", "long_name": "flux"}


class FluxTimeDataArray(DataArray):
    """Flux through a surface in the time-domain.

    Example
    -------
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(t=t)
    >>> data = FluxTimeDataArray(np.random.random(3), coords=coords)
    """

    __slots__ = ("t",)
    _data_attrs = {"units": "W", "long_name": "flux"}


class ModeAmpsDataArray(DataArray):
    """Forward and backward propagating complex-valued mode amplitudes.

    Example
    -------
    >>> direction = ["+", "-"]
    >>> f = [1e14, 2e14, 3e14]
    >>> mode_index = np.arange(4)
    >>> coords = dict(direction=direction, f=f, mode_index=mode_index)
    >>> data = ModeAmpsDataArray((1+1j) * np.random.random((2, 3, 4)), coords=coords)
    """

    __slots__ = ("direction", "f", "mode_index")
    _data_attrs = {"units": "sqrt(W)", "long_name": "mode amplitudes"}


class ModeIndexDataArray(DataArray):
    """Complex-valued effective propagation index of a mode.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(4)
    >>> coords = dict(f=f, mode_index=mode_index)
    >>> data = ModeIndexDataArray((1+1j) * np.random.random((2,4)), coords=coords)
    """

    __slots__ = ("f", "mode_index")
    _data_attrs = {"long_name": "Propagation index"}
