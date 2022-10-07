"""Storing tidy3d data at it's most fundamental level as xr.DataArray objects"""
from __future__ import annotations
from typing import Dict

import xarray as xr
import numpy as np

from ...constants import HERTZ, SECOND, MICROMETER, RADIAN
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
    "theta": {"units": RADIAN, "long_name": "elevation angle"},
    "phi": {"units": RADIAN, "long_name": "azimuth angle"},
    "ux": {"long_name": "normalized kx"},
    "uy": {"long_name": "normalized ky"},
    "orders_x": {"long_name": "diffraction order"},
    "orders_y": {"long_name": "diffraction order"},
    "polarization": {"long_name": "polarization"},
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

        # fix case if data of empty list or tuple is supplied as first arg
        data_arg = args[0]
        is_empty_array = isinstance(data_arg, np.ndarray) and (data_arg.size == 0)
        is_empty_list = (isinstance(data_arg, list)) and (data_arg == [])
        is_empty_tuple = (isinstance(data_arg, tuple)) and (data_arg == ())
        if is_empty_array or is_empty_list or is_empty_tuple:
            shape = tuple(len(values) for _, values, _ in coords)
            new_args = list(args)
            new_args[0] = np.zeros(shape=shape)
            args = tuple(new_args)

        super().__init__(*args, **kwargs)

    @classmethod
    def __get_validators__(cls):
        """Defines which validator function to use for ComplexNumber."""
        yield cls.validate

    @classmethod
    def validate(cls, value):
        """What gets called when you construct a DataArray."""
        return cls(value)

    @classmethod
    def __modify_schema__(cls, field_schema):
        """Sets the schema of DataArray object."""

        schema = dict(
            title="DataArray",
            type="xr.DataArray",
            properties=dict(
                __slots__=dict(
                    title="__slots__",
                    type="Tuple[str, ...]",
                ),
            ),
            required=["__slots__"],
        )
        field_schema.update(schema)

    def __eq__(self, other) -> bool:
        """Whether two data array objects are equal."""
        if not np.all(self.data == other.data):
            return False
        for key, val in self.coords.items():
            if not np.all(np.array(val) == np.array(other.coords[key])):
                return False
        return True

    @property
    def abs(self):
        """Absolute value of data array."""
        return abs(self)

    def to_hdf5(self, fname: str, group_path: str) -> None:
        """Save an xr.DataArray to the hdf5 file with a given path to the group."""
        self.to_netcdf(fname, group=group_path, engine="h5netcdf", invalid_netcdf=True, mode="a")

    @classmethod
    def from_hdf5(cls, fname: str, group_path: str) -> DataArray:
        """Load an DataArray from an hdf5 file with a given path to the group."""
        return xr.open_dataarray(fname, group=group_path, engine="h5netcdf", invalid_netcdf=True)


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


class Near2FarAngleDataArray(DataArray):
    """Radiation vectors in frequency domain as a function of angles theta and phi.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> coords = dict(f=f, theta=theta, phi=phi)
    >>> values = (1+1j) * np.random.random((len(theta), len(phi), len(f)))
    >>> data = Near2FarAngleDataArray(values, coords=coords)
    """

    __slots__ = ("theta", "phi", "f")
    _data_attrs = {"long_name": "radiation vectors"}


class Near2FarCartesianDataArray(DataArray):
    """Radiation vectors in frequency domain as a function of local x and y coordinates.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> x = np.linspace(0, 5, 10)
    >>> y = np.linspace(0, 10, 20)
    >>> coords = dict(f=f, x=x, y=y)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(f)))
    >>> data = Near2FarCartesianDataArray(values, coords=coords)
    """

    __slots__ = ("x", "y", "f")
    _data_attrs = {"long_name": "radiation vectors"}


class Near2FarKSpaceDataArray(DataArray):
    """Radiation vector in frequency domain as a function of normalized
    kx and ky vectors on the observation plane.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> ux = np.linspace(0, 5, 10)
    >>> uy = np.linspace(0, 10, 20)
    >>> coords = dict(f=f, ux=ux, uy=uy)
    >>> values = (1+1j) * np.random.random((len(ux), len(uy), len(f)))
    >>> data = Near2FarKSpaceDataArray(values, coords=coords)
    """

    __slots__ = ("ux", "uy", "f")
    _data_attrs = {"long_name": "radiation vectors"}


class DiffractionDataArray(DataArray):
    """Diffraction power amplitudes as a function of diffraction orders and frequency.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> orders_x = np.linspace(-1, 1, 3)
    >>> orders_y = np.linspace(-2, 2, 5)
    >>> pol = ["s", "p"]
    >>> coords = dict(f=f, orders_x=orders_x, orders_y=orders_y, polarization=pol)
    >>> values = (1+1j) * np.random.random((len(orders_x), len(orders_y), len(pol), len(f)))
    >>> data = DiffractionDataArray(values, coords=coords)
    """

    __slots__ = ("orders_x", "orders_y", "polarization", "f")
    _data_attrs = {"long_name": "diffraction amplitude"}
