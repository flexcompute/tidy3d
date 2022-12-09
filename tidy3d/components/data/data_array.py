"""Storing tidy3d data at it's most fundamental level as xr.DataArray objects"""
from __future__ import annotations
from typing import Dict

import xarray as xr
import numpy as np

from ...constants import HERTZ, SECOND, MICROMETER, RADIAN
from ...log import DataError, FileError

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
}

# string that gets written to the json file
DATA_ARRAY_TAG = "XR.DATAARRAY"


class DataArray(xr.DataArray):
    """Subclass of ``xr.DataArray`` that requires _dims to match the keys of the coords."""

    # Always set __slots__ = () to avoid xarray warnings
    __slots__ = ()
    # stores an ordered tuple of strings corresponding to the data dimensions
    _dims = ()
    # stores a dictionary of attributes corresponding to the data values
    _data_attrs: Dict[str, str] = {}

    @classmethod
    def __get_validators__(cls):
        """Validators that get run when :class:`.DataArray` objects are added to pydantic models."""
        yield cls.check_unloaded_data
        yield cls.validate_dims
        yield cls.assign_data_attrs
        yield cls.assign_coord_attrs

    @classmethod
    def check_unloaded_data(cls, val):
        """If the data comes in as the raw data array string, raise a custom warning."""
        if isinstance(val, str):
            raise DataError(
                f"Trying to load {cls.__name__} but the data is not present. "
                "Note that data will not be saved to .json file. "
                "use .hdf5 format instead if data present."
            )
        return cls(val)

    @classmethod
    def validate_dims(cls, val):
        """Make sure the dims are the same as _dims, then put them in the correct order."""
        if set(val.dims) != set(cls._dims):
            raise ValueError(f"wrong dims, expected '{cls._dims}', got '{val.dims}'")
        return val.transpose(*cls._dims)

    @classmethod
    def assign_data_attrs(cls, val):
        """Assign the correct data attributes to the :class:`.DataArray`."""

        for attr_name, attr in cls._data_attrs.items():
            val.attrs[attr_name] = attr
        return val

    @classmethod
    def assign_coord_attrs(cls, val):
        """Assign the correct coordinate attributes to the :class:`.DataArray`."""

        for dim in cls._dims:
            dim_attrs = DIM_ATTRS.get(dim)
            if dim_attrs is not None:
                for attr_name, attr in dim_attrs.items():
                    val.coords[dim].attrs[attr_name] = attr
        return val

    @classmethod
    def __modify_schema__(cls, field_schema):
        """Sets the schema of DataArray object."""

        schema = dict(
            title="DataArray",
            type="xr.DataArray",
            properties=dict(
                _dims=dict(
                    title="_dims",
                    type="Tuple[str, ...]",
                ),
            ),
            required=["_dims"],
        )
        field_schema.update(schema)

    @classmethod
    def _json_encoder(cls, val):  # pylint:disable=unused-argument
        """What function to call when writing a DataArray to json."""
        return DATA_ARRAY_TAG

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
        return xr.load_dataarray(fname, group=group_path, engine="h5netcdf", invalid_netcdf=True)

    @classmethod
    def from_file(cls, fname: str, group_path: str) -> DataArray:
        """Load an DataArray from an hdf5 file with a given path to the group."""
        if ".hdf5" not in fname:
            raise FileError(
                "DataArray objects must be written to '.hdf5' format. "
                f"Given filename of {fname}."
            )
        return cls.from_hdf5(fname=fname, group_path=group_path)


class FreqDataArray(DataArray):
    """Frequency-domain array.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> fd = FreqDataArray((1+1j) * np.random.random((2,)), coords=dict(f=f))
    """

    __slots__ = ()
    _dims = "f"


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


class TimeDataArray(DataArray):
    """Time-domain array.

    Example
    -------
    >>> t = [0, 1e-12, 2e-12]
    >>> td = TimeDataArray((1+1j) * np.random.random((3,)), coords=dict(t=t))
    """

    __slots__ = ()
    _dims = "t"


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

    __slots__ = ()
    _dims = ("x", "y", "z", "f")
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

    __slots__ = ()
    _dims = ("x", "y", "z", "t")
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

    __slots__ = ()
    _dims = ("x", "y", "z", "f", "mode_index")
    _data_attrs = {"long_name": "field value"}


class FluxDataArray(DataArray):
    """Flux through a surface in the frequency-domain.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> coords = dict(f=f)
    >>> fd = FluxDataArray(np.random.random(2), coords=coords)
    """

    __slots__ = ()
    _dims = ("f",)
    _data_attrs = {"units": "W", "long_name": "flux"}


class FluxTimeDataArray(DataArray):
    """Flux through a surface in the time-domain.

    Example
    -------
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(t=t)
    >>> data = FluxTimeDataArray(np.random.random(3), coords=coords)
    """

    __slots__ = ()
    _dims = ("t",)
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

    __slots__ = ()
    _dims = ("direction", "f", "mode_index")
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

    __slots__ = ()
    _dims = ("f", "mode_index")
    _data_attrs = {"long_name": "Propagation index"}


class FieldProjectionAngleDataArray(DataArray):
    """Far fields in frequency domain as a function of angles theta and phi.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> r = np.atleast_1d(5)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> coords = dict(r=r, theta=theta, phi=phi, f=f)
    >>> values = (1+1j) * np.random.random((len(r), len(theta), len(phi), len(f)))
    >>> data = FieldProjectionAngleDataArray(values, coords=coords)
    """

    __slots__ = ()
    _dims = ("r", "theta", "phi", "f")
    _data_attrs = {"long_name": "radiation vectors"}


class FieldProjectionCartesianDataArray(DataArray):
    """Far fields in frequency domain as a function of local x and y coordinates.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> x = np.linspace(0, 5, 10)
    >>> y = np.linspace(0, 10, 20)
    >>> z = np.atleast_1d(5)
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(z), len(f)))
    >>> data = FieldProjectionCartesianDataArray(values, coords=coords)
    """

    __slots__ = ()
    _dims = ("x", "y", "z", "f")
    _data_attrs = {"long_name": "radiation vectors"}


class FieldProjectionKSpaceDataArray(DataArray):
    """Far fields in frequency domain as a function of normalized
    kx and ky vectors on the observation plane.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> r = np.atleast_1d(5)
    >>> ux = np.linspace(0, 5, 10)
    >>> uy = np.linspace(0, 10, 20)
    >>> coords = dict(ux=ux, uy=uy, r=r, f=f)
    >>> values = (1+1j) * np.random.random((len(ux), len(uy), len(r), len(f)))
    >>> data = FieldProjectionKSpaceDataArray(values, coords=coords)
    """

    __slots__ = ()
    _dims = ("ux", "uy", "r", "f")
    _data_attrs = {"long_name": "radiation vectors"}


class DiffractionDataArray(DataArray):
    """Diffraction power amplitudes as a function of diffraction orders and frequency.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> orders_x = np.linspace(-1, 1, 3)
    >>> orders_y = np.linspace(-2, 2, 5)
    >>> coords = dict(orders_x=orders_x, orders_y=orders_y, f=f)
    >>> values = (1+1j) * np.random.random((len(orders_x), len(orders_y), len(f)))
    >>> data = DiffractionDataArray(values, coords=coords)
    """

    __slots__ = ()
    _dims = ("orders_x", "orders_y", "f")
    _data_attrs = {"long_name": "diffraction amplitude"}
