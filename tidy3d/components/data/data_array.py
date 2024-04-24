"""Storing tidy3d data at it's most fundamental level as xr.DataArray objects"""
from __future__ import annotations
from typing import Dict, List, Union
from abc import ABC

import xarray as xr
import numpy as np
import dask
import h5py

from ...constants import (
    HERTZ,
    MICROMETER,
    PICOSECOND_PER_NANOMETER_PER_KILOMETER,
    RADIAN,
    SECOND,
    WATT,
)
from ...exceptions import DataError, FileError
from ..types import Bound, Axis

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
    "face_index": {"long_name": "face index"},
    "vertex_index": {"long_name": "vertex index"},
    "axis": {"long_name": "axis"},
}


# name of the DataArray.values in the hdf5 file (xarray's default name too)
DATA_ARRAY_VALUE_NAME = "__xarray_dataarray_variable__"


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
        if isinstance(val, str) and val in DATA_ARRAY_MAP:
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
    def _json_encoder(cls, val):
        """What function to call when writing a DataArray to json."""
        return type(val).__name__

    def __eq__(self, other) -> bool:
        """Whether two data array objects are equal."""

        if not isinstance(other, xr.DataArray):
            return False

        if not self.data.shape == other.data.shape or not np.all(self.data == other.data):
            return False
        for key, val in self.coords.items():
            if not np.all(np.array(val) == np.array(other.coords[key])):
                return False
        return True

    @property
    def abs(self):
        """Absolute value of data array."""
        return abs(self)

    def to_hdf5(self, fname: Union[str, h5py.File], group_path: str) -> None:
        """Save an xr.DataArray to the hdf5 file or file handle with a given path to the group."""

        # file name passed
        if isinstance(fname, str):
            with h5py.File(fname, "w") as f_handle:
                self.to_hdf5_handle(f_handle=f_handle, group_path=group_path)

        # file handle passed
        else:
            self.to_hdf5_handle(f_handle=fname, group_path=group_path)

    def to_hdf5_handle(self, f_handle: h5py.File, group_path: str) -> None:
        """Save an xr.DataArray to the hdf5 file handle with a given path to the group."""

        sub_group = f_handle.create_group(group_path)
        sub_group[DATA_ARRAY_VALUE_NAME] = self.values
        for key, val in self.coords.items():
            if val.dtype == "<U1":
                sub_group[key] = val.values.tolist()
            else:
                sub_group[key] = val

    @classmethod
    def from_hdf5(cls, fname: str, group_path: str) -> DataArray:
        """Load an DataArray from an hdf5 file with a given path to the group."""
        with h5py.File(fname, "r") as f:
            sub_group = f[group_path]
            values = np.array(sub_group[DATA_ARRAY_VALUE_NAME])
            coords = {dim: np.array(sub_group[dim]) for dim in cls._dims}
            for key, val in coords.items():
                if val.dtype == "O":
                    coords[key] = [byte_string.decode() for byte_string in val.tolist()]
            return cls(values, coords=coords)

    @classmethod
    def from_file(cls, fname: str, group_path: str) -> DataArray:
        """Load an DataArray from an hdf5 file with a given path to the group."""
        if ".hdf5" not in fname:
            raise FileError(
                "DataArray objects must be written to '.hdf5' format. "
                f"Given filename of {fname}."
            )
        return cls.from_hdf5(fname=fname, group_path=group_path)

    def __hash__(self) -> int:
        """Generate hash value for a :class:.`DataArray` instance, needed for custom components."""
        token_str = dask.base.tokenize(self)
        return hash(token_str)

    def multiply_at(self, value: complex, coord_name: str, indices: List[int]) -> DataArray:
        """Multiply self by value at indices into ."""
        self_mult = self.copy()
        self_mult[{coord_name: indices}] *= value
        return self_mult


class FreqDataArray(DataArray):
    """Frequency-domain array.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> fd = FreqDataArray((1+1j) * np.random.random((2,)), coords=dict(f=f))
    """

    __slots__ = ()
    _dims = ("f",)


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


class AbstractSpatialDataArray(DataArray, ABC):
    """Spatial distribution."""

    __slots__ = ()
    _dims = ("x", "y", "z")
    _data_attrs = {"long_name": "field value"}

    def sel_inside(self, bounds: Bound) -> SpatialDataArray:
        """Return a new SpatialDataArray that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        SpatialDataArray
            Extracted spatial data array.
        """

        inds_list = []

        coords = (self.x, self.y, self.z)

        for coord, smin, smax in zip(coords, bounds[0], bounds[1]):
            length = len(coord)

            # one point along direction, assume invariance
            if length == 1:
                comp_inds = [0]
            else:
                # if data does not cover structure at all take the closest index
                if smax < coord[0]:  # structure is completely on the left side
                    # take 2 if possible, so that linear iterpolation is possible
                    comp_inds = np.arange(0, max(2, length))

                elif smin > coord[-1]:  # structure is completely on the right side
                    # take 2 if possible, so that linear iterpolation is possible
                    comp_inds = np.arange(min(0, length - 2), length)

                else:
                    if smin < coord[0]:
                        ind_min = 0
                    else:
                        ind_min = max(0, (coord >= smin).argmax().data - 1)

                    if smax > coord[-1]:
                        ind_max = length - 1
                    else:
                        ind_max = (coord >= smax).argmax().data

                    comp_inds = np.arange(ind_min, ind_max + 1)

            inds_list.append(comp_inds)

        return self.isel(x=inds_list[0], y=inds_list[1], z=inds_list[2])

    def does_cover(self, bounds: Bound) -> bool:
        """Check whether data fully covers specified by ``bounds`` spatial region. If data contains
        only one point along a given direction, then it is assumed the data is constant along that
        direction and coverage is not checked.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        bool
            Full cover check outcome.
        """

        coords = (self.x, self.y, self.z)
        return all(
            (coord[0] <= smin and coord[-1] >= smax) or len(coord) == 1
            for coord, smin, smax in zip(coords, bounds[0], bounds[1])
        )


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

    def reflect(self, axis: Axis, center: float) -> SpatialDataArray:
        """Reflect data across the plane define by parameters ``axis`` and ``center`` from right to
        left.

        Parameters
        ----------
        axis : Literal[0, 1, 2]
            Normal direction of the reflection plane.
        center : float
            Location of the reflection plane along its normal direction.

        Returns
        -------
        SpatialDataArray
            Data after reflection is performed.
        """

        coords = list(self.coords.values())
        data = np.array(self.data)

        if np.isclose(center, coords[axis].data[0]):
            num_duplicates = 1
        elif center > coords[axis].data[0]:
            raise DataError("Reflection center must be outside and on the left of the data region.")
        else:
            num_duplicates = 0

        shape = np.array(np.shape(data))
        old_len = shape[axis]
        shape[axis] = 2 * old_len - num_duplicates

        ind_left = [slice(shape[0]), slice(shape[1]), slice(shape[2])]
        ind_right = [slice(shape[0]), slice(shape[1]), slice(shape[2])]

        ind_left[axis] = slice(old_len - 1, None, -1)
        ind_right[axis] = slice(old_len - num_duplicates, None)

        new_data = np.zeros(shape)

        new_data[ind_left[0], ind_left[1], ind_left[2]] = data
        new_data[ind_right[0], ind_right[1], ind_right[2]] = data

        new_coords = np.zeros(shape[axis])
        new_coords[old_len - num_duplicates :] = coords[axis]
        new_coords[old_len - 1 :: -1] = 2 * center - coords[axis]

        coords[axis] = new_coords
        coords_dict = dict(zip("xyz", coords))

        return SpatialDataArray(new_data, coords=coords_dict)


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
    _data_attrs = {"units": WATT, "long_name": "flux"}


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
    _data_attrs = {"units": WATT, "long_name": "flux"}


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


class GroupIndexDataArray(DataArray):
    """Group index of a mode.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(4)
    >>> coords = dict(f=f, mode_index=mode_index)
    >>> data = GroupIndexDataArray((1+1j) * np.random.random((2,4)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "mode_index")
    _data_attrs = {"long_name": "Group index"}


class ModeDispersionDataArray(DataArray):
    """Dispersion parameter of a mode.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(4)
    >>> coords = dict(f=f, mode_index=mode_index)
    >>> data = ModeDispersionDataArray((1+1j) * np.random.random((2,4)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "mode_index")
    _data_attrs = {
        "long_name": "Dispersion parameter",
        "units": PICOSECOND_PER_NANOMETER_PER_KILOMETER,
    }


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


class TriangleMeshDataArray(DataArray):
    """Data of the triangles of a surface mesh as in the STL file format."""

    __slots__ = ()
    _dims = ("face_index", "vertex_index", "axis")
    _data_attrs = {"long_name": "surface mesh triangles"}


class HeatDataArray(DataArray):
    """Heat data array.

    Example
    -------
    >>> T = [0, 1e-12, 2e-12]
    >>> td = HeatDataArray((1+1j) * np.random.random((3,)), coords=dict(T=T))
    """

    __slots__ = ()
    _dims = "T"


class ChargeDataArray(DataArray):
    """Charge data array.

    Example
    -------
    >>> n = [0, 1e-12, 2e-12]
    >>> p = [0, 3e-12, 4e-12]
    >>> td = ChargeDataArray((1+1j) * np.random.random((3,3)), coords=dict(n=n, p=p))
    """

    __slots__ = ()
    _dims = ("n", "p")


class PointDataArray(DataArray):
    """A two-dimensional array that stores coordinates of a collection of points.
    Dimension ``index`` denotes the index of a point in the collection, and dimension ``axis``
    denotes the point's coordinate along that axis.

    Example
    -------
    >>> point_array = PointDataArray(
    ...     (1+1j) * np.random.random((5, 3)), coords=dict(index=np.arange(5), axis=np.arange(3)),
    ... )
    >>> # get coordinates of a point number 3
    >>> point3 = point_array.sel(index=3)
    >>> # get x coordinates of all points
    >>> x_coords = point_array.sel(axis=0)
    """

    __slots__ = ()
    _dims = ("index", "axis")


class CellDataArray(DataArray):
    """A two-dimensional array that stores indices of points composing each cell in a collection of
    cells of the same type (for example: triangles, tetrahedra, etc). Dimension ``cell_index``
    denotes the index of a cell in the collection, and dimension ``vertex_index`` denotes placement
    (index) of a point in a cell (for example: 0, 1, or 2 for triangles; 0, 1, 2, or 3 for
    tetrahedra).

    Example
    -------
    >>> cell_array = CellDataArray(
    ...     (1+1j) * np.random.random((4, 3)),
    ...     coords=dict(cell_index=np.arange(4), vertex_index=np.arange(3)),
    ... )
    >>> # get indices of points composing cell number 3
    >>> cell3 = cell_array.sel(cell_index=3)
    >>> # get indices of points that represent the first vertex in each cell
    >>> first_vertices = cell_array.sel(vertex_index=0)
    """

    __slots__ = ()
    _dims = ("cell_index", "vertex_index")


class IndexedDataArray(DataArray):
    """Stores a one-dimensional array enumerated by coordinate ``index``. It is typically used
    in conjuction with a ``PointDataArray`` to store point-associated data or a ``CellDataArray``
    to store cell-associated data.

    Example
    -------
    >>> indexed_array = IndexedDataArray(
    ...     (1+1j) * np.random.random((3,)), coords=dict(index=np.arange(3))
    ... )
    """

    __slots__ = ()
    _dims = ("index",)


DATA_ARRAY_TYPES = [
    SpatialDataArray,
    ScalarFieldDataArray,
    ScalarFieldTimeDataArray,
    ScalarModeFieldDataArray,
    FluxDataArray,
    FluxTimeDataArray,
    ModeAmpsDataArray,
    ModeIndexDataArray,
    GroupIndexDataArray,
    ModeDispersionDataArray,
    FieldProjectionAngleDataArray,
    FieldProjectionCartesianDataArray,
    FieldProjectionKSpaceDataArray,
    DiffractionDataArray,
    FreqModeDataArray,
    FreqDataArray,
    TimeDataArray,
    FreqModeDataArray,
    TriangleMeshDataArray,
    HeatDataArray,
    ChargeDataArray,
    PointDataArray,
    CellDataArray,
    IndexedDataArray,
]
DATA_ARRAY_MAP = {data_array.__name__: data_array for data_array in DATA_ARRAY_TYPES}
