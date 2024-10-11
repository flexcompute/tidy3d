"""Storing tidy3d data at it's most fundamental level as xr.DataArray objects"""

from __future__ import annotations

import warnings
from abc import ABC
from typing import Any, Dict, List, Mapping, Union

import autograd.numpy as anp
import dask
import h5py
import numpy as np
import pandas
import xarray as xr
from autograd.tracer import isbox
from xarray.core import alignment, missing
from xarray.core.indexes import PandasIndex
from xarray.core.indexing import _outer_to_numpy_indexer
from xarray.core.types import InterpOptions, Self
from xarray.core.utils import OrderedSet, either_dict_or_kwargs
from xarray.core.variable import as_variable

from ...constants import (
    HERTZ,
    MICROMETER,
    PICOSECOND_PER_NANOMETER_PER_KILOMETER,
    RADIAN,
    SECOND,
    WATT,
)
from ...exceptions import DataError, FileError
from ..autograd import TidyArrayBox, get_static, interpn, is_tidy_box
from ..types import Axis, Bound

# maps the dimension names to their attributes
DIM_ATTRS = {
    "x": {"units": MICROMETER, "long_name": "x position"},
    "y": {"units": MICROMETER, "long_name": "y position"},
    "z": {"units": MICROMETER, "long_name": "z position"},
    "f": {"units": HERTZ, "long_name": "frequency"},
    "t": {"units": SECOND, "long_name": "time"},
    "direction": {"long_name": "propagation direction"},
    "mode_index": {"long_name": "mode index"},
    "eme_port_index": {"long_name": "EME port index"},
    "eme_cell_index": {"long_name": "EME cell index"},
    "mode_index_in": {"long_name": "mode index in"},
    "mode_index_out": {"long_name": "mode index out"},
    "sweep_index": {"long_name": "sweep index"},
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

    def __init__(self, data, *args, **kwargs):
        # if data is a vanilla autograd box, convert to our box
        if isbox(data) and not is_tidy_box(data):
            data = TidyArrayBox.from_arraybox(data)
        # do the same for xr.Variable or xr.DataArray type
        elif (
            isinstance(data, (xr.Variable, xr.DataArray))
            and isbox(data.data)
            and not is_tidy_box(data.data)
        ):
            data.data = TidyArrayBox.from_arraybox(data.data)
        super().__init__(data, *args, **kwargs)

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

    def _interp_validator(self, field_name: str = None) -> None:
        """Make sure we can interp()/sel() the data.

        NOTE
        ----
        This does not check every 'DataArray' by default. Instead, when required, this check can be
        called from a validator, as is the case with 'CustomMedium' and 'CustomFieldSource'.
        """
        if field_name is None:
            field_name = "DataArray"

        dims = self.coords.dims

        for dim in dims:
            # in case we encounter some /0 or /NaN we'll ignore the warnings here
            with np.errstate(divide="ignore", invalid="ignore"):
                # check that we can interpolate
                try:
                    x0 = np.array(self.coords[dim][0])
                    self.interp({dim: x0}, method="linear")
                    self.interp({dim: x0}, method="nearest")
                    # self.interp_like(self.isel({self.dim: 0}))
                except pandas.errors.InvalidIndexError as e:
                    raise DataError(
                        f"'{field_name}.interp()' fails to interpolate along {dim} which is used by the solver. "
                        "This may be caused, for instance, by duplicated data "
                        f"in this dimension (you can verify this by running "
                        f"'{field_name}={field_name}.drop_duplicates(dim=\"{dim}\")' "
                        f"and interpolate with the new '{field_name}'). "
                        "Plase make sure data can be interpolated."
                    ) from e
                # in case it can interpolate, try also to sel
                try:
                    x0 = np.array(self.coords[dim][0])
                    self.sel({dim: x0}, method="nearest")
                except pandas.errors.InvalidIndexError as e:
                    raise DataError(
                        f"'{field_name}.sel()' fails to select along {dim} which is used by the solver. "
                        "This may be caused, for instance, by duplicated data "
                        f"in this dimension (you can verify this by running "
                        f"'{field_name}={field_name}.drop_duplicates(dim=\"{dim}\")' "
                        f"and run 'sel()' with the new '{field_name}'). "
                        "Plase make sure 'sel()' can be used on the 'DataArray'."
                    ) from e

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
    def values(self):
        """
        The array's data converted to a numpy.ndarray.

        Returns
        -------
        np.ndarray
            The values of the DataArray.
        """
        if isbox(self.data):
            warnings.warn(
                "'DataArray.values' for automatic differentiation is deprecated, "
                "please use 'DataArray.data' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.data
        return super().values

    @values.setter
    def values(self, value: Any) -> None:
        self.variable.values = value

    @property
    def abs(self):
        """Absolute value of data array."""
        return abs(self)

    @property
    def is_uniform(self):
        """Whether each element is of equal value in the data array"""
        raw_data = self.data.ravel()
        return np.allclose(raw_data, raw_data[0])

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
        sub_group[DATA_ARRAY_VALUE_NAME] = get_static(self.data)
        for key, val in self.coords.items():
            if val.dtype == "<U1":
                sub_group[key] = val.values.tolist()
            else:
                sub_group[key] = val

    @classmethod
    def from_hdf5(cls, fname: str, group_path: str) -> Self:
        """Load an DataArray from an hdf5 file with a given path to the group."""
        with h5py.File(fname, "r") as f:
            sub_group = f[group_path]
            values = np.array(sub_group[DATA_ARRAY_VALUE_NAME])
            coords = {dim: np.array(sub_group[dim]) for dim in cls._dims if dim in sub_group}
            for key, val in coords.items():
                if val.dtype == "O":
                    coords[key] = [byte_string.decode() for byte_string in val.tolist()]
            return cls(values, coords=coords, dims=cls._dims)

    @classmethod
    def from_file(cls, fname: str, group_path: str) -> Self:
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

    def multiply_at(self, value: complex, coord_name: str, indices: List[int]) -> Self:
        """Multiply self by value at indices."""
        if isbox(self.data) or isbox(value):
            return self._ag_multiply_at(value, coord_name, indices)

        self_mult = self.copy()
        self_mult[{coord_name: indices}] *= value
        return self_mult

    def _ag_multiply_at(self, value: complex, coord_name: str, indices: List[int]) -> Self:
        """Autograd multiply_at override when tracing."""
        key = {coord_name: indices}
        _, index_tuple, _ = self.variable._broadcast_indexes(key)
        idx = _outer_to_numpy_indexer(index_tuple, self.data.shape)
        mask = np.zeros(self.data.shape, dtype="?")
        mask[idx] = True
        return self.copy(deep=False, data=anp.where(mask, self.data * value, self.data))

    def interp(
        self,
        coords: Mapping[Any, Any] | None = None,
        method: InterpOptions = "linear",
        assume_sorted: bool = False,
        kwargs: Mapping[str, Any] | None = None,
        **coords_kwargs: Any,
    ) -> Self:
        """Interpolate this DataArray to new coordinate values.

        Parameters
        ----------
        coords : Union[Mapping[Any, Any], None] = None
            A mapping from dimension names to new coordinate labels.
        method : InterpOptions = "linear"
            The interpolation method to use.
        assume_sorted : bool = False
            If True, skip sorting of coordinates.
        kwargs : Union[Mapping[str, Any], None] = None
            Additional keyword arguments to pass to the interpolation function.
        **coords_kwargs : Any
            The keyword arguments form of coords.

        Returns
        -------
        DataArray
            A new DataArray with interpolated values.

        Raises
        ------
        KeyError
            If any of the specified coordinates are not in the DataArray.
        """
        if isbox(self.data):
            return self._ag_interp(coords, method, assume_sorted, kwargs, **coords_kwargs)

        return super().interp(coords, method, assume_sorted, kwargs, **coords_kwargs)

    def _ag_interp(
        self,
        coords: Union[Mapping[Any, Any], None] = None,
        method: InterpOptions = "linear",
        assume_sorted: bool = False,
        kwargs: Union[Mapping[str, Any], None] = None,
        **coords_kwargs: Any,
    ) -> Self:
        """Autograd interp override when tracing over self.data.

        This implementation closely follows the interp implementation of xarray
        to match its behavior as closely as possible while supporting autograd.

        See:
        - https://docs.xarray.dev/en/latest/generated/xarray.DataArray.interp.html
        - https://docs.xarray.dev/en/latest/generated/xarray.Dataset.interp.html
        """
        if kwargs is None:
            kwargs = {}

        ds = self._to_temp_dataset()

        coords = either_dict_or_kwargs(coords, coords_kwargs, "interp")
        indexers = dict(ds._validate_interp_indexers(coords))

        if coords:
            # Find shared dimensions between the dataset and the indexers
            sdims = (
                set(ds.dims)
                .intersection(*[set(nx.dims) for nx in indexers.values()])
                .difference(coords.keys())
            )
            indexers.update({d: ds.variables[d] for d in sdims})

        obj = ds if assume_sorted else ds.sortby(list(coords))

        # workaround to get a variable for a dimension without a coordinate
        validated_indexers = {
            k: (obj._variables.get(k, as_variable((k, range(obj.sizes[k])))), v)
            for k, v in indexers.items()
        }

        for k, v in validated_indexers.items():
            obj, newidx = missing._localize(obj, {k: v})
            validated_indexers[k] = newidx[k]

        variables = {}
        reindex = False
        for name, var in obj._variables.items():
            if name in indexers:
                continue
            dtype_kind = var.dtype.kind
            if dtype_kind in "uifc":
                # Interpolation for numeric types
                var_indexers = {k: v for k, v in validated_indexers.items() if k in var.dims}
                variables[name] = self._ag_interp_func(var, var_indexers, method, **kwargs)
            elif dtype_kind in "ObU" and (validated_indexers.keys() & var.dims):
                # Stepwise interpolation for non-numeric types
                reindex = True
            elif all(d not in indexers for d in var.dims):
                # Keep variables not dependent on interpolated coords
                variables[name] = var

        if reindex:
            # Reindex for non-numeric types
            reindex_indexers = {k: v for k, (_, v) in validated_indexers.items() if v.dims == (k,)}
            reindexed = alignment.reindex(
                obj,
                indexers=reindex_indexers,
                method="nearest",
                exclude_vars=variables.keys(),
            )
            indexes = dict(reindexed._indexes)
            variables.update(reindexed.variables)
        else:
            # Get the indexes that are not being interpolated along
            indexes = {k: v for k, v in obj._indexes.items() if k not in indexers}

        # Get the coords that also exist in the variables
        coord_names = obj._coord_names & variables.keys()
        selected = ds._replace_with_new_dims(variables.copy(), coord_names, indexes=indexes)

        # Attach indexer as coordinate
        for k, v in indexers.items():
            if v.dims == (k,):
                index = PandasIndex(v, k, coord_dtype=v.dtype)
                index_vars = index.create_variables({k: v})
                indexes[k] = index
                variables.update(index_vars)
            else:
                variables[k] = v

        # Extract coordinates from indexers
        coord_vars, new_indexes = selected._get_indexers_coords_and_indexes(coords)
        variables.update(coord_vars)
        indexes.update(new_indexes)

        coord_names = obj._coord_names & variables.keys() | coord_vars.keys()
        ds = ds._replace_with_new_dims(variables, coord_names, indexes=indexes)
        return self._from_temp_dataset(ds)

    @staticmethod
    def _ag_interp_func(var, indexes_coords, method, **kwargs):
        """
        Interpolate the variable `var` along the coordinates specified in `indexes_coords` using the given `method`.

        The implementation follows xarray's interp implementation in xarray.core.missing,
        but replaces some of the pre-processing as well as the actual interpolation
        function with an autograd-compatible approach.


        Parameters
        ----------
        var : xr.Variable
            The variable to be interpolated.
        indexes_coords : dict
            A dictionary mapping dimension names to coordinate values for interpolation.
        method : str
            The interpolation method to use.
        **kwargs : dict
            Additional keyword arguments to pass to the interpolation function.

        Returns
        -------
        xr.Variable
            The interpolated variable.
        """
        if not indexes_coords:
            return var.copy()
        result = var
        for indep_indexes_coords in missing.decompose_interp(indexes_coords):
            var = result

            # target dimensions
            dims = list(indep_indexes_coords)
            x, new_x = zip(*[indep_indexes_coords[d] for d in dims])
            destination = missing.broadcast_variables(*new_x)

            broadcast_dims = [d for d in var.dims if d not in dims]
            original_dims = broadcast_dims + dims
            new_dims = broadcast_dims + list(destination[0].dims)

            x, new_x = missing._floatize_x(x, new_x)

            permutation = [var.dims.index(dim) for dim in original_dims]
            combined_permutation = permutation[-len(x) :] + permutation[: -len(x)]
            data = anp.transpose(var.data, combined_permutation)
            xi = anp.stack([anp.ravel(new_xi.data) for new_xi in new_x], axis=-1)

            result = interpn(
                [xn.data for xn in x],
                data,
                xi,
                method=method,
            )

            result = anp.moveaxis(result, 0, -1)
            result = anp.reshape(result, result.shape[:-1] + new_x[0].shape)

            result = xr.Variable(new_dims, result, attrs=var.attrs, fastpath=True)

            out_dims: OrderedSet = OrderedSet()
            for d in var.dims:
                if d in dims:
                    out_dims.update(indep_indexes_coords[d][1].dims)
                else:
                    out_dims.add(d)
            if len(out_dims) > 1:
                result = result.transpose(*out_dims)
        return result


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

    @property
    def _spatially_sorted(self) -> SpatialDataArray:
        """Check whether sorted and sort if not."""
        needs_sorting = []
        for axis in "xyz":
            axis_coords = self.coords[axis].values
            if len(axis_coords) > 1 and np.any(axis_coords[1:] < axis_coords[:-1]):
                needs_sorting.append(axis)

        if len(needs_sorting) > 0:
            return self.sortby(needs_sorting)

        return self

    def sel_inside(self, bounds: Bound) -> SpatialDataArray:
        """Return a new SpatialDataArray that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``. Note that the returned data is sorted with respect
        to spatial coordinates.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        SpatialDataArray
            Extracted spatial data array.
        """
        if any(bmin > bmax for bmin, bmax in zip(*bounds)):
            raise DataError(
                "Min and max bounds must be packaged as '(minx, miny, minz), (maxx, maxy, maxz)'."
            )

        # make sure data is sorted with respect to coordinates
        sorted_self = self._spatially_sorted

        inds_list = []

        coords = (sorted_self.x, sorted_self.y, sorted_self.z)

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

        return sorted_self.isel(x=inds_list[0], y=inds_list[1], z=inds_list[2])

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
        if any(bmin > bmax for bmin, bmax in zip(*bounds)):
            raise DataError(
                "Min and max bounds must be packaged as '(minx, miny, minz), (maxx, maxy, maxz)'."
            )

        coords = (self.x, self.y, self.z)
        return all(
            (np.min(coord) <= smin and np.max(coord) >= smax) or len(coord) == 1
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

    def reflect(self, axis: Axis, center: float, reflection_only: bool = False) -> SpatialDataArray:
        """Reflect data across the plane define by parameters ``axis`` and ``center`` from right to
        left. Note that the returned data is sorted with respect to spatial coordinates.

        Parameters
        ----------
        axis : Literal[0, 1, 2]
            Normal direction of the reflection plane.
        center : float
            Location of the reflection plane along its normal direction.
        reflection_only : bool = False
            Return only reflected data.

        Returns
        -------
        SpatialDataArray
            Data after reflection is performed.
        """

        sorted_self = self._spatially_sorted

        coords = [sorted_self.x.values, sorted_self.y.values, sorted_self.z.values]
        data = np.array(sorted_self.data)

        data_left_bound = coords[axis][0]

        if np.isclose(center, data_left_bound):
            num_duplicates = 1
        elif center > data_left_bound:
            raise DataError("Reflection center must be outside and to the left of the data region.")
        else:
            num_duplicates = 0

        if reflection_only:
            coords[axis] = 2 * center - coords[axis]
            coords_dict = dict(zip("xyz", coords))

            tmp_arr = SpatialDataArray(sorted_self.data, coords=coords_dict)

            return tmp_arr.sortby("xyz"[axis])

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


class EMEScalarModeFieldDataArray(AbstractSpatialDataArray):
    """Spatial distribution of a mode in frequency-domain as a function of mode index
    and EME cell index.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3]
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> eme_cell_index = np.arange(5)
    >>> coords = dict(x=x, y=y, z=z, f=f, mode_index=mode_index, eme_cell_index=eme_cell_index)
    >>> fd = EMEScalarModeFieldDataArray((1+1j) * np.random.random((2,3,1,2,5,5)), coords=coords)
    """

    __slots__ = ()
    _dims = ("x", "y", "z", "f", "sweep_index", "eme_cell_index", "mode_index")


class EMEFreqModeDataArray(DataArray):
    """Array over frequency, mode index, and EME cell index.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> eme_cell_index = np.arange(5)
    >>> coords = dict(f=f, mode_index=mode_index, eme_cell_index=eme_cell_index)
    >>> fd = EMEFreqModeDataArray((1+1j) * np.random.random((2, 5, 5)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "sweep_index", "eme_cell_index", "mode_index")


class EMEScalarFieldDataArray(AbstractSpatialDataArray):
    """Spatial distribution of a field excited from an EME port in frequency-domain as a
    function of mode index at the EME port and the EME port index.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> eme_port_index = [0, 1]
    >>> coords = dict(x=x, y=y, z=z, f=f, mode_index=mode_index, eme_port_index=eme_port_index)
    >>> fd = EMEScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2,5,2)), coords=coords)
    """

    __slots__ = ()
    _dims = ("x", "y", "z", "f", "sweep_index", "eme_port_index", "mode_index")


class EMECoefficientDataArray(DataArray):
    """EME expansion coefficient of the mode `mode_index_out` in the EME cell
    `eme_cell_index`, when excited from mode `mode_index_in` of EME port `eme_port_index`.

    Example
    -------
    >>> mode_index_in = [0, 1]
    >>> mode_index_out = [0, 1]
    >>> eme_cell_index = np.arange(5)
    >>> eme_port_index = [0, 1]
    >>> f = [2e14]
    >>> coords = dict(
    ...     f=f,
    ...     mode_index_out=mode_index_out,
    ...     mode_index_in=mode_index_in,
    ...     eme_cell_index=eme_cell_index,
    ...     eme_port_index=eme_port_index
    ... )
    >>> fd = EMECoefficientDataArray((1 + 1j) * np.random.random((1, 2, 2, 5, 2)), coords=coords)
    """

    __slots__ = ()
    _dims = (
        "f",
        "sweep_index",
        "eme_port_index",
        "eme_cell_index",
        "mode_index_out",
        "mode_index_in",
    )
    _data_attrs = {"long_name": "mode expansion coefficient"}


class EMESMatrixDataArray(DataArray):
    """Scattering matrix elements for a fixed pair of ports, possibly with an extra
    sweep index.

    Example
    -------
    >>> mode_index_in = [0, 1]
    >>> mode_index_out = [0, 1, 2]
    >>> f = [2e14]
    >>> sweep_index = np.arange(10)
    >>> coords = dict(
    ...     f=f,
    ...     mode_index_out=mode_index_out,
    ...     mode_index_in=mode_index_in,
    ...     sweep_index=sweep_index
    ... )
    >>> fd = EMESMatrixDataArray((1 + 1j) * np.random.random((1, 3, 2, 10)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "sweep_index", "mode_index_out", "mode_index_in")
    _data_attrs = {"long_name": "scattering matrix element"}


class EMEModeIndexDataArray(DataArray):
    """Complex-valued effective propagation index of an EME mode,
    also indexed by EME cell.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(4)
    >>> eme_cell_index = np.arange(5)
    >>> coords = dict(f=f, mode_index=mode_index, eme_cell_index=eme_cell_index)
    >>> data = EMEModeIndexDataArray((1+1j) * np.random.random((2,4,5)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "sweep_index", "eme_cell_index", "mode_index")
    _data_attrs = {"long_name": "Propagation index"}


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
    EMEScalarFieldDataArray,
    EMEScalarModeFieldDataArray,
    EMESMatrixDataArray,
    EMECoefficientDataArray,
    EMEModeIndexDataArray,
    EMEFreqModeDataArray,
    ChargeDataArray,
    PointDataArray,
    CellDataArray,
    IndexedDataArray,
]
DATA_ARRAY_MAP = {data_array.__name__: data_array for data_array in DATA_ARRAY_TYPES}
