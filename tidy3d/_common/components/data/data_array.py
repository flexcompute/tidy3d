"""Storing tidy3d data at it's most fundamental level as xr.DataArray objects"""

from __future__ import annotations

import pathlib
from abc import ABC
from typing import TYPE_CHECKING, Any

import autograd.numpy as anp
import h5py
import numpy as np
import xarray as xr
from autograd.tracer import isbox
from pydantic_core import core_schema
from xarray.core import missing
from xarray.core.indexes import PandasIndex
from xarray.core.indexing import _outer_to_numpy_indexer
from xarray.core.utils import OrderedSet, either_dict_or_kwargs
from xarray.core.variable import as_variable

from tidy3d._common.compat import alignment
from tidy3d._common.components.autograd import TidyArrayBox, get_static, interpn, is_tidy_box
from tidy3d._common.components.geometry.bound_ops import bounds_contains
from tidy3d._common.constants import (
    HERTZ,
    MICROMETER,
    RADIAN,
    SECOND,
)
from tidy3d._common.exceptions import DataError, FileError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike
    from typing import Optional, Union

    from numpy.typing import NDArray
    from pydantic.annotated_handlers import GetCoreSchemaHandler
    from pydantic.json_schema import GetJsonSchemaHandler, JsonSchemaValue
    from xarray.core.types import InterpOptions, Self

    from tidy3d._common.components.autograd import InterpolationType
    from tidy3d._common.components.types.base import Axis, Bound


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
    _data_attrs: dict[str, str] = {}

    def __init__(self, data: Any, *args: Any, **kwargs: Any) -> None:
        # if data is a vanilla autograd box, convert to our box
        if isbox(data) and not is_tidy_box(data):
            data = TidyArrayBox.from_arraybox(data)
        # do the same for xr.Variable or xr.DataArray type
        elif isinstance(data, (xr.Variable, xr.DataArray)):
            if isbox(data.data) and not is_tidy_box(data.data):
                data.data = TidyArrayBox.from_arraybox(data.data)
        super().__init__(data, *args, **kwargs)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Core schema definition for validation & serialization."""

        def _initial_parser(value: Any) -> Self:
            if isinstance(value, cls):
                return value

            if isinstance(value, str) and value == cls.__name__:
                raise DataError(
                    f"Trying to load '{cls.__name__}' from string placeholder '{value}' "
                    "but the actual data is missing. DataArrays are not typically stored "
                    "in JSON. Load from HDF5 or ensure the DataArray object is provided."
                )

            try:
                instance = cls(value)
                if not isinstance(instance, cls):
                    raise TypeError(
                        f"Constructor for {cls.__name__} returned unexpected type {type(instance)}"
                    )
                return instance
            except Exception as e:
                raise ValueError(
                    f"Could not construct '{cls.__name__}' from input of type '{type(value)}'. "
                    f"Ensure input is compatible with xarray.DataArray constructor. Original error: {e}"
                ) from e

        validation_schema = core_schema.no_info_plain_validator_function(_initial_parser)
        validation_schema = core_schema.no_info_after_validator_function(
            cls._validate_dims, validation_schema
        )
        validation_schema = core_schema.no_info_after_validator_function(
            cls._assign_data_attrs, validation_schema
        )
        validation_schema = core_schema.no_info_after_validator_function(
            cls._assign_coord_attrs, validation_schema
        )

        def _serialize_to_name(instance: Self) -> str:
            return type(instance).__name__

        # serialization behavior:
        # - for JSON ('json' mode), use the _serialize_to_name function.
        # - for Python ('python' mode), use Pydantic's default for the object type
        serialization_schema = core_schema.plain_serializer_function_ser_schema(
            _serialize_to_name,
            return_schema=core_schema.str_schema(),
            when_used="json",
        )

        return core_schema.json_or_python_schema(
            python_schema=validation_schema,
            json_schema=validation_schema,  # Use same validation rules for JSON input
            serialization=serialization_schema,
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema_obj: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """JSON schema definition (defines how it LOOKS in a schema, not the data)."""
        return {
            "type": "string",
            "title": cls.__name__,
            "description": (
                f"Placeholder for a '{cls.__name__}' object. Actual data is typically "
                "serialized separately (e.g., via HDF5) and not embedded in JSON."
            ),
        }

    @classmethod
    def _validate_dims(cls, val: Self) -> Self:
        """Make sure the dims are the same as ``_dims``, then put them in the correct order."""
        if set(val.dims) != set(cls._dims):
            raise ValueError(
                f"Wrong dims for {cls.__name__}, expected '{cls._dims}', got '{val.dims}'"
            )
        if val.dims != cls._dims:
            val = val.transpose(*cls._dims)
        return val

    @classmethod
    def _assign_data_attrs(cls, val: Self) -> Self:
        """Assign the correct data attributes to the :class:`.DataArray`."""
        for attr_name, attr_val in cls._data_attrs.items():
            val.attrs[attr_name] = attr_val
        return val

    @classmethod
    def _assign_coord_attrs(cls, val: Self) -> Self:
        """Assign the correct coordinate attributes to the :class:`.DataArray`."""
        target_dims = set(val.dims) & set(cls._dims) & set(val.coords)
        for dim in target_dims:
            template = DIM_ATTRS.get(dim)
            if not template:
                continue

            coord_attrs = val.coords[dim].attrs
            missing = {k: v for k, v in template.items() if coord_attrs.get(k) != v}
            coord_attrs.update(missing)
        return val

    def _interp_validator(self, field_name: Optional[str] = None) -> None:
        """Ensure the data can be interpolated or selected by checking for duplicate coordinates.

        NOTE
        ----
        This does not check every 'DataArray' by default. Instead, when required, this check can be
        called from a validator, as is the case with 'CustomMedium' and 'CustomFieldSource'.
        """
        if field_name is None:
            field_name = self.__class__.__name__

        for dim, coord in self.coords.items():
            if coord.to_index().duplicated().any():
                raise DataError(
                    f"Field '{field_name}' contains duplicate coordinates in dimension '{dim}'. "
                    "Duplicates can be removed by running "
                    f"'{field_name}={field_name}.drop_duplicates(dim=\"{dim}\")'."
                )

    def __eq__(self, other: Any) -> bool:
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
    def values(self) -> NDArray:
        """
        The array's data converted to a numpy.ndarray.

        Returns
        -------
        np.ndarray
            The values of the DataArray.
        """
        return self.data if isbox(self.data) else super().values

    @values.setter
    def values(self, value: Any) -> None:
        self.variable.values = value

    def to_numpy(self) -> np.ndarray:
        """Return `.data` when traced to avoid `dtype=object` NumPy conversion."""
        return self.data if isbox(self.data) else super().to_numpy()

    @property
    def abs(self) -> Self:
        """Absolute value of data array."""
        return abs(self)

    @property
    def angle(self) -> Self:
        """Angle or phase value of data array."""
        values = np.angle(self.values)
        return type(self)(values, coords=self.coords)

    @property
    def is_uniform(self) -> bool:
        """Whether each element is of equal value in the data array"""
        raw_data = self.data.ravel()
        return np.allclose(raw_data, raw_data[0])

    def to_hdf5(self, fname: Union[PathLike, h5py.File], group_path: str) -> None:
        """Save an ``xr.DataArray`` to the hdf5 file or file handle with a given path to the group."""
        if isinstance(fname, (str, pathlib.Path)):
            path = pathlib.Path(fname)
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as f_handle:
                self.to_hdf5_handle(f_handle=f_handle, group_path=group_path)
        else:
            self.to_hdf5_handle(f_handle=fname, group_path=group_path)

    def to_hdf5_handle(self, f_handle: h5py.File, group_path: str) -> None:
        """Save an ``xr.DataArray`` to the hdf5 file handle with a given path to the group."""
        sub_group = f_handle.create_group(group_path)
        sub_group[DATA_ARRAY_VALUE_NAME] = get_static(self.data)
        for key, val in self.coords.items():
            if val.dtype == "<U1":
                sub_group[key] = val.values.tolist()
            else:
                sub_group[key] = val

    @classmethod
    def from_hdf5(cls, fname: PathLike, group_path: str) -> Self:
        """Load a DataArray from an hdf5 file with a given path to the group."""
        path = pathlib.Path(fname)
        with h5py.File(path, "r") as f:
            sub_group = f[group_path]
            values = np.array(sub_group[DATA_ARRAY_VALUE_NAME])
            coords = {dim: np.array(sub_group[dim]) for dim in cls._dims if dim in sub_group}
            for key, val in coords.items():
                if val.dtype == "O":
                    coords[key] = [byte_string.decode() for byte_string in val.tolist()]
            return cls(values, coords=coords, dims=cls._dims)

    @classmethod
    def from_file(cls, fname: PathLike, group_path: str) -> Self:
        """Load a DataArray from an hdf5 file with a given path to the group."""
        path = pathlib.Path(fname)
        if not any(suffix.lower() == ".hdf5" for suffix in path.suffixes):
            raise FileError(
                f"'DataArray' objects must be written to '.hdf5' format. Given filename of {path}."
            )
        return cls.from_hdf5(fname=path, group_path=group_path)

    def __hash__(self) -> int:
        """Generate hash value for a :class:`.DataArray` instance, needed for custom components."""
        import dask

        token_str = dask.base.tokenize(self)
        return hash(token_str)

    def multiply_at(self, value: complex, coord_name: str, indices: list[int]) -> Self:
        """Multiply self by value at indices."""
        if isbox(self.data) or isbox(value):
            return self._ag_multiply_at(value, coord_name, indices)

        self_mult = self.copy()
        self_mult[{coord_name: indices}] *= value
        return self_mult

    def _ag_multiply_at(self, value: complex, coord_name: str, indices: list[int]) -> Self:
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
    def _ag_interp_func(
        var: xr.Variable,
        indexes_coords: dict[str, tuple[xr.Variable, xr.Variable]],
        method: InterpolationType,
        **kwargs: Any,
    ) -> xr.Variable:
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
        method : Literal["nearest", "linear"]
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

            result = interpn([xn.data for xn in x], data, xi, method=method, **kwargs)

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

    def _with_updated_data(self, data: np.ndarray, coords: dict[str, Any]) -> DataArray:
        """Make copy of ``DataArray`` with ``data`` at specified ``coords``, autograd compatible

        Constraints / Edge cases:
            - `coords` must map to a specific value eg {x: '1'}, does not broadcast to arrays
            - `data` will be reshaped to try to match `self.shape` except where `coords` present
        """

        # make mask
        mask = xr.zeros_like(self, dtype=bool)
        mask.loc[coords] = True

        # reshape `data` to line up with `self.dims`, with shape of 1 along the selected axis
        old_data = self.data
        new_shape = list(old_data.shape)
        for i, dim in enumerate(self.dims):
            if dim in coords:
                new_shape[i] = 1
        try:
            new_data = data.reshape(new_shape)
        except ValueError as e:
            raise ValueError(
                "Couldn't reshape the supplied 'data' to update 'DataArray'. The provided data was "
                f"of shape {data.shape} and tried to reshape to {new_shape}. If you encounter this "
                "error please raise an issue on the tidy3d github repository with the context."
            ) from e

        # broadcast data to repeat data along the selected dimensions to match mask
        new_data = new_data + np.zeros_like(old_data)

        new_data = np.where(mask, new_data, old_data)

        return self.copy(deep=True, data=new_data)


class FreqDataArray(DataArray):
    """Frequency-domain array.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> fd = FreqDataArray((1+1j) * np.random.random((2,)), coords=dict(f=f))
    """

    __slots__ = ()
    _dims = ("f",)


class AbstractSpatialDataArray(DataArray, ABC):
    """Spatial distribution."""

    __slots__ = ()
    _dims = ("x", "y", "z")
    _data_attrs = {"long_name": "field value"}

    @property
    def _spatially_sorted(self) -> Self:
        """Check whether sorted and sort if not."""
        needs_sorting = []
        for axis in "xyz":
            axis_coords = self.coords[axis].values
            if len(axis_coords) > 1 and np.any(axis_coords[1:] < axis_coords[:-1]):
                needs_sorting.append(axis)

        if len(needs_sorting) > 0:
            return self.sortby(needs_sorting)

        return self

    def sel_inside(self, bounds: Bound) -> Self:
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
                        ind_min = max(0, (coord >= smin).data.argmax() - 1)

                    if smax > coord[-1]:
                        ind_max = length - 1
                    else:
                        ind_max = (coord >= smax).data.argmax()

                    comp_inds = np.arange(ind_min, ind_max + 1)

            inds_list.append(comp_inds)

        return sorted_self.isel(x=inds_list[0], y=inds_list[1], z=inds_list[2])

    def does_cover(self, bounds: Bound, rtol: float = 0.0, atol: float = 0.0) -> bool:
        """Check whether data fully covers specified by ``bounds`` spatial region. If data contains
        only one point along a given direction, then it is assumed the data is constant along that
        direction and coverage is not checked.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        rtol : float = 0.0
            Relative tolerance for comparing bounds
        atol : float = 0.0
            Absolute tolerance for comparing bounds

        Returns
        -------
        bool
            Full cover check outcome.
        """
        if any(bmin > bmax for bmin, bmax in zip(*bounds)):
            raise DataError(
                "Min and max bounds must be packaged as '(minx, miny, minz), (maxx, maxy, maxz)'."
            )
        xyz = [self.x, self.y, self.z]
        self_min = [0] * 3
        self_max = [0] * 3
        for dim in range(3):
            coords = xyz[dim]
            if len(coords) == 1:
                self_min[dim] = bounds[0][dim]
                self_max[dim] = bounds[1][dim]
            else:
                self_min[dim] = np.min(coords)
                self_max[dim] = np.max(coords)
        self_bounds = (tuple(self_min), tuple(self_max))
        return bounds_contains(self_bounds, bounds, rtol=rtol, atol=atol)


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


class TriangleMeshDataArray(DataArray):
    """Data of the triangles of a surface mesh as in the STL file format."""

    __slots__ = ()
    _dims = ("face_index", "vertex_index", "axis")
    _data_attrs = {"long_name": "surface mesh triangles"}


class TimeDataArray(DataArray):
    """Time-domain array.

    Example
    -------
    >>> t = [0, 1e-12, 2e-12]
    >>> td = TimeDataArray((1+1j) * np.random.random((3,)), coords=dict(t=t))
    """

    __slots__ = ()
    _dims = ("t",)


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

    def reflect(self, axis: Axis, center: float, reflection_only: bool = False) -> Self:
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


DATA_ARRAY_TYPES: list[type[DataArray]] = [
    FreqDataArray,
    TriangleMeshDataArray,
    TimeDataArray,
    SpatialDataArray,
    ScalarFieldDataArray,
]
DATA_ARRAY_MAP = {data_array.__name__: data_array for data_array in DATA_ARRAY_TYPES}
