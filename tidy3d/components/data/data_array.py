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

from tidy3d.compat import alignment
from tidy3d.components.autograd import TidyArrayBox, get_static, interpn, is_tidy_box
from tidy3d.components.geometry.bound_ops import bounds_contains
from tidy3d.constants import (
    AMP,
    COULOMB,
    HERTZ,
    MICROMETER,
    OHM,
    PICOSECOND_PER_NANOMETER_PER_KILOMETER,
    RADIAN,
    SECOND,
    STERADIAN,
    VOLT,
    WATT,
)
from tidy3d.exceptions import DataError, FileError, format_chained_exception_message

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping
    from os import PathLike

    from numpy.typing import NDArray
    from pydantic.annotated_handlers import GetCoreSchemaHandler
    from pydantic.json_schema import GetJsonSchemaHandler, JsonSchemaValue
    from xarray.core.types import InterpOptions, Self

    from tidy3d.components.autograd import InterpolationType
    from tidy3d.components.grid.grid import Coords
    from tidy3d.components.types import Axis, Bound, BoundOptional
    from tidy3d.components.types.base import Coordinate

# maps the dimension names to their attributes
DIPOLE_EMISSION_INTENSITY_UNITS = f"{WATT}/({STERADIAN} * ({COULOMB}*{MICROMETER})^2)"

DIM_ATTRS = {
    "x": {"units": MICROMETER, "long_name": "x position"},
    "y": {"units": MICROMETER, "long_name": "y position"},
    "z": {"units": MICROMETER, "long_name": "z position"},
    "f": {"units": HERTZ, "long_name": "frequency"},
    "t": {"units": SECOND, "long_name": "time"},
    "v": {"units": VOLT, "long_name": "voltage"},
    "direction": {"long_name": "propagation direction"},
    "mode_index": {"long_name": "mode index"},
    "terminal_label": {"long_name": "terminal label"},
    "terminal_label_out": {"long_name": "output terminal label"},
    "terminal_label_in": {"long_name": "input terminal label"},
    "eme_port_index": {"long_name": "EME port index"},
    "eme_cell_index": {"long_name": "EME cell index"},
    "eme_interface_index": {"long_name": "EME interface index"},
    "mode_index_in": {"long_name": "mode index in"},
    "mode_index_out": {"long_name": "mode index out"},
    "trace_index_in": {"long_name": "trace index in"},
    "trace_index_out": {"long_name": "trace index out"},
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
    "dipole_axis": {"long_name": "dipole orientation axis"},
    "polarization": {"long_name": "emission polarization"},
    "angle": {"long_name": "emission angle index"},
    "spherical_coordinate": {"long_name": "spherical coordinate"},
}


# name of the DataArray.values in the hdf5 file (xarray's default name too)
DATA_ARRAY_VALUE_NAME = "__xarray_dataarray_variable__"


class _SanitizedPlotProxy:
    """Proxy xarray's plot accessor through a plotting-safe DataArray copy."""

    __slots__ = ("_data_array",)

    def __init__(self, data_array: xr.DataArray) -> None:
        self._data_array = data_array

    def _accessor(self) -> xr.plot.accessor.DataArrayPlotAccessor:
        """Construct xarray's plot accessor on a static plotting-safe copy."""
        # Local import avoids a circular dependency because ``data.utils`` also
        # imports tidy3d DataArray subclasses for spatial data typing.
        from .utils import static_dataarray_for_plot

        return xr.DataArray.plot(static_dataarray_for_plot(self._data_array))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call xarray's plotting entrypoint on the sanitized accessor."""
        return self._accessor()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to xarray's plotting accessor."""
        return getattr(self._accessor(), name)


class _StructuredSpatialPlotProxy(_SanitizedPlotProxy):
    """Plot proxy that preserves structured spatial plot validation."""

    def __call__(self, *args: Any, field: bool = True, grid: bool = False, **kwargs: Any) -> Any:
        """Validate structured spatial-only kwargs before plotting."""
        if grid:
            raise DataError("The 'grid' argument is only supported for unstructured data.")
        if not field:
            raise DataError("The 'field' argument is only supported for unstructured data.")
        return super().__call__(*args, **kwargs)


class DataArray(xr.DataArray):
    """Subclass of ``xr.DataArray`` that requires _dims to match the keys of the coords."""

    # Always set __slots__ = () to avoid xarray warnings
    __slots__ = ()
    # stores an ordered tuple of strings corresponding to the data dimensions
    _dims: tuple[str, ...] = ()
    # stores a dictionary of attributes corresponding to the data values
    _data_attrs: dict[str, str] = {}

    def __init__(self, data: Any, *args: Any, **kwargs: Any) -> None:
        # if data is a vanilla autograd box, convert to our box
        if isbox(data) and not is_tidy_box(data):
            data = TidyArrayBox.from_arraybox(data)
        # do the same for xr.Variable or xr.DataArray type
        elif isinstance(data, xr.Variable | xr.DataArray):
            if isbox(data.data) and not is_tidy_box(data.data):
                data.data = TidyArrayBox.from_arraybox(data.data)
        super().__init__(data, *args, **kwargs)

    @classmethod
    def _constructor_kwargs_for_pydantic(cls) -> dict[str, Any]:
        """Additional constructor kwargs used only by pydantic parsing."""
        return {}

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
                instance = cls(value, **cls._constructor_kwargs_for_pydantic())
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

    def _interp_validator(self, field_name: str | None = None) -> None:
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
    def plot(self) -> _SanitizedPlotProxy:
        """xarray-style plotting accessor on a plotting-safe DataArray copy."""
        return _SanitizedPlotProxy(self)

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

    def to_hdf5(self, fname: PathLike | h5py.File, group_path: str) -> None:
        """Save an ``xr.DataArray`` to the hdf5 file or file handle with a given path to the group."""
        if isinstance(fname, str | pathlib.Path):
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
        for key in self._dims:
            if key not in self.coords:
                continue
            val = self.coords[key]
            if val.dtype.kind == "U":
                # Convert Unicode strings to list for HDF5 storage
                sub_group[key] = val.values.tolist()
            else:
                sub_group[key] = val

    @classmethod
    def _from_hdf5_handle(cls, f_handle: h5py.File, group_path: str) -> Self:
        """Load a DataArray from an open hdf5 file handle with a given group path."""
        sub_group = f_handle[group_path]
        values = np.array(sub_group[DATA_ARRAY_VALUE_NAME])
        coords = {dim: np.array(sub_group[dim]) for dim in cls._dims if dim in sub_group}
        for key, val in coords.items():
            if val.dtype == "O":
                coords[key] = [byte_string.decode() for byte_string in val.tolist()]
        return cls(values, coords=coords, dims=cls._dims)

    @classmethod
    def from_hdf5(cls, fname: PathLike | h5py.File, group_path: str) -> Self:
        """Load a DataArray from an hdf5 file or open file handle with a given group path."""
        if isinstance(fname, h5py.File):
            return cls._from_hdf5_handle(f_handle=fname, group_path=group_path)

        path = pathlib.Path(fname)
        with h5py.File(path, "r") as f_handle:
            return cls._from_hdf5_handle(f_handle=f_handle, group_path=group_path)

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
        coords: Mapping[Any, Any] | None = None,
        method: InterpOptions = "linear",
        assume_sorted: bool = False,
        kwargs: Mapping[str, Any] | None = None,
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
                format_chained_exception_message(
                    "Couldn't reshape the supplied 'data' to update 'DataArray'. The provided "
                    f"data was of shape {data.shape} and tried to reshape to {new_shape}. If "
                    "you encounter this error please raise an issue on the tidy3d github "
                    "repository with this context.",
                    e,
                )
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


class FreqVoltageDataArray(DataArray):
    """Frequency-domain array.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> v = [0.1, 0.2, 0.3]
    >>> coords = dict(f=f, v=v)
    >>> fd = FreqVoltageDataArray((1+1j) * np.random.random((2, 3)), coords=coords)
    """

    __slots__ = ()
    _dims = (
        "f",
        "v",
    )


class ModeDataArray(DataArray):
    """Mode index data array.
    Example
    -------
    >>> mode_index = np.arange(4)
    >>> coords = dict(mode_index=mode_index)
    >>> data = ModeDataArray((1+1j) * np.random.random(4), coords=coords)
    """

    __slots__ = ()
    _dims = ("mode_index",)


class TerminalDataArray(DataArray):
    """Terminal index data array.
    Example
    -------
    >>> terminal_label = ["t0", "t1", "t2", "t3", "t4"]
    >>> coords = dict(terminal_label=terminal_label)
    >>> data = TerminalDataArray((1+1j) * np.random.random(5), coords=coords)
    """

    __slots__ = ()
    _dims = ("terminal_label",)


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


class FreqTerminalDataArray(DataArray):
    """Array over frequency and terminal index.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> terminal_label = ["t0", "t1", "t2", "t3", "t4"]
    >>> coords = dict(f=f, terminal_label=terminal_label)
    >>> fd = FreqTerminalDataArray((1+1j) * np.random.random((2, 5)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "terminal_label")


class FreqModeModeDataArray(DataArray):
    """Array over frequency, mode index, and mode index.
    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index_out = np.arange(5)
    >>> mode_index_in = np.arange(5)
    >>> coords = dict(f=f, mode_index_out=mode_index_out, mode_index_in=mode_index_in)
    >>> fd = FreqModeModeDataArray((1+1j) * np.random.random((2, 5, 5)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "mode_index_out", "mode_index_in")


class FreqTerminalModeDataArray(DataArray):
    """Array over frequency, terminal index, and mode index.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> terminal_label = ["t0", "t1"]
    >>> coords = dict(f=f, terminal_label=terminal_label, mode_index=mode_index)
    >>> fd = FreqTerminalModeDataArray((1+1j) * np.random.random((2, 2, 5)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "terminal_label", "mode_index")


class FreqModeTerminalDataArray(DataArray):
    """Array over frequency, mode index, and terminal index.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> terminal_label = ["t0", "t1"]
    >>> coords = dict(f=f, mode_index=mode_index, terminal_label=terminal_label)
    >>> fd = FreqModeTerminalDataArray((1+1j) * np.random.random((2, 5, 2)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "mode_index", "terminal_label")


class FreqTerminalTerminalDataArray(DataArray):
    """Array over frequency, terminal index, and terminal index.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> terminal_label_out = ["t0", "t1"]
    >>> terminal_label_in = ["t0", "t1"]
    >>> coords = dict(f=f, terminal_label_out=terminal_label_out, terminal_label_in=terminal_label_in)
    >>> fd = FreqTerminalTerminalDataArray((1+1j) * np.random.random((2, 2, 2)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "terminal_label_out", "terminal_label_in")


class TimeDataArray(DataArray):
    """Time-domain array.

    Example
    -------
    >>> t = [0, 1e-12, 2e-12]
    >>> td = TimeDataArray((1+1j) * np.random.random((3,)), coords=dict(t=t))
    """

    __slots__ = ()
    _dims = ("t",)


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
    def plot(self) -> _StructuredSpatialPlotProxy:
        """Plot the spatial data.

        Accepts the same arguments as xarray's ``DataArray.plot()``. The extra
        ``grid`` and ``field`` keyword arguments are accepted for API
        compatibility with :meth:`TriangularGridDataset.plot`, but grid overlay
        is not supported on structured data.

        Parameters
        ----------
        field : bool = True
            Whether to plot the data field. Must be ``True`` for structured
            data.
        grid : bool = False
            Not supported for structured data. Raises ``DataError`` if
            ``True``.
        """
        return _StructuredSpatialPlotProxy(self)

    @property
    def _spatially_sorted(self) -> Self:
        """Check whether sorted and sort if not."""
        needs_sorting = []
        for axis in "xyz":
            axis_coords = np.atleast_1d(self.coords[axis].values)
            if axis_coords.size > 1 and np.any(axis_coords[1:] < axis_coords[:-1]):
                needs_sorting.append(axis)

        if len(needs_sorting) > 0:
            return self.sortby(needs_sorting)

        return self

    def shifted_spatial_coords(self, center: Coordinate) -> Self:
        """Return a copy with spatial coordinates shifted by ``center``."""
        shifted = self
        for axis, dim in enumerate("xyz"):
            if dim in shifted.coords:
                coord_vals = np.asarray(shifted.coords[dim].data)
                shifted = shifted.assign_coords({dim: coord_vals + center[axis]})
        return shifted

    def interpolate_to_grid(
        self,
        grid: Coords,
        *,
        offset: Coordinate | None = None,
        method: InterpOptions = "linear",
        target_dims: tuple[str, ...] | None = None,
    ) -> Self:
        """Interpolate onto a target grid, with optional spatial offset and output ordering."""
        if offset is None:
            interpolated = grid.spatial_interp(self, method)
        else:
            interpolated = grid.spatial_interp(self.shifted_spatial_coords(offset), method)
        if target_dims is not None and tuple(interpolated.dims) != tuple(target_dims):
            interpolated = interpolated.transpose(*target_dims)
        return interpolated

    def interp_within_domain(
        self,
        interp_coords: dict,
        clip_bounds: BoundOptional,
        assume_sorted: bool = False,
    ) -> Self:
        """Interpolate to ``interp_coords``, clipping source data to ``clip_bounds``.

        This is a bounds-aware variant of :meth:`interp` that:

        1. **Clips** the source data to ``clip_bounds``, keeping only data
           points whose coordinates lie on or within the bounds.
        2. **Clamps** all target coordinates to the clipped data range,
           so any target outside that range receives the nearest edge value.
        3. **Interpolates** on the clipped data using the clamped
           coordinates.
        4. **Zeros** the result at target coordinates that are strictly
           outside ``clip_bounds``.

        These steps allow for regular interpolation for strictly interior points,
        while points on the ``clip_bounds`` effectively choose the nearest value from the
        source data, and points strictly outside the bounds are set to 0.

        Each of the six bounds (min/max for x/y/z) is independently
        optional.  A ``None`` value leaves that side unclipped.

        Parameters
        ----------
        interp_coords : dict
            Target interpolation coordinates, keyed by dimension name
            (``"x"``, ``"y"``, ``"z"``).
        clip_bounds : BoundOptional
            ``(min_tuple, max_tuple)`` where each element is a 3-tuple of
            ``Optional[float]``.  Axes set to ``None`` are not clipped.
        assume_sorted : bool = False
            If True, skip sorting of coordinates.

        Returns
        -------
        Self
            Interpolated data with coordinates matching ``interp_coords``.
            Values outside ``clip_bounds`` are zero.
        """
        bound_min, bound_max = clip_bounds

        # Build clip selector from bounds
        clip_sel = {}
        for dim in interp_coords:
            ax = "xyz".index(dim)
            lo = bound_min[ax]
            hi = bound_max[ax]
            if lo is not None or hi is not None:
                clip_sel[dim] = slice(lo, hi)

        clipped = self.sel(clip_sel) if clip_sel else self

        # Clamp target coordinates to clipped data range
        # Emulates picking nearest source data for edge values
        clamped = {
            dim: np.clip(
                vals,
                float(clipped.coords[dim][0]),
                float(clipped.coords[dim][-1]),
            )
            for dim, vals in interp_coords.items()
        }

        result = clipped.interp(**clamped, assume_sorted=assume_sorted)
        # Assign the result coordinates back to the original interpolation points.
        result = result.assign_coords(interp_coords)

        # Zero out values at coordinates strictly outside the clip bounds.
        # Clamping mapped these to edge values, but they should be zero.
        for dim, vals in interp_coords.items():
            ax = "xyz".index(dim)
            lo = bound_min[ax]
            hi = bound_max[ax]
            if lo is None and hi is None:
                continue
            mask = np.zeros(len(vals), dtype=bool)
            if lo is not None:
                mask |= np.asarray(vals) < lo
            if hi is not None:
                mask |= np.asarray(vals) > hi
            if np.any(mask):
                axis_num = result.get_axis_num(dim)
                shape = [1] * result.ndim
                shape[axis_num] = -1
                result.values = np.where(mask.reshape(shape), 0, result.values)

        return result

    def sel_inside(self, bounds: Bound, *, include_interp_padding: bool = True) -> Self:
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
        include_interp_padding : bool = True
            If ``True`` (default), include neighbor points around bounds to support interpolation.
            If ``False``, keep only points whose coordinates are inside bounds.
        """
        if any(bmin > bmax for bmin, bmax in zip(*bounds)):
            raise DataError(
                "Min and max bounds must be packaged as '(minx, miny, minz), (maxx, maxy, maxz)'."
            )

        # make sure data is sorted with respect to coordinates
        sorted_self = self._spatially_sorted

        if not include_interp_padding:
            selected = sorted_self
            for coord, smin, smax, dim in zip(
                (sorted_self.x, sorted_self.y, sorted_self.z),
                bounds[0],
                bounds[1],
                "xyz",
            ):
                coord_vals = np.atleast_1d(coord.values)
                if coord_vals.size <= 1:
                    continue
                selected = selected.sel({dim: slice(smin, smax)})
            return selected

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

    def reflect(
        self, axis: Axis, center: float, reflection_only: bool = False, symmetry: float = 1
    ) -> Self:
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
        symmetry : float = 1
            Symmetry factor of the reflection.

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

            tmp_arr = SpatialDataArray(sorted_self.data * symmetry, coords=coords_dict)

            return tmp_arr.sortby("xyz"[axis])

        shape = np.array(np.shape(data))
        old_len = shape[axis]
        shape[axis] = 2 * old_len - num_duplicates

        ind_left = [slice(shape[0]), slice(shape[1]), slice(shape[2])]
        ind_right = [slice(shape[0]), slice(shape[1]), slice(shape[2])]

        ind_left[axis] = slice(old_len - 1, None, -1)
        ind_right[axis] = slice(old_len - num_duplicates, None)

        new_data = np.zeros(shape)

        new_data[ind_left[0], ind_left[1], ind_left[2]] = data * symmetry
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


class ModeAmpsTimeDataArray(DataArray):
    """Forward and backward propagating complex-valued mode amplitudes in the time domain.

    Example
    -------
    >>> direction = ["+", "-"]
    >>> t = [0, 1e-12, 2e-12]
    >>> mode_index = np.arange(4)
    >>> coords = dict(direction=direction, t=t, mode_index=mode_index)
    >>> data = ModeAmpsTimeDataArray((1+1j) * np.random.random((2, 3, 4)), coords=coords)
    """

    __slots__ = ()
    _dims = ("direction", "t", "mode_index")
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


class DipoleEmissionDataArray(DataArray):
    """Radiation intensity stored by a single dipole-emission monitor.

    Example
    -------
    >>> dipole_axis = ["x", "y", "z"]
    >>> f = np.linspace(1e14, 2e14, 3)
    >>> coords = dict(dipole_axis=dipole_axis, f=f)
    >>> values = np.random.random((len(dipole_axis), len(f)))
    >>> data = DipoleEmissionDataArray(values, coords=coords)
    """

    __slots__ = ()
    _dims = ("dipole_axis", "f")
    _data_attrs = {
        "long_name": "angular radiation intensity per dipole moment squared",
        "units": DIPOLE_EMISSION_INTENSITY_UNITS,
    }


class DipoleEmissionPositionDataArray(DataArray):
    """Position samples stored by a single dipole-emission monitor.

    Example
    -------
    >>> index = np.arange(2)
    >>> dipole_axis = ["x", "y", "z"]
    >>> f = np.linspace(1e14, 2e14, 3)
    >>> coords = dict(index=index, dipole_axis=dipole_axis, f=f)
    >>> values = np.random.random((len(index), len(dipole_axis), len(f)))
    >>> data = DipoleEmissionPositionDataArray(values, coords=coords)
    """

    __slots__ = ()
    _dims = ("index", "dipole_axis", "f")
    _data_attrs = {
        "long_name": "position-resolved angular radiation intensity per dipole moment squared",
        "units": DIPOLE_EMISSION_INTENSITY_UNITS,
    }


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
    _dims = ("T",)


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


class EMEInterfaceSMatrixDataArray(DataArray):
    """Scattering matrix elements at a single cell interface for a fixed pair of ports,
    possibly with an extra sweep index.
    Example
    -------
    >>> mode_index_in = [0, 1]
    >>> mode_index_out = [0, 1, 2]
    >>> eme_cell_index = [2, 4]
    >>> f = [2e14]
    >>> sweep_index = np.arange(10)
    >>> coords = dict(
    ...     f=f,
    ...     sweep_index=sweep_index,
    ...     eme_cell_index=eme_cell_index,
    ...     mode_index_out=mode_index_out,
    ...     mode_index_in=mode_index_in,
    ... )
    >>> fd = EMEInterfaceSMatrixDataArray((1 + 1j) * np.random.random((1, 10, 2, 3, 2)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "sweep_index", "eme_cell_index", "mode_index_out", "mode_index_in")
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


class EMEFluxDataArray(DataArray):
    """Power flux of an EME mode, also indexed by EME cell.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> sweep_index = np.arange(2)
    >>> eme_cell_index = np.arange(5)
    >>> mode_index = np.arange(4)
    >>> coords = dict(f=f, sweep_index=sweep_index, eme_cell_index=eme_cell_index, mode_index=mode_index)
    >>> data = EMEFluxDataArray(np.random.random((2,2,5,4)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "sweep_index", "eme_cell_index", "mode_index")
    _data_attrs = {"units": WATT, "long_name": "flux"}


class EMETraceMetricDataArray(DataArray):
    """Field-metric matrix used by advanced EME interface diagnostics."""

    __slots__ = ()
    _dims = ("f", "sweep_index", "trace_index_out", "trace_index_in")
    _data_attrs = {"long_name": "interface trace metric"}


class EMEInterfaceCellIndexDataArray(DataArray):
    """Left or right EME cell index for each aggregated EME interface."""

    __slots__ = ()
    _dims = ("eme_interface_index",)
    _data_attrs = {"long_name": "EME interface cell index"}


class EMEInterfaceDiagnosticDataArray(DataArray):
    """Scalar EME interface diagnostic value.

    Indexed by interface, incident port, and incident mode.
    """

    __slots__ = ()
    _dims = ("f", "sweep_index", "eme_interface_index", "eme_port_index", "mode_index")
    _data_attrs = {"long_name": "interface diagnostic"}


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


class SteadyVoltageDataArray(DataArray):
    """Steady voltage data array. Data array used with steady state
    simulations with voltage as dimension.

    Example
    -------
    >>> import tidy3d as td
    >>> intensities = [0, 1, 4]
    >>> V = [-1, -0.5, 0]
    >>> voltage_dataarray = td.SteadyVoltageDataArray(data=intensities, coords={"v": V})
    """

    __slots__ = ()
    _dims = ("v",)


class ConvergenceHistoryDataArray(DataArray):
    """Per-iteration residual trace across a voltage sweep and solution components.

    Dimensions:

    * ``v`` -- sweep bias [V]
    * ``pseudo_step`` -- Newton iteration index (0-based); ragged biases are
      NaN-padded on this axis to the longest trace in the sweep.
    * ``component`` -- solution component label (e.g. ``"potential"``,
      ``"electrons"``, ``"holes"``, ``"temperature"``).

    Values are residual norms in the native units of the corresponding solution
    variable (volts for ``"potential"``; um^-3 for the carrier concentrations).
    NaN at ``(v, k, c)`` means bias ``v`` did not reach iteration ``k`` for
    component ``c``.

    Example
    -------
    >>> import numpy as np
    >>> import tidy3d as td
    >>> V = [-1.0, 0.0]
    >>> steps = [0, 1, 2]
    >>> comps = ["potential"]
    >>> data = np.array([[[3.2e-1], [1.7e-1], [1.1e-8]],
    ...                  [[2.1e-1], [9.0e-2], [np.nan]]])
    >>> history = td.ConvergenceHistoryDataArray(
    ...     data=data, coords={"v": V, "pseudo_step": steps, "component": comps}
    ... )
    """

    __slots__ = ()
    _dims = ("v", "pseudo_step", "component")


class PointDataArray(DataArray):
    """A two-dimensional array that stores coordinates/field components for a collection of points.
    Dimension ``index`` denotes the index of a point in the collection, and dimension ``axis``
    denotes the field component (or point coordinate) in that direction.

    Example
    -------
    >>> point_array = PointDataArray(
    ...     (1+1j) * np.random.random((5, 3)), coords=dict(index=np.arange(5), axis=np.arange(3)),
    ... )
    >>> # get coordinates of a point number 3
    >>> point3 = point_array.sel(index=3)
    >>> # get x coordinates of all points
    >>> x_coords = point_array.sel(axis=0)
    >>>
    >>> field_da = PointDataArray(
    ...     np.random.random((120, 3)), coords=dict(index=np.arange(120), axis=np.arange(3)),
    ... )
    >>> # get field of point number 90
    >>> field_point90 = field_da.sel(index=90)
    >>> # get z component of all points
    >>> z_field = field_da.sel(axis=2)
    """

    __slots__ = ()
    _dims = ("index", "axis")

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
                and shape[1] == 3
                and not isinstance(data, xr.Variable | xr.DataArray)
            ):
                kwargs["dims"] = ("index", "axis")
                kwargs["coords"] = {"index": np.arange(shape[0]), "axis": np.arange(3)}
        super().__init__(data, *args, **kwargs)

    @classmethod
    def _constructor_kwargs_for_pydantic(cls) -> dict[str, Any]:
        """Keep pydantic parsing strict for raw lists and arrays."""
        return {"_skip_bare_canonicalization": True}


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


class IndexedVoltageDataArray(DataArray):
    """Stores a two-dimensional array with coordinates ``index`` and ``voltage``, where
    ``index`` is usually associated with ``PointDataArray`` and ``voltage`` indicates at what
    bias/DC-voltage the data was obtained with.

    Example
    -------
    >>> indexed_array = IndexedVoltageDataArray(
    ...     (1+1j) * np.random.random((3,2)), coords=dict(index=np.arange(3), voltage=[-1, 1])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "voltage")


class IndexedTimeDataArray(DataArray):
    """Stores a two-dimensional array with coordinates ``index`` and ``t``, where
    ``index`` is usually associated with ``PointDataArray`` and ``t`` indicates at what
    simulated time the data was obtained.

    Example
    -------
    >>> indexed_array = IndexedTimeDataArray(
    ...     (1+1j) * np.random.random((3,2)), coords=dict(index=np.arange(3), t=[0, 1])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "t")


class IndexedFieldVoltageDataArray(DataArray):
    """Stores indexed values of vector fields for different voltages. It is typically used
    in conjuction with a ``PointDataArray`` to store point-associated vector data.
    Example
    -------
    >>> indexed_array = IndexedFieldVoltageDataArray(
    ...     (1+1j) * np.random.random((4,3,2)), coords=dict(index=np.arange(4), axis=np.arange(3), voltage=[-1, 1])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "axis", "voltage")


class SpatialVoltageDataArray(AbstractSpatialDataArray):
    """Spatial distribution with voltage mapping.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> v = [-1, 1]
    >>> coords = dict(x=x, y=y, z=z, voltage=v)
    >>> fd = SpatialVoltageDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    """

    __slots__ = ()
    _dims = ("x", "y", "z", "voltage")


class PerturbationCoefficientDataArray(DataArray):
    __slots__ = ()
    _dims = ("wvl", "coeff")


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


class IndexedSurfaceFreqDataArray(DataArray):
    """Stores indexed values of scalar fields on the sides of a surface. It is typically used
    in conjuction with a ``PointDataArray`` to store point-associated scalar data.

    Example
    -------
    >>> surface_side_array = IndexedSurfaceFreqDataArray(
    ...     (1+1j) * np.random.random((4,2,1)), coords=dict(index=np.arange(4), side=["outside", "inside"], f=[1e9])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "side", "f")


class IndexedSurfaceTimeDataArray(DataArray):
    """Stores indexed values of scalar fields on the sides of a surface. It is typically used
    in conjuction with a ``PointDataArray`` to store point-associated scalar data.

    Example
    -------
    >>> surface_side_array = IndexedSurfaceTimeDataArray(
    ...     (1+1j) * np.random.random((4,2,1)), coords=dict(index=np.arange(4), side=["outside", "inside"], t=[1e9])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "side", "t")


class IndexedSurfaceFieldDataArray(DataArray):
    """Stores indexed values of vector fields on the sides of a surface in frequency domain.
    It is typically used in conjuction with a ``PointDataArray`` to store point-associated vector data.

    Example
    -------
    >>> indexed_array = IndexedSurfaceFieldDataArray(
    ...     (1+1j) * np.random.random((4,2,3,1)), coords=dict(index=np.arange(4), side=["outside", "inside"], axis=np.arange(3), f=[1e9])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "side", "axis", "f")


class IndexedSurfaceFieldTimeDataArray(DataArray):
    """Stores indexed values of vector fields on the sides of a surface in time domain.
    It is typically used in conjuction with a ``PointDataArray`` to store point-associated vector data.

    Example
    -------
    >>> indexed_array = IndexedSurfaceFieldTimeDataArray(
    ...     (1+1j) * np.random.random((4,2,3,1)), coords=dict(index=np.arange(4), side=["outside", "inside"], axis=np.arange(3), t=[0])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "side", "axis", "t")


class IndexedFieldDataArray(DataArray):
    """Stores indexed values of vector fields in frequency domain. It is typically used
    in conjuction with a ``PointDataArray`` to store point-associated vector data.

    Example
    -------
    >>> indexed_array = IndexedFieldDataArray(
    ...     (1+1j) * np.random.random((4,3,1)), coords=dict(index=np.arange(4), axis=np.arange(3), f=[1e9])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "axis", "f")


class IndexedFieldTimeDataArray(DataArray):
    """Stores indexed values of vector fields in time domain. It is typically used
    in conjuction with a ``PointDataArray`` to store point-associated vector data.

    Example
    -------
    >>> indexed_array = IndexedFieldTimeDataArray(
    ...     (1+1j) * np.random.random((4,3,1)), coords=dict(index=np.arange(4), axis=np.arange(3), t=[0])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "axis", "t")


class IndexedFreqDataArray(DataArray):
    """Stores indexed values of scalar fields in frequency domain. It is typically used
    in conjuction with a ``PointDataArray`` to store point-associated vector data.

    Example
    -------
    >>> indexed_array = IndexedFreqDataArray(
    ...     (1+1j) * np.random.random((4,1)), coords=dict(index=np.arange(4), f=[1e9])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "f")


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


DATA_ARRAY_TYPES = [
    SpatialDataArray,
    ScalarFieldDataArray,
    ScalarFieldTimeDataArray,
    ScalarModeFieldDataArray,
    ScalarTerminalFieldDataArray,
    FluxDataArray,
    FluxTimeDataArray,
    ModeAmpsDataArray,
    ModeAmpsTimeDataArray,
    ModeIndexDataArray,
    GroupIndexDataArray,
    ModeDispersionDataArray,
    FieldProjectionAngleDataArray,
    FieldProjectionCartesianDataArray,
    FieldProjectionKSpaceDataArray,
    SphericalAngleDataArray,
    DipoleEmissionDataArray,
    DipoleEmissionPositionDataArray,
    DiffractionDataArray,
    ModeDataArray,
    TerminalDataArray,
    FreqModeDataArray,
    FreqDataArray,
    TimeDataArray,
    FreqVoltageDataArray,
    FreqTerminalDataArray,
    FreqTerminalModeDataArray,
    FreqModeTerminalDataArray,
    FreqTerminalTerminalDataArray,
    TriangleMeshDataArray,
    HeatDataArray,
    EMEScalarFieldDataArray,
    EMEScalarModeFieldDataArray,
    EMESMatrixDataArray,
    EMEInterfaceSMatrixDataArray,
    EMECoefficientDataArray,
    EMEModeIndexDataArray,
    EMEFluxDataArray,
    EMEFreqModeDataArray,
    EMETraceMetricDataArray,
    EMEInterfaceCellIndexDataArray,
    EMEInterfaceDiagnosticDataArray,
    MixedModeDataArray,
    ChargeDataArray,
    SteadyVoltageDataArray,
    ConvergenceHistoryDataArray,
    PointDataArray,
    CellDataArray,
    IndexedDataArray,
    IndexedFieldVoltageDataArray,
    IndexedVoltageDataArray,
    SpatialVoltageDataArray,
    PerturbationCoefficientDataArray,
    IndexedTimeDataArray,
    VoltageFreqDataArray,
    VoltageTimeDataArray,
    VoltageFreqModeDataArray,
    VoltageFreqTerminalDataArray,
    VoltageFreqTerminalModeDataArray,
    VoltageFreqModeTerminalDataArray,
    CurrentFreqDataArray,
    CurrentTimeDataArray,
    CurrentFreqModeDataArray,
    CurrentFreqTerminalDataArray,
    CurrentFreqTerminalModeDataArray,
    ImpedanceModeDataArray,
    ImpedanceTerminalDataArray,
    ImpedanceFreqDataArray,
    ImpedanceTimeDataArray,
    ImpedanceFreqModeDataArray,
    FreqModeModeDataArray,
    ImpedanceFreqModeModeDataArray,
    ImpedanceFreqTerminalTerminalDataArray,
    IndexedSurfaceFieldDataArray,
    IndexedSurfaceFieldTimeDataArray,
    IndexedFieldDataArray,
    IndexedFieldTimeDataArray,
    IndexedFreqDataArray,
    IndexedSurfaceFreqDataArray,
    IndexedSurfaceTimeDataArray,
]

DATA_ARRAY_MAP = {data_array.__name__: data_array for data_array in DATA_ARRAY_TYPES}

IndexedDataArrayTypes = (
    IndexedDataArray
    | IndexedVoltageDataArray
    | IndexedSurfaceFieldDataArray
    | IndexedSurfaceFieldTimeDataArray
    | IndexedFieldDataArray
    | IndexedFieldTimeDataArray
    | IndexedFreqDataArray
    | IndexedTimeDataArray
    | IndexedFieldVoltageDataArray
    | IndexedSurfaceFreqDataArray
    | IndexedSurfaceTimeDataArray
    | PointDataArray
)

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


class _TracedDataset(xr.Dataset):
    """Dataset subclass that preserves traced tidy3d DataArrays when accessed.

    When xr.Dataset constructor is called with tidy3d DataArray objects,
    xarray extracts the data and stores it as Variables internally. When
    items are accessed via __getitem__, xarray wraps these Variables in
    vanilla xr.DataArray, losing the custom .values property that's needed
    for autograd compatibility.

    This subclass overrides _construct_dataarray to return tidy3d DataArray
    objects, preserving the custom .values behavior that returns .data
    directly when tracing (avoiding np.asarray which breaks autodiff).
    """

    __slots__ = ()

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> _TracedDataset:
        """Construct a traced dataset from an existing Dataset instance."""
        return cls._construct_direct(
            variables=dataset._variables.copy(),
            coord_names=dataset._coord_names.copy(),
            dims=dict(dataset._dims),
            attrs=None if dataset._attrs is None else dataset._attrs.copy(),
            indexes=dataset._indexes.copy(),
            encoding=None if dataset._encoding is None else dataset._encoding.copy(),
            close=dataset._close,
        )

    def _construct_dataarray(self, name: Hashable) -> DataArray:
        """Construct a tidy3d DataArray by indexing this dataset."""
        return DataArray(super()._construct_dataarray(name))
