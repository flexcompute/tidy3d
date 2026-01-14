"""Storing tidy3d data at it's most fundamental level as xr.DataArray objects"""

from __future__ import annotations

import pathlib
import re
import warnings
from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any

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
    from os import PathLike
    from typing import Optional, Union

    from numpy.typing import NDArray
    from pydantic.annotated_handlers import GetCoreSchemaHandler
    from pydantic.json_schema import GetJsonSchemaHandler, JsonSchemaValue
    from xarray.core.types import InterpOptions, Self

    from tidy3d._common.components.autograd import InterpolationType
    from tidy3d._common.components.types.base import Bound

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

# Toggle for emitting deprecation warnings from legacy xarray shims.
LEGACY_SHIM_WARNINGS = True
# Toggle for installing legacy xarray shims on import.
LEGACY_SHIM_ENABLED = True


@dataclass(frozen=True)
class DataArraySpec:
    """Declarative schema for an ``xarray.DataArray`` field."""

    id: str
    dims: tuple[str, ...]
    data_attrs: Mapping[str, Any] = field(default_factory=dict)
    coord_attrs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    require_unique_coords: bool = False

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_after_validator_function(
            self._validate,
            core_schema.any_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                self._serialize, info_arg=True, when_used="json"
            ),
        )

    def __get_pydantic_json_schema__(
        self, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        json_schema = handler(schema)
        json_schema.update(
            {
                "title": "xarray.DataArray",
                "type": "object",
                "td_schema": self.id,
                "td_dims": list(self.dims),
            }
        )
        return json_schema

    def _validate(self, value: Any, info: core_schema.ValidationInfo) -> xr.DataArray:
        data_array = self._coerce_to_dataarray(value, info)
        return self.validate_data_array(data_array)

    def _serialize(self, value: xr.DataArray, info: core_schema.SerializationInfo) -> str:
        # Preserve existing JSON placeholder behavior by default.
        return self.id

    def _coerce_to_dataarray(self, value: Any, info: core_schema.ValidationInfo) -> xr.DataArray:
        if isinstance(value, DataArray):
            return value

        if isinstance(value, xr.DataArray):
            return DataArray(
                value.data,
                coords=value.coords,
                dims=value.dims,
                name=value.name,
                attrs=dict(value.attrs),
            )

        if isinstance(value, str) and is_data_array_name(value):
            raise DataError(
                "Trying to load a DataArray from a string placeholder but the data is missing. "
                "DataArrays are not typically stored in JSON. Load from HDF5 or ensure the "
                "DataArray object is provided."
            )

        if isinstance(value, Mapping) and "__td_dataarray__" in value:
            payload = value.get("__td_dataarray__", {})
            schema = payload.get("schema")
            if schema != self.id:
                raise ValueError(f"schema mismatch: expected {self.id!r}, got {schema!r}")
            inline = payload.get("inline")
            if isinstance(inline, Mapping):
                return self._from_inline(inline)

            raise ValueError("unsupported DataArray payload; missing inline data")

        raise ValueError("expected an xarray.DataArray or serialized DataArray payload")

    def _from_inline(self, inline: Mapping[str, Any]) -> xr.DataArray:
        dims = inline.get("dims", self.dims)
        if isinstance(dims, str):
            dims = (dims,)
        dims = tuple(dims)
        coords = dict(inline.get("coords", {}))
        data = np.asarray(inline.get("data"))
        return DataArray(data, coords=coords, dims=dims)

    def from_hdf5(self, fname: PathLike, group_path: str) -> xr.DataArray:
        """Load a DataArray from an hdf5 file using this spec's dimensions."""
        path = pathlib.Path(fname)
        with h5py.File(path, "r") as f:
            sub_group = f[group_path]
            values = np.array(sub_group[DATA_ARRAY_VALUE_NAME])
            coords = {dim: np.array(sub_group[dim]) for dim in self.dims if dim in sub_group}
            for key, val in coords.items():
                if val.dtype == "O":
                    coords[key] = [byte_string.decode() for byte_string in val.tolist()]
            data_array = DataArray(values, coords=coords, dims=self.dims)
            return self.validate_data_array(data_array)

    def validate_data_array(self, data_array: xr.DataArray) -> xr.DataArray:
        expected = tuple(self.dims)
        given = tuple(str(d) for d in data_array.dims)
        if set(given) != set(expected):
            raise ValueError(f"wrong dims: expected {expected}, got {given}")

        if given != expected:
            data_array = data_array.transpose(*expected)

        data_array = data_array.copy(deep=False)

        if self.data_attrs:
            data_array.attrs.update(self.data_attrs)
        for dim, attrs in self.coord_attrs.items():
            if dim in data_array.coords:
                data_array.coords[dim].attrs.update(attrs)

        if self.require_unique_coords:
            for dim in expected:
                if data_array.coords[dim].to_index().duplicated().any():
                    raise ValueError(f"duplicate coordinates in dimension {dim!r}")

        if type(data_array) is not DataArray:
            data_array = DataArray(
                data_array.data,
                coords=data_array.coords,
                dims=data_array.dims,
                name=data_array.name,
                attrs=dict(data_array.attrs),
            )

        return data_array

    def matches(self, data_array: xr.DataArray) -> bool:
        try:
            self.validate_data_array(data_array)
        except ValueError:
            return False
        return True


DATA_ARRAY_SPEC_MAP: dict[str, DataArraySpec] = {}
DATA_ARRAY_SCHEMA_MAP: dict[str, type[DataArray]] = {}


def _camel_to_snake(name: str) -> str:
    step1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    step2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step1)
    return step2.lower()


def _default_schema_id(data_array_type: type[DataArray]) -> str:
    override = getattr(data_array_type, "_schema_id", None)
    if override:
        return override
    base = data_array_type.__name__
    if base == "DataArray":
        return "tidy3d.data.data_array"
    if base.endswith("DataArray"):
        base = base[: -len("DataArray")]
    if not base:
        return "tidy3d.data.data_array"
    return f"tidy3d.data.{_camel_to_snake(base)}"


def _default_spec_for_type(data_array_type: type[DataArray]) -> DataArraySpec:
    dims = tuple(getattr(data_array_type, "_dims", ()))
    data_attrs = dict(getattr(data_array_type, "_data_attrs", {}))
    coord_attrs = {dim: DIM_ATTRS[dim] for dim in dims if dim in DIM_ATTRS}
    return DataArraySpec(
        id=_default_schema_id(data_array_type),
        dims=dims,
        data_attrs=data_attrs,
        coord_attrs=coord_attrs,
    )


def data_array_spec_for_type(data_array_type: type[DataArray]) -> DataArraySpec:
    spec = data_array_type.__spec__
    if not isinstance(spec, DataArraySpec):
        raise TypeError(f"{data_array_type.__name__} is missing a DataArraySpec.")
    return spec


def data_array_annotated_type(data_array_type: type[DataArray]) -> Any:
    """Return an ``Annotated[DataArray, DataArraySpec]`` alias for a DataArray class."""
    return Annotated[DataArray, data_array_spec_for_type(data_array_type)]


def _isinstance(value: Any, data_array_type: type[DataArray]) -> bool:
    """Spec-based check that replaces subclass ``isinstance`` usage."""
    if not isinstance(value, xr.DataArray):
        return False
    spec = data_array_spec_for_type(data_array_type)
    return spec.matches(value)


def register_data_array_spec(spec: DataArraySpec, data_array_type: type[DataArray]) -> None:
    """Register a DataArraySpec for schema lookup and legacy compatibility."""
    DATA_ARRAY_SPEC_MAP[spec.id] = spec
    DATA_ARRAY_SCHEMA_MAP[spec.id] = data_array_type


def data_array_spec_from_name(name: str) -> DataArraySpec | None:
    spec = DATA_ARRAY_SPEC_MAP.get(name)
    if spec is not None:
        return spec
    if name.endswith("DataArray"):
        base = name[: -len("DataArray")]
        if base:
            legacy_id = f"tidy3d.data.{_camel_to_snake(base)}"
            return DATA_ARRAY_SPEC_MAP.get(legacy_id)
    return None


def data_array_type_from_name(name: str) -> type[DataArray] | None:
    spec = data_array_spec_from_name(name)
    if spec is None:
        return None
    return DataArray


def iter_data_array_names() -> tuple[str, ...]:
    names = list(DATA_ARRAY_SPEC_MAP.keys())
    names.extend(da_type.__name__ for da_type in DATA_ARRAY_SCHEMA_MAP.values())
    return tuple(dict.fromkeys(names))


def is_data_array_name(value: Any) -> bool:
    return isinstance(value, str) and data_array_spec_from_name(value) is not None


class DataArray(xr.DataArray):
    """Subclass of ``xr.DataArray`` that requires _dims to match the keys of the coords."""

    # Always set __slots__ = () to avoid xarray warnings
    __slots__ = ()
    # stores an ordered tuple of strings corresponding to the data dimensions
    _dims = ()
    # stores a dictionary of attributes corresponding to the data values
    _data_attrs: dict[str, str] = {}
    # optional stable schema id (defaults to class name if not set)
    _schema_id: str | None = None
    # schema metadata for spec-based validation
    __spec__: DataArraySpec = DataArraySpec(id="tidy3d.data.data_array", dims=())

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is DataArray:
            return
        spec = cls.__dict__.get("__spec__")
        if spec is None:
            spec = _default_spec_for_type(cls)
            cls.__spec__ = spec
        elif not isinstance(spec, DataArraySpec):
            raise TypeError(f"{cls.__name__}.__spec__ must be a DataArraySpec.")
        register_data_array_spec(spec, cls)

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
        """Delegate pydantic validation/serialization to the attached spec."""
        spec = data_array_spec_for_type(cls)
        return spec.__get_pydantic_core_schema__(source_type, handler)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Delegate JSON schema generation to the attached spec."""
        spec = data_array_spec_for_type(cls)
        return spec.__get_pydantic_json_schema__(schema, handler)

    @classmethod
    def schema_id(cls) -> str:
        return data_array_spec_for_type(cls).id

    @property
    def spec(self) -> DataArraySpec:
        return data_array_spec_for_type(type(self))

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
        write_data_array_to_hdf5(self, f_handle=f_handle, group_path=group_path)

    @classmethod
    def from_hdf5(cls, fname: PathLike, group_path: str) -> Self:
        """Load a DataArray from an hdf5 file with a given path to the group."""
        spec = data_array_spec_for_type(cls)
        return spec.from_hdf5(fname=fname, group_path=group_path)

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


def write_data_array_to_hdf5(
    data_array: xr.DataArray, f_handle: h5py.File, group_path: str
) -> None:
    """Save an ``xr.DataArray`` to an hdf5 file handle at the given group path."""
    sub_group = f_handle.create_group(group_path)
    sub_group[DATA_ARRAY_VALUE_NAME] = get_static(data_array.data)
    for key, val in data_array.coords.items():
        if val.dtype == "<U1":
            sub_group[key] = val.values.tolist()
        else:
            sub_group[key] = val


def _spatially_sorted_data_array(data_array: xr.DataArray) -> xr.DataArray:
    needs_sorting = []
    for axis in "xyz":
        if axis not in data_array.coords:
            raise DataError(
                "Spatial DataArray methods require coordinates for 'x', 'y', and 'z' dimensions."
            )
        axis_coords = data_array.coords[axis].values
        if len(axis_coords) > 1 and np.any(axis_coords[1:] < axis_coords[:-1]):
            needs_sorting.append(axis)

    if needs_sorting:
        result = data_array.sortby(needs_sorting)
        return _cast_data_array(result, data_array)

    return data_array


def _sel_inside_data_array(data_array: xr.DataArray, bounds: Bound) -> xr.DataArray:
    if any(bmin > bmax for bmin, bmax in zip(*bounds)):
        raise DataError(
            "Min and max bounds must be packaged as '(minx, miny, minz), (maxx, maxy, maxz)'."
        )

    sorted_data = _spatially_sorted_data_array(data_array)
    inds_list = []

    coords = (sorted_data.coords["x"], sorted_data.coords["y"], sorted_data.coords["z"])

    for coord, smin, smax in zip(coords, bounds[0], bounds[1]):
        length = len(coord)

        # one point along direction, assume invariance
        if length == 1:
            comp_inds = [0]
        else:
            # if data does not cover structure at all take the closest index
            if smax < coord[0]:
                comp_inds = np.arange(0, max(2, length))
            elif smin > coord[-1]:
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

    result = sorted_data.isel(x=inds_list[0], y=inds_list[1], z=inds_list[2])
    return _cast_data_array(result, data_array)


def _does_cover_data_array(
    data_array: xr.DataArray, bounds: Bound, rtol: float = 0.0, atol: float = 0.0
) -> bool:
    if any(bmin > bmax for bmin, bmax in zip(*bounds)):
        raise DataError(
            "Min and max bounds must be packaged as '(minx, miny, minz), (maxx, maxy, maxz)'."
        )

    for axis in "xyz":
        if axis not in data_array.coords:
            raise DataError(
                "Spatial DataArray methods require coordinates for 'x', 'y', and 'z' dimensions."
            )

    xyz = [data_array.coords["x"], data_array.coords["y"], data_array.coords["z"]]
    data_min = [0.0, 0.0, 0.0]
    data_max = [0.0, 0.0, 0.0]
    for dim in range(3):
        coords = xyz[dim]
        if len(coords) == 1:
            data_min[dim] = bounds[0][dim]
            data_max[dim] = bounds[1][dim]
        else:
            data_min[dim] = np.min(coords)
            data_max[dim] = np.max(coords)
    data_bounds = (tuple(data_min), tuple(data_max))
    return bounds_contains(data_bounds, bounds, rtol=rtol, atol=atol)


def _is_uniform_data_array(data_array: xr.DataArray) -> bool:
    raw_data = np.asarray(data_array.data).ravel()
    if raw_data.size == 0:
        return True
    return np.allclose(raw_data, raw_data[0])


def _angle_data_array(data_array: xr.DataArray) -> xr.DataArray:
    values = np.angle(np.asarray(data_array.data))
    result = xr.DataArray(values, coords=data_array.coords, dims=data_array.dims)
    return _cast_data_array(result, data_array)


def _with_updated_data_array(
    data_array: xr.DataArray, data: np.ndarray, coords: dict[str, Any]
) -> xr.DataArray:
    mask = xr.zeros_like(data_array, dtype=bool)
    mask.loc[coords] = True

    old_data = np.asarray(data_array.data)
    new_shape = list(old_data.shape)
    for i, dim in enumerate(data_array.dims):
        if dim in coords:
            new_shape[i] = 1
    try:
        new_data = data.reshape(new_shape)
    except ValueError as e:
        raise ValueError(
            "Couldn't reshape the supplied 'data' to update 'DataArray'. The provided data was "
            f"of shape {data.shape} and tried to reshape to {new_shape}."
        ) from e

    new_data = new_data + np.zeros_like(old_data)
    updated = np.where(mask, new_data, old_data)
    result = data_array.copy(deep=True, data=updated)
    return _cast_data_array(result, data_array)


def _reflect_data_array(
    data_array: xr.DataArray, axis: int, center: float, reflection_only: bool = False
) -> xr.DataArray:
    sorted_data = _spatially_sorted_data_array(data_array)

    coords = [
        sorted_data.coords["x"].values,
        sorted_data.coords["y"].values,
        sorted_data.coords["z"].values,
    ]
    data = np.array(sorted_data.data)

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
        reflected = type(sorted_data)(data, coords=coords_dict, dims=sorted_data.dims)
        return _cast_data_array(reflected.sortby("xyz"[axis]), data_array)

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

    reflected = type(sorted_data)(new_data, coords=coords_dict, dims=sorted_data.dims)
    return _cast_data_array(reflected, data_array)


def _cast_data_array(result: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Preserve subclass type when possible."""
    ref_type = type(reference)
    if ref_type is xr.DataArray or isinstance(result, ref_type):
        return result
    return ref_type(
        result.data,
        coords=result.coords,
        dims=result.dims,
        name=result.name,
        attrs=dict(result.attrs),
    )


@xr.register_dataarray_accessor("td")
class Tidy3DAccessor:
    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def validate(self, spec: DataArraySpec | None = None) -> xr.DataArray:
        if spec is None:
            if isinstance(self._obj, DataArray):
                spec = data_array_spec_for_type(type(self._obj))
            else:
                raise ValueError("A DataArraySpec must be provided for plain xarray objects.")
        return spec.validate_data_array(self._obj)

    def sel_inside(self, bounds: Bound) -> xr.DataArray:
        if isinstance(self._obj, DataArray):
            return self._obj.sel_inside(bounds)
        return _sel_inside_data_array(self._obj, bounds)

    def does_cover(self, bounds: Bound, rtol: float = 0.0, atol: float = 0.0) -> bool:
        if isinstance(self._obj, DataArray):
            return self._obj.does_cover(bounds, rtol=rtol, atol=atol)
        return _does_cover_data_array(self._obj, bounds, rtol=rtol, atol=atol)

    @property
    def is_uniform(self) -> bool:
        if isinstance(self._obj, DataArray):
            return self._obj.is_uniform
        return _is_uniform_data_array(self._obj)

    @property
    def angle(self) -> xr.DataArray:
        if isinstance(self._obj, DataArray):
            return self._obj.angle
        return _angle_data_array(self._obj)

    @property
    def abs(self) -> xr.DataArray:
        if isinstance(self._obj, DataArray):
            return self._obj.abs
        return abs(self._obj)

    def reflect(self, axis: int, center: float, reflection_only: bool = False) -> xr.DataArray:
        if isinstance(self._obj, DataArray):
            return self._obj.reflect(axis, center, reflection_only=reflection_only)
        return _reflect_data_array(self._obj, axis, center, reflection_only=reflection_only)

    def with_updated_data(self, data: np.ndarray, coords: dict[str, Any]) -> xr.DataArray:
        if isinstance(self._obj, DataArray):
            return self._obj._with_updated_data(data=data, coords=coords)
        return _with_updated_data_array(self._obj, data=data, coords=coords)

    def _with_updated_data(self, data: np.ndarray, coords: dict[str, Any]) -> xr.DataArray:
        return self.with_updated_data(data=data, coords=coords)


def td_validate(data_array: xr.DataArray, spec: DataArraySpec) -> xr.DataArray:
    return data_array.td.validate(spec=spec)


def td_sel_inside(data_array: xr.DataArray, bounds: Bound) -> xr.DataArray:
    return data_array.td.sel_inside(bounds)


def td_does_cover(
    data_array: xr.DataArray, bounds: Bound, rtol: float = 0.0, atol: float = 0.0
) -> bool:
    return data_array.td.does_cover(bounds, rtol=rtol, atol=atol)


def td_reflect(
    data_array: xr.DataArray, axis: int, center: float, reflection_only: bool = False
) -> xr.DataArray:
    return data_array.td.reflect(axis, center, reflection_only=reflection_only)


def td_with_updated_data(
    data_array: xr.DataArray, data: np.ndarray, coords: dict[str, Any]
) -> xr.DataArray:
    return data_array.td.with_updated_data(data=data, coords=coords)


def td_angle(data_array: xr.DataArray) -> xr.DataArray:
    return data_array.td.angle


def td_abs(data_array: xr.DataArray) -> xr.DataArray:
    return data_array.td.abs


def install_legacy_shims() -> None:
    """Install deprecated xarray.DataArray methods that forward to ``da.td.*``."""

    if not hasattr(xr.DataArray, "sel_inside"):

        def _sel_inside(self: xr.DataArray, bounds: Bound) -> xr.DataArray:
            if LEGACY_SHIM_WARNINGS:
                warnings.warn(
                    "xr.DataArray.sel_inside(...) is deprecated; use `da.td.sel_inside(...)` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return _sel_inside_data_array(self, bounds)

        xr.DataArray.sel_inside = _sel_inside

    if not hasattr(xr.DataArray, "does_cover"):

        def _does_cover(
            self: xr.DataArray, bounds: Bound, rtol: float = 0.0, atol: float = 0.0
        ) -> bool:
            if LEGACY_SHIM_WARNINGS:
                warnings.warn(
                    "xr.DataArray.does_cover(...) is deprecated; use `da.td.does_cover(...)` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return _does_cover_data_array(self, bounds, rtol=rtol, atol=atol)

        xr.DataArray.does_cover = _does_cover

    if not hasattr(xr.DataArray, "is_uniform"):

        @property
        def _is_uniform(self: xr.DataArray) -> bool:
            if LEGACY_SHIM_WARNINGS:
                warnings.warn(
                    "xr.DataArray.is_uniform is deprecated; use `da.td.is_uniform` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return _is_uniform_data_array(self)

        xr.DataArray.is_uniform = _is_uniform

    if not hasattr(xr.DataArray, "angle"):

        @property
        def _angle(self: xr.DataArray) -> xr.DataArray:
            if LEGACY_SHIM_WARNINGS:
                warnings.warn(
                    "xr.DataArray.angle is deprecated; use `da.td.angle` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return _angle_data_array(self)

        xr.DataArray.angle = _angle

    if not hasattr(xr.DataArray, "abs"):

        @property
        def _abs(self: xr.DataArray) -> xr.DataArray:
            if LEGACY_SHIM_WARNINGS:
                warnings.warn(
                    "xr.DataArray.abs is deprecated; use `da.td.abs` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return abs(self)

        xr.DataArray.abs = _abs

    if not hasattr(xr.DataArray, "reflect"):

        def _reflect(
            self: xr.DataArray, axis: int, center: float, reflection_only: bool = False
        ) -> xr.DataArray:
            if LEGACY_SHIM_WARNINGS:
                warnings.warn(
                    "xr.DataArray.reflect(...) is deprecated; use `da.td.reflect(...)` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return _reflect_data_array(self, axis, center, reflection_only=reflection_only)

        xr.DataArray.reflect = _reflect

    if not hasattr(xr.DataArray, "_with_updated_data"):

        def _with_updated_data(
            self: xr.DataArray, data: np.ndarray, coords: dict[str, Any]
        ) -> xr.DataArray:
            if LEGACY_SHIM_WARNINGS:
                warnings.warn(
                    "xr.DataArray._with_updated_data(...) is deprecated; use `da.td.with_updated_data(...)` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return _with_updated_data_array(self, data=data, coords=coords)

        xr.DataArray._with_updated_data = _with_updated_data


register_data_array_spec(DataArray.__spec__, DataArray)


if LEGACY_SHIM_ENABLED:
    install_legacy_shims()


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
        return _spatially_sorted_data_array(self)

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
        return _sel_inside_data_array(self, bounds)

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
        return _does_cover_data_array(self, bounds, rtol=rtol, atol=atol)


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
