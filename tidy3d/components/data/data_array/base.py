"""Base ``DataArray`` class: validation, serialization hooks, and xarray overrides."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from autograd.tracer import isbox
from pydantic_core import core_schema

from tidy3d.components.autograd import TidyArrayBox, is_tidy_box
from tidy3d.exceptions import DataError

from . import _autograd as _autograd_methods
from . import _io as _io_methods
from ._constants import DIM_ATTRS
from ._plot import _SanitizedPlotProxy

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray
    from pydantic.annotated_handlers import GetCoreSchemaHandler
    from pydantic.json_schema import GetJsonSchemaHandler, JsonSchemaValue
    from xarray.core.types import InterpOptions, Self


class DataArray(xr.DataArray):
    """Subclass of ``xr.DataArray`` that requires _dims to match the keys of the coords."""

    # Always set __slots__ = () to avoid xarray warnings
    __slots__ = ()
    # stores an ordered tuple of strings corresponding to the data dimensions
    _dims: tuple[str, ...] = ()
    # Previous dimensions accepted for legacy serialized or in-memory arrays. Any
    # dimensions absent from ``_dims`` must be singleton and are normalized away.
    _legacy_dims: tuple[str, ...] | None = None
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
        """Normalize supported legacy dimensions, validate them, and order them."""
        for dim in set(cls._legacy_dims or ()) - set(cls._dims):
            if dim not in val.dims:
                continue
            if val.sizes[dim] != 1:
                raise ValueError(
                    f"{cls.__name__} does not support multiple '{dim}' entries. "
                    f"Only a singleton legacy '{dim}' dimension can be normalized. "
                    f"Select one '{dim}' entry before constructing this array, or migrate "
                    "saved legacy result data with an earlier Tidy3D version."
                )
            val = val.squeeze(dim=dim, drop=True)

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

    to_hdf5 = _io_methods.to_hdf5
    to_hdf5_handle = _io_methods.to_hdf5_handle
    _from_hdf5_handle = classmethod(_io_methods._from_hdf5_handle)
    from_hdf5 = classmethod(_io_methods.from_hdf5)
    from_file = classmethod(_io_methods.from_file)

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

    _ag_multiply_at = _autograd_methods._ag_multiply_at

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

    _ag_interp = _autograd_methods._ag_interp
    _ag_interp_func = staticmethod(_autograd_methods._ag_interp_func)
    _with_updated_data = _autograd_methods._with_updated_data


# Bind 'DataArray' in the modules that define these methods, so that runtime annotation
# lookups such as 'typing.get_type_hints' can resolve it as they could before the split.
_autograd_methods.DataArray = DataArray
_io_methods.DataArray = DataArray
