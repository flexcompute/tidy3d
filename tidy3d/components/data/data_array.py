"""Storing tidy3d data at it's most fundamental level as xr.DataArray objects"""
from __future__ import annotations
from typing import Dict, List

import xarray as xr
import numpy as np
import dask
import h5py

from pydantic_core import core_schema
from typing_extensions import Annotated

from pydantic import (
    BaseModel,
    GetJsonSchemaHandler,
    ValidationError,
)
from pydantic.json_schema import JsonSchemaValue

from ...constants import HERTZ, SECOND, MICROMETER, RADIAN
from ...exceptions import DataError, FileError

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


class _DataArray(xr.DataArray):
    """Subclass of ``xr.DataArray`` that requires _dims to match the keys of the coords."""

    # Always set __slots__ = () to avoid xarray warnings
    __slots__ = ()
    # stores an ordered tuple of strings corresponding to the data dimensions
    _dims = ()
    # stores a dictionary of attributes corresponding to the data values
    _data_attrs: Dict[str, str] = {}

    @classmethod
    def _mod_schema(cls, field_schema):
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
        return type(val).__name__

    def __eq__(self, other) -> bool:
        """Whether two data array objects are equal."""
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

    def to_hdf5(self, fname: str, group_path: str) -> None:
        """Save an xr.DataArray to the hdf5 file with a given path to the group."""
        sub_group = fname.create_group(group_path)
        sub_group[DATA_ARRAY_VALUE_NAME] = self.values
        for key, val in self.coords.items():
            # sub_group[key] = val
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


class _DataArrayAnnotation:
    """Annotation for DataArray to add validation and serialization."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        """
        We return a pydantic_core.CoreSchema that behaves in the following ways:

        * dictionaries will be parsed as `complex` instances using real and imaginary parts
        * `complex`, `float`, and `int` instances will be parsed as `complex` instances
        * Nothing else will pass validation
        * Serialization will always return a dict with real and imaginary parts
        """

        # def handle_dict(val):
        #     """If the data comes in as the raw data array string, raise a custom warning."""
        #     if isinstance(val, dict):
        #         import pdb; pdb.set_trace()
        #     return val

        def check_unloaded_data(val):
            """If the data comes in as the raw data array string, raise a custom warning."""
            if isinstance(val, str) and val in DATA_ARRAY_MAP:
                raise DataError(
                    f"Trying to load {_source_type.__name__} but the data is not present. "
                    "Note that data will not be saved to .json file. "
                    "use .hdf5 format instead if data present."
                )
            return _source_type(val)

        def validate_dims(val):
            """Make sure the dims are the same as _dims, then put them in the correct order."""
            if set(val.dims) != set(_source_type._dims):
                raise ValueError(f"wrong dims, expected '{_source_type._dims}', got '{val.dims}'")
            return val.transpose(*_source_type._dims)

        def assign_data_attrs(val):
            """Assign the correct data attributes to the :class:`.DataArray`."""

            for attr_name, attr in _source_type._data_attrs.items():
                val.attrs[attr_name] = attr
            return val

        def assign_coord_attrs(val):
            """Assign the correct coordinate attributes to the :class:`.DataArray`."""

            for dim in _source_type._dims:
                dim_attrs = DIM_ATTRS.get(dim)
                if dim_attrs is not None:
                    for attr_name, attr in dim_attrs.items():
                        val.coords[dim].attrs[attr_name] = attr
            return val

        # def correct_type(val):
        #     """Ensure an `xr.DataArray` is returned."""
        #     try:
        #         return _source_type(val)

        from_dict_schema = core_schema.chain_schema(
            [
                # core_schema.no_info_plain_validator_function(handle_dict),
                core_schema.no_info_plain_validator_function(check_unloaded_data),
                core_schema.no_info_plain_validator_function(validate_dims),
                core_schema.no_info_plain_validator_function(assign_data_attrs),
                core_schema.no_info_plain_validator_function(assign_coord_attrs),
                # core_schema.no_info_plain_validator_function(correct_type),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_dict_schema,
            python_schema=from_dict_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: type(instance).__name__
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Use the same schema that would be used for `int`
        return handler(core_schema.dict_schema())


# We now create an `Annotated` wrapper that we'll use as the annotation for fields on `BaseModel`s, etc.
DataArray = Annotated[_DataArray, _DataArrayAnnotation]


class _FreqDataArray(DataArray):
    """Frequency-domain array.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> fd = FreqDataArray((1+1j) * np.random.random((2,)), coords=dict(f=f))
    """

    __slots__ = ()
    _dims = ("f",)


class _FreqModeDataArray(DataArray):
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


class _TimeDataArray(DataArray):
    """Time-domain array.

    Example
    -------
    >>> t = [0, 1e-12, 2e-12]
    >>> td = TimeDataArray((1+1j) * np.random.random((3,)), coords=dict(t=t))
    """

    __slots__ = ()
    _dims = "t"


class _MixedModeDataArray(DataArray):
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


class _SpatialDataArray(DataArray):
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
    _dims = ("x", "y", "z")
    _data_attrs = {"long_name": "field value"}


class _ScalarFieldDataArray(DataArray):
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


class _ScalarFieldTimeDataArray(DataArray):
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


class _ScalarModeFieldDataArray(DataArray):
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


class _FluxDataArray(DataArray):
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


class _FluxTimeDataArray(DataArray):
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


class _ModeAmpsDataArray(DataArray):
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


class _ModeIndexDataArray(DataArray):
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


class _FieldProjectionAngleDataArray(DataArray):
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


class _FieldProjectionCartesianDataArray(DataArray):
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


class _FieldProjectionKSpaceDataArray(DataArray):
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


class _DiffractionDataArray(DataArray):
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


class _TriangleMeshDataArray(DataArray):
    """Data of the triangles of a surface mesh as in the STL file format."""

    __slots__ = ()
    _dims = ("face_index", "vertex_index", "axis")
    _data_attrs = {"long_name": "surface mesh triangles"}


SpatialDataArray = Annotated[_SpatialDataArray, _DataArrayAnnotation]
ScalarFieldDataArray = Annotated[_ScalarFieldDataArray, _DataArrayAnnotation]
ScalarFieldTimeDataArray = Annotated[_ScalarFieldTimeDataArray, _DataArrayAnnotation]
ScalarModeFieldDataArray = Annotated[_ScalarModeFieldDataArray, _DataArrayAnnotation]
FluxDataArray = Annotated[_FluxDataArray, _DataArrayAnnotation]
FluxTimeDataArray = Annotated[_FluxTimeDataArray, _DataArrayAnnotation]
ModeAmpsDataArray = Annotated[_ModeAmpsDataArray, _DataArrayAnnotation]
ModeIndexDataArray = Annotated[_ModeIndexDataArray, _DataArrayAnnotation]
MixedModeDataArray = Annotated[_MixedModeDataArray, _DataArrayAnnotation]
FieldProjectionAngleDataArray = Annotated[_FieldProjectionAngleDataArray, _DataArrayAnnotation]
FieldProjectionCartesianDataArray = Annotated[
    _FieldProjectionCartesianDataArray, _DataArrayAnnotation
]
FieldProjectionKSpaceDataArray = Annotated[_FieldProjectionKSpaceDataArray, _DataArrayAnnotation]
DiffractionDataArray = Annotated[_DiffractionDataArray, _DataArrayAnnotation]
FreqModeDataArray = Annotated[_FreqModeDataArray, _DataArrayAnnotation]
FreqDataArray = Annotated[_FreqDataArray, _DataArrayAnnotation]
TimeDataArray = Annotated[_TimeDataArray, _DataArrayAnnotation]
FreqModeDataArray = Annotated[_FreqModeDataArray, _DataArrayAnnotation]
TriangleMeshDataArray = Annotated[_TriangleMeshDataArray, _DataArrayAnnotation]

DATA_ARRAY_MAP = dict(
    _SpatialDataArray=SpatialDataArray,
    _ScalarFieldDataArray=ScalarFieldDataArray,
    _ScalarFieldTimeDataArray=ScalarFieldTimeDataArray,
    _ScalarModeFieldDataArray=ScalarModeFieldDataArray,
    _FluxDataArray=FluxDataArray,
    _FluxTimeDataArray=FluxTimeDataArray,
    _ModeAmpsDataArray=ModeAmpsDataArray,
    _ModeIndexDataArray=ModeIndexDataArray,
    _MixedModeDataArray=MixedModeDataArray,
    _FieldProjectionAngleDataArray=FieldProjectionAngleDataArray,
    _FieldProjectionCartesianDataArray=FieldProjectionCartesianDataArray,
    _FieldProjectionKSpaceDataArray=FieldProjectionKSpaceDataArray,
    _DiffractionDataArray=DiffractionDataArray,
    _FreqModeDataArray=FreqModeDataArray,
    _FreqDataArray=FreqDataArray,
    _TimeDataArray=TimeDataArray,
    _TriangleMeshDataArray=TriangleMeshDataArray,
)
DATA_ARRAY_TYPES = DATA_ARRAY_MAP.keys()

# DATA_ARRAY_MAP = {data_array.__name__: data_array for data_array in DATA_ARRAY_TYPES}

# import pdb; pdb.set_trace()
