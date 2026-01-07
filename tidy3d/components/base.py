"""global configuration / base class for pydantic models used to make simulation."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from functools import wraps
from math import ceil
from os import PathLike
from pathlib import Path
from types import UnionType
from typing import Any, Callable, Literal, Optional, Union, get_args, get_origin

import h5py
import numpy as np
import pydantic.v1 as pydantic
import rich
import xarray as xr
import yaml
from autograd.builtins import dict as dict_ag
from autograd.tracer import isbox
from pydantic.v1.fields import ModelField
from pydantic.v1.json import custom_pydantic_encoder
from typing_extensions import Self

from tidy3d.exceptions import FileError
from tidy3d.log import log

from .autograd.types import AutogradFieldMap, Box
from .autograd.utils import get_static
from .data.data_array import DATA_ARRAY_MAP, DataArray
from .file_util import compress_file_to_gzip, extract_gzip_file
from .types import TYPE_TAG_STR, ComplexNumber

INDENT_JSON_FILE = 4  # default indentation of json string in json files
INDENT = None  # default indentation of json string used internally
JSON_TAG = "JSON_STRING"
# If json string is larger than ``MAX_STRING_LENGTH``, split the string when storing in hdf5
MAX_STRING_LENGTH = 1_000_000_000
FORBID_SPECIAL_CHARACTERS = ["/"]
TRACED_FIELD_KEYS_ATTR = "__tidy3d_traced_field_keys__"
TYPE_TO_CLASS_MAP: dict[str, type[Tidy3dBaseModel]] = {}


def cache(prop):
    """Decorates a property to cache the first computed value and return it on subsequent calls."""

    # note, we could also just use `prop` as dict key, but hashing property might be slow
    prop_name = prop.__name__

    @wraps(prop)
    def cached_property_getter(self):
        """The new property method to be returned by decorator."""

        stored_value = self._cached_properties.get(prop_name)

        if stored_value is not None:
            return stored_value

        computed_value = prop(self)
        self._cached_properties[prop_name] = computed_value
        return computed_value

    return cached_property_getter


def cached_property(cached_property_getter):
    """Shortcut for property(cache()) of a getter."""

    return property(cache(cached_property_getter))


def cached_property_guarded(key_func):
    """Like cached_property, but invalidates when the key_func(self) changes."""

    def _decorator(getter):
        prop_name = getter.__name__

        @wraps(getter)
        def _guarded(self):
            cache_store = self._cached_properties.get(prop_name)
            current_key = key_func(self)
            if cache_store is not None:
                cached_key, cached_value = cache_store
                if cached_key == current_key:
                    return cached_value
            value = getter(self)
            self._cached_properties[prop_name] = (current_key, value)
            return value

        return property(_guarded)

    return _decorator


def ndarray_encoder(val):
    """How a ``np.ndarray`` gets handled before saving to json."""
    if np.any(np.iscomplex(val)):
        return {"real": val.real.tolist(), "imag": val.imag.tolist()}
    return val.real.tolist()


def make_json_compatible(json_string: str) -> str:
    """Makes the string compatible with json standards, notably for infinity."""

    tmp_string = "<<TEMPORARY_INFINITY_STRING>>"
    json_string = json_string.replace("-Infinity", tmp_string)
    json_string = json_string.replace('""-Infinity""', tmp_string)
    json_string = json_string.replace("Infinity", '"Infinity"')
    json_string = json_string.replace('""Infinity""', '"Infinity"')
    return json_string.replace(tmp_string, '"-Infinity"')


def _get_valid_extension(fname: PathLike) -> str:
    """Return the file extension from fname, validated to accepted ones."""
    valid_extensions = [".json", ".yaml", ".hdf5", ".h5", ".hdf5.gz"]
    path = Path(fname)
    extensions = [s.lower() for s in path.suffixes[-2:]]
    if len(extensions) == 0:
        raise FileError(f"File '{path}' missing extension.")
    single_extension = extensions[-1]
    if single_extension in valid_extensions:
        return single_extension
    double_extension = "".join(extensions)
    if double_extension in valid_extensions:
        return double_extension
    raise FileError(
        f"File extension must be one of {', '.join(valid_extensions)}; file '{path}' does not "
        "match any of those."
    )


def skip_if_fields_missing(fields: list[str], root=False):
    """Decorate ``validator`` to check that other fields have passed validation."""

    def actual_decorator(validator):
        @wraps(validator)
        def _validator(cls, *args: Any, **kwargs: Any):
            """New validator function."""
            values = kwargs.get("values")
            if values is None:
                values = args[0] if root else args[1]
            for field in fields:
                if field not in values:
                    log.warning(
                        f"Could not execute validator '{validator.__name__}' because field "
                        f"'{field}' failed validation."
                    )
                    if root:
                        return values
                    return kwargs.get("val") if "val" in kwargs else args[0]

            return validator(cls, *args, **kwargs)

        return _validator

    return actual_decorator


def field_allows_scalar(field: ModelField) -> bool:
    annotation = field.outer_type_

    def allows_scalar(a: Any) -> bool:
        origin = get_origin(a)
        if origin in (Union, UnionType):
            args = (arg for arg in get_args(a) if arg is not type(None))
            return any(allows_scalar(arg) for arg in args)
        if origin is not None:
            return False
        return isinstance(a, type) and issubclass(a, (float, int, np.generic))

    return allows_scalar(annotation)


class Tidy3dBaseModel(pydantic.BaseModel):
    """Base pydantic model that all Tidy3d components inherit from.
    Defines configuration for handling data structures
    as well as methods for importing, exporting, and hashing tidy3d objects.
    For more details on pydantic base models, see:
    `Pydantic Models <https://pydantic-docs.helpmanual.io/usage/models/>`_
    """

    def __hash__(self) -> int:
        """Hash method."""
        try:
            return super().__hash__(self)
        except TypeError:
            return hash(self.json())

    def _hash_self(self) -> str:
        """Hash this component with ``hashlib`` in a way that is the same every session."""
        bf = io.BytesIO()
        self.to_hdf5(bf)
        return hashlib.md5(bf.getvalue()).hexdigest()

    def __init__(self, **kwargs: Any) -> None:
        """Init method, includes post-init validators."""
        log.begin_capture()
        super().__init__(**kwargs)
        self._post_init_validators()
        log.end_capture(self)

    @pydantic.validator("*", pre=True, allow_reuse=True)
    def coerce_numpy_scalars_for_model(cls, v: Any, field: ModelField) -> Any:
        """
        Wildcard field validator: coerce numpy scalars / size-1 arrays to native Python
        scalars, but only for fields whose annotations allow scalars.
        """
        if not field_allows_scalar(field):
            return v

        if isinstance(v, np.generic) or (isinstance(v, np.ndarray) and v.size == 1):
            return v.item()

        return v

    def _post_init_validators(self) -> None:
        """Call validators taking ``self`` that get run after init, implement in subclasses."""

    def __init_subclass__(cls) -> None:
        """Things that are done to each of the models."""

        cls.add_type_field()
        cls.generate_docstring()
        type_value = cls.__fields__.get(TYPE_TAG_STR)
        if type_value and type_value.default:
            TYPE_TO_CLASS_MAP[type_value.default] = cls

    @classmethod
    def _get_type_value(cls, obj: dict[str, Any]) -> str:
        """Return the type tag from a raw dictionary."""
        if not isinstance(obj, dict):
            raise TypeError("Input must be a dict")
        try:
            type_value = obj[TYPE_TAG_STR]
        except KeyError as exc:
            raise ValueError(f'Missing "{TYPE_TAG_STR}" in data') from exc
        if not isinstance(type_value, str) or not type_value:
            raise ValueError(f'Invalid "{TYPE_TAG_STR}" value: {type_value!r}')
        return type_value

    @classmethod
    def _get_registered_class(cls, type_value: str) -> type[Tidy3dBaseModel]:
        try:
            return TYPE_TO_CLASS_MAP[type_value]
        except KeyError as exc:
            raise ValueError(f"Unknown type: {type_value}") from exc

    @classmethod
    def _should_dispatch_to(cls, target_cls: type[Tidy3dBaseModel]) -> bool:
        """Return True if ``cls`` allows auto-dispatch to ``target_cls``."""
        return issubclass(target_cls, cls)

    @classmethod
    def _resolve_dispatch_target(cls, obj: dict[str, Any]) -> type[Tidy3dBaseModel]:
        """Determine which subclass should receive ``obj``."""
        type_value = cls._get_type_value(obj)
        target_cls = cls._get_registered_class(type_value)
        if cls._should_dispatch_to(target_cls):
            return target_cls
        if target_cls is cls:
            return cls
        raise ValueError(
            f'Cannot parse type "{type_value}" using {cls.__name__}; expected subclass of {cls.__name__}.'
        )

    @classmethod
    def _target_cls_from_file(
        cls, fname: PathLike, group_path: Optional[str] = None
    ) -> type[Tidy3dBaseModel]:
        """Peek the file metadata to determine the subclass to instantiate."""
        model_dict = cls.dict_from_file(
            fname=fname,
            group_path=group_path,
            load_data_arrays=False,
        )
        return cls._resolve_dispatch_target(model_dict)

    @classmethod
    def _parse_obj(cls, obj: dict[str, Any], **parse_obj_kwargs: Any) -> Tidy3dBaseModel:
        """Dispatch ``obj`` to the correct subclass registered in the type map."""
        target_cls = cls._resolve_dispatch_target(obj)
        if target_cls is cls:
            return super().parse_obj(obj, **parse_obj_kwargs)
        return target_cls.parse_obj(obj, **parse_obj_kwargs)

    @classmethod
    def _parse_model_dict(
        cls, model_dict: dict[str, Any], **parse_obj_kwargs: Any
    ) -> Tidy3dBaseModel:
        """Parse ``model_dict`` while optionally auto-dispatching when called on the base class."""
        if cls is Tidy3dBaseModel:
            return cls._parse_obj(model_dict, **parse_obj_kwargs)
        return cls.parse_obj(model_dict, **parse_obj_kwargs)

    class Config:
        """Sets config for all :class:`Tidy3dBaseModel` objects.

        Configuration Options
        ---------------------
        allow_population_by_field_name : bool = True
            Allow properties to stand in for fields(?).
        arbitrary_types_allowed : bool = True
            Allow types like numpy arrays.
        extra : str = 'forbid'
            Forbid extra kwargs not specified in model.
        json_encoders : Dict[type, Callable]
            Defines how to encode type in json file.
        validate_all : bool = True
            Validate default values just to be safe.
        validate_assignment : bool
            Re-validate after re-assignment of field in model.
        """

        arbitrary_types_allowed = True
        validate_all = True
        extra = "forbid"
        validate_assignment = True
        allow_population_by_field_name = True
        json_encoders = {
            np.ndarray: ndarray_encoder,
            complex: lambda x: ComplexNumber(real=x.real, imag=x.imag),
            xr.DataArray: DataArray._json_encoder,
            Box: lambda x: x._value,
        }
        frozen = True
        allow_mutation = False
        copy_on_model_validation = "none"

    _cached_properties = pydantic.PrivateAttr({})
    _has_tracers: Optional[bool] = pydantic.PrivateAttr(default=None)

    @pydantic.root_validator(skip_on_failure=True)
    def _special_characters_not_in_name(cls, values):
        name = values.get("name")
        if name:
            for character in FORBID_SPECIAL_CHARACTERS:
                if character in name:
                    raise ValueError(
                        f"Special character '{character}' not allowed in component name {name}."
                    )
        return values

    attrs: dict = pydantic.Field(
        {},
        title="Attributes",
        description="Dictionary storing arbitrary metadata for a Tidy3D object. "
        "This dictionary can be freely used by the user for storing data without affecting the "
        "operation of Tidy3D as it is not used internally. "
        "Note that, unlike regular Tidy3D fields, ``attrs`` are mutable. "
        "For example, the following is allowed for setting an ``attr`` ``obj.attrs['foo'] = bar``. "
        "Also note that Tidy3D will raise a ``TypeError`` if ``attrs`` contain objects "
        "that can not be serialized. One can check if ``attrs`` are serializable "
        "by calling ``obj.json()``.",
    )

    def _attrs_digest(self) -> str:
        """Stable digest of `attrs` using the same JSON encoding rules as pydantic .json()."""
        encoders = getattr(self.__config__, "json_encoders", {}) or {}

        def _default(o):
            return custom_pydantic_encoder(encoders, o)

        json_str = json.dumps(
            self.attrs,
            default=_default,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        json_str = make_json_compatible(json_str)

        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def copy(self, deep: bool = True, validate: bool = True, **kwargs: Any) -> Self:
        """Copy a Tidy3dBaseModel.  With ``deep=True`` and ``validate=True`` as default."""
        kwargs.update(deep=deep)
        new_copy = pydantic.BaseModel.copy(self, **kwargs)
        if validate:
            return self.validate(new_copy.dict())
        # cached property is cleared automatically when validation is on, but it
        # needs to be manually cleared when validation is off
        new_copy._cached_properties = {}
        new_copy._has_tracers = None
        return new_copy

    def updated_copy(
        self, path: Optional[str] = None, deep: bool = True, validate: bool = True, **kwargs: Any
    ) -> Self:
        """Make copy of a component instance with ``**kwargs`` indicating updated field values.

        Note
        ----
        If ``path`` supplied, applies the updated copy with the update performed on the sub-
        component corresponding to the path. For indexing into a tuple or list, use the integer
        value.

        Example
        -------
        >>> sim = simulation.updated_copy(size=new_size, path=f"structures/{i}/geometry") # doctest: +SKIP
        """

        if not path:
            return self._updated_copy(**kwargs, deep=deep, validate=validate)

        path_components = path.split("/")

        field_name = path_components[0]

        try:
            sub_component = getattr(self, field_name)
        except AttributeError as e:
            raise AttributeError(
                f"Could not field field '{field_name}' in the sub-component `path`. "
                f"Found fields of '{tuple(self.__fields__.keys())}'. "
                "Please double check the `path` passed to `.updated_copy()`."
            ) from e

        if isinstance(sub_component, (list, tuple)):
            integer_index_path = path_components[1]

            try:
                index = int(integer_index_path)
            except ValueError:
                raise ValueError(
                    f"Could not grab integer index from path '{path}'. "
                    f"Please correct the sub path containing '{integer_index_path}' to be an "
                    f"integer index into '{field_name}' (containing {len(sub_component)} elements)."
                ) from None

            sub_component_list = list(sub_component)
            sub_component = sub_component_list[index]
            sub_path = "/".join(path_components[2:])

            sub_component_list[index] = sub_component.updated_copy(
                path=sub_path, deep=deep, validate=validate, **kwargs
            )
            new_component = tuple(sub_component_list)
        else:
            sub_path = "/".join(path_components[1:])
            new_component = sub_component.updated_copy(
                path=sub_path, deep=deep, validate=validate, **kwargs
            )

        return self._updated_copy(deep=deep, validate=validate, **{field_name: new_component})

    def _updated_copy(self, deep: bool = True, validate: bool = True, **kwargs: Any) -> Self:
        """Make copy of a component instance with ``**kwargs`` indicating updated field values."""
        return self.copy(update=kwargs, deep=deep, validate=validate)

    def help(self, methods: bool = False) -> None:
        """Prints message describing the fields and methods of a :class:`Tidy3dBaseModel`.

        Parameters
        ----------
        methods : bool = False
            Whether to also print out information about object's methods.

        Example
        -------
        >>> simulation.help(methods=True) # doctest: +SKIP
        """
        rich.inspect(self, methods=methods)

    @classmethod
    def from_file(
        cls,
        fname: PathLike,
        group_path: Optional[str] = None,
        lazy: bool = False,
        on_load: Optional[Callable] = None,
        **parse_obj_kwargs: Any,
    ) -> Self:
        """Loads a :class:`Tidy3dBaseModel` from .yaml, .json, .hdf5, or .hdf5.gz file.

        Parameters
        ----------
        fname : PathLike
            Full path to the file to load the :class:`Tidy3dBaseModel` from.
        group_path : str | None = None
            Path to a group inside the file to use as the base level. Only for hdf5 files.
            Starting `/` is optional.
        lazy : bool = False
            Whether to load the actual data (``lazy=False``) or return a proxy that loads
            the data when accessed (``lazy=True``).
        on_load : Callable | None = None
            Callback function executed once the model is fully materialized.
            Only used if ``lazy=True``. The callback is invoked with the loaded
            instance as its sole argument, enabling post-processing such as
            validation, logging, or warnings checks.
        **parse_obj_kwargs
            Keyword arguments passed to either pydantic's ``parse_obj`` function when loading model.

        Returns
        -------
        Self
            An instance of the component class calling ``load``.

        Example
        -------
        >>> simulation = Simulation.from_file(fname='folder/sim.json') # doctest: +SKIP
        """
        if lazy:
            target_cls = cls._target_cls_from_file(fname=fname, group_path=group_path)
            Proxy = _make_lazy_proxy(target_cls, on_load=on_load)
            return Proxy(fname, group_path, parse_obj_kwargs)
        model_dict = cls.dict_from_file(fname=fname, group_path=group_path)
        obj = cls._parse_model_dict(model_dict, **parse_obj_kwargs)
        if not lazy and on_load is not None:
            on_load(obj)
        return obj

    @classmethod
    def dict_from_file(
        cls, fname: PathLike, group_path: Optional[str] = None, *, load_data_arrays: bool = True
    ) -> dict:
        """Loads a dictionary containing the model from a .yaml, .json, .hdf5, or .hdf5.gz file.

        Parameters
        ----------
        fname : PathLike
            Full path to the file to load the :class:`Tidy3dBaseModel` from.
        group_path : str, optional
            Path to a group inside the file to use as the base level.

        Returns
        -------
        dict
            A dictionary containing the model.

        Example
        -------
        >>> simulation = Simulation.from_file(fname='folder/sim.json') # doctest: +SKIP
        """
        fname_path = Path(fname)
        extension = _get_valid_extension(fname_path)
        kwargs = {"fname": fname_path}

        if group_path is not None:
            if extension in {".hdf5", ".hdf5.gz", ".h5"}:
                kwargs["group_path"] = group_path
            else:
                log.warning("'group_path' provided, but this feature only works with hdf5 files.")

        if extension in {".hdf5", ".hdf5.gz", ".h5"}:
            kwargs["load_data_arrays"] = load_data_arrays

        converter = {
            ".json": cls.dict_from_json,
            ".yaml": cls.dict_from_yaml,
            ".hdf5": cls.dict_from_hdf5,
            ".hdf5.gz": cls.dict_from_hdf5_gz,
            ".h5": cls.dict_from_hdf5,
        }[extension]
        return converter(**kwargs)

    def to_file(self, fname: PathLike) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .yaml, .json, or .hdf5 file

        Parameters
        ----------
        fname : PathLike
            Full path to the .yaml or .json file to save the :class:`Tidy3dBaseModel` to.

        Example
        -------
        >>> simulation.to_file(fname='folder/sim.json') # doctest: +SKIP
        """
        extension = _get_valid_extension(fname)
        converter = {
            ".json": self.to_json,
            ".yaml": self.to_yaml,
            ".hdf5": self.to_hdf5,
            ".hdf5.gz": self.to_hdf5_gz,
        }[extension]
        return converter(fname=fname)

    @classmethod
    def from_json(cls, fname: PathLike, **parse_obj_kwargs: Any) -> Self:
        """Load a :class:`Tidy3dBaseModel` from .json file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .json file to load the :class:`Tidy3dBaseModel` from.

        Returns
        -------
        Self
            An instance of the component class calling `load`.
        **parse_obj_kwargs
            Keyword arguments passed to pydantic's ``parse_obj`` method.

        Example
        -------
        >>> simulation = Simulation.from_json(fname='folder/sim.json') # doctest: +SKIP
        """
        model_dict = cls.dict_from_json(fname=fname)
        return cls._parse_model_dict(model_dict, **parse_obj_kwargs)

    @classmethod
    def dict_from_json(cls, fname: PathLike) -> dict:
        """Load dictionary of the model from a .json file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .json file to load the :class:`Tidy3dBaseModel` from.

        Returns
        -------
        dict
            A dictionary containing the model.

        Example
        -------
        >>> sim_dict = Simulation.dict_from_json(fname='folder/sim.json') # doctest: +SKIP
        """
        with open(fname, encoding="utf-8") as json_fhandle:
            model_dict = json.load(json_fhandle)
        return model_dict

    def to_json(self, fname: PathLike) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .json file

        Parameters
        ----------
        fname : PathLike
            Full path to the .json file to save the :class:`Tidy3dBaseModel` to.

        Example
        -------
        >>> simulation.to_json(fname='folder/sim.json') # doctest: +SKIP
        """
        export_model = self.to_static()
        json_string = export_model._json(indent=INDENT_JSON_FILE)
        self._warn_if_contains_data(json_string)
        path = Path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            file_handle.write(json_string)

    @classmethod
    def from_yaml(cls, fname: PathLike, **parse_obj_kwargs: Any) -> Self:
        """Loads :class:`Tidy3dBaseModel` from .yaml file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .yaml file to load the :class:`Tidy3dBaseModel` from.
        **parse_obj_kwargs
            Keyword arguments passed to pydantic's ``parse_obj`` method.

        Returns
        -------
        Self
            An instance of the component class calling `from_yaml`.

        Example
        -------
        >>> simulation = Simulation.from_yaml(fname='folder/sim.yaml') # doctest: +SKIP
        """
        model_dict = cls.dict_from_yaml(fname=fname)
        return cls._parse_model_dict(model_dict, **parse_obj_kwargs)

    @classmethod
    def dict_from_yaml(cls, fname: PathLike) -> dict:
        """Load dictionary of the model from a .yaml file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .yaml file to load the :class:`Tidy3dBaseModel` from.

        Returns
        -------
        dict
            A dictionary containing the model.

        Example
        -------
        >>> sim_dict = Simulation.dict_from_yaml(fname='folder/sim.yaml') # doctest: +SKIP
        """
        with open(fname, encoding="utf-8") as yaml_in:
            model_dict = yaml.safe_load(yaml_in)
        return model_dict

    def to_yaml(self, fname: PathLike) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .yaml file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .yaml file to save the :class:`Tidy3dBaseModel` to.

        Example
        -------
        >>> simulation.to_yaml(fname='folder/sim.yaml') # doctest: +SKIP
        """
        export_model = self.to_static()
        json_string = export_model._json()
        self._warn_if_contains_data(json_string)
        model_dict = json.loads(json_string)
        path = Path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w+", encoding="utf-8") as file_handle:
            yaml.dump(model_dict, file_handle, indent=INDENT_JSON_FILE)

    @staticmethod
    def _warn_if_contains_data(json_str: str) -> None:
        """Log a warning if the json string contains data, used in '.json' and '.yaml' file."""
        if any((key in json_str for key, _ in DATA_ARRAY_MAP.items())):
            log.warning(
                "Data contents found in the model to be written to file. "
                "Note that this data will not be included in '.json' or '.yaml' formats. "
                "As a result, it will not be possible to load the file back to the original model."
                "Instead, use `.hdf5` extension in filename passed to 'to_file()'."
            )

    @staticmethod
    def _construct_group_path(group_path: str) -> str:
        """Construct a group path with the leading forward slash if not supplied."""

        # empty string or None
        if not group_path:
            return "/"

        # missing leading forward slash
        if group_path[0] != "/":
            return f"/{group_path}"

        return group_path

    @staticmethod
    def get_tuple_group_name(index: int) -> str:
        """Get the group name of a tuple element."""
        return str(int(index))

    @staticmethod
    def get_tuple_index(key_name: str) -> int:
        """Get the index into the tuple based on its group name."""
        return int(str(key_name))

    @classmethod
    def tuple_to_dict(cls, tuple_values: tuple) -> dict:
        """How we generate a dictionary mapping new keys to tuple values for hdf5."""
        return {cls.get_tuple_group_name(index=i): val for i, val in enumerate(tuple_values)}

    @classmethod
    def get_sub_model(cls, group_path: str, model_dict: dict | list) -> dict:
        """Get the sub model for a given group path."""

        for key in group_path.split("/"):
            if key:
                if isinstance(model_dict, list):
                    tuple_index = cls.get_tuple_index(key_name=key)
                    model_dict = model_dict[tuple_index]
                else:
                    model_dict = model_dict[key]
        return model_dict

    @staticmethod
    def _json_string_key(index: int) -> str:
        """Get json string key for string chunk number ``index``."""
        if index:
            return f"{JSON_TAG}_{index}"
        return JSON_TAG

    @classmethod
    def _json_string_from_hdf5(cls, fname: PathLike) -> str:
        """Load the model json string from an hdf5 file."""
        with h5py.File(fname, "r") as f_handle:
            num_string_parts = len([key for key in f_handle.keys() if JSON_TAG in key])
            json_string = b""
            for ind in range(num_string_parts):
                json_string += f_handle[cls._json_string_key(ind)][()]
        return json_string

    @classmethod
    def dict_from_hdf5(
        cls,
        fname: PathLike,
        group_path: str = "",
        custom_decoders: Optional[list[Callable]] = None,
        load_data_arrays: bool = True,
    ) -> dict:
        """Loads a dictionary containing the model contents from a .hdf5 file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .hdf5 file to load the :class:`Tidy3dBaseModel` from.
        group_path : str, optional
            Path to a group inside the file to selectively load a sub-element of the model only.
        custom_decoders : List[Callable]
            List of functions accepting
            (fname: str, group_path: str, model_dict: dict, key: str, value: Any) that store the
            value in the model dict after a custom decoding.

        Returns
        -------
        dict
            Dictionary containing the model.

        Example
        -------
        >>> sim_dict = Simulation.dict_from_hdf5(fname='folder/sim.hdf5') # doctest: +SKIP
        """

        def is_data_array(value: Any) -> bool:
            """Whether a value is supposed to be a data array based on the contents."""
            return isinstance(value, str) and value in DATA_ARRAY_MAP

        fname_path = Path(fname)

        def load_data_from_file(model_dict: dict, group_path: str = "") -> None:
            """For every DataArray item in dictionary, load path of hdf5 group as value."""

            for key, value in model_dict.items():
                subpath = f"{group_path}/{key}"

                # apply custom validation to the key value pair and modify model_dict
                if custom_decoders:
                    for custom_decoder in custom_decoders:
                        custom_decoder(
                            fname=str(fname_path),
                            group_path=subpath,
                            model_dict=model_dict,
                            key=key,
                            value=value,
                        )

                # write the path to the element of the json dict where the data_array should be
                if is_data_array(value):
                    data_array_type = DATA_ARRAY_MAP[value]
                    model_dict[key] = data_array_type.from_hdf5(
                        fname=fname_path, group_path=subpath
                    )
                    continue

                # if a list, assign each element a unique key, recurse
                if isinstance(value, (list, tuple)):
                    value_dict = cls.tuple_to_dict(tuple_values=value)
                    load_data_from_file(model_dict=value_dict, group_path=subpath)

                    # handle case of nested list of DataArray elements
                    val_tuple = list(value_dict.values())
                    for ind, (model_item, value_item) in enumerate(zip(model_dict[key], val_tuple)):
                        if is_data_array(model_item):
                            model_dict[key][ind] = value_item

                # if a dict, recurse
                elif isinstance(value, dict):
                    load_data_from_file(model_dict=value, group_path=subpath)

        model_dict = json.loads(cls._json_string_from_hdf5(fname=fname_path))
        group_path = cls._construct_group_path(group_path)
        model_dict = cls.get_sub_model(group_path=group_path, model_dict=model_dict)
        if load_data_arrays:
            load_data_from_file(model_dict=model_dict, group_path=group_path)
        return model_dict

    @classmethod
    def from_hdf5(
        cls,
        fname: PathLike,
        group_path: str = "",
        custom_decoders: Optional[list[Callable]] = None,
        **parse_obj_kwargs: Any,
    ) -> Self:
        """Loads :class:`Tidy3dBaseModel` instance to .hdf5 file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .hdf5 file to load the :class:`Tidy3dBaseModel` from.
        group_path : str, optional
            Path to a group inside the file to selectively load a sub-element of the model only.
            Starting `/` is optional.
        custom_decoders : List[Callable]
            List of functions accepting
            (fname: str, group_path: str, model_dict: dict, key: str, value: Any) that store the
            value in the model dict after a custom decoding.
        **parse_obj_kwargs
            Keyword arguments passed to pydantic's ``parse_obj`` method.

        Example
        -------
        >>> simulation = Simulation.from_hdf5(fname='folder/sim.hdf5') # doctest: +SKIP
        """

        group_path = cls._construct_group_path(group_path)
        model_dict = cls.dict_from_hdf5(
            fname=fname,
            group_path=group_path,
            custom_decoders=custom_decoders,
        )
        return cls._parse_model_dict(model_dict, **parse_obj_kwargs)

    def to_hdf5(
        self,
        fname: PathLike | io.BytesIO,
        custom_encoders: Optional[list[Callable]] = None,
    ) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .hdf5 file.

        Parameters
        ----------
        fname : PathLike | BytesIO
            Full path to the .hdf5 file or buffer to save the :class:`Tidy3dBaseModel` to.
        custom_encoders : List[Callable]
            List of functions accepting (fname: str, group_path: str, value: Any) that take
            the ``value`` supplied and write it to the hdf5 ``fname`` at ``group_path``.

        Example
        -------
        >>> simulation.to_hdf5(fname='folder/sim.hdf5') # doctest: +SKIP
        """

        export_model = self.to_static()
        traced_keys_payload = export_model.attrs.get(TRACED_FIELD_KEYS_ATTR)

        if traced_keys_payload is None:
            traced_keys_payload = self.attrs.get(TRACED_FIELD_KEYS_ATTR)
        if traced_keys_payload is None:
            traced_keys_payload = self._serialized_traced_field_keys()
        path = Path(fname) if isinstance(fname, PathLike) else fname
        with h5py.File(path, "w") as f_handle:
            json_str = export_model._json()
            for ind in range(ceil(len(json_str) / MAX_STRING_LENGTH)):
                ind_start = int(ind * MAX_STRING_LENGTH)
                ind_stop = min(int(ind + 1) * MAX_STRING_LENGTH, len(json_str))
                f_handle[self._json_string_key(ind)] = json_str[ind_start:ind_stop]

            def add_data_to_file(data_dict: dict, group_path: str = "") -> None:
                """For every DataArray item in dictionary, write path of hdf5 group as value."""

                for key, value in data_dict.items():
                    # append the key to the path
                    subpath = f"{group_path}/{key}"

                    if custom_encoders:
                        for custom_encoder in custom_encoders:
                            custom_encoder(fname=f_handle, group_path=subpath, value=value)

                    # write the path to the element of the json dict where the data_array should be
                    if isinstance(value, xr.DataArray):
                        value.to_hdf5(fname=f_handle, group_path=subpath)

                    # if a tuple, assign each element a unique key
                    if isinstance(value, (list, tuple)):
                        value_dict = export_model.tuple_to_dict(tuple_values=value)
                        add_data_to_file(data_dict=value_dict, group_path=subpath)

                    # if a dict, recurse
                    elif isinstance(value, dict):
                        add_data_to_file(data_dict=value, group_path=subpath)

            add_data_to_file(data_dict=export_model.dict())
            if traced_keys_payload:
                f_handle.attrs[TRACED_FIELD_KEYS_ATTR] = traced_keys_payload

    @classmethod
    def dict_from_hdf5_gz(
        cls,
        fname: PathLike,
        group_path: str = "",
        custom_decoders: Optional[list[Callable]] = None,
        load_data_arrays: bool = True,
    ) -> dict:
        """Loads a dictionary containing the model contents from a .hdf5.gz file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .hdf5.gz file to load the :class:`Tidy3dBaseModel` from.
        group_path : str, optional
            Path to a group inside the file to selectively load a sub-element of the model only.
        custom_decoders : List[Callable]
            List of functions accepting
            (fname: str, group_path: str, model_dict: dict, key: str, value: Any) that store the
            value in the model dict after a custom decoding.

        Returns
        -------
        dict
            Dictionary containing the model.

        Example
        -------
        >>> sim_dict = Simulation.dict_from_hdf5(fname='folder/sim.hdf5.gz') # doctest: +SKIP
        """
        file_descriptor, extracted = tempfile.mkstemp(".hdf5")
        os.close(file_descriptor)
        extracted_path = Path(extracted)
        try:
            extract_gzip_file(fname, extracted_path)
            result = cls.dict_from_hdf5(
                extracted_path,
                group_path=group_path,
                custom_decoders=custom_decoders,
                load_data_arrays=load_data_arrays,
            )
        finally:
            extracted_path.unlink(missing_ok=True)

        return result

    @classmethod
    def from_hdf5_gz(
        cls,
        fname: PathLike,
        group_path: str = "",
        custom_decoders: Optional[list[Callable]] = None,
        **parse_obj_kwargs: Any,
    ) -> Self:
        """Loads :class:`Tidy3dBaseModel` instance to .hdf5.gz file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .hdf5.gz file to load the :class:`Tidy3dBaseModel` from.
        group_path : str, optional
            Path to a group inside the file to selectively load a sub-element of the model only.
            Starting `/` is optional.
        custom_decoders : List[Callable]
            List of functions accepting
            (fname: str, group_path: str, model_dict: dict, key: str, value: Any) that store the
            value in the model dict after a custom decoding.
        **parse_obj_kwargs
            Keyword arguments passed to pydantic's ``parse_obj`` method.

        Example
        -------
        >>> simulation = Simulation.from_hdf5_gz(fname='folder/sim.hdf5.gz') # doctest: +SKIP
        """

        group_path = cls._construct_group_path(group_path)
        model_dict = cls.dict_from_hdf5_gz(
            fname=fname,
            group_path=group_path,
            custom_decoders=custom_decoders,
        )
        return cls._parse_model_dict(model_dict, **parse_obj_kwargs)

    def to_hdf5_gz(
        self, fname: PathLike | io.BytesIO, custom_encoders: Optional[list[Callable]] = None
    ) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .hdf5.gz file.

        Parameters
        ----------
        fname : PathLike | BytesIO
            Full path to the .hdf5.gz file or buffer to save the :class:`Tidy3dBaseModel` to.
        custom_encoders : List[Callable]
            List of functions accepting (fname: str, group_path: str, value: Any) that take
            the ``value`` supplied and write it to the hdf5 ``fname`` at ``group_path``.

        Example
        -------
        >>> simulation.to_hdf5_gz(fname='folder/sim.hdf5.gz') # doctest: +SKIP
        """
        file, decompressed = tempfile.mkstemp(".hdf5")
        os.close(file)
        try:
            self.to_hdf5(decompressed, custom_encoders=custom_encoders)
            compress_file_to_gzip(decompressed, fname)
        finally:
            os.unlink(decompressed)

    def __lt__(self, other):
        """define < for getting unique indices based on hash."""
        return hash(self) < hash(other)

    def __gt__(self, other):
        """define > for getting unique indices based on hash."""
        return hash(self) > hash(other)

    def __le__(self, other):
        """define <= for getting unique indices based on hash."""
        return hash(self) <= hash(other)

    def __ge__(self, other):
        """define >= for getting unique indices based on hash."""
        return hash(self) >= hash(other)

    def __eq__(self, other):
        """Define == for two Tidy3dBaseModels."""
        if other is None:
            return False

        def check_equal(dict1: dict, dict2: dict) -> bool:
            """Check if two dictionaries are equal, with special handlings."""

            # if different keys, automatically fail
            if not dict1.keys() == dict2.keys():
                return False

            # loop through elements in each dict
            for key in dict1:  # noqa: PLC0206
                val1 = dict1[key]
                val2 = dict2[key]

                val1 = get_static(val1)
                val2 = get_static(val2)

                # if one of val1 or val2 is None (exclusive OR)
                if (val1 is None) != (val2 is None):
                    return False

                # convert tuple to dict to use this recursive function
                if isinstance(val1, tuple) and isinstance(val2, tuple):
                    val1 = dict(zip(range(len(val1)), val1))
                    val2 = dict(zip(range(len(val2)), val2))

                # if dictionaries, recurse
                if isinstance(val1, dict) or isinstance(val2, dict):
                    are_equal = check_equal(val1, val2)
                    if not are_equal:
                        return False

                # if numpy arrays, use numpy to do equality check
                elif isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
                    if not np.array_equal(val1, val2):
                        return False

                # everything else
                else:
                    # note: this logic is because != is handled differently in DataArrays apparently
                    if not val1 == val2:
                        return False

            return True

        return check_equal(self.dict(), other.dict())

    @cached_property_guarded(lambda self: self._attrs_digest())
    def _json_string(self) -> str:
        """Returns string representation of a :class:`Tidy3dBaseModel`.

        Returns
        -------
        str
            Json-formatted string holding :class:`Tidy3dBaseModel` data.
        """
        return self._json()

    def _json(self, indent=INDENT, exclude_unset=False, **kwargs: Any) -> str:
        """Overwrites the model ``json`` representation with some extra customized handling.

        Parameters
        -----------
        **kwargs : kwargs passed to `self.json()`

        Returns
        -------
        str
            Json-formatted string holding :class:`Tidy3dBaseModel` data.
        """

        json_string = self.json(indent=indent, exclude_unset=exclude_unset, **kwargs)
        json_string = make_json_compatible(json_string)
        return json_string

    def _strip_traced_fields(
        self, starting_path: tuple[str, ...] = (), include_untraced_data_arrays: bool = False
    ) -> AutogradFieldMap:
        """Extract a dictionary mapping paths in the model to the data traced by ``autograd``.

        Parameters
        ----------
        starting_path : tuple[str, ...] = ()
            If provided, starts recursing in self.dict() from this path of field names
        include_untraced_data_arrays : bool = False
            Whether to include ``DataArray`` objects without tracers.
            We need to include these when returning data, but are unnecessary for structures.

        Returns
        -------
        dict
            mapping of traced fields used by ``autograd``

        """

        path = tuple(starting_path)
        if self._has_tracers is False and not include_untraced_data_arrays:
            return dict_ag()

        field_mapping = {}

        def handle_value(x: Any, path: tuple[str, ...]) -> None:
            """recursively update ``field_mapping`` with path to the autograd data."""

            # this is a leaf node that we want to trace, add this path and data to the mapping
            if isbox(x):
                field_mapping[path] = x

            # for data arrays, need to be more careful as their tracers are stored in .data
            elif isinstance(x, xr.DataArray) and (isbox(x.data) or include_untraced_data_arrays):
                field_mapping[path] = x.data

            # for sequences, add (i,) to the path and handle each value individually
            elif isinstance(x, (list, tuple)):
                for i, val in enumerate(x):
                    handle_value(val, path=(*path, i))

            # for dictionaries, add the (key,) to the path and handle each value individually
            elif isinstance(x, dict):
                for key, val in x.items():
                    handle_value(val, path=(*path, key))

        # recursively parse the dictionary of this object
        self_dict = self.dict()

        # if an include_only string was provided, only look at that subset of the dict
        if path:
            for key in path:
                self_dict = self_dict[key]

        handle_value(self_dict, path=path)

        if field_mapping:
            if not include_untraced_data_arrays:
                self._has_tracers = True
            return dict_ag(field_mapping)

        if not include_untraced_data_arrays and not path:
            self._has_tracers = False
        return dict_ag()

    def _insert_traced_fields(self, field_mapping: AutogradFieldMap) -> Self:
        """Recursively insert a map of paths to autograd-traced fields into a copy of this obj."""

        self_dict = self.dict()

        def insert_value(x, path: tuple[str, ...], sub_dict: dict) -> None:
            """Insert a value into the path into a dictionary."""
            current_dict = sub_dict
            for key in path[:-1]:
                if isinstance(current_dict[key], tuple):
                    current_dict[key] = list(current_dict[key])
                current_dict = current_dict[key]

            final_key = path[-1]
            if isinstance(current_dict[final_key], tuple):
                current_dict[final_key] = list(current_dict[final_key])

            sub_element = current_dict[final_key]
            if isinstance(sub_element, xr.DataArray):
                current_dict[final_key] = sub_element.copy(deep=False, data=x)

            else:
                current_dict[final_key] = x

        for path, value in field_mapping.items():
            insert_value(value, path=path, sub_dict=self_dict)

        return type(self)._parse_model_dict(self_dict)

    def _serialized_traced_field_keys(
        self, field_mapping: AutogradFieldMap | None = None
    ) -> Optional[str]:
        """Return a serialized, order-independent representation of traced field paths."""

        if field_mapping is None:
            field_mapping = self._strip_traced_fields()
        if not field_mapping:
            return None

        # TODO: remove this deferred import once TracerKeys is decoupled from Tidy3dBaseModel.
        from tidy3d.components.autograd.field_map import TracerKeys

        tracer_keys = TracerKeys.from_field_mapping(field_mapping)
        return tracer_keys.json(separators=(",", ":"), ensure_ascii=True)

    def to_static(self) -> Self:
        """Version of object with all autograd-traced fields removed."""

        if self._has_tracers is False:
            return self

        # get dictionary of all traced fields
        field_mapping = self._strip_traced_fields()

        # shortcut to just return self if no tracers found, for performance
        if not field_mapping:
            self._has_tracers = False
            return self

        # convert all fields to static values
        field_mapping_static = {key: get_static(val) for key, val in field_mapping.items()}

        # insert the static values into a copy of self
        static_self = self._insert_traced_fields(field_mapping_static)
        static_self._has_tracers = False
        return static_self

    @classmethod
    def add_type_field(cls) -> None:
        """Automatically place "type" field with model name in the model field dictionary."""

        value = cls.__name__
        annotation = Literal[value]

        tag_field = ModelField.infer(
            name=TYPE_TAG_STR,
            value=value,
            annotation=annotation,
            class_validators=None,
            config=cls.__config__,
        )
        cls.__fields__[TYPE_TAG_STR] = tag_field

    @classmethod
    def generate_docstring(cls) -> str:
        """Generates a docstring for a Tidy3D mode and saves it to the __doc__ of the class."""

        # store the docstring in here
        doc = ""

        # if the model already has a docstring, get the first lines and save the rest
        original_docstrings = []
        if cls.__doc__:
            original_docstrings = cls.__doc__.split("\n\n")
            class_description = original_docstrings.pop(0)
            doc += class_description
        original_docstrings = "\n\n".join(original_docstrings)

        # create the list of parameters (arguments) for the model
        doc += "\n\n    Parameters\n    ----------\n"
        for field_name, field in cls.__fields__.items():
            # ignore the type tag
            if field_name == TYPE_TAG_STR:
                continue

            # get data type
            data_type = field._type_display()

            # get default values
            default_val = field.get_default()
            if "=" in str(default_val):
                # handle cases where default values are pydantic models
                default_val = f"{default_val.__class__.__name__}({default_val})"
                default_val = (", ").join(default_val.split(" "))

            # make first line: name : type = default
            default_str = "" if field.required else f" = {default_val}"
            doc += f"    {field_name} : {data_type}{default_str}\n"

            # get field metadata
            field_info = field.field_info
            doc += "        "

            # add units (if present)
            units = field_info.extra.get("units")
            if units is not None:
                if isinstance(units, (tuple, list)):
                    unitstr = "("
                    for unit in units:
                        unitstr += str(unit)
                        unitstr += ", "
                    unitstr = unitstr[:-2]
                    unitstr += ")"
                else:
                    unitstr = units
                doc += f"[units = {unitstr}].  "

            # add description
            description_str = field_info.description
            if description_str is not None:
                doc += f"{description_str}\n"

        # add in remaining things in the docs
        if original_docstrings:
            doc += "\n"
            doc += original_docstrings

        doc += "\n"
        cls.__doc__ = doc

    def get_submodels_by_hash(self) -> dict[int, list[Union[str, tuple[str, int]]]]:
        """Return a dictionary of this object's sub-models indexed by their hash values."""
        fields = {}
        for key in self.__fields__:
            field = getattr(self, key)

            if isinstance(field, Tidy3dBaseModel):
                hash_ = hash(field)
                if hash_ not in fields:
                    fields[hash_] = []
                fields[hash_].append(key)

            # Do we need to consider np.ndarray here?
            elif isinstance(field, (list, tuple, np.ndarray)):
                for index, sub_field in enumerate(field):
                    if isinstance(sub_field, Tidy3dBaseModel):
                        hash_ = hash(sub_field)
                        if hash_ not in fields:
                            fields[hash_] = []
                        fields[hash_].append((key, index))

            elif isinstance(field, dict):
                for index, sub_field in field.items():
                    if isinstance(sub_field, Tidy3dBaseModel):
                        hash_ = hash(sub_field)
                        if hash_ not in fields:
                            fields[hash_] = []
                        fields[hash_].append((key, index))

        return fields

    @staticmethod
    def _scientific_notation(
        min_val: float, max_val: float, min_digits: int = 4
    ) -> tuple[str, str]:
        """
        Convert numbers to scientific notation, displaying only digits up to the point of difference,
        with a minimum number of significant digits specified by `min_digits`.
        """

        def to_sci(value: float, exponent: int, precision: int) -> str:
            normalized_value = value / (10**exponent)
            return f"{normalized_value:.{precision}f}e{exponent}"

        if min_val == 0 or max_val == 0:
            return f"{min_val:.0e}", f"{max_val:.0e}"

        exponent_min = math.floor(math.log10(abs(min_val)))
        exponent_max = math.floor(math.log10(abs(max_val)))

        common_exponent = min(exponent_min, exponent_max)
        normalized_min = min_val / (10**common_exponent)
        normalized_max = max_val / (10**common_exponent)

        if normalized_min == normalized_max:
            precision = min_digits
        else:
            precision = 0
            while round(normalized_min, precision) == round(normalized_max, precision):
                precision += 1

        precision = max(precision, min_digits)

        sci_min = to_sci(min_val, common_exponent, precision)
        sci_max = to_sci(max_val, common_exponent, precision)

        return sci_min, sci_max


def _make_lazy_proxy(
    target_cls: type,
    on_load: Optional[Callable[[Any], None]] = None,
) -> type:
    """
    Return a lazy-loading proxy subclass of ``target_cls``.

    Parameters
    ----------
    target_cls : type
        Must implement ``dict_from_file`` and ``parse_obj``.
    on_load : Callable[[Any], None] | None = None
        A function to call with the fully loaded instance once loaded.

    Returns
    -------
    type
        A class named ``<TargetClsName>Proxy`` with init args:
        ``(fname, group_path, parse_obj_kwargs)``.
    """

    proxy_name = f"{target_cls.__name__}Proxy"

    class _LazyProxy(target_cls):
        def __init__(
            self,
            fname: PathLike,
            group_path: Optional[str],
            parse_obj_kwargs: Any,
        ):
            object.__setattr__(self, "_lazy_fname", Path(fname))
            object.__setattr__(self, "_lazy_group_path", group_path)
            object.__setattr__(self, "_lazy_parse_obj_kwargs", dict(parse_obj_kwargs or {}))

        def copy(self, **kwargs: Any):
            """Return another lazy proxy instead of materializing."""
            return _LazyProxy(
                self._lazy_fname,
                self._lazy_group_path,
                {**self._lazy_parse_obj_kwargs, **kwargs},
            )

        def __getattribute__(self, name: str):
            if name in (
                "__class__",
                "__dict__",
                "__weakref__",
                "__post_root_validators__",
                "copy",  # <-- avoid materializing just for copy
            ) or name.startswith("_lazy_"):
                return object.__getattribute__(self, name)

            d = object.__getattribute__(self, "__dict__")
            if "_lazy_fname" in d:  # sentinel: not loaded yet
                fname = d["_lazy_fname"]
                group_path = d["_lazy_group_path"]
                kwargs = d["_lazy_parse_obj_kwargs"]

                model_dict = target_cls.dict_from_file(fname=fname, group_path=group_path)
                target = target_cls._parse_model_dict(model_dict, **kwargs)

                d.clear()
                d.update(target.__dict__)
                object.__setattr__(self, "__class__", target.__class__)
                object.__setattr__(self, "__fields_set__", set(target.__fields_set__))
                private_attrs = getattr(target, "__private_attributes__", {}) or {}
                for attr_name in private_attrs:
                    object.__setattr__(self, attr_name, getattr(target, attr_name))

                if on_load is not None:
                    on_load(self)

            return object.__getattribute__(self, name)

    _LazyProxy.__name__ = proxy_name
    return _LazyProxy
