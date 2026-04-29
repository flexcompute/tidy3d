"""global configuration / base class for pydantic models used to make simulation."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import total_ordering, wraps
from math import ceil
from os import PathLike
from pathlib import Path
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import h5py
import numpy as np
import rich
import xarray as xr
import yaml
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.tracer import isbox
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic import (
    ValidationError as PydanticValidationError,
)
from pydantic_core import InitErrorDetails, PydanticCustomError

from tidy3d.exceptions import FileError, Tidy3dError, format_chained_exception_message
from tidy3d.log import log

from .autograd.types import TracedDict
from .autograd.utils import get_static
from .data.data_array import DATA_ARRAY_MAP
from .docstrings import (
    _DOCSTRING_RAW_ATTR,
    _clean_default_repr,
    _fmt_ann_literal,
    _format_model_default,
)
from .file_util import compress_file_to_gzip, extract_gzip_file
from .types import TYPE_TAG_STR, Undefined

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import NoReturn

    from pydantic.fields import FieldInfo
    from pydantic.functional_validators import ModelWrapValidatorHandler

    from tidy3d.compat import Self

    from .autograd.types import AutogradFieldMap

INDENT_JSON_FILE = 4  # default indentation of json string in json files
INDENT = None  # default indentation of json string used internally
JSON_TAG = "JSON_STRING"
# If json string is larger than ``MAX_STRING_LENGTH``, split the string when storing in hdf5
MAX_STRING_LENGTH = 1_000_000_000
FORBID_SPECIAL_CHARACTERS = ["/"]
TRACED_FIELD_KEYS_ATTR = "__tidy3d_traced_field_keys__"
TYPE_TO_CLASS_MAP: dict[str, type[Tidy3dBaseModel]] = {}
_LAZY_PROXY_UNHANDLED = object()

_CacheReturn = TypeVar("_CacheReturn")


def cache(prop: Callable[[Any], _CacheReturn]) -> Callable[[Any], _CacheReturn]:
    """Decorates a property to cache the first computed value and return it on subsequent calls."""

    # note, we could also just use `prop` as dict key, but hashing property might be slow
    prop_name = prop.__name__

    @wraps(prop)
    def cached_property_getter(self: Any) -> _CacheReturn:
        """The new property method to be returned by decorator."""

        stored_value = self._cached_properties.get(prop_name)

        if stored_value is not None:
            return stored_value

        computed_value = prop(self)
        self._cached_properties[prop_name] = computed_value
        return computed_value

    return cached_property_getter


def cached_property(cached_property_getter: Callable[[Any], _CacheReturn]) -> property:
    """Shortcut for property(cache()) of a getter."""

    return property(cache(cached_property_getter))


def _default_keyed_cache_key(
    signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Build a canonical cache key from bound method arguments."""

    bound = signature.bind(None, *args, **kwargs)
    bound.apply_defaults()
    bound.arguments.pop("self", None)
    return tuple(bound.arguments.items())


def keyed_cache(
    key_func: Callable[..., Any] | None = None,
) -> Callable[[Callable[..., _CacheReturn]], Callable[..., _CacheReturn]]:
    """Decorate a method so it caches multiple values keyed by call arguments.

    Cached entries are stored in ``self._cached_properties[method_name]``. By default,
    the key is derived from the method signature, so equivalent calls such as
    ``fn(1)`` and ``fn(x=1)`` share a cache entry. Pass ``key_func`` to provide a
    custom key builder for unhashable or normalized arguments.
    """

    def _decorator(method: Callable[..., _CacheReturn]) -> Callable[..., _CacheReturn]:
        signature = inspect.signature(method)
        cache_name = method.__name__

        @wraps(method)
        def _cached(self: Any, *args: Any, **kwargs: Any) -> _CacheReturn:
            key = (
                key_func(self, *args, **kwargs)
                if key_func is not None
                else _default_keyed_cache_key(signature, args, kwargs)
            )
            return self._get_cached_value_by_key(
                cache_name, key, lambda: method(self, *args, **kwargs)
            )

        return _cached

    return _decorator


_GuardedReturn = TypeVar("_GuardedReturn")


def cached_property_guarded(
    key_func: Callable[[Any], Any],
) -> Callable[[Callable[[Any], _GuardedReturn]], property]:
    """Like cached_property, but invalidates when the key_func(self) changes."""

    def _decorator(getter: Callable[[Any], _GuardedReturn]) -> property:
        prop_name = getter.__name__

        @wraps(getter)
        def _guarded(self: Any) -> _GuardedReturn:
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


def make_json_compatible(json_string: str) -> str:
    """Makes the string compatible with json standards, notably for infinity."""

    tmp_string = "<<TEMPORARY_INFINITY_STRING>>"
    json_string = json_string.replace("-Infinity", tmp_string)
    json_string = json_string.replace('""-Infinity""', tmp_string)
    json_string = json_string.replace("Infinity", '"Infinity"')
    json_string = json_string.replace('""Infinity""', '"Infinity"')
    return json_string.replace(tmp_string, '"-Infinity"')


def _strip_json_exponent_plus_signs(json_string: str) -> str:
    """Strip `+` from JSON number exponents while preserving JSON strings.

    Assumes valid pydantic_core JSON. Its versions disagree on exponent spelling; Tidy3D keeps
    the no-plus form so serialized model bytes stay stable across versions.
    """

    normalized = []
    in_string = False
    escaped = False

    for index, character in enumerate(json_string):
        if in_string:
            normalized.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif (
            character == "+"
            and normalized
            and normalized[-1] in {"e", "E"}
            and index + 1 < len(json_string)
            and json_string[index + 1].isdigit()
        ):
            continue
        normalized.append(character)

    return "".join(normalized)


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


T = TypeVar("T", bound="Tidy3dBaseModel")


def field_allows_scalar(field: FieldInfo) -> bool:
    annotation = field.annotation

    def allows_scalar(a: Any) -> bool:
        origin = get_origin(a)
        if origin in (Union, UnionType):
            args = (arg for arg in get_args(a) if arg is not type(None))
            return any(allows_scalar(arg) for arg in args)
        if origin is not None:
            return False
        return isinstance(a, type) and issubclass(a, (float, int, np.generic))

    return allows_scalar(annotation)


@total_ordering
class Tidy3dBaseModel(BaseModel):
    """Base pydantic model that all Tidy3d components inherit from.
    Defines configuration for handling data structures
    as well as methods for importing, exporting, and hashing tidy3d objects.
    For more details on pydantic base models, see:
    `Pydantic models <https://docs.pydantic.dev/latest/concepts/models/>`_
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        defer_build=True,
        validate_default=True,
        populate_by_name=True,
        ser_json_inf_nan="strings",
        extra="forbid",
        frozen=True,
    )

    _DOCSTRING_SHOW_DEFAULT_ARGS: ClassVar[bool] = False
    _DOCSTRING_INCLUDE_ATTRS: ClassVar[bool] = False

    attrs: dict = Field(
        default_factory=dict,
        title="Attributes",
        description="Dictionary storing arbitrary metadata for a Tidy3D object. "
        "This dictionary can be freely used by the user for storing data without affecting the "
        "operation of Tidy3D as it is not used internally. "
        "Note that, unlike regular Tidy3D fields, ``attrs`` are mutable. "
        "For example, the following is allowed for setting an ``attr`` ``obj.attrs['foo'] = bar``. "
        "Also note that Tidy3D will raise a ``TypeError`` if ``attrs`` contain objects "
        "that can not be serialized. One can check if ``attrs`` are serializable "
        "by calling ``obj.model_dump_json()``.",
    )

    _cached_properties: dict = PrivateAttr(default_factory=dict)
    _has_tracers: bool | None = PrivateAttr(default=None)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        """Serialize with stable float exponent formatting across pydantic_core versions."""
        return _strip_json_exponent_plus_signs(super().model_dump_json(*args, **kwargs))

    def _get_keyed_cache_store(self, cache_name: str) -> dict[Any, Any]:
        """Return a keyed cache dict stored under ``cache_name``."""

        cache_store = self._cached_properties.get(cache_name)
        if cache_store is None:
            cache_store = {}
            self._cached_properties[cache_name] = cache_store
        elif not isinstance(cache_store, dict):
            raise TypeError(
                f"Cached entry {cache_name!r} is not a keyed cache store: "
                f"{type(cache_store).__name__}."
            )
        return cache_store

    def _get_cached_value_by_key(
        self,
        cache_name: str,
        key: Any,
        builder: Callable[[], _CacheReturn],
    ) -> _CacheReturn:
        """Return a keyed cached value, computing and storing it on a cache miss."""

        cache_store = self._get_keyed_cache_store(cache_name)
        if key not in cache_store:
            cache_store[key] = builder()
        return cache_store[key]

    @field_validator("name", check_fields=False)
    @classmethod
    def _validate_name_no_special_characters(cls: type[T], name: str | None) -> str | None:
        if name is None:
            return name
        for character in FORBID_SPECIAL_CHARACTERS:
            if character in name:
                raise ValueError(
                    f"Special character '{character}' not allowed in component name {name}."
                )
        return name

    def __init_subclass__(cls: type[T], **kwargs: Any) -> None:
        """Injects a constant discriminator field before Pydantic builds the model.

        Adds
            type: Literal["<ClassName>"] = "<ClassName>"
        to every concrete subclass so it can participate in a
        `Field(discriminator="type")` union without manual boilerplate.

        Must run *before* `super().__init_subclass__()`; that call lets Pydantic
        see the injected field during its normal schema/validator generation.
        See also: https://peps.python.org/pep-0487/
        """
        tag = cls.__name__
        cls.__annotations__[TYPE_TAG_STR] = Literal[tag]
        setattr(cls, TYPE_TAG_STR, tag)
        TYPE_TO_CLASS_MAP[tag] = cls

        if "__tidy3d_end_capture__" not in cls.__dict__:

            @model_validator(mode="after")
            def __tidy3d_end_capture__(self: T) -> T:
                if log._capture:
                    log.end_capture(self)
                return self

            cls.__tidy3d_end_capture__ = __tidy3d_end_capture__

        super().__init_subclass__(**kwargs)

    @classmethod
    def __pydantic_init_subclass__(cls: type[T], **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)

        # add docstring once pydantic is done constructing the class
        if _DOCSTRING_RAW_ATTR not in cls.__dict__:
            setattr(cls, _DOCSTRING_RAW_ATTR, cls.__doc__ or "")
        cls.__doc__ = cls.generate_docstring()

    @classmethod
    def model_rebuild(
        cls,
        *,
        force: bool = False,
        raise_errors: bool = True,
        _parent_namespace_depth: int = 2,
        _types_namespace: Mapping[str, Any] | None = None,
    ) -> bool | None:
        rebuilt = super().model_rebuild(
            force=force,
            raise_errors=raise_errors,
            _parent_namespace_depth=_parent_namespace_depth + 1,
            _types_namespace=_types_namespace,
        )
        if _DOCSTRING_RAW_ATTR not in cls.__dict__:
            setattr(cls, _DOCSTRING_RAW_ATTR, cls.__doc__ or "")
        cls.__doc__ = cls.generate_docstring()
        return rebuilt

    @model_validator(mode="wrap")
    @classmethod
    def _capture_validation_warnings(
        cls: type[T],
        data: Any,
        handler: ModelWrapValidatorHandler[T],
    ) -> T:
        if not log._capture:
            return handler(data)

        log.begin_capture()
        try:
            return handler(data)
        except Exception:
            log.abort_capture()
            raise

    def _raise_validation_error_at_loc(
        self, message: Any, *loc: Any, log_error: bool = True
    ) -> NoReturn:
        """Raise a Pydantic validation error anchored to a specific field path."""
        message_str = str(message)
        if log_error and not isinstance(message, Tidy3dError):
            log.error(message_str)
        raise PydanticValidationError.from_exception_data(
            type(self).__name__,
            [
                InitErrorDetails(
                    type=PydanticCustomError("value_error", message_str),
                    loc=loc,
                    input=None,
                )
            ],
            hide_input=True,
        )

    def _call_with_validation_loc(
        self, loc: Sequence[Any], func: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Call a helper that may raise ``Tidy3dError`` and attach a concrete location."""
        try:
            return func(*args, **kwargs)
        except Tidy3dError as error:
            self._raise_validation_error_at_loc(str(error), *loc, log_error=False)

    def __hash__(self) -> int:
        """Hash method."""
        return self._recursive_hash(self)

    @staticmethod
    def _recursive_hash(value: Any) -> int:
        # Handle Autograd ArrayBoxes
        if isinstance(value, ArrayBox):
            # Unwrap the underlying numpy array and recurse
            return Tidy3dBaseModel._recursive_hash(value._value)
        if isinstance(value, np.ndarray):
            # numpy arrays are not hashable by default, use byte representation
            v_hash = hashlib.md5(value.tobytes()).hexdigest()
            return hash(v_hash)
        if isinstance(value, (xr.DataArray, xr.Dataset)):
            # we choose to not hash data arrays as this would require a lot of careful handling of units, metadata.
            # technically this is incorrect, but should never lead to bugs in current implementation
            return hash(str(value.__class__.__name__))
        if isinstance(value, str):
            # this if-case is necessary because length-1 string would lead to infinite recursion in sequence case below
            return hash(value)
        if isinstance(value, Sequence):
            # this assumes all objects in lists are hashable by default and do not require special handling
            v_hash = tuple([Tidy3dBaseModel._recursive_hash(vi) for vi in value])
            return hash(v_hash)
        if isinstance(value, dict):
            to_hash_list = []
            for k, v in value.items():
                v_hash = Tidy3dBaseModel._recursive_hash(v)
                to_hash_list.append((k, v_hash))
            return hash(tuple(to_hash_list))
        if isinstance(value, Tidy3dBaseModel):
            # This function needs to take special care because of mutable attributes inside of frozen pydantic models
            to_hash_list = []
            for k in type(value).model_fields:
                if k == "attrs":
                    continue
                v_hash = Tidy3dBaseModel._recursive_hash(getattr(value, k))
                to_hash_list.append((k, v_hash))
            extra = getattr(value, "__pydantic_extra__", None)
            if extra:
                for k, v in extra.items():
                    v_hash = Tidy3dBaseModel._recursive_hash(v)
                    to_hash_list.append((k, v_hash))
            # attrs is mutable, use serialized output as safe hashing option
            if value.attrs:
                attrs_str = value._attrs_digest()
                attrs_hash = hash(attrs_str)
                to_hash_list.append(("attrs", attrs_hash))
            return hash(tuple(to_hash_list))
        return hash(value)

    def _hash_self(self) -> str:
        """Hash this component with ``hashlib`` in a way that is the same every session."""
        bf = io.BytesIO()
        self.to_hdf5(bf)
        return hashlib.md5(bf.getvalue()).hexdigest()

    @model_validator(mode="before")
    @classmethod
    def coerce_numpy_scalars_for_model(cls, data: Any) -> Any:
        """
        coerce numpy scalars / size-1 arrays to native Python
        scalars, but only for fields whose annotations allow scalars.
        """
        if not isinstance(data, dict):
            return data

        for name, field in cls.model_fields.items():
            if name not in data or not field_allows_scalar(field):
                continue

            v = data[name]
            if isinstance(v, np.generic) or (isinstance(v, np.ndarray) and v.size == 1):
                data[name] = v.item()

        return data

    @classmethod
    def _get_type_value(cls, obj: dict[str, Any]) -> str:
        """Return the type tag from a raw dictionary."""
        if not isinstance(obj, dict):
            raise TypeError("Input must be a dict")
        try:
            type_value = obj[TYPE_TAG_STR]
        except KeyError as exc:
            raise ValueError(
                format_chained_exception_message(f'Missing "{TYPE_TAG_STR}" in data', exc)
            ) from exc
        if not isinstance(type_value, str) or not type_value:
            raise ValueError(f'Invalid "{TYPE_TAG_STR}" value: {type_value!r}')
        return type_value

    @classmethod
    def _get_registered_class(cls, type_value: str) -> type[Tidy3dBaseModel]:
        try:
            return TYPE_TO_CLASS_MAP[type_value]
        except KeyError as exc:
            raise ValueError(
                format_chained_exception_message(f"Unknown type: {type_value}", exc)
            ) from exc

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
        cls, fname: PathLike, group_path: str | None = None
    ) -> type[Tidy3dBaseModel]:
        """Peek the file metadata to determine the subclass to instantiate."""
        model_dict = cls.dict_from_file(
            fname=fname,
            group_path=group_path,
            load_data_arrays=False,
        )
        return cls._resolve_dispatch_target(model_dict)

    @classmethod
    def _model_validate(cls, obj: dict[str, Any], **parse_obj_kwargs: Any) -> Tidy3dBaseModel:
        """Dispatch ``obj`` to the correct subclass registered in the type map."""
        target_cls = cls._resolve_dispatch_target(obj)
        if target_cls is cls:
            return super().model_validate(obj, **parse_obj_kwargs)
        return target_cls.model_validate(obj, **parse_obj_kwargs)

    @classmethod
    def _validate_model_dict(
        cls, model_dict: dict[str, Any], **parse_obj_kwargs: Any
    ) -> Tidy3dBaseModel:
        """Parse ``model_dict`` while optionally auto-dispatching when called on the base class."""
        if cls is Tidy3dBaseModel:
            return cls._model_validate(model_dict, **parse_obj_kwargs)
        return cls.model_validate(model_dict, **parse_obj_kwargs)

    def _preprocess_update_values(self, update: Mapping[str, Any]) -> dict[str, Any]:
        """Preprocess update values to convert lists to tuples where appropriate.

        This helps avoid Pydantic v2 serialization warnings when using `model_copy()`
        with list values for tuple fields.
        """
        if not update:
            return {}

        def get_tuple_element_type(annotation: Any) -> type | None:
            """Get the element type of a tuple annotation if it has one consistent type."""
            origin = get_origin(annotation)
            if origin is tuple:
                args = get_args(annotation)
                if args:
                    # Check if it's a homogeneous tuple like tuple[bool, ...] or tuple[str, ...]
                    if len(args) == 2 and args[1] is ...:
                        return args[0]
                    # Check if all elements have the same type
                    if all(arg == args[0] for arg in args):
                        return args[0]
            return None

        def should_convert_to_tuple(annotation: Any) -> tuple[bool, type[Any] | None]:
            """Check if the given annotation represents a tuple type and return element type if any."""
            origin = get_origin(annotation)

            if origin is tuple:
                return True, get_tuple_element_type(annotation)

            # Union types containing tuple
            if origin is Union:
                args = get_args(annotation)
                for arg in args:
                    if get_origin(arg) is tuple:
                        return True, get_tuple_element_type(arg)

            return False, None

        def convert_value(value: Any, field_info: FieldInfo) -> Any:
            """Convert value based on field type information."""
            annotation = field_info.annotation

            # Handle list/tuple to tuple conversion with proper element types
            is_tuple, element_type = should_convert_to_tuple(annotation)

            # Check if value is a numpy array and needs to be converted to tuple
            try:
                import numpy as np

                if isinstance(value, np.ndarray) and is_tuple:
                    # Convert numpy array to list first
                    value = value.tolist()
            except ImportError:
                pass

            # Handle autograd SequenceBox - convert to tuple
            if (
                is_tuple
                and hasattr(value, "__class__")
                and value.__class__.__name__ == "SequenceBox"
            ):
                # SequenceBox is iterable, so convert it to tuple
                return tuple(value)

            if isinstance(value, (list, tuple)) and is_tuple:
                # Convert elements based on element type
                if element_type is bool:
                    # Convert integers to booleans
                    value = [bool(item) if isinstance(item, int) else item for item in value]
                elif element_type is str:
                    # Ensure all elements are strings
                    value = [str(item) if not isinstance(item, str) else item for item in value]
                else:
                    # Check if it's a numpy array or contains numpy types
                    try:
                        import numpy as np

                        if any(isinstance(item, np.generic) for item in value):
                            # Convert numpy types to Python types
                            value = [
                                item.item() if isinstance(item, np.generic) else item
                                for item in value
                            ]
                    except ImportError:
                        pass
                return tuple(value)

            # Handle int to bool conversion
            if annotation is bool and isinstance(value, int):
                return bool(value)

            # Handle dict to Tidy3dBaseModel conversion
            if isinstance(value, dict):
                # Check if the annotation is a Tidy3dBaseModel subclass
                origin = get_origin(annotation)
                if origin is None:
                    # Not a generic type, check if it's a direct subclass
                    try:
                        if isinstance(annotation, type) and issubclass(annotation, Tidy3dBaseModel):
                            return annotation(**value)
                    except (TypeError, AttributeError):
                        pass
                elif origin is Union:
                    # For Union types, try to convert to the first matching Tidy3dBaseModel type
                    args = get_args(annotation)
                    for arg in args:
                        try:
                            if isinstance(arg, type) and issubclass(arg, Tidy3dBaseModel):
                                return arg(**value)
                        except (TypeError, AttributeError, ValueError):
                            continue

            return value

        processed = {}
        for field_name, value in update.items():
            if field_name in type(self).model_fields:
                field_info = type(self).model_fields[field_name]
                processed[field_name] = convert_value(value, field_info)
            else:
                processed[field_name] = value

        return processed

    def copy(
        self,
        deep: bool = True,
        *,
        validate: bool = True,
        update: Mapping[str, Any] | None = None,
    ) -> Self:
        """Return a copy of the model.

        Parameters
        ----------
        deep : bool = True
            Whether to make a deep copy of all fields before applying *update*.
            With ``deep=False``, fields **not** listed in *update* become shared
            references with the original instance. This is only safe when no
            code path will later mutate those shared objects in-place (e.g.
            ``array[:] = ...``). Since tidy3d models are generally treated as
            immutable, ``deep=False`` is usually fine for internal
            copy-and-update patterns.
        validate : bool = True
            If ``True``, run full Pydantic validation on the copied data.

            **When is** ``validate=False`` **safe?** Only when *all* update values
            are already the correct pydantic field types (validated model
            instances, plain Python scalars, ``None``, etc.).

            It is **not safe** when update values come from xarray operations
            (arithmetic, ``sel``, ``isel``, ``interp``, ``rename``, ``drop_vars``,
            numpy ufuncs, ``.real/.imag/.abs``, ...) because those return plain
            ``xr.DataArray`` and lose the tidy3d subclass
            (``ScalarFieldDataArray``, ``IndexedDataArray``, ...). Validation is
            needed to coerce them back. It is also **not safe** when the model
            has validators that enforce business logic (e.g.
            ``Simulation._check_normalize_index``).
        update : Optional[Mapping[str, Any]] = None
            Optional mapping of fields to overwrite (passed straight
            through to ``model_copy(update=...)``).
        """
        if update and self.model_config.get("extra") == "forbid":
            invalid = set(update) - set(type(self).model_fields)
            if invalid:
                raise KeyError(f"'{self.type}' received invalid fields on copy: {invalid}")

        # preprocess update values to convert lists to tuples where appropriate
        if update:
            update = self._preprocess_update_values(update)

        new_model = self.model_copy(deep=deep, update=update)

        if validate:
            return self.__class__.model_validate(new_model.model_dump())
        else:
            # make sure cache is always cleared
            new_model._cached_properties = {}

        new_model._has_tracers = None
        return new_model

    def updated_copy(
        self,
        path: str | None = None,
        *,
        deep: bool = True,
        validate: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Make copy of a component instance with ``**kwargs`` indicating updated field values.

        Note
        ----
        If ``path`` is supplied, applies the updated copy with the update performed on the sub-
        component corresponding to the path. For indexing into a tuple or list, use the integer
        value.

        Example
        -------
        >>> sim = simulation.updated_copy(size=new_size, path=f"structures/{i}/geometry") # doctest: +SKIP
        """
        if not path:
            return self.copy(deep=deep, validate=validate, update=kwargs)

        path_parts = path.split("/")
        field_name, *rest = path_parts

        try:
            sub_component = getattr(self, field_name)
        except AttributeError as exc:
            raise AttributeError(
                format_chained_exception_message(
                    f"Could not find field '{field_name}' in path '{path}'. "
                    f"Available top-level fields: {tuple(type(self).model_fields)}",
                    exc,
                )
            ) from exc

        if isinstance(sub_component, (list, tuple)):
            try:
                index = int(rest[0])
            except (IndexError, ValueError):
                raise ValueError(
                    f"Expected integer index into '{field_name}' in path '{path}'."
                ) from None
            sub_component_list = list(sub_component)
            sub_component_list[index] = sub_component_list[index].updated_copy(
                path="/".join(rest[1:]),
                deep=deep,
                validate=validate,
                **kwargs,
            )
            new_value = type(sub_component)(sub_component_list)
        else:
            new_value = sub_component.updated_copy(
                path="/".join(rest),
                deep=deep,
                validate=validate,
                **kwargs,
            )

        return self.copy(deep=deep, validate=validate, update={field_name: new_value})

    @staticmethod
    def _core_model_traversal(
        current_obj: Any, current_path_segments: tuple[str, ...]
    ) -> Iterator[tuple[Self, tuple[str, ...]]]:
        """
        Recursively traverses a model structure yielding Tidy3dBaseModel instances and their paths.

        This is an internal helper method used by :meth:`find_paths` and :meth:`find_submodels`
        to navigate nested :class:`~tidy3d.Tidy3dBaseModel` structures.

        Parameters
        ----------
        current_obj : Any
            The current object in the traversal, which can be a :class:`~tidy3d.Tidy3dBaseModel`,
            list, tuple, or other type.
        current_path_segments : tuple[str, ...]
            A tuple of strings representing the path segments from the initial model
            to the ``current_obj``.

        Returns
        -------
        Iterator[tuple[Self, tuple[str, ...]]]
            An iterator yielding tuples, where the first element is a found :class:`~tidy3d.Tidy3dBaseModel` instance
            and the second is a tuple of strings representing the path to that instance
            from the initial object. The path for the top-level model itself will be an empty tuple.
        """
        if isinstance(current_obj, Tidy3dBaseModel):
            yield current_obj, current_path_segments

            for field_name in type(current_obj).model_fields:
                if (
                    field_name == "type"
                    and getattr(current_obj, field_name, None) == current_obj.__class__.__name__
                ):
                    continue

                field_value = getattr(current_obj, field_name)
                yield from Tidy3dBaseModel._core_model_traversal(
                    field_value, (*current_path_segments, field_name)
                )
        elif isinstance(current_obj, (list, tuple)):
            for index, item in enumerate(current_obj):
                yield from Tidy3dBaseModel._core_model_traversal(
                    item, (*current_path_segments, str(index))
                )

    def find_paths(self, target_field_name: str, target_field_value: Any = Undefined) -> list[str]:
        """
        Finds paths to nested model instances that have a specific field, optionally matching a value.

        The paths are string representations like ``"structures/0/geometry"``, designed for direct
        use with the :meth:`updated_copy` method to modify specific parts of this model.
        An empty string ``""`` in the returned list indicates that this model instance
        itself (the one ``find_paths`` is called on) matches the criteria.

        Parameters
        ----------
        target_field_name : str
            The name of the attribute (field) to search for within nested
            :class:`~tidy3d.Tidy3dBaseModel` instances. For example, ``"name"`` or ``"permittivity"``.
        target_field_value : Any, optional
            If provided, only paths to model instances where ``target_field_name`` also has this
            specific value will be returned. If omitted, paths are returned if the
            ``target_field_name`` exists, regardless of its value.

        Returns
        -------
        list[str]
            A sorted list of unique string paths. Each path points to a
            :class:`~tidy3d.Tidy3dBaseModel` instance that possesses the ``target_field_name``
            (and optionally matches ``target_field_value``).

        Example
        -------
        >>> # Assume 'sim' is a Tidy3D simulation object
        >>> # Find all geometries named "waveguide"
        >>> paths = sim.find_paths(target_field_name="name", target_field_value="waveguide") # doctest: +SKIP
        >>> # paths might be ['structures/0', 'structures/3']
        >>> # Update the size of the first found "waveguide"
        >>> new_sim = sim.updated_copy(path=paths[0], size=(1.0, 0.5, 0.22)) # doctest: +SKIP
        """
        found_paths_set = set()

        for sub_model_instance, path_segments_to_sub_model in Tidy3dBaseModel._core_model_traversal(
            self, ()
        ):
            if target_field_name in type(sub_model_instance).model_fields:
                passes_value_filter = True
                if target_field_value is not Undefined:
                    actual_value = getattr(sub_model_instance, target_field_name)
                    if actual_value != target_field_value:
                        passes_value_filter = False

                if passes_value_filter:
                    path_str = "/".join(path_segments_to_sub_model)
                    found_paths_set.add(path_str)

        return sorted(found_paths_set)

    def find_submodels(self, target_type: Self) -> list[Self]:
        """
        Finds all unique nested instances of a specific Tidy3D model type within this model.

        This method traverses the model structure and collects all instances that are of
        the ``target_type`` (e.g., :class:`~tidy3d.Structure`, :class:`~tidy3d.Medium`,
        :class:`~tidy3d.Box`).
        Uniqueness is determined by the model's content. The order of models
        in the returned list corresponds to their first encounter during a depth-first traversal.

        Parameters
        ----------
        target_type : Tidy3dBaseModel
            The specific Tidy3D class (e.g., ``Structure``, ``Medium``, ``Box``) to search for.
            This class must be a subclass of :class:`~tidy3d.Tidy3dBaseModel`.

        Returns
        -------
        list[Tidy3dBaseModel]
            A list of unique instances found within this model that are of the
            provided ``target_type``.

        Example
        -------
        >>> # Assume 'sim' is a Tidy3D Simulation object
        >>> # Find all Structure instances within the simulation
        >>> all_structures = sim.find_submodels(td.Structure) # doctest: +SKIP
        >>> for struct in all_structures:
        ...     print(f"Structure: {struct.name}, medium: {struct.medium}") # doctest: +SKIP

        >>> # Find all Box geometries within the simulation
        >>> all_boxes = sim.find_submodels(td.Box) # doctest: +SKIP
        >>> for box in all_boxes:
        ...     print(f"Found Box with size: {box.size}") # doctest: +SKIP

        >>> # Find all Medium instances (useful for checking materials)
        >>> all_media = sim.find_submodels(td.Medium) # doctest: +SKIP
        >>> # Note: This would find td.Medium instances, but not td.PECMedium or td.PoleResidue
        >>> # unless they inherit directly from td.Medium and not just Tidy3dBaseModel or td.AbstractMedium.
        >>> # To find all medium types, one might search for td.AbstractMedium if that's a common base.
        """
        found_models_dict = {}

        for sub_model_candidate, _ in Tidy3dBaseModel._core_model_traversal(self, ()):
            if isinstance(sub_model_candidate, target_type):
                if sub_model_candidate not in found_models_dict:
                    found_models_dict[sub_model_candidate] = True

        return list(found_models_dict.keys())

    def help(self, methods: bool = False) -> None:
        """Prints message describing the fields and methods of a :class:`~tidy3d.Tidy3dBaseModel`.

        Parameters
        ----------
        methods : bool = False
            Whether to also print out information about object's methods.

        Example
        -------
        >>> simulation.help(methods=True) # doctest: +SKIP
        """
        rich.inspect(type(self), methods=methods)

    @classmethod
    def _lazy_proxy_copy_state_keys(cls) -> tuple[str, ...]:
        """Return names of extra lazy-proxy state fields to preserve across ``copy()``."""

        return ()

    @classmethod
    def _lazy_proxy_resolve_attr(cls, proxy: Any, name: str, lazy_state: dict[str, Any]) -> Any:
        """Return a lazily resolved attribute value or ``_LAZY_PROXY_UNHANDLED``."""

        return _LAZY_PROXY_UNHANDLED

    @classmethod
    def _lazy_proxy_materialize(cls, lazy_state: dict[str, Any]) -> Self:
        """Build the fully loaded target instance for a lazy proxy."""

        model_dict = cls.dict_from_file(
            fname=lazy_state["_lazy_fname"],
            group_path=lazy_state["_lazy_group_path"],
        )
        return cls._validate_model_dict(model_dict, **lazy_state["_lazy_parse_obj_kwargs"])

    @classmethod
    def from_file(
        cls,
        fname: PathLike,
        group_path: str | None = None,
        lazy: bool = False,
        on_load: Callable[[Any], None] | None = None,
        **parse_obj_kwargs: Any,
    ) -> Self:
        """Loads a :class:`~tidy3d.Tidy3dBaseModel` from .yaml, .json, .hdf5, or .hdf5.gz file.

        Parameters
        ----------
        fname : PathLike
            Full path to the file to load the :class:`~tidy3d.Tidy3dBaseModel` from.
        group_path : Optional[str] = None
            Path to a group inside the file to use as the base level. Only for hdf5 files.
            Starting `/` is optional.
        lazy : bool = False
            Whether to load the actual data (``lazy=False``) or return a proxy that loads
            the data when accessed (``lazy=True``).
        on_load : Optional[Callable[[Any], None]] = None
            Callback function executed once the model is fully materialized.
            Only used if ``lazy=True``. The callback is invoked with the loaded
            instance as its sole argument, enabling post-processing such as
            validation, logging, or warnings checks.
        **model_validate_kwargs
            Keyword arguments passed to pydantic's ``model_validate`` method when loading model.

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
        obj = cls._validate_model_dict(model_dict, **parse_obj_kwargs)
        if not lazy and on_load is not None:
            on_load(obj)
        return obj

    @classmethod
    def dict_from_file(
        cls: type[T],
        fname: PathLike,
        group_path: str | None = None,
        *,
        load_data_arrays: bool = True,
    ) -> dict:
        """Loads a dictionary containing the model from a .yaml, .json, .hdf5, or .hdf5.gz file.

        Parameters
        ----------
        fname : PathLike
            Full path to the file to load the :class:`~tidy3d.Tidy3dBaseModel` from.
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
        """Exports :class:`~tidy3d.Tidy3dBaseModel` instance to .yaml, .json, or .hdf5 file

        Parameters
        ----------
        fname : PathLike
            Full path to the .yaml or .json file to save the :class:`~tidy3d.Tidy3dBaseModel` to.

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
    def from_json(cls: type[T], fname: PathLike, **model_validate_kwargs: Any) -> Self:
        """Load a :class:`~tidy3d.Tidy3dBaseModel` from .json file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .json file to load the :class:`~tidy3d.Tidy3dBaseModel` from.

        Returns
        -------
        Self
            An instance of the component class calling `load`.
        **model_validate_kwargs
            Keyword arguments passed to pydantic's ``model_validate`` method.

        Example
        -------
        >>> simulation = Simulation.from_json(fname='folder/sim.json') # doctest: +SKIP
        """
        model_dict = cls.dict_from_json(fname=fname)
        return cls._validate_model_dict(model_dict, **model_validate_kwargs)

    @classmethod
    def dict_from_json(cls: type[T], fname: PathLike) -> dict:
        """Load dictionary of the model from a .json file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .json file to load the :class:`~tidy3d.Tidy3dBaseModel` from.

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
        """Exports :class:`~tidy3d.Tidy3dBaseModel` instance to .json file

        Parameters
        ----------
        fname : PathLike
            Full path to the .json file to save the :class:`~tidy3d.Tidy3dBaseModel` to.

        Example
        -------
        >>> simulation.to_json(fname='folder/sim.json') # doctest: +SKIP
        """
        export_model = self.to_static()
        json_string = export_model.model_dump_json(indent=INDENT_JSON_FILE)
        self._warn_if_contains_data(json_string)
        path = Path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            file_handle.write(json_string)

    @classmethod
    def from_yaml(cls: type[T], fname: PathLike, **model_validate_kwargs: Any) -> Self:
        """Loads :class:`~tidy3d.Tidy3dBaseModel` from .yaml file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .yaml file to load the :class:`~tidy3d.Tidy3dBaseModel` from.
        **model_validate_kwargs
            Keyword arguments passed to pydantic's ``model_validate`` method.

        Returns
        -------
        Self
            An instance of the component class calling `from_yaml`.

        Example
        -------
        >>> simulation = Simulation.from_yaml(fname='folder/sim.yaml') # doctest: +SKIP
        """
        model_dict = cls.dict_from_yaml(fname=fname)
        return cls._validate_model_dict(model_dict, **model_validate_kwargs)

    @classmethod
    def dict_from_yaml(cls: type[T], fname: PathLike) -> dict:
        """Load dictionary of the model from a .yaml file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .yaml file to load the :class:`~tidy3d.Tidy3dBaseModel` from.

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
        """Exports :class:`~tidy3d.Tidy3dBaseModel` instance to .yaml file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .yaml file to save the :class:`~tidy3d.Tidy3dBaseModel` to.

        Example
        -------
        >>> simulation.to_yaml(fname='folder/sim.yaml') # doctest: +SKIP
        """
        export_model = self.to_static()
        # We intentionally round-trip through JSON to preserve the exact JSON-mode serialization
        # behavior in YAML output (notably `ser_json_inf_nan="strings"` for Infinity/-Infinity/NaN).
        json_string = export_model.model_dump_json()
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
                "As a result, it will not be possible to load the file back to the original model. "
                "Instead, use '.hdf5' extension in filename passed to 'to_file()'."
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
    def tuple_to_dict(cls: type[T], tuple_values: tuple) -> dict:
        """How we generate a dictionary mapping new keys to tuple values for hdf5."""
        return {cls.get_tuple_group_name(index=i): val for i, val in enumerate(tuple_values)}

    @classmethod
    def get_sub_model(
        cls: type[T], group_path: str, model_dict: dict[str, Any] | list[Any]
    ) -> dict:
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
    def _json_string_from_hdf5(cls: type[T], fname: PathLike | h5py.File) -> str:
        """Load the model json string from an hdf5 file path or open file handle."""
        if isinstance(fname, h5py.File):
            f_handle = fname
            num_string_parts = len([key for key in f_handle.keys() if JSON_TAG in key])
            json_string = b""
            for ind in range(num_string_parts):
                json_string += f_handle[cls._json_string_key(ind)][()]
            return json_string

        with h5py.File(fname, "r") as f_handle:
            return cls._json_string_from_hdf5(f_handle)

    @classmethod
    def _load_data_from_file(
        cls: type[T],
        fname: PathLike | h5py.File,
        model_dict: dict,
        group_path: str = "",
        custom_decoders: list[Callable] | None = None,
        should_load_path: Callable[[str], bool] | None = None,
    ) -> None:
        """Materialize DataArray payloads referenced by an HDF5-backed model dict in place."""

        def is_data_array(value: Any) -> bool:
            """Whether a value is supposed to be a data array based on the contents."""
            return isinstance(value, str) and value in DATA_ARRAY_MAP

        if not isinstance(fname, h5py.File):
            with h5py.File(Path(fname), "r") as f_handle:
                cls._load_data_from_file(
                    fname=f_handle,
                    model_dict=model_dict,
                    group_path=group_path,
                    custom_decoders=custom_decoders,
                    should_load_path=should_load_path,
                )
            return

        f_handle = fname
        fname_path = Path(f_handle.filename)

        for key, value in model_dict.items():
            subpath = f"{group_path}/{key}"

            if should_load_path is not None and not should_load_path(subpath):
                continue

            if custom_decoders:
                for custom_decoder in custom_decoders:
                    custom_decoder(
                        fname=str(fname_path),
                        group_path=subpath,
                        model_dict=model_dict,
                        key=key,
                        value=value,
                    )

            if is_data_array(value):
                data_array_type = DATA_ARRAY_MAP[value]
                model_dict[key] = data_array_type.from_hdf5(fname=f_handle, group_path=subpath)
                continue

            if isinstance(value, (list, tuple)):
                value_dict = cls.tuple_to_dict(tuple_values=value)
                cls._load_data_from_file(
                    fname=f_handle,
                    model_dict=value_dict,
                    group_path=subpath,
                    custom_decoders=custom_decoders,
                    should_load_path=should_load_path,
                )

                # handle case of nested list of DataArray elements
                val_tuple = list(value_dict.values())
                for ind, (model_item, value_item) in enumerate(zip(model_dict[key], val_tuple)):
                    if is_data_array(model_item):
                        model_dict[key][ind] = value_item

            elif isinstance(value, dict):
                cls._load_data_from_file(
                    fname=f_handle,
                    model_dict=value,
                    group_path=subpath,
                    custom_decoders=custom_decoders,
                    should_load_path=should_load_path,
                )

    @classmethod
    def dict_from_hdf5(
        cls: type[T],
        fname: PathLike | h5py.File,
        group_path: str = "",
        custom_decoders: list[Callable] | None = None,
        load_data_arrays: bool = True,
    ) -> dict:
        """Loads a dictionary containing the model contents from a .hdf5 file.

        Parameters
        ----------
        fname : PathLike or h5py.File
            Full path to the .hdf5 file, or an open HDF5 file handle, used to load the
            :class:`~tidy3d.Tidy3dBaseModel` from.
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
        if not isinstance(fname, h5py.File):
            with h5py.File(Path(fname), "r") as f_handle:
                return cls.dict_from_hdf5(
                    fname=f_handle,
                    group_path=group_path,
                    custom_decoders=custom_decoders,
                    load_data_arrays=load_data_arrays,
                )

        f_handle = fname
        model_dict = json.loads(cls._json_string_from_hdf5(fname=f_handle))
        group_path = cls._construct_group_path(group_path)
        model_dict = cls.get_sub_model(group_path=group_path, model_dict=model_dict)
        if load_data_arrays:
            cls._load_data_from_file(
                fname=f_handle,
                model_dict=model_dict,
                group_path=group_path,
                custom_decoders=custom_decoders,
            )
        return model_dict

    @classmethod
    def from_hdf5(
        cls: type[T],
        fname: PathLike | h5py.File,
        group_path: str = "",
        custom_decoders: list[Callable] | None = None,
        **model_validate_kwargs: Any,
    ) -> Self:
        """Loads :class:`~tidy3d.Tidy3dBaseModel` instance to .hdf5 file.

        Parameters
        ----------
        fname : PathLike or h5py.File
            Full path to the .hdf5 file, or an open HDF5 file handle, used to load the
            :class:`~tidy3d.Tidy3dBaseModel` from.
        group_path : str, optional
            Path to a group inside the file to selectively load a sub-element of the model only.
            Starting `/` is optional.
        custom_decoders : List[Callable]
            List of functions accepting
            (fname: str, group_path: str, model_dict: dict, key: str, value: Any) that store the
            value in the model dict after a custom decoding.
        **model_validate_kwargs
            Keyword arguments passed to pydantic's ``model_validate`` method.

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
        return cls._validate_model_dict(model_dict, **model_validate_kwargs)

    def to_hdf5(
        self,
        fname: PathLike | io.BytesIO,
        custom_encoders: list[Callable] | None = None,
    ) -> None:
        """Exports :class:`~tidy3d.Tidy3dBaseModel` instance to .hdf5 file.

        Parameters
        ----------
        fname : Union[PathLike, BytesIO]
            Full path to the .hdf5 file or buffer to save the :class:`~tidy3d.Tidy3dBaseModel` to.
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
            json_str = export_model.model_dump_json()
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

            add_data_to_file(data_dict=export_model.model_dump())
            if traced_keys_payload:
                f_handle.attrs[TRACED_FIELD_KEYS_ATTR] = traced_keys_payload

    @classmethod
    def dict_from_hdf5_gz(
        cls: type[T],
        fname: PathLike,
        group_path: str = "",
        custom_decoders: list[Callable] | None = None,
        load_data_arrays: bool = True,
    ) -> dict:
        """Loads a dictionary containing the model contents from a .hdf5.gz file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .hdf5.gz file to load the :class:`~tidy3d.Tidy3dBaseModel` from.
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
        cls: type[T],
        fname: PathLike,
        group_path: str = "",
        custom_decoders: list[Callable] | None = None,
        **model_validate_kwargs: Any,
    ) -> Self:
        """Loads :class:`~tidy3d.Tidy3dBaseModel` instance to .hdf5.gz file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .hdf5.gz file to load the :class:`~tidy3d.Tidy3dBaseModel` from.
        group_path : str, optional
            Path to a group inside the file to selectively load a sub-element of the model only.
            Starting `/` is optional.
        custom_decoders : List[Callable]
            List of functions accepting
            (fname: str, group_path: str, model_dict: dict, key: str, value: Any) that store the
            value in the model dict after a custom decoding.
        **model_validate_kwargs
            Keyword arguments passed to pydantic's ``model_validate`` method.

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
        return cls._validate_model_dict(model_dict, **model_validate_kwargs)

    def to_hdf5_gz(
        self,
        fname: PathLike | io.BytesIO,
        custom_encoders: list[Callable] | None = None,
    ) -> None:
        """Exports :class:`~tidy3d.Tidy3dBaseModel` instance to .hdf5.gz file.

        Parameters
        ----------
        fname : Union[PathLike, BytesIO]
            Full path to the .hdf5.gz file or buffer to save the :class:`~tidy3d.Tidy3dBaseModel` to.
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

    def __lt__(self, other: object) -> bool:
        """define < for getting unique indices based on hash."""
        return hash(self) < hash(other)

    def __eq__(self, other: object) -> bool:
        """Two models are equal when origins match and every public or extra field matches."""
        if not isinstance(other, BaseModel):
            return NotImplemented

        self_origin = (
            getattr(self, "__pydantic_generic_metadata__", {}).get("origin") or self.__class__
        )
        other_origin = (
            getattr(other, "__pydantic_generic_metadata__", {}).get("origin") or other.__class__
        )
        if self_origin is not other_origin:
            return False

        if getattr(self, "__pydantic_extra__", None) != getattr(other, "__pydantic_extra__", None):
            return False

        def _fields_equal(a: Any, b: Any) -> bool:
            a = get_static(a)
            b = get_static(b)

            if a is b:
                return True
            if type(a) is not type(b):
                if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
                    return False
            if isinstance(a, np.ndarray):
                return np.array_equal(a, b)
            if isinstance(a, (xr.DataArray, xr.Dataset)):
                return a.equals(b)
            if isinstance(a, Mapping):
                if a.keys() != b.keys():
                    return False
                return all(_fields_equal(a[k], b[k]) for k in a)
            if isinstance(a, Sequence) and not isinstance(a, (str, bytes)):
                if len(a) != len(b):
                    return False
                return all(_fields_equal(x, y) for i, (x, y) in enumerate(zip(a, b)))
            if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
                return True
            return a == b

        for name in type(self).model_fields:
            if not _fields_equal(getattr(self, name), getattr(other, name)):
                return False

        return True

    def _attrs_digest(self) -> str:
        """Stable digest of `attrs` using the same JSON encoding rules as `model_dump_json()`."""
        # encoders = getattr(self.__config__, "json_encoders", {}) or {}

        # def _default(o):
        #     return custom_pydantic_encoder(encoders, o)

        json_str = json.dumps(
            self.attrs,
            # default=_default,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        json_str = make_json_compatible(json_str)

        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    @cached_property_guarded(lambda self: self._attrs_digest())
    def _json_string(self) -> str:
        """Returns string representation of a :class:`~tidy3d.Tidy3dBaseModel`.

        Returns
        -------
        str
            Json-formatted string holding :class:`~tidy3d.Tidy3dBaseModel` data.
        """
        return self.model_dump_json(indent=INDENT, exclude_unset=False)

    def _strip_traced_fields(
        self,
        starting_paths: tuple[tuple[str, ...], ...] = (),
        include_untraced_data_arrays: bool = False,
    ) -> AutogradFieldMap:
        """Extract a dictionary mapping paths in the model to the data traced by ``autograd``.

        Parameters
        ----------
        starting_paths : tuple[tuple[str, ...], ...] = ()
            If provided, starts recursing in self.model_dump() from these paths of field names.
            Can be a single path tuple or multiple path tuples.
        include_untraced_data_arrays : bool = False
            Whether to include ``DataArray`` objects without tracers.
            We need to include these when returning data, but are unnecessary for structures.

        Returns
        -------
        dict
            mapping of traced fields used by ``autograd``

        """

        paths = tuple(starting_paths)
        if self._has_tracers is False and not include_untraced_data_arrays:
            return TracedDict()

        field_mapping = {}

        def handle_value(x: Any, path: tuple[str, ...]) -> None:
            """recursively update ``field_mapping`` with path to the autograd data."""

            # this is a leaf node that we want to trace, add this path and data to the mapping
            if isbox(x):
                field_mapping[path] = x

            # for data arrays, need to be more careful as their tracers are stored in .data
            elif isinstance(x, xr.DataArray):
                data = x.data
                if isbox(data) or any(isbox(el) for el in np.asarray(data).ravel()):
                    field_mapping[path] = x.data
                elif include_untraced_data_arrays:
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
        self_dict = self.model_dump(round_trip=True)

        # Handle multiple starting paths
        if paths:
            # If paths is a single tuple, convert to tuple of tuples
            if isinstance(paths[0], str):
                paths = (paths,)

            # Process each starting path
            for starting_path in paths:
                # Navigate to the starting path in the dictionary
                current_dict = self_dict
                for key in starting_path:
                    current_dict = current_dict[key]

                # Handle the subtree starting from this path
                handle_value(current_dict, path=starting_path)
        else:
            # No starting paths specified, process entire dictionary
            handle_value(self_dict, path=())

        if field_mapping:
            if not include_untraced_data_arrays:
                self._has_tracers = True
            return TracedDict(field_mapping)

        if not include_untraced_data_arrays and not paths:
            self._has_tracers = False
        return TracedDict()

    def _insert_traced_fields(self, field_mapping: AutogradFieldMap) -> Self:
        """Recursively insert a map of paths to autograd-traced fields into a copy of this obj."""
        self_dict = self.model_dump(round_trip=True)

        def insert_value(x: Any, path: tuple[str, ...], sub_dict: dict[str, Any]) -> None:
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

        return self.__class__.model_validate(self_dict)

    def _serialized_traced_field_keys(
        self, field_mapping: AutogradFieldMap | None = None
    ) -> str | None:
        """Return a serialized, order-independent representation of traced field paths."""

        if field_mapping is None:
            field_mapping = self._strip_traced_fields()
        if not field_mapping:
            return None

        # TODO: remove this deferred import once TracerKeys is decoupled from Tidy3dBaseModel.
        from tidy3d.components.autograd.field_map import TracerKeys

        tracer_keys = TracerKeys.from_field_mapping(field_mapping)
        return tracer_keys.model_dump_json()

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
    def generate_docstring(
        cls,
        show_default_args: bool | None = None,
        include_attrs: bool | None = None,
    ) -> str:
        """Generates a docstring for a Tidy3D model."""
        if show_default_args is None:
            show_default_args = cls._DOCSTRING_SHOW_DEFAULT_ARGS
        if include_attrs is None:
            include_attrs = cls._DOCSTRING_INCLUDE_ATTRS

        doc = ""

        # keep any pre-existing class description
        original_docstrings = []
        raw_doc = cls.__dict__.get(_DOCSTRING_RAW_ATTR)
        if raw_doc is None:
            raw_doc = cls.__doc__ or ""
        if raw_doc:
            original_docstrings = raw_doc.split("\n\n")
            doc += original_docstrings.pop(0)
        original_docstrings = "\n\n".join(original_docstrings)

        # parameters
        doc += "\n\n    Parameters\n    ----------\n"
        for field_name, field in cls.model_fields.items():  # v2
            if field_name == TYPE_TAG_STR:
                continue
            if field_name == "attrs" and not include_attrs:
                continue

            # type
            ann = getattr(field, "annotation", None)
            field_metadata = getattr(field, "metadata", None)
            data_type = _fmt_ann_literal(ann, field_metadata=field_metadata)

            # default / default_factory
            if field.default_factory is not None:
                try:
                    default_val = field.default_factory()
                except Exception:
                    default_val = f"{field.default_factory.__name__}()"
            else:
                default_val = field.get_default(call_default_factory=False)

            if isinstance(default_val, BaseModel):
                default_val = _format_model_default(
                    default_val, show_default_args=show_default_args
                )
            elif "=" in str(default_val) if default_val is not None else False:
                default_val = _clean_default_repr(
                    str(f"{default_val.__class__.__name__}({default_val})")
                )

            default_str = "" if field.is_required() else f" = {default_val}"
            doc += f"    {field_name} : {data_type}{default_str}\n"

            parts = []

            # units
            units = None
            extra = getattr(field, "json_schema_extra", None)
            if isinstance(extra, dict):
                units = extra.get("units")
            if units is None and hasattr(field, "metadata"):
                for meta in field.metadata:
                    if isinstance(meta, dict) and "units" in meta:
                        units = meta["units"]
                        break
            if units is not None:
                unitstr = (
                    f"({', '.join(str(u) for u in units)})"
                    if isinstance(units, (list, tuple))
                    else str(units)
                )
                parts.append(f"[units = {unitstr}].")

            # description
            desc = getattr(field, "description", None)
            if desc:
                parts.append(desc)

            if parts:
                doc += "        " + "  ".join(parts) + "\n"

        if original_docstrings:
            doc += "\n" + original_docstrings
        doc += "\n"

        return doc

    def get_submodels_by_hash(self) -> dict[int, list[str | tuple[str, int]]]:
        """
        Return a mapping ``{hash(submodel): [field_path, ...]}`` for every
        nested ``Tidy3dBaseModel`` inside this model.
        """
        out = defaultdict(list)

        for name in type(self).model_fields:
            value = getattr(self, name)

            if isinstance(value, Tidy3dBaseModel):
                out[hash(value)].append(name)
                continue

            if isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    if isinstance(item, Tidy3dBaseModel):
                        out[hash(item)].append((name, idx))

            elif isinstance(value, np.ndarray):
                for idx, item in enumerate(value.flat):
                    if isinstance(item, Tidy3dBaseModel):
                        out[hash(item)].append((name, idx))

            elif isinstance(value, dict):
                for k, item in value.items():
                    if isinstance(item, Tidy3dBaseModel):
                        out[hash(item)].append((name, k))

        return dict(out)

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

    def __rich_repr__(self) -> rich.repr.Result:
        """How to pretty-print instances of ``Tidy3dBaseModel``."""
        for name in type(self).model_fields:
            value = getattr(self, name)

            # don't print the type field we add to the models
            if name == "type":
                continue

            # skip `attrs` if it's an empty dictionary
            if name == "attrs" and isinstance(value, dict) and not value:
                continue

            yield name, value

    def __repr__(self) -> str:
        """Return a concise string representation of the model."""
        try:
            return _format_model_default(self, show_default_args=False)
        except Exception:
            return super().__repr__()

    def __str__(self) -> str:
        """Return a pretty-printed string representation of the model."""
        from io import StringIO

        from rich.console import Console

        sio = StringIO()
        console = Console(file=sio, force_jupyter=False)
        console.print(self)
        output = sio.getvalue()
        return output.rstrip("\n")


def _make_lazy_proxy(
    target_cls: type[Tidy3dBaseModel],
    on_load: Callable[[Any], None] | None = None,
) -> type[Tidy3dBaseModel]:
    """
    Return a lazy-loading proxy subclass of ``target_cls``.

    Parameters
    ----------
    target_cls : type
        Must implement ``dict_from_file`` and ``model_validate``.
    on_load : Optional[Callable[[Any], None]] = None
        A function to call with the fully loaded instance once loaded.

    Returns
    -------
    type
        A class named ``<TargetClsName>Proxy`` with init args:
        ``(fname, group_path, parse_obj_kwargs)``.
    """

    proxy_name = f"{target_cls.__name__}Proxy"

    def materialize_proxy(proxy: Tidy3dBaseModel, target: Tidy3dBaseModel) -> None:
        """Replace a lazy proxy instance with a fully loaded target instance."""

        d = object.__getattribute__(proxy, "__dict__")
        d.clear()
        d.update(target.__dict__)

        object.__setattr__(proxy, "__class__", target.__class__)
        fields_set = getattr(target, "__pydantic_fields_set__", None)
        if fields_set is not None:
            object.__setattr__(proxy, "__pydantic_fields_set__", set(fields_set))

        pvt = getattr(target, "__pydantic_private__", None)
        if pvt is not None:
            object.__setattr__(proxy, "__pydantic_private__", pvt)

        object.__setattr__(proxy, "__pydantic_extra__", getattr(target, "__pydantic_extra__", None))

        if on_load is not None:
            on_load(proxy)

    class _LazyProxy(target_cls):  # type: ignore[misc]
        def __init__(
            self,
            fname: PathLike,
            group_path: str | None,
            parse_obj_kwargs: Any,
            **lazy_state: Any,
        ) -> None:
            # store lazy context only in __dict__
            object.__setattr__(self, "_lazy_fname", Path(fname))
            object.__setattr__(self, "_lazy_group_path", group_path)
            object.__setattr__(self, "_lazy_parse_obj_kwargs", dict(parse_obj_kwargs or {}))
            for key, value in lazy_state.items():
                object.__setattr__(self, f"_lazy_{key}", value)

        def copy(self, **kwargs: Any) -> Self:
            """Return another lazy proxy instead of materializing."""
            d = object.__getattribute__(self, "__dict__")
            copy_state = {
                key: d[f"_lazy_{key}"]
                for key in target_cls._lazy_proxy_copy_state_keys()
                if f"_lazy_{key}" in d
            }
            return _LazyProxy(
                object.__getattribute__(self, "_lazy_fname"),
                object.__getattribute__(self, "_lazy_group_path"),
                {
                    **object.__getattribute__(self, "_lazy_parse_obj_kwargs"),
                    **kwargs,
                },
                **copy_state,
            )

        def __getattribute__(self, name: str) -> Any:
            # Attributes that must *not* trigger materialization
            if name.startswith("_lazy_") or name in {
                "__class__",
                "__dict__",
                "__weakref__",
                "__post_root_validators__",
                "__pydantic_decorators__",
                "copy",  # don't materialize just for .copy()
            }:
                return object.__getattribute__(self, name)

            d = object.__getattribute__(self, "__dict__")

            if "_lazy_fname" in d:
                resolved = target_cls._lazy_proxy_resolve_attr(self, name, d)
                if resolved is not _LAZY_PROXY_UNHANDLED:
                    return resolved
                target = target_cls._lazy_proxy_materialize(d)
                materialize_proxy(self, target)

            return object.__getattribute__(self, name)

    _LazyProxy.__name__ = proxy_name
    return _LazyProxy
