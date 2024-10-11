"""global configuration / base class for pydantic models used to make simulation."""

from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import tempfile
from functools import wraps
from math import ceil
from typing import Any, Callable, Dict, List, Tuple, Union

import h5py
import numpy as np
import pydantic.v1 as pydantic
import rich
import xarray as xr
import yaml
from autograd.builtins import dict as dict_ag
from autograd.tracer import isbox
from pydantic.v1.fields import ModelField

from ..exceptions import FileError
from ..log import log
from .autograd.types import AutogradFieldMap, Box
from .autograd.utils import get_static
from .data.data_array import DATA_ARRAY_MAP, DataArray
from .file_util import compress_file_to_gzip, extract_gzip_file
from .types import TYPE_TAG_STR, ComplexNumber, Literal

INDENT_JSON_FILE = 4  # default indentation of json string in json files
INDENT = None  # default indentation of json string used internally
JSON_TAG = "JSON_STRING"
# If json string is larger than ``MAX_STRING_LENGTH``, split the string when storing in hdf5
MAX_STRING_LENGTH = 1_000_000_000
FORBID_SPECIAL_CHARACTERS = ["/"]


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


def ndarray_encoder(val):
    """How a ``np.ndarray`` gets handled before saving to json."""
    if np.any(np.iscomplex(val)):
        return dict(real=val.real.tolist(), imag=val.imag.tolist())
    return val.real.tolist()


def _get_valid_extension(fname: str) -> str:
    """Return the file extension from fname, validated to accepted ones."""
    valid_extensions = [".json", ".yaml", ".hdf5", ".h5", ".hdf5.gz"]
    extensions = [s.lower() for s in pathlib.Path(fname).suffixes[-2:]]
    if len(extensions) == 0:
        raise FileError(f"File '{fname}' missing extension.")
    single_extension = extensions[-1]
    if single_extension in valid_extensions:
        return single_extension
    double_extension = "".join(extensions)
    if double_extension in valid_extensions:
        return double_extension
    raise FileError(
        f"File extension must be one of {', '.join(valid_extensions)}; file '{fname}' does not "
        "match any of those."
    )


def skip_if_fields_missing(fields: List[str], root=False):
    """Decorate ``validator`` to check that other fields have passed validation."""

    def actual_decorator(validator):
        @wraps(validator)
        def _validator(cls, *args, **kwargs):
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
                    else:
                        return kwargs.get("val") if "val" in kwargs.keys() else args[0]

            return validator(cls, *args, **kwargs)

        return _validator

    return actual_decorator


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
        return hashlib.sha256(bf.getvalue()).hexdigest()

    def __init__(self, **kwargs):
        """Init method, includes post-init validators."""
        log.begin_capture()
        super().__init__(**kwargs)
        self._post_init_validators()
        log.end_capture(self)

    def _post_init_validators(self) -> None:
        """Call validators taking ``self`` that get run after init, implement in subclasses."""

    def __init_subclass__(cls) -> None:
        """Things that are done to each of the models."""

        cls.add_type_field()
        cls.generate_docstring()

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
        "Also note that `Tidy3D`` will raise a ``TypeError`` if ``attrs`` contain objects "
        "that can not be serialized. One can check if ``attrs`` are serializable "
        "by calling ``obj.json()``.",
    )

    def copy(self, deep: bool = True, **kwargs) -> Tidy3dBaseModel:
        """Copy a Tidy3dBaseModel.  With ``deep=True`` as default."""
        kwargs.update(deep=deep)
        new_copy = pydantic.BaseModel.copy(self, **kwargs)
        return self.validate(new_copy.dict())

    def updated_copy(self, path: str = None, deep: bool = True, **kwargs) -> Tidy3dBaseModel:
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
            return self._updated_copy(**kwargs, deep=deep)

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
                )

            sub_component_list = list(sub_component)
            sub_component = sub_component_list[index]
            sub_path = "/".join(path_components[2:])

            sub_component_list[index] = sub_component.updated_copy(
                path=sub_path, deep=deep, **kwargs
            )
            new_component = tuple(sub_component_list)
        else:
            sub_path = "/".join(path_components[1:])
            new_component = sub_component.updated_copy(path=sub_path, deep=deep, **kwargs)

        return self._updated_copy(deep=deep, **{field_name: new_component})

    def _updated_copy(self, deep: bool = True, **kwargs) -> Tidy3dBaseModel:
        """Make copy of a component instance with ``**kwargs`` indicating updated field values."""
        return self.copy(update=kwargs, deep=deep)

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
    def from_file(cls, fname: str, group_path: str = None, **parse_obj_kwargs) -> Tidy3dBaseModel:
        """Loads a :class:`Tidy3dBaseModel` from .yaml, .json, .hdf5, or .hdf5.gz file.

        Parameters
        ----------
        fname : str
            Full path to the file to load the :class:`Tidy3dBaseModel` from.
        group_path : str, optional
            Path to a group inside the file to use as the base level. Only for hdf5 files.
            Starting `/` is optional.
        **parse_obj_kwargs
            Keyword arguments passed to either pydantic's ``parse_obj`` function when loading model.

        Returns
        -------
        :class:`Tidy3dBaseModel`
            An instance of the component class calling ``load``.

        Example
        -------
        >>> simulation = Simulation.from_file(fname='folder/sim.json') # doctest: +SKIP
        """
        model_dict = cls.dict_from_file(fname=fname, group_path=group_path)
        return cls.parse_obj(model_dict, **parse_obj_kwargs)

    @classmethod
    def dict_from_file(cls, fname: str, group_path: str = None) -> dict:
        """Loads a dictionary containing the model from a .yaml, .json, .hdf5, or .hdf5.gz file.

        Parameters
        ----------
        fname : str
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

        extension = _get_valid_extension(fname)
        kwargs = {"fname": fname}

        if group_path is not None:
            if extension == ".hdf5" or extension == ".hdf5.gz":
                kwargs["group_path"] = group_path
            else:
                log.warning("'group_path' provided, but this feature only works with hdf5 files.")

        converter = {
            ".json": cls.dict_from_json,
            ".yaml": cls.dict_from_yaml,
            ".hdf5": cls.dict_from_hdf5,
            ".hdf5.gz": cls.dict_from_hdf5_gz,
            ".h5": cls.dict_from_hdf5,
        }[extension]
        return converter(**kwargs)

    def to_file(self, fname: str) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .yaml, .json, or .hdf5 file

        Parameters
        ----------
        fname : str
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
    def from_json(cls, fname: str, **parse_obj_kwargs) -> Tidy3dBaseModel:
        """Load a :class:`Tidy3dBaseModel` from .json file.

        Parameters
        ----------
        fname : str
            Full path to the .json file to load the :class:`Tidy3dBaseModel` from.

        Returns
        -------
        :class:`Tidy3dBaseModel`
            An instance of the component class calling `load`.
        **parse_obj_kwargs
            Keyword arguments passed to pydantic's ``parse_obj`` method.

        Example
        -------
        >>> simulation = Simulation.from_json(fname='folder/sim.json') # doctest: +SKIP
        """
        model_dict = cls.dict_from_json(fname=fname)
        return cls.parse_obj(model_dict, **parse_obj_kwargs)

    @classmethod
    def dict_from_json(cls, fname: str) -> dict:
        """Load dictionary of the model from a .json file.

        Parameters
        ----------
        fname : str
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

    def to_json(self, fname: str) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .json file

        Parameters
        ----------
        fname : str
            Full path to the .json file to save the :class:`Tidy3dBaseModel` to.

        Example
        -------
        >>> simulation.to_json(fname='folder/sim.json') # doctest: +SKIP
        """
        json_string = self._json(indent=INDENT_JSON_FILE)
        self._warn_if_contains_data(json_string)
        with open(fname, "w", encoding="utf-8") as file_handle:
            file_handle.write(json_string)

    @classmethod
    def from_yaml(cls, fname: str, **parse_obj_kwargs) -> Tidy3dBaseModel:
        """Loads :class:`Tidy3dBaseModel` from .yaml file.

        Parameters
        ----------
        fname : str
            Full path to the .yaml file to load the :class:`Tidy3dBaseModel` from.
        **parse_obj_kwargs
            Keyword arguments passed to pydantic's ``parse_obj`` method.

        Returns
        -------
        :class:`Tidy3dBaseModel`
            An instance of the component class calling `from_yaml`.

        Example
        -------
        >>> simulation = Simulation.from_yaml(fname='folder/sim.yaml') # doctest: +SKIP
        """
        model_dict = cls.dict_from_yaml(fname=fname)
        return cls.parse_obj(model_dict, **parse_obj_kwargs)

    @classmethod
    def dict_from_yaml(cls, fname: str) -> dict:
        """Load dictionary of the model from a .yaml file.

        Parameters
        ----------
        fname : str
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

    def to_yaml(self, fname: str) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .yaml file.

        Parameters
        ----------
        fname : str
            Full path to the .yaml file to save the :class:`Tidy3dBaseModel` to.

        Example
        -------
        >>> simulation.to_yaml(fname='folder/sim.yaml') # doctest: +SKIP
        """
        json_string = self._json_string
        self._warn_if_contains_data(json_string)
        model_dict = json.loads(json_string)
        with open(fname, "w+", encoding="utf-8") as file_handle:
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
    def _json_string_from_hdf5(cls, fname: str) -> str:
        """Load the model json string from an hdf5 file."""
        with h5py.File(fname, "r") as f_handle:
            num_string_parts = len([key for key in f_handle.keys() if JSON_TAG in key])
            json_string = b""
            for ind in range(num_string_parts):
                json_string += f_handle[cls._json_string_key(ind)][()]
        return json_string

    @classmethod
    def dict_from_hdf5(
        cls, fname: str, group_path: str = "", custom_decoders: List[Callable] = None
    ) -> dict:
        """Loads a dictionary containing the model contents from a .hdf5 file.

        Parameters
        ----------
        fname : str
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

        def load_data_from_file(model_dict: dict, group_path: str = "") -> None:
            """For every DataArray item in dictionary, load path of hdf5 group as value."""

            for key, value in model_dict.items():
                subpath = f"{group_path}/{key}"

                # apply custom validation to the key value pair and modify model_dict
                if custom_decoders:
                    for custom_decoder in custom_decoders:
                        custom_decoder(
                            fname=fname,
                            group_path=subpath,
                            model_dict=model_dict,
                            key=key,
                            value=value,
                        )

                # write the path to the element of the json dict where the data_array should be
                if is_data_array(value):
                    data_array_type = DATA_ARRAY_MAP[value]
                    model_dict[key] = data_array_type.from_hdf5(fname=fname, group_path=subpath)
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

        model_dict = json.loads(cls._json_string_from_hdf5(fname=fname))
        group_path = cls._construct_group_path(group_path)
        model_dict = cls.get_sub_model(group_path=group_path, model_dict=model_dict)
        load_data_from_file(model_dict=model_dict, group_path=group_path)
        return model_dict

    @classmethod
    def from_hdf5(
        cls,
        fname: str,
        group_path: str = "",
        custom_decoders: List[Callable] = None,
        **parse_obj_kwargs,
    ) -> Tidy3dBaseModel:
        """Loads :class:`Tidy3dBaseModel` instance to .hdf5 file.

        Parameters
        ----------
        fname : str
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
            fname=fname, group_path=group_path, custom_decoders=custom_decoders
        )
        return cls.parse_obj(model_dict, **parse_obj_kwargs)

    def to_hdf5(self, fname: str, custom_encoders: List[Callable] = None) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .hdf5 file.

        Parameters
        ----------
        fname : str
            Full path to the .hdf5 file to save the :class:`Tidy3dBaseModel` to.
        custom_encoders : List[Callable]
            List of functions accepting (fname: str, group_path: str, value: Any) that take
            the ``value`` supplied and write it to the hdf5 ``fname`` at ``group_path``.

        Example
        -------
        >>> simulation.to_hdf5(fname='folder/sim.hdf5') # doctest: +SKIP
        """

        with h5py.File(fname, "w") as f_handle:
            json_str = self._json_string
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
                        value_dict = self.tuple_to_dict(tuple_values=value)
                        add_data_to_file(data_dict=value_dict, group_path=subpath)

                    # if a dict, recurse
                    elif isinstance(value, dict):
                        add_data_to_file(data_dict=value, group_path=subpath)

            add_data_to_file(data_dict=self.dict())

    @classmethod
    def dict_from_hdf5_gz(
        cls, fname: str, group_path: str = "", custom_decoders: List[Callable] = None
    ) -> dict:
        """Loads a dictionary containing the model contents from a .hdf5.gz file.

        Parameters
        ----------
        fname : str
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
        file, extracted = tempfile.mkstemp(".hdf5")
        os.close(file)
        try:
            extract_gzip_file(fname, extracted)
            result = cls.dict_from_hdf5(
                extracted, group_path=group_path, custom_decoders=custom_decoders
            )
        finally:
            os.unlink(extracted)

        return result

    @classmethod
    def from_hdf5_gz(
        cls,
        fname: str,
        group_path: str = "",
        custom_decoders: List[Callable] = None,
        **parse_obj_kwargs,
    ) -> Tidy3dBaseModel:
        """Loads :class:`Tidy3dBaseModel` instance to .hdf5.gz file.

        Parameters
        ----------
        fname : str
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
            fname=fname, group_path=group_path, custom_decoders=custom_decoders
        )
        return cls.parse_obj(model_dict, **parse_obj_kwargs)

    def to_hdf5_gz(self, fname: str, custom_encoders: List[Callable] = None) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .hdf5.gz file.

        Parameters
        ----------
        fname : str
            Full path to the .hdf5.gz file to save the :class:`Tidy3dBaseModel` to.
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
            for key in dict1.keys():
                val1 = dict1[key]
                val2 = dict2[key]

                val1 = get_static(val1)
                val2 = get_static(val2)

                # if one of val1 or val2 is None (exclusive OR)
                if (val1 is None) != (val2 is None):
                    return False

                # convert tuple to dict to use this recursive function
                if isinstance(val1, tuple) or isinstance(val2, tuple):
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

    @cached_property
    def _json_string(self) -> str:
        """Returns string representation of a :class:`Tidy3dBaseModel`.

        Returns
        -------
        str
            Json-formatted string holding :class:`Tidy3dBaseModel` data.
        """
        return self._json()

    def _json(self, indent=INDENT, exclude_unset=False, **kwargs) -> str:
        """Overwrites the model ``json`` representation with some extra customized handling.

        Parameters
        -----------
        **kwargs : kwargs passed to `self.json()`

        Returns
        -------
        str
            Json-formatted string holding :class:`Tidy3dBaseModel` data.
        """

        def make_json_compatible(json_string: str) -> str:
            """Makes the string compatible with json standards, notably for infinity."""

            tmp_string = "<<TEMPORARY_INFINITY_STRING>>"
            json_string = json_string.replace("-Infinity", tmp_string)
            json_string = json_string.replace("Infinity", '"Infinity"')
            return json_string.replace(tmp_string, '"-Infinity"')

        json_string = self.json(indent=indent, exclude_unset=exclude_unset, **kwargs)
        json_string = make_json_compatible(json_string)
        return json_string

    def strip_traced_fields(
        self, starting_path: tuple[str] = (), include_untraced_data_arrays: bool = False
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
                    handle_value(val, path=path + (i,))

            # for dictionaries, add the (key,) to the path and handle each value individually
            elif isinstance(x, dict):
                for key, val in x.items():
                    handle_value(val, path=path + (key,))

        # recursively parse the dictionary of this object
        self_dict = self.dict()

        # if an include_only string was provided, only look at that subset of the dict
        if starting_path:
            for key in starting_path:
                self_dict = self_dict[key]

        handle_value(self_dict, path=starting_path)

        # convert the resulting field_mapping to an autograd-traced dictionary
        return dict_ag(field_mapping)

    def insert_traced_fields(self, field_mapping: AutogradFieldMap) -> Tidy3dBaseModel:
        """Recursively insert a map of paths to autograd-traced fields into a copy of this obj."""

        self_dict = self.dict()

        def insert_value(x, path: tuple[str, ...], sub_dict: dict):
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

        return self.parse_obj(self_dict)

    def to_static(self) -> Tidy3dBaseModel:
        """Version of object with all autograd-traced fields removed."""

        # get dictionary of all traced fields
        field_mapping = self.strip_traced_fields()

        # shortcut to just return self if no tracers found, for performance
        if not field_mapping:
            return self

        # convert all fields to static values
        field_mapping_static = {key: get_static(val) for key, val in field_mapping.items()}

        # insert the static values into a copy of self
        return self.insert_traced_fields(field_mapping_static)

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

    def get_submodels_by_hash(self) -> Dict[int, List[Union[str, Tuple[str, int]]]]:
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
