"""global configuration / base class for pydantic models used to make simulation."""
from __future__ import annotations

import json
from typing import Any, Union, Optional
from functools import wraps
from typing_extensions import _AnnotatedAlias

import rich
import pydantic
from pydantic.fields import ModelField
import yaml
import numpy as np
import h5py
import xarray as xr
from dask.base import tokenize

from .types import ComplexNumber, Literal, TYPE_TAG_STR  # , DataObject
from ..log import FileError, log, Tidy3dKeyError

# default indentation (# spaces) in files
INDENT = 4


class Tidy3dBaseModel(pydantic.BaseModel):
    """Base pydantic model that all Tidy3d components inherit from.
    Defines configuration for handling data structures
    as well as methods for imporing, exporting, and hashing tidy3d objects.
    For more details on pydantic base models, see:
    `Pydantic Models <https://pydantic-docs.helpmanual.io/usage/models/>`_
    """

    def __init_subclass__(cls) -> None:
        """Things that are done to each of the models."""

        cls.add_type_field()
        cls.generate_docstring()

    class Config:  # pylint: disable=too-few-public-methods
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
            np.ndarray: lambda x: tuple(x.tolist()),
            complex: lambda x: ComplexNumber(real=x.real, imag=x.imag),
            xr.DataArray: lambda x: x.to_dict(),  # pylint:disable=unhashable-member
        }
        frozen = True
        allow_mutation = False

    _cached_properties = pydantic.PrivateAttr({})

    def copy(self, validate: bool = True, **kwargs) -> Tidy3dBaseModel:
        """Copy a Tidy3dBaseModel.  With ``deep=True`` as default."""
        if "deep" in kwargs and kwargs["deep"] is False:
            raise ValueError("Can't do shallow copy of component, set `deep=True` in copy().")
        kwargs.update(dict(deep=True))
        new_copy = pydantic.BaseModel.copy(self, **kwargs)
        return self.validate(new_copy.dict()) if validate else new_copy

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
    def load_without_validation(cls, **values) -> Tidy3dBaseModel:
        """Creates a new model setting from trusted or pre-validated data.

        Parameters
        ----------
        **values
            The fields and values of the model as keyword arguments.
        """

        # dict of the pd.Field() values for this class, used to get type info later
        fields = cls.__fields__

        def is_dict_with_type(value: Any) -> bool:
            """Is this value a dictionary with a type field? Probably Tidy3dBaseModel."""
            return isinstance(value, dict) and "type" in value

        def get_cls_type(value: Any, outer_type: type, model_type: type) -> type:
            """Gets the class / type needed to load the data in `value`."""

            # if the value could be a tidy3d object, we infer its type from the type annotation.
            if is_dict_with_type(value):

                # if an annotated type, get the union from the first `arg`
                if isinstance(model_type, _AnnotatedAlias):
                    outer_type = model_type.__args__[0]

                # of a union of types, try to grab the correct type using `type` field
                origin_dict = outer_type.__dict__.get("__origin__")
                if origin_dict == Union:

                    # make a dictionary containing the type_name : tidy3d_type for each union member
                    union_types = {}
                    for union_type in outer_type.__dict__["__args__"]:
                        if isinstance(union_type, type) and issubclass(union_type, Tidy3dBaseModel):
                            type_name = union_type.__fields__["type"].type_.__dict__["__args__"][0]
                            union_types[type_name] = union_type

                    # try to get the value type from the dict of unions, error if not there
                    value_type = value["type"]
                    if value_type not in union_types:
                        raise Tidy3dKeyError(
                            f"trying to construct {value} with type {value_type},"
                            f"but this type is not present in the field type union {union_types}"
                        )
                    return union_types[value["type"]]

            # at this point, the type annotation didnt have a union of types, so we just return
            return model_type

        def convert_list_to_tuple(val_list: list) -> tuple:
            """Recursively convert a list to tuple."""
            if not isinstance(val_list, list):
                return val_list
            return tuple(convert_list_to_tuple(v) for v in val_list)

        def parse_raw_value(value: Any) -> Any:
            """Special rules for parsing a single value."""
            if isinstance(value, list):
                value = convert_list_to_tuple(value)
            if isinstance(value, tuple):
                value = tuple(parse_raw_value(v) for v in value)
            if value == "Infinity":
                value = np.inf
            if value == "-Infinity":
                value = -np.inf
            if isinstance(value, dict) and "real" in value and "imag" in value:
                value = value.get("real") + 1j * value.get("imag")
            return value

        def construct_value(name: str, value: Any) -> Any:
            """Load a single value without validation."""

            # infer the type information from the pydantic model
            model_field = fields[name]
            model_type = model_field.type_
            outer_type = model_field.outer_type_

            # if it's a union type, and there is a 'type' in the value dictionary
            if is_dict_with_type(value):

                # grab the type and load the data without validation using that type
                model_type = get_cls_type(value, outer_type, model_type)
                return model_type.load_without_validation(**value)

            # if the type of the field is a tuple and the supplied value is not None
            if (
                value is not None
                and hasattr(outer_type, "_name")
                and outer_type._name == "Tuple"  # pylint:disable=protected-access
            ):

                # construct new tuple by loading
                new_vals = []
                for val in value:
                    if is_dict_with_type(val):
                        model_sub_type = get_cls_type(val, outer_type, model_type)
                        new_vals.append(model_sub_type.load_without_validation(**val))
                    else:
                        val = parse_raw_value(val)
                        new_vals.append(val)
                return tuple(new_vals)

            # regular type, cast to the correct value
            if isinstance(model_type, type) and value is not None:
                value = parse_raw_value(value)
                return model_type(value)

            return parse_raw_value(value)

        # loop through supplied values as a dictionary
        constructed_values = {k: construct_value(name=k, value=v) for k, v in values.items()}

        # use the constructed values to call pydantic's construct function
        return cls.construct(**constructed_values)

    @classmethod
    def from_file(cls, fname: str, **parse_kwargs) -> Tidy3dBaseModel:
        """Loads a :class:`Tidy3dBaseModel` from .yaml, .json, or .hdf5 file.

        Parameters
        ----------
        fname : str
            Full path to the .yaml or .json file to load the :class:`Tidy3dBaseModel` from.
        **parse_kwargs
            Keyword arguments passed to either pydantic's ``parse_file`` or ``parse_raw`` methods
            for ``.json`` and ``.yaml`` file formats, respectively.
        Returns
        -------
        :class:`Tidy3dBaseModel`
            An instance of the component class calling `load`.

        Example
        -------
        >>> simulation = Simulation.from_file(fname='folder/sim.json') # doctest: +SKIP
        """
        if ".json" in fname:
            return cls.from_json(fname=fname, **parse_kwargs)
        if ".yaml" in fname:
            return cls.from_yaml(fname=fname, **parse_kwargs)
        if ".hdf5" in fname:
            return cls.from_hdf5(fname=fname, **parse_kwargs)

        raise FileError(f"File must be .json, .yaml, or .hdf5 type, given {fname}")

    def to_file(self, fname: str, data_file: Optional[str] = None) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .yaml, .json, or .hdf5 file

        Parameters
        ----------
        fname : str
            Full path to the .yaml or .json file to save the :class:`Tidy3dBaseModel` to.
        data_file : str = None
            Path to a separate hdf5 file to write :class:`.DataArray` objects to.

        Example
        -------
        >>> simulation.to_file(fname='folder/sim.json') # doctest: +SKIP
        """

        if ".json" in fname:
            return self.to_json(fname=fname, data_file=data_file)
        if ".yaml" in fname:
            return self.to_yaml(fname=fname, data_file=data_file)
        if ".hdf5" in fname:
            if data_file:
                log.warning("`data_file` has no effect when already writing to `hdf5` file.")
            return self.to_hdf5(fname=fname)

        raise FileError(f"File must be .json, .yaml, or .hdf5 type, given {fname}")

    @classmethod
    def from_json(cls, fname: str, **parse_file_kwargs) -> Tidy3dBaseModel:
        """Load a :class:`Tidy3dBaseModel` from .json file.

        Parameters
        ----------
        fname : str
            Full path to the .json file to load the :class:`Tidy3dBaseModel` from.

        Returns
        -------
        :class:`Tidy3dBaseModel`
            An instance of the component class calling `load`.
        **parse_file_kwargs
            Keyword arguments passed to pydantic's ``parse_file`` method.

        Example
        -------
        >>> simulation = Simulation.from_json(fname='folder/sim.json') # doctest: +SKIP
        """
        return cls.parse_file(fname, **parse_file_kwargs)

    def to_json(self, fname: str, data_file: Optional[str] = None) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .json file

        Parameters
        ----------
        fname : str
            Full path to the .json file to save the :class:`Tidy3dBaseModel` to.
        data_file : str = None
            Path to a separate hdf5 file to write :class:`.DataArray` objects to.

        Example
        -------
        >>> simulation.to_json(fname='folder/sim.json') # doctest: +SKIP
        """
        json_string = self._json_string(data_file=data_file)
        with open(fname, "w", encoding="utf-8") as file_handle:
            file_handle.write(json_string)

    @classmethod
    def from_yaml(cls, fname: str, **parse_raw_kwargs) -> Tidy3dBaseModel:
        """Loads :class:`Tidy3dBaseModel` from .yaml file.

        Parameters
        ----------
        fname : str
            Full path to the .yaml file to load the :class:`Tidy3dBaseModel` from.
        **parse_raw_kwargs
            Keyword arguments passed to pydantic's ``parse_raw`` method.

        Returns
        -------
        :class:`Tidy3dBaseModel`
            An instance of the component class calling `from_yaml`.

        Example
        -------
        >>> simulation = Simulation.from_yaml(fname='folder/sim.yaml') # doctest: +SKIP
        """
        with open(fname, "r", encoding="utf-8") as yaml_in:
            json_dict = yaml.safe_load(yaml_in)
        json_raw = json.dumps(json_dict, indent=INDENT)
        return cls.parse_raw(json_raw, **parse_raw_kwargs)

    def to_yaml(self, fname: str, data_file: Optional[str] = None) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .yaml file.

        Parameters
        ----------
        fname : str
            Full path to the .yaml file to save the :class:`Tidy3dBaseModel` to.
        data_file : str = None
            Path to a separate hdf5 file to write :class:`.DataArray` objects to.

        Example
        -------
        >>> simulation.to_yaml(fname='folder/sim.yaml') # doctest: +SKIP
        """
        json_string = self._json_string(data_file=data_file)
        json_dict = json.loads(json_string)
        with open(fname, "w+", encoding="utf-8") as file_handle:
            yaml.dump(json_dict, file_handle, indent=INDENT)

    @classmethod
    def from_hdf5(cls, fname: str, **parse_raw_kwargs) -> Tidy3dBaseModel:
        """Loads :class:`Tidy3dBaseModel` from .yaml file.

        Parameters
        ----------
        fname : str
            Full path to the .hdf5 file to load the :class:`Tidy3dBaseModel` from.
        **parse_raw_kwargs
            Keyword arguments passed to pydantic's ``parse_raw`` method.

        Returns
        -------
        :class:`Tidy3dBaseModel`
            An instance of the component class calling `from_hdf5`.

        Example
        -------
        >>> simulation = Simulation.from_hdf5(fname='folder/sim.hdf5') # doctest: +SKIP
        """
        self_data_dict = cls.hdf5_to_dict(fname)
        return cls.parse_obj(self_data_dict, **parse_raw_kwargs)

    def to_hdf5(self, fname: str) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .hdf5 file.

        Parameters
        ----------
        fname : str
            Full path to the .hdf5 file to save the :class:`Tidy3dBaseModel` to.

        Example
        -------
        >>> simulation.to_hdf5(fname='folder/sim.hdf5') # doctest: +SKIP
        """

        data_dict = self.dict()
        self.dump_hdf5(data_dict, fname)

    """=============================================================================================
    Code modified from the hdfdict package: https://github.com/SiggiGue/hdfdict

    MIT License

    Copyright (c) 2018 Siegfried Gündert
    Copyright Flexcompute 2022

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    # pylint:disable=too-many-return-statements
    @staticmethod
    def unpack_dataset(dataset: h5py.Dataset, keep_numpy: bool = False) -> Any:
        """Gets the value contained in a dataset in a form ready to insert into final dict.

        Parameters
        ----------
        item : h5py.Dataset
            The raw value coming from the dataset, which needs to be decoded.
        keep_numpy : bool = False
            Whether to load a ``np.ndarray`` as such or convert it to list.

        Returns
        -------
        Value taken from the dataset ready to insert into returned dict.
        """
        value = dataset[()]

        # decoding numpy arrays
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return ()
            if isinstance(value[0], bytes):
                return [val.decode("utf-8") for val in value]
            if value.dtype == bool:
                return value.astype(bool)
            if not keep_numpy:
                return value.tolist()
            return value

        # decoding special types
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if np.isnan(value):
            return None

        return value

    @classmethod
    def load_from_handle(cls, hdf5_group: h5py.Group, **kwargs) -> Tidy3dBaseModel:
        """Loads an instance of the class from an hdf5 group,

        Parameters
        ----------
        hdf5_group : h5py.Group
            The hdf5 group containing data correponding to the ``dict()`` of the object.
        kwargs : dict
            Keyword arguments passed to ``pydantic.BaseModel.parse_obj``

        Returns
        -------
        Tidy3dBaseModel
        """
        data_dict = cls._load_group_data(data_dict={}, hdf5_group=hdf5_group)
        return cls.parse_obj(data_dict, **kwargs)

    @classmethod
    def _load_group_data(
        cls, data_dict: dict, hdf5_group: h5py.Group, keep_numpy: bool = False
    ) -> dict:
        """Recusively load the data from the group with dataset unpacking as base case."""

        if "keep_numpy" in hdf5_group:
            keep_numpy = hdf5_group["keep_numpy"]

        for key, value in hdf5_group.items():

            if key == "keep_numpy":
                continue

            # recurive case, try to load the group into data_dict[key]
            if isinstance(value, h5py.Group):
                data_dict[key] = cls._load_group_data(
                    data_dict={}, hdf5_group=value, keep_numpy=keep_numpy
                )

            # base case, unpack the value in the dataset
            elif isinstance(value, h5py.Dataset):
                data_dict[key] = cls.unpack_dataset(value, keep_numpy=keep_numpy)

        if any("TUPLE_ELEMENT_" in key for key in data_dict.keys()):
            return tuple(data_dict.values())

        return data_dict

    @classmethod
    def hdf5_to_dict(cls, fname: str) -> dict:
        """Load an hdf5 file into a dictionary storing its unpacked contents.

        Parameters
        ----------
        fname : str
            Path to the hdf5 file.

        Returns
        -------
        dict
            The dictionary containing all group names as keys and datasets as values.
        """

        # open the file and load its data recursively into a dictionary
        with h5py.File(fname, "r") as f:
            return cls._load_group_data(data_dict={}, hdf5_group=f)

    @staticmethod
    def pack_dataset(hdf5_group: h5py.Group, key: str, value: Any) -> None:
        """Loads a key value pair as a dataset in the hdf5 group."""

        # handle special cases
        if value is None:
            value = np.nan
        if isinstance(value, str):
            value = value.encode("utf-8")

        # numpy array containing strings (usually direction=['-','+'])
        elif isinstance(value, np.ndarray) and (value.dtype == "<U1"):
            value = value.tolist()

        _ = hdf5_group.create_dataset(name=key, data=value)

    def add_to_handle(self, hdf5_group: h5py.Group) -> None:
        """Saves a :class:`.Tidy3dBaesModel` instance to an hdf5 group,

        Parameters
        ----------
        hdf5_group : h5py.Group
            The hdf5 group containing data correponding to the ``dict()`` of the object.
        """
        self_dict = self.dict()
        self._save_group_data(data_dict=self_dict, hdf5_group=hdf5_group)

    @classmethod
    def _save_group_data(cls, data_dict: dict, hdf5_group: h5py.Group) -> None:
        """Recursively save the data to a group with a non-dict data as base case."""

        for key, value in data_dict.items():

            # if tuple as key, combine into a single string
            if isinstance(key, tuple):
                key = "_".join((str(i) for i in key))

            if isinstance(value, xr.DataArray):
                coords = {key: np.array(val) for key, val in value.coords.items()}
                value = dict(data=value.data, coords=coords, keep_numpy=True)

            # if a tuple of dicts, convert to a dict with special
            elif isinstance(value, tuple) and any(isinstance(val, dict) for val in value):
                value = {f"TUPLE_ELEMENT_{i}": val for i, val in enumerate(value)}

            # if dictionary as item in dict, create subgroup and recurse
            if isinstance(value, dict):
                hdf5_subgroup = hdf5_group.create_group(key)
                cls._save_group_data(data_dict=value, hdf5_group=hdf5_subgroup)

            # otherwise (actual data), just encode it and save it to the group as a dataset
            else:
                cls.pack_dataset(hdf5_group=hdf5_group, key=key, value=value)

    @classmethod
    def dump_hdf5(cls, data_dict: dict, fname: str) -> None:
        """Writes a dictionary of data into an hdf5 file.

        Parameters
        ----------
        data_dict : dict
            A dictionary of data with strings or tuples as keys and data or other dicts as values.
        fname : str
            path to the .hdf5 file.
        """

        # open the file and write to it recursively
        with h5py.File(fname, "w") as f:
            cls._save_group_data(data_dict=data_dict, hdf5_group=f)

    """End hdfdict modification
    ============================================================================================="""

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

    def _json_string(self, include_unset: bool = True, data_file: Optional[str] = None) -> str:
        """Returns string representation of a :class:`Tidy3dBaseModel`.

        Parameters
        ----------
        include_unset : bool = True
            Whether to include default fields in json string.
        data_file : str = None
            Path to a separate hdf5 file to write :class:`.DataArray` objects to.

        Returns
        -------
        str
            Json-formatted string holding :class:`Tidy3dBaseModel` data.
        """
        exclude_unset = not include_unset

        def get_json_string() -> str:
            """Return json string with function settings applied."""
            return self.json(indent=INDENT, exclude_unset=exclude_unset)

        def json_with_separate_data() -> str:
            """Return json string with data written to separate file."""

            original_encoder = self.__config__.json_encoders[xr.DataArray]

            # Create/overwrite data file
            with h5py.File(data_file, "w") as fhandle:

                def write_data(x):
                    """Write data to group inside hdf5 file."""
                    group_name = tokenize(x)
                    if group_name not in fhandle.keys():
                        group = fhandle.create_group(group_name)
                        coords = {key: np.array(val) for key, val in x.coords.items()}
                        data_dict = dict(data=x.data, coords=coords, keep_numpy=True)
                        self._save_group_data(data_dict=data_dict, hdf5_group=group)
                    return dict(group_name=group_name, data_file=data_file, tag="DATA_ITEM")

                self.__config__.json_encoders[xr.DataArray] = write_data

                # put infinity and -infinity in quotes
                json_string = get_json_string()

                # re-set the json encoder for data
                self.__config__.json_encoders[xr.DataArray] = original_encoder

                return json_string

        json_string = json_with_separate_data() if data_file else get_json_string()
        tmp_string = "<<TEMPORARY_INFINITY_STRING>>"
        json_string = json_string.replace("-Infinity", tmp_string)
        json_string = json_string.replace("Infinity", '"Infinity"')
        json_string = json_string.replace(tmp_string, '"-Infinity"')

        return json_string

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
            data_type = field._type_display()  # pylint:disable=protected-access

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


def cache(prop):
    """Decorates a property to cache the first computed value and return it on subsequent calls."""

    # note, we could also just use `prop` as dict key, but hashing property might be slow
    prop_name = prop.__name__

    @wraps(prop)
    def cached_property_getter(self):
        """The new property method to be returned by decorator."""

        stored_value = self._cached_properties.get(prop_name)  # pylint:disable=protected-access

        if stored_value is not None:
            return stored_value

        computed_value = prop(self)
        self._cached_properties[prop_name] = computed_value  # pylint:disable=protected-access
        return computed_value

    return cached_property_getter


def cached_property(cached_property_getter):
    """Shortcut for property(cache()) of a getter."""

    return property(cache(cached_property_getter))
