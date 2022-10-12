"""global configuration / base class for pydantic models used to make simulation."""
from __future__ import annotations

import json

# from typing import Optional
from functools import wraps

import rich

import pydantic
from pydantic.fields import ModelField
import yaml
import numpy as np
import h5py
import xarray as xr

# from dask.base import tokenize

from .types import ComplexNumber, Literal, TYPE_TAG_STR
from ..log import FileError
from .data.data_array import DataArray

# default indentation (# spaces) in files
INDENT = 4
JSON_TAG = "JSON_STRING"
DATA_ARRAY_TAG = "XR.DATAARRAY"


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
            xr.DataArray: lambda x: DATA_ARRAY_TAG,  # pylint:disable=unhashable-member
        }
        frozen = True
        allow_mutation = False
        copy_on_model_validation = "none"

    _cached_properties = pydantic.PrivateAttr({})

    def copy(self, **kwargs) -> Tidy3dBaseModel:
        """Copy a Tidy3dBaseModel.  With ``deep=True`` as default."""
        if "deep" in kwargs and kwargs["deep"] is False:
            raise ValueError("Can't do shallow copy of component, set `deep=True` in copy().")
        kwargs.update(dict(deep=True))
        new_copy = pydantic.BaseModel.copy(self, **kwargs)
        return self.validate(new_copy.dict())

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

        if ".json" in fname:
            return self.to_json(fname=fname)
        if ".yaml" in fname:
            return self.to_yaml(fname=fname)
        if ".hdf5" in fname:
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
        json_string = self._json_string
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
        json_dict = json.loads(json_string)
        with open(fname, "w+", encoding="utf-8") as file_handle:
            yaml.dump(json_dict, file_handle, indent=INDENT)

    @staticmethod
    def tuple_to_dict(tuple_name: str, tuple_values: tuple) -> dict:
        """How we generate a dictionary mapping new keys to tuple values for hdf5."""
        return {f"{tuple_name}_{i}": val for i, val in enumerate(tuple_values)}

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

        def load_data_from_file(json_dict: dict, group_path: str = "/") -> None:
            """For every DataArray item in dictionary, write path of hdf5 group as value."""

            for key, value in json_dict.items():

                subpath = f"{group_path}/{key}"

                # write the path to the element of the json dict where the data_array should be
                if value == DATA_ARRAY_TAG:
                    json_dict[key] = DataArray.from_hdf5(fname=fname, group_path=subpath)
                    continue

                # if a list, assign each element a unique key, recurse
                if isinstance(value, (list, tuple)):
                    value_dict = cls.tuple_to_dict(tuple_name=key, tuple_values=value)
                    load_data_from_file(json_dict=value_dict, group_path=subpath)

                # if a dict, recurse
                elif isinstance(value, dict):
                    load_data_from_file(json_dict=value, group_path=subpath)

        with h5py.File(fname, "r") as f_handle:
            json_string = f_handle[JSON_TAG][()]

        json_dict = json.loads(json_string)
        load_data_from_file(json_dict, group_path="/")

        return cls.parse_obj(json_dict, **parse_raw_kwargs)

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

        def add_data_to_file(data_dict: dict, group_path: str = "/") -> None:
            """For every DataArray item in dictionary, write path of hdf5 group as value."""

            for key, value in data_dict.items():

                # append the key to the path
                subpath = f"{group_path}/{key}"

                # write the path to the element of the json dict where the data_array should be
                if isinstance(value, xr.DataArray):
                    value.to_hdf5(fname=fname, group_path=subpath)

                # if a tuple, assign each element a unique key
                if isinstance(value, (list, tuple)):
                    value_dict = self.tuple_to_dict(tuple_name=key, tuple_values=value)
                    add_data_to_file(data_dict=value_dict, group_path=subpath)

                # if a dict, recurse
                elif isinstance(value, dict):
                    add_data_to_file(data_dict=value, group_path=subpath)

        json_string = self._json_string
        with h5py.File(fname, "w") as f_handle:
            f_handle[JSON_TAG] = json_string

        add_data_to_file(data_dict=self.dict(), group_path="/")

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

    @property
    def _json_string(self) -> str:
        """Returns string representation of a :class:`Tidy3dBaseModel`.

        Returns
        -------
        str
            Json-formatted string holding :class:`Tidy3dBaseModel` data.
        """

        def make_json_compatible(json_string: str) -> str:
            """Makes the string compatiable with json standards, notably for infinity."""

            tmp_string = "<<TEMPORARY_INFINITY_STRING>>"
            json_string = json_string.replace("-Infinity", tmp_string)
            json_string = json_string.replace("Infinity", '"Infinity"')
            return json_string.replace(tmp_string, '"-Infinity"')

        json_string = self.json(indent=INDENT, exclude_unset=False)
        json_string = make_json_compatible(json_string)
        json_dict = json.loads(json_string)

        return json.dumps(json_dict)

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
