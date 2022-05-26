"""global configuration / base class for pydantic models used to make simulation."""

import json
from functools import wraps

import rich
import pydantic
import yaml
import numpy as np
from pydantic.fields import ModelField

from .types import ComplexNumber, NumpyArray, Literal
from ..log import FileError

# default indentation (# spaces) in files
INDENT = 4

# type tag default name
TYPE_TAG_STR = "type"


class Tidy3dBaseModel(pydantic.BaseModel):
    """Base pydantic model that all Tidy3d components inherit from.
    Defines configuration for handling data structures
    as well as methods for imporing, exporting, and hashing tidy3d objects.
    For more details on pydantic base models, see:
    `Pydantic Models <https://pydantic-docs.helpmanual.io/usage/models/>`_
    """

    def __init_subclass__(cls):
        """Things that are done to each of the models."""

        add_type_field(cls)
        cls.__doc__ = generate_docstring(cls)

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
            np.ndarray: lambda x: NumpyArray(data_list=x.tolist()),
            complex: lambda x: ComplexNumber(real=x.real, imag=x.imag),
        }

    def copy(self, deep: bool = True, **kwargs) -> "Self":
        """Copy a Tidy3dBaseModel.  With ``deep=True`` as default."""
        return super().copy(deep=deep, **kwargs)

    def help(self, methods: bool = False) -> None:
        """Prints message describing the fields and methods of a :class:`Tidy3dBaseModel`.

        Parameters
        ----------
        methods : bool = False
            Whether to also print out information about object's methods.

        Example
        -------
        >>> simulation.help(methods=True)
        """
        rich.inspect(self, methods=methods)

    @classmethod
    def from_file(cls, fname: str, **parse_kwargs):
        """Loads a :class:`Tidy3dBaseModel` from .yaml or .json file.

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
        >>> simulation = Simulation.from_file(fname='folder/sim.json')
        """
        if ".json" in fname:
            return cls.from_json(fname=fname, **parse_kwargs)
        if ".yaml" in fname:
            return cls.from_yaml(fname=fname, **parse_kwargs)
        raise FileError(f"File must be .json or .yaml, given {fname}")

    def to_file(self, fname: str) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .yaml or .json file

        Parameters
        ----------
        fname : str
            Full path to the .yaml or .json file to save the :class:`Tidy3dBaseModel` to.

        Example
        -------
        >>> simulation.to_file(fname='folder/sim.json')
        """
        if ".json" in fname:
            return self.to_json(fname=fname)
        if ".yaml" in fname:
            return self.to_yaml(fname=fname)
        raise FileError(f"File must be .json or .yaml, given {fname}")

    @classmethod
    def from_json(cls, fname: str, **parse_file_kwargs):
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
        >>> simulation = Simulation.from_json(fname='folder/sim.json')
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
        >>> simulation.to_json(fname='folder/sim.json')
        """
        json_string = self._json_string()
        with open(fname, "w", encoding="utf-8") as file_handle:
            file_handle.write(json_string)

    @classmethod
    def from_yaml(cls, fname: str, **parse_raw_kwargs):
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
        >>> simulation = Simulation.from_yaml(fname='folder/sim.yaml')
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
        >>> simulation.to_yaml(fname='folder/sim.yaml')
        """
        json_string = self._json_string()
        json_dict = json.loads(json_string)
        with open(fname, "w+", encoding="utf-8") as file_handle:
            yaml.dump(json_dict, file_handle, indent=INDENT)

    def __hash__(self) -> int:
        """Hash a :class:`Tidy3dBaseModel` objects using its json string."""
        return hash(self.json())

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
        """define == for checking whether two base models are equal unique indices based on hash."""
        return hash(self) == hash(other)

    def _json_string(self, include_unset: bool = True) -> str:
        """Returns string representation of a :class:`Tidy3dBaseModel`.

        Parameters
        ----------
        include_unset : bool = True
            Whether to include default fields in json string.

        Returns
        -------
        str
            Json-formatted string holding :class:`Tidy3dBaseModel` data.
        """
        exclude_unset = not include_unset

        # put infinity and -infinity in quotes
        tmp_string = "<<TEMPORARY_INFINITY_STRING>>"
        json_string = self.json(indent=INDENT, exclude_unset=exclude_unset)
        json_string = json_string.replace("-Infinity", tmp_string)
        json_string = json_string.replace("Infinity", '"Infinity"')
        json_string = json_string.replace(tmp_string, '"-Infinity"')

        return json_string

    @classmethod
    def construct(cls, _fields_set=None, *, __recursive__=True, **values):
        """Recursive construct."""
        if not __recursive__:
            return super().construct(_fields_set, **values)

        m = cls.__new__(cls)

        fields_values = {}
        for name, field in cls.__fields__.items():
            if name in values:
                if issubclass(field.type_, BaseModel):
                    fields_values[name] = field.outer_type_.construct(
                        **values[name], __recursive__=True
                    )
                else:
                    fields_values[name] = values[name]
            elif not field.required:
                fields_values[name] = field.get_default()

        object.__setattr__(m, "__dict__", fields_values)
        if _fields_set is None:
            _fields_set = set(values.keys())
        object.__setattr__(m, "__fields_set__", _fields_set)
        m._init_private_attributes()
        return m


def add_type_field(cls):
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


def generate_docstring(cls) -> str:
    """Generates a docstring for a Tidy3D model."""

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
        default_str = f" = {default_val}" if default_val != ... else ""
        doc += f"    {field_name} : {data_type}{default_str}\n"

        # get field metadata
        field_info = field.field_info
        doc += "        "

        # add units (if present)
        units = field_info.extra.get("units")
        if units is not None:
            doc += f"[units = {units}].  "

        # add description
        description_str = field_info.description
        if description_str is not None:
            doc += f"{description_str}\n"

    # add in remaining things in the docs
    if original_docstrings:
        doc += "\n"
        doc += original_docstrings

    doc += "\n"

    return doc


def cache(prop):
    """Decorates a property to cache the previously computed values based on a hash of self."""

    stored_values = {}

    @wraps(prop)
    def cached_property(self):
        """The new property method to be returned by decorator."""

        hash_key = hash(self)

        # if the value has been computed before, we can simply return the stored value
        if hash_key in stored_values:
            return stored_values[hash_key]

        # otherwise, we need to recompute the value, store it in the storage dict, and return
        return_value = prop(self)
        stored_values[hash_key] = return_value
        return return_value

    return cached_property
