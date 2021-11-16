"""global configuration / base class for pydantic models used to make simulation."""

import json

import rich
import pydantic
import yaml
import numpy as np

from .types import ComplexNumber
from ..log import FileError

# default indentation (# spaces) in files
INDENT = 4


class Tidy3dBaseModel(pydantic.BaseModel):
    """Base pydantic model that all Tidy3d components inherit from.
    Defines configuration for handling data structures
    as well as methods for imporing, exporting, and hashing tidy3d objects.
    For more details on pydantic base models, see:
    `Pydantic Models <https://pydantic-docs.helpmanual.io/usage/models/>`_
    """

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
            np.ndarray: lambda x: x.tolist(),
            complex: lambda x: ComplexNumber(real=np.real(x), imag=np.imag(x)),
        }
        json_decoders = {ComplexNumber: lambda x: x.real + 1j * x.imag}

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
    def from_file(cls, fname: str):
        """Loads a :class:`Tidy3dBaseModel` from .yaml or .json file.

        Parameters
        ----------
        fname : str
            Full path to the .yaml or .json file to load the :class:`Tidy3dBaseModel` from.

        Returns
        -------
        :class:`Tidy3dBaseModel`
            An instance of the component class calling `load`.

        Example
        -------
        >>> simulation = Simulation.from_file(fname='folder/sim.json')
        """
        if ".json" in fname:
            return cls.from_json(fname=fname)
        if ".yaml" in fname:
            return cls.from_yaml(fname=fname)
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
    def from_json(cls, fname: str):
        """Load a :class:`Tidy3dBaseModel` from .json file.

        Parameters
        ----------
        fname : str
            Full path to the .json file to load the :class:`Tidy3dBaseModel` from.

        Returns
        -------
        :class:`Tidy3dBaseModel`
            An instance of the component class calling `load`.

        Example
        -------
        >>> simulation = Simulation.from_json(fname='folder/sim.json')
        """
        return cls.parse_file(fname)

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
    def from_yaml(cls, fname: str):
        """Loads :class:`Tidy3dBaseModel` from .yaml file.

        Parameters
        ----------
        fname : str
            Full path to the .yaml file to load the :class:`Tidy3dBaseModel` from.

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
        return cls.parse_raw(json_raw)

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
        """Hash a :class:`Tidy3dBaseModel` objects using its json string.

        Returns
        -------
        int
            Integer representation of the hash of the :class:`Tidy3dBaseModel`.

        Example
        -------
        >>> hash_integer = hash(simulation)
        """
        return hash(self.json())

    def __lt__(self, other):
        """define < for getting unique indices based on hash."""
        return hash(self) < hash(other)

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
        return self.json(indent=INDENT, exclude_unset=exclude_unset)
