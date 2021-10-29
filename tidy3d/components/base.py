""" global configuration / base class for pydantic models used to make simulation """

import json

import rich
import pydantic
import yaml
import numpy as np

# default indentation (# spaces) in files
INDENT = 4


class Tidy3dBaseModel(pydantic.BaseModel):
    """https://pydantic-docs.helpmanual.io/usage/model_config/"""

    class Config:  # pylint: disable=too-few-public-methods
        """sets config for all Tidy3dBaseModel objects"""

        arbitrary_types_allowed = True
        validate_all = True  # validate default values too
        extra = "forbid"  # forbid extra kwargs not specified in model
        validate_assignment = True  # validate when attributes are set after initialization
        allow_population_by_field_name = True
        json_encoders = {np.ndarray: lambda x: x.tolist()}  # pylint: disable=unnecessary-lambda

    def help(self, methods: bool = False) -> None:
        """get help for this object"""
        rich.inspect(self, methods=methods)

    def __hash__(self) -> int:
        """hash tidy3dBaseModel objects using their json strings"""
        return hash(self.json())

    def _json_string(self, exclude_unset: bool = False) -> str:
        """returns string representation of self"""
        return self.json(indent=INDENT, exclude_unset=exclude_unset)

    @classmethod
    def load(cls, fname: str):
        """load Simulation from .json file"""
        return cls.parse_file(fname)

    def export(self, fname: str) -> None:
        """Exports Tidy3dBaseModel instance to .json file"""
        json_string = self._json_string()
        with open(fname, "w", encoding="utf-8") as file_handle:
            file_handle.write(json_string)

    @classmethod
    def load_yaml(cls, fname: str):
        """load Simulation from .yaml file"""
        with open(fname, "r", encoding="utf-8") as yaml_in:
            json_dict = yaml.safe_load(yaml_in)
        json_raw = json.dumps(json_dict, indent=INDENT)
        return cls.parse_raw(json_raw)

    def export_yaml(self, fname: str) -> None:
        """Exports Tidy3dBaseModel instance to .yaml file"""
        json_string = self._json_string()
        json_dict = json.loads(json_string)
        with open(fname, "w+", encoding="utf-8") as file_handle:
            yaml.dump(json_dict, file_handle, indent=INDENT)
