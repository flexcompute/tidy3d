""" global configuration / base class for pydantic models used to make simulation """

import json
import yaml

import pydantic

# default indentation (# spaces) in files
INDENT = 4


class Tidy3dBaseModel(pydantic.BaseModel):
    """https://pydantic-docs.helpmanual.io/usage/model_config/"""

    class Config:  # pylint: disable=too-few-public-methods
        """sets config for all Tidy3dBaseModel objects"""

        validate_all = True  # validate default values too
        extra = "forbid"  # forbid extra kwargs not specified in model
        validate_assignment = True  # validate when attributes are set after initialization
        allow_population_by_field_name = True

    def _json_string(self) -> str:
        """returns string representation of self"""
        return self.json(
            indent=INDENT
        )  # , exclude_unset=True) # if I exclude unset, it throws away info

    def export(self, fname: str) -> None:
        """Exports Tidy3dBaseModel instance to .json file"""
        json_string = self._json_string()
        with open(fname, "w", encoding="utf-8") as file_handle:
            file_handle.write(json_string)

    def export_yaml(self, fname: str) -> None:
        """Exports Tidy3dBaseModel instance to .yaml file"""
        json_string = self._json_string()
        json_dict = json.loads(json_string)
        with open(fname, "w+", encoding="utf-8") as file_handle:
            yaml.dump(json_dict, file_handle, indent=INDENT)

    @classmethod
    def load(cls, fname: str):
        """load Simulation from .json file"""
        return cls.parse_file(fname)

    @classmethod
    def load_yaml(cls, fname: str):
        """load Simulation from .yaml file"""
        with open(fname, "r", encoding="utf-8") as yaml_in:
            json_dict = yaml.safe_load(yaml_in)
        json_raw = json.dumps(json_dict, indent=INDENT)
        return cls.parse_raw(json_raw)


def register_subclasses(fields: tuple):
    """attempt at a decorator factory"""

    field_map = {field.__name__: field for field in fields}

    def _register_subclasses(cls):
        """attempt at a decorator"""

        orig_init = cls.__init__

        class _class:
            class_name: str

            def __init__(self, **kwargs):
                print(kwargs)
                class_name = type(self).__name__
                kwargs["class_name"] = class_name
                print(kwargs)
                orig_init(**kwargs)

            @classmethod
            def __get_validators__(cls):
                yield cls.validate

            @classmethod
            def validate(cls, v):
                if isinstance(v, dict):
                    class_name = v.get("class_name")
                    json_string = json.dumps(v)
                else:
                    class_name = v.class_name
                    json_string = v.json()
                cls_type = field_map[class_name]
                return cls_type.parse_raw(json_string)

        return _class

    return _register_subclasses
