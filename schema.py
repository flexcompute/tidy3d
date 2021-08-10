import json
from jsonschema import Draft3Validator
from jsonschema.validators import extend

VALIDATOR = Draft3Validator

def load_schema(fname_schema: str = "schema.json") -> dict:
    """loads schema from json file into dict"""
    with open(fname_schema, "r") as fp:
        schema_dict = json.load(fp)
    return schema_dict


def _accepts_tuple(checker, instance):
    """custom validator for array types, accept tuple as ok"""
    is_array = VALIDATOR.TYPE_CHECKER.is_type(instance, "array")
    is_tuple = isinstance(instance, tuple)
    return is_array or is_tuple


def _create_validator(schema: dict) -> Draft3Validator:
    """generates custom validator with array type accepting tuples"""
    array_checker = VALIDATOR.TYPE_CHECKER.redefine("array", _accepts_tuple)
    custom_validator = extend(VALIDATOR, type_checker=array_checker)
    validator = custom_validator(schema)
    return validator


def validate_schema(sim_dict: dict, fname_schema: str = "schema.json") -> None:
    """makes sure simulation dict is consistent with schema json"""
    schema_dict = load_schema(fname_schema)
    validator = _create_validator(schema_dict)
    validator.is_valid(sim_dict)
