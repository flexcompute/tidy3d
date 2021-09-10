import json
from jsonschema import Draft3Validator
from jsonschema.validators import extend

""" creates schema validator and function for validating simulation dict """

VALIDATOR = Draft3Validator
SCHEMA_PATH = 'schema.json'
SCHEMA_DICT = load_json(SCHEMA_PATH)
SCHEMA_VALIDATOR = _create_validator(SCHEMA_DICT)

def load_json(fname: str) -> dict:
    """loads a json file into dictionary """
    with open(fname, "r") as fp:
        data_dict = json.load(fp)
    return data_dict

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

def validate_dict(sim_dict: dict) -> None:
    """makes sure a simulation dict is consistent with schema """
    SCHEMA_VALIDATOR.is_valid(sim_dict)
