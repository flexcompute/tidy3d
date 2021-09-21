import json
from jsonschema import Draft3Validator
from jsonschema.validators import extend
import jsonschema2md

import sys

sys.path.append("../../")

from tidy3d import Simulation

""" creates schema validator and function for validating simulation dict """


def load_json(fname: str) -> dict:
    """loads a json file into dictionary"""
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
    """makes sure a simulation dict is consistent with schema"""
    SCHEMA_VALIDATOR.is_valid(sim_dict)


def generate_schema_docs(fname_schema: str, fname_readme: str = "SCHEMA.md") -> None:
    """make some docs, note this doesnt really work yet"""
    parser = jsonschema2md.Parser()
    with open(fname_schema, "r") as fp:
        md_lines = parser.parse_schema(json.load(fp))
    with open(fname_readme, "w") as fp:
        for line in md_lines:
            fp.write(md_lines)


def write_schema(fname_schema: str = "schema.json") -> None:
    """saves simulation object schema to json"""
    schema_str = Simulation.schema_json(indent=2)
    with open(fname_schema, "w") as fp:
        fp.write(schema_str)


VALIDATOR = Draft3Validator
SCHEMA_PATH = "schema.json"
SCHEMA_DICT = load_json(SCHEMA_PATH)
SCHEMA_VALIDATOR = _create_validator(SCHEMA_DICT)

if __name__ == "__main__":
    write_schema(SCHEMA_PATH)
    SCHEMA_DICT = load_json(SCHEMA_PATH)
    SCHEMA_VALIDATOR = _create_validator(SCHEMA_DICT)
    # generate_schema_docs(SCHEMA_PATH)
