import dataclasses
# import typing

from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class SimpleSimulation:
    param1: int
    param2: float

simple_example = SimpleSimulation(1, 2.2)

# Encoding to JSON. Note the output is a string, not a dictionary.
json_string = simple_example.to_json()  # {"int_field": 1}
print(json_string)

# Encoding to a (JSON) dict
json_dict = simple_example.to_dict()  # {'int_field': 1}
print(json_dict)

# Decoding from JSON. Note the input is a string, not a dictionary.
SimpleSimulation.from_json('{"param1": 1, "param2":2.2}')  # SimpleExample(1)

# Decoding from a (JSON) dict
SimpleSimulation.from_dict({'param1': 1, 'param2': 2.2})  # SimpleExample(1)