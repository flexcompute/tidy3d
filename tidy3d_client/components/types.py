import pydantic
from typing import Tuple, Dict, List, Callable, Any, Union

# Literal only available in python 3.8 + so try import otherwise use extensions
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

""" Defines 'types' that various fields can be """

# tuple containing three non-negative floats
Size = Tuple[
    pydantic.NonNegativeFloat, pydantic.NonNegativeFloat, pydantic.NonNegativeFloat
]

# tuple containing three floats
Coordinate = Tuple[float, float, float]
Coordinate2D = Tuple[float, float]

# tuple containing min coordinate (in each x,y,z) and max coordinate
Bound = Tuple[Coordinate, Coordinate]

# grid size
GridSize = Union[pydantic.PositiveFloat, Tuple[pydantic.PositiveFloat, ...]]

# axis type
Axis = Literal[0, 1, 2]