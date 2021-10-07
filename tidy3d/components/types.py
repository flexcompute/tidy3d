""" Defines 'types' that various fields can be """

from typing import Tuple, List, Union

# Literal only available in python 3.8 + so try import otherwise use extensions
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import pydantic
import numpy as np
from matplotlib.axes._subplots import Axes

# tuple containing three non-negative floats
Size = Tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat, pydantic.NonNegativeFloat]

# tuple containing three floats
Coordinate = Tuple[float, float, float]
Coordinate2D = Tuple[float, float]

# tuple containing min coordinate (in each x,y,z) and max coordinate
Bound = Tuple[Coordinate, Coordinate]

# grid size
GridSize = Union[pydantic.PositiveFloat, List[pydantic.PositiveFloat]]

# axis index
Axis = Literal[0, 1, 2]
Vertices = List[Coordinate2D]

# pole-residue poles (each pole has two complex numbers)
Complex = Tuple[float, float]
PoleAndResidue = Tuple[Complex, Complex]

# sources
Polarization = Literal["Jx", "Jy", "Jz", "Mx", "My", "Mz"]
Direction = Literal["+", "-"]

# monitors
EMField = Literal["E", "H"]
Component = Literal["x", "y", "z"]

Numpy = np.ndarray

Symmetry = Literal[0, -1, 1]

Ax = Axes

ArrayLike = Union[List[float], Numpy]

class TypedArray(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        return np.array(val, dtype=cls.inner_type)

class ArrayMeta(type):
    def __getitem__(self, t):
        return type('Array', (TypedArray,), {'inner_type': t})

class Array(np.ndarray, metaclass=ArrayMeta):
    """ Type of numpy array (Array[float], Array[complex]) """
    pass

def numpy_encoding(array):
    """ json encoding of numpy array """
    if np.array([1+1j]).dtype == 'complex':
        return {'re': list(array.real), 'im': list(array.imag)}
    else:
        return list(array)