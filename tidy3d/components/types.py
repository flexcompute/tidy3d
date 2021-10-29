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
from shapely.geometry.base import BaseGeometry

from ..constants import inf

""" abstract """

Inf = Literal[inf]

""" geometric """

Size1D = Union[pydantic.NonNegativeFloat, Inf]
Size = Tuple[Size1D, Size1D, Size1D]
Coordinate = Tuple[float, float, float]
Coordinate2D = Tuple[float, float]
Bound = Tuple[Coordinate, Coordinate]
GridSize = Union[pydantic.PositiveFloat, List[pydantic.PositiveFloat]]
Axis = Literal[0, 1, 2]
Vertices = List[Coordinate2D]
Shapely = BaseGeometry

""" grid """


""" medium """

Complex = Tuple[float, float]
PoleAndResidue = Tuple[Complex, Complex]
FreqBound = Union[float, Inf, Literal[-inf]]

""" symmetries """

Symmetry = Literal[0, -1, 1]

""" sources """

Polarization = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
Direction = Literal["+", "-"]

""" monitors """

EMField = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
FieldType = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

""" plotting """

Ax = Axes

""" Numpy """

# generic numpy array
Numpy = np.ndarray


class TypedArray(np.ndarray):
    """A numpy array with a type given by cls.inner_type"""

    @classmethod
    def __get_validators__(cls):
        """boilerplate"""
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        """validator"""
        # need to fix, doesnt work for simulationdata_export and load?
        return np.array(val, dtype=cls.inner_type)  # pylint: disable=no-member


class ArrayMeta(type):
    """metclass for Array, enables Array[type] -> TypedArray"""

    def __getitem__(cls, t):
        """Array[t] -> TypedArray"""
        return type("Array", (TypedArray,), {"inner_type": t})


class Array(np.ndarray, metaclass=ArrayMeta):
    """type of numpy array with annotated type (Array[float], Array[complex])"""


# lists or np.ndarrays of certain type
FloatArrayLike = Union[List[float], Array[float]]
IntArrayLike = Union[List[int], Array[int]]
ComplexArrayLike = Union[List[complex], Array[complex]]

# encoding for JSON in pydantic models
def numpy_encoding(array):
    """json encoding of numpy array"""
    if array.dtype == "complex":
        return {"re": list(array.real), "im": list(array.imag)}
    return list(array)
