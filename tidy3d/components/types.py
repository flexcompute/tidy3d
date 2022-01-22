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

from ..constants import LARGE_NUMBER

""" infinity """


class Inf(pydantic.BaseModel):
    """Infinity.  Can use built-in instance: ``tidy3d.inf``."""

    def __neg__(self):
        """Negative Infinity"""
        return NegInf()

    def __truediv__(self, other):
        """dividing by something equals a large number"""
        return LARGE_NUMBER


class NegInf(pydantic.BaseModel):
    """Negative infinity.  Can use built-in instance as: ``-tidy3d.inf``."""

    def __neg__(self):
        """Positive Infinity"""
        return Inf()

    def __truediv__(self, other):
        """dividing by something equals a very large negative number"""
        return -LARGE_NUMBER


# built in instance of Inf ()
inf = LARGE_NUMBER  # comment out to use Inf().
# inf = Inf()


""" Complex Values """


class ComplexNumber(pydantic.BaseModel):
    real: float
    imag: float

    @property
    def z(self):
        return self.real + 1j * self.imag


class tidycomplex(complex):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value, field):

        if isinstance(value, ComplexNumber):
            return value.z
        elif isinstance(value, dict):
            c = ComplexNumber(**value)
            return c.z
        else:
            return cls(value)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(ComplexNumber.schema())


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

""" medium """

# Complex = Union[complex, ComplexNumber]
Complex = Union[tidycomplex, ComplexNumber]
PoleAndResidue = Tuple[Complex, Complex]

# PoleAndResidue = Tuple[Tuple[float, float], Tuple[float, float]]
FreqBoundMax = Union[float, Inf]
FreqBoundMin = Union[float, NegInf]
FreqBound = Tuple[FreqBoundMin, FreqBoundMax]

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


class NumpyArray(pydantic.BaseModel):
    data_list: List

    @property
    def arr(self):
        return np.array(self.data_list)


class tidynumpy(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value, field):
        if isinstance(value, NumpyArray):
            return value.arr
        elif isinstance(value, dict):
            n = NumpyArray(**value)
            return n.arr
        elif isinstance(value, list):
            return value
        else:
            return np.array(value)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(NumpyArray.schema())


""" note:
    ^ this is the best way to declare numpy types if you know dtype.
    for example: ``field_amps: Array[float] = np.random.random(5)``.
"""

ArrayLike = Union[tidynumpy, NumpyArray, List]


# lists or np.ndarrays of certain type
FloatArrayLike = Union[List[float], Array[float]]
IntArrayLike = Union[List[int], Array[int]]
ComplexArrayLike = Union[List[complex], Array[complex]]

# encoding for JSON in tidy3d data
# technically not used yet since the tidy3d data has separate load and export methods.
def numpy_encoding(array):
    """json encoding of a (maybe complex-valued) numpy array."""
    if array.dtype == "complex":
        return {"re": list(array.real), "im": list(array.imag)}
    return list(array)
