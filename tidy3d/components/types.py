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


""" Complex Values """


class ComplexNumber(pydantic.BaseModel):
    """Complex number with a well defined schema."""

    real: float
    imag: float

    @property
    def as_complex(self):
        """return complex representation of ComplexNumber."""
        return self.real + 1j * self.imag


class tidycomplex(complex):  # pylint: disable=invalid-name
    """complex type that we can use in our models."""

    @classmethod
    def __get_validators__(cls):
        """Defines which validator function to use for NumpyArray."""
        yield cls.validate

    @classmethod
    def validate(cls, value):
        """What gets called when you construct a tidycomplex."""

        if isinstance(value, ComplexNumber):
            return value.as_complex
        if isinstance(value, dict):
            c = ComplexNumber(**value)
            return c.as_complex
        return cls(value)

    @classmethod
    def __modify_schema__(cls, field_schema):
        """Sets the schema of NumpyArray."""
        field_schema.update(ComplexNumber.schema())


""" geometric """

Size1D = pydantic.NonNegativeFloat
Size = Tuple[Size1D, Size1D, Size1D]
Coordinate = Tuple[float, float, float]
Coordinate2D = Tuple[float, float]
Bound = Tuple[Coordinate, Coordinate]
GridSize = Union[pydantic.PositiveFloat, List[pydantic.PositiveFloat]]
Axis = Literal[0, 1, 2]
Axis2D = Literal[0, 1]
Vertices = List[Coordinate2D]
Shapely = BaseGeometry

""" medium """

# Complex = Union[complex, ComplexNumber]
Complex = Union[tidycomplex, ComplexNumber]
PoleAndResidue = Tuple[Complex, Complex]

# PoleAndResidue = Tuple[Tuple[float, float], Tuple[float, float]]
FreqBoundMax = float
FreqBoundMin = float
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

    @classmethod
    def __modify_schema__(cls, field_schema):
        """Sets the schema of NumpyArray."""
        field_schema.update(NumpyArray.schema())


class ArrayMeta(type):
    """metclass for Array, enables Array[type] -> TypedArray"""

    def __getitem__(cls, t):
        """Array[t] -> TypedArray"""
        return type("Array", (TypedArray,), {"inner_type": t})


class Array(np.ndarray, metaclass=ArrayMeta):
    """type of numpy array with annotated type (Array[float], Array[complex])"""


""" note:
    ^ this is the best way to declare numpy types if you know dtype.
    for example: ``field_amps: Array[float] = np.random.random(5)``.
"""


class NumpyArray(pydantic.BaseModel):
    """Wrapper around numpy arrays that has a well defined json schema."""

    data_list: List

    @property
    def arr(self):
        """Contructs a numpy array representation of the NumpyArray."""
        return np.array(self.data_list)


class tidynumpy(np.ndarray):  # pylint: disable=invalid-name
    """Numpy array type that we can use in place of np.ndarray."""

    @classmethod
    def __get_validators__(cls):
        """Defines which validator function to use for NumpyArray."""
        yield cls.validate

    @classmethod
    def validate(cls, value):
        """What gets called when you construct a tidynumpy."""
        if isinstance(value, NumpyArray):
            return value.arr
        if isinstance(value, dict):
            numpy_array = NumpyArray(**value)
            return numpy_array.arr
        if isinstance(value, list):
            return value
        return np.array(value)

    @classmethod
    def __modify_schema__(cls, field_schema):
        """Sets the schema of NumpyArray."""
        field_schema.update(NumpyArray.schema())


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
