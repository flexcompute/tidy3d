""" Defines 'types' that various fields can be """

from typing import Tuple, Union

# Literal only available in python 3.8 + so try import otherwise use extensions
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing_extensions import Annotated

import pydantic
import numpy as np
from matplotlib.axes._subplots import Axes
from shapely.geometry.base import BaseGeometry
from ..log import ValidationError

# import warnings
# warnings.filterwarnings(action="error", category=np.ComplexWarning)
""" Numpy Arrays """

# type tag default name
TYPE_TAG_STR = "type"


def annotate_type(UnionType):  # pylint:disable=invalid-name
    """Annotated union type using TYPE_TAG_STR as discriminator."""
    return Annotated[UnionType, pydantic.Field(discriminator=TYPE_TAG_STR)]


# generic numpy array
Numpy = np.ndarray


class TypedArray(np.ndarray):
    """A numpy array with a type given by cls.inner_type"""

    @classmethod
    def __get_validators__(cls):
        """boilerplate"""
        yield cls.validate_to_numpy

    @classmethod
    def validate_to_numpy(cls, val):
        """convert to numpy array"""
        return np.array(val)

    @classmethod
    def __modify_schema__(cls, field_schema):
        """Sets the schema of NumpyArray."""
        field_schema.update(NumpyArray.schema())


class NumpyArray(pydantic.BaseModel):
    """Wrapper around numpy arrays that has a well defined json schema."""

    data_list: list

    @property
    def arr(self):
        """Contructs a numpy array representation of the NumpyArray."""
        return np.array(self.data_list)


class ArrayMeta(type):
    """metclass for Array, enables Array[type] -> TypedArray"""

    def __getitem__(cls, t):
        """Array[t] -> TypedArray"""
        return type("Array", (TypedArray,), {"inner_type": t})


class Array(np.ndarray, metaclass=ArrayMeta):
    """type of numpy array with annotated type (Array[float], Array[complex])"""


class TypedArrayLike(np.ndarray):
    """A numpy array with a type given by cls.inner_type"""

    @classmethod
    def make_tuple(cls, v):
        """Converts a nested list of lists into a list of tuples."""
        return (
            tuple(cls.make_tuple(x) for x in v)
            if isinstance(v, list)
            else cls.inner_type(v)  # pylint:disable=no-member
        )

    @classmethod
    def __get_validators__(cls):
        """boilerplate"""
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        """validator"""
        # need to fix, doesnt work for simulationdata_export and load?

        if isinstance(val, np.ndarray):
            val_ndims = len(val.shape)
            cls_ndims = cls.ndims  # pylint:disable=no-member

            if (cls_ndims is not None) and (cls_ndims != val_ndims):
                raise ValidationError(
                    "wrong number of dimensions given. " f"Given {val_ndims}, expected {cls_ndims}."
                )
            return cls.make_tuple(val.tolist())

        return tuple(val)

    @classmethod
    def __modify_schema__(cls, field_schema):
        """Sets the schema of ArrayLike."""

        field_schema.update(
            dict(
                title="Array Like",
                description="Accepts sequence (tuple, list, numpy array) and converts to tuple.",
                type="tuple",
                properties={},
                required=[],
            )
        )


class ArrayLikeMeta(type):
    """metclass for Array, enables Array[type] -> TypedArray"""

    def __getitem__(cls, type_ndims):
        """Array[type, ndims] -> TypedArrayLike"""
        desired_type, ndims = type_ndims
        return type("Array", (TypedArrayLike,), {"inner_type": desired_type, "ndims": ndims})


class ArrayLike(np.ndarray, metaclass=ArrayLikeMeta):
    """type of numpy array with annotated type (Array[float], Array[complex])"""


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
        """Defines which validator function to use for ComplexNumber."""
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
        """Sets the schema of ComplexNumber."""
        field_schema.update(ComplexNumber.schema())


""" Data """


class DataObject(pydantic.BaseModel):
    """An object used in tidy3d model that subclasses xr.DataArray."""

    data_dict: dict
    data_type: type

    @property
    def as_data_array(self):
        """return DataArray respresentation of this DataObject."""
        # is this used?
        return self.data_type.from_dict(self.data_dict)


""" geometric """

Size1D = pydantic.NonNegativeFloat
Size = Tuple[Size1D, Size1D, Size1D]
Coordinate = Tuple[float, float, float]
Coordinate2D = Tuple[float, float]
Bound = Tuple[Coordinate, Coordinate]
GridSize = Union[pydantic.PositiveFloat, Tuple[pydantic.PositiveFloat, ...]]
Axis = Literal[0, 1, 2]
Axis2D = Literal[0, 1]
Shapely = BaseGeometry
Vertices = Union[Tuple[Coordinate2D, ...], ArrayLike[float, 2]]

""" medium """

# Complex = Union[complex, ComplexNumber]
Complex = Union[tidycomplex, ComplexNumber]
PoleAndResidue = Tuple[Complex, Complex]

# PoleAndResidue = Tuple[Tuple[float, float], Tuple[float, float]]
FreqBoundMax = float
FreqBoundMin = float
FreqBound = Tuple[FreqBoundMin, FreqBoundMax]

""" sources """

Polarization = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
Direction = Literal["+", "-"]

""" monitors """

EMField = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
FieldType = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
FreqArray = Union[Tuple[float, ...], ArrayLike[float, 1]]
ObsGridArray = Union[Tuple[float, ...], ArrayLike[float, 1]]
RadVec = Literal["Ntheta", "Nphi", "Ltheta", "Lphi"]

""" plotting """

Ax = Axes
