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
from matplotlib.axes import Axes
from shapely.geometry.base import BaseGeometry
from ..exceptions import ValidationError

# type tag default name
TYPE_TAG_STR = "type"


def annotate_type(UnionType):  # pylint:disable=invalid-name
    """Annotated union type using TYPE_TAG_STR as discriminator."""
    return Annotated[UnionType, pydantic.Field(discriminator=TYPE_TAG_STR)]


""" Numpy Arrays """


def _totuple(arr: np.ndarray) -> tuple:
    """Convert a numpy array to a nested tuple."""
    if arr.ndim > 1:
        return tuple(_totuple(val) for val in arr)
    return tuple(arr)


# generic numpy array
Numpy = np.ndarray


class ArrayLike:
    """Type that stores a numpy array."""

    ndim = None
    dtype = None
    shape = None

    @classmethod
    def __get_validators__(cls):
        yield cls.load_complex
        yield cls.convert_to_numpy
        yield cls.check_dims
        yield cls.check_shape

    @classmethod
    def load_complex(cls, val):
        """Special handling to load a complex-valued np.ndarray saved to file."""
        if not isinstance(val, dict):
            return val
        if "real" not in val or "imag" not in val:
            raise ValueError("ArrayLike real and imaginary parts not stored properly.")
        arr_real = np.array(val["real"])
        arr_imag = np.array(val["imag"])
        return arr_real + 1j * arr_imag

    @classmethod
    def convert_to_numpy(cls, val):
        """Convert the value to np.ndarray and provide some casting."""
        arr_numpy = np.array(val, ndmin=1, dtype=cls.dtype, copy=True)
        arr_tidy3d = np.ndarray(shape=arr_numpy.shape, dtype=arr_numpy.dtype)
        arr_tidy3d[:] = arr_numpy
        return arr_tidy3d

    @classmethod
    def check_dims(cls, val):
        """Make sure the number of dimensions is correct."""
        if cls.ndim and val.ndim != cls.ndim:
            raise ValidationError(f"Expected {cls.ndim} dimensions for ArrayLike, got {val.ndim}.")
        return val

    @classmethod
    def check_shape(cls, val):
        """Make sure the shape is correct."""
        if cls.shape and val.shape != cls.shape:
            raise ValidationError(f"Expected shape {cls.shape} for ArrayLike, got {val.shape}.")
        return val

    @classmethod
    def __modify_schema__(cls, field_schema):
        """Sets the schema of DataArray object."""

        schema = dict(
            title="ArrayLike",
            type="ArrayLike",
        )
        field_schema.update(schema)


def constrained_array(
    dtype: type = None, ndim: int = None, shape: Tuple[pydantic.NonNegativeInt, ...] = None
) -> type:
    """Generate an ArrayLike sub-type with constraints built in."""

    # note, a unique name is required for each subclass of ArrayLike with constraints
    type_name = "ArrayLike"
    if dtype is not None:
        type_name += f"_dtype={dtype}"
    if ndim is not None:
        type_name += f"_ndim={ndim}"
    if shape is not None:
        type_name += f"_shape={shape}"
    return type(type_name, (ArrayLike,), dict(dtype=dtype, ndim=ndim, shape=shape))


# pre-define a set of commonly used array like instances for import and use in type hints
ArrayFloat1D = constrained_array(dtype=float, ndim=1)
ArrayFloat2D = constrained_array(dtype=float, ndim=2)
ArrayFloat3D = constrained_array(dtype=float, ndim=3)
ArrayFloat4D = constrained_array(dtype=float, ndim=4)
ArrayComplex1D = constrained_array(dtype=complex, ndim=1)
ArrayComplex2D = constrained_array(dtype=complex, ndim=2)
ArrayComplex3D = constrained_array(dtype=complex, ndim=3)
ArrayComplex4D = constrained_array(dtype=complex, ndim=4)

TensorReal = constrained_array(dtype=float, ndim=2, shape=(3, 3))

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


""" symmetry """

Symmetry = Literal[0, -1, 1]

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
Vertices = Union[Tuple[Coordinate2D, ...], ArrayFloat2D]
Vertices = constrained_array(ndim=2, dtype=float)
PlanePosition = Literal["bottom", "middle", "top"]

""" medium """

# custom medium
InterpMethod = Literal["nearest", "linear"]

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
FreqArray = Union[Tuple[float, ...], ArrayFloat1D]
ObsGridArray = Union[Tuple[float, ...], ArrayFloat1D]

""" plotting """

Ax = Axes
PlotVal = Literal["real", "imag", "abs"]
FieldVal = Literal["real", "imag", "abs", "abs^2", "phase"]
PlotScale = Literal["lin", "dB"]
ColormapType = Literal["divergent", "sequential", "cyclic"]

""" mode solver """

ModeSolverType = Literal["tensorial", "diagonal"]

""" mode tracking """

TrackFreq = Literal["central", "lowest", "highest"]
