"""Defines 'types' that various fields can be"""

from typing import (
    Literal,  # We support py3.9+, so direct typing import is fine.
    Optional,
    Tuple,
    Union,
)

import autograd.numpy as np
import pydantic.v1 as pydantic

try:
    from matplotlib.axes import Axes
except ImportError:
    Axes = None
from shapely.geometry.base import BaseGeometry
from typing_extensions import Annotated

from ..exceptions import ValidationError

# type tag default name
TYPE_TAG_STR = "type"


def annotate_type(UnionType):
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
        yield cls.assert_non_null

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
        return arr_numpy

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
    def assert_non_null(cls, val):
        """Make sure array is not None."""
        if np.any(np.isnan(val)):
            raise ValidationError("'ArrayLike' field contained None or nan values.")
        return val

    @classmethod
    def __modify_schema__(cls, field_schema):
        """Sets the schema of DataArray object."""

        schema = dict(
            type="ArrayLike",
        )
        field_schema.update(schema)


def constrained_array(
    dtype: type = None, ndim: int = None, shape: Tuple[pydantic.NonNegativeInt, ...] = None
) -> type:
    """Generate an ArrayLike sub-type with constraints built in."""

    # note, a unique name is required for each subclass of ArrayLike with constraints
    type_name = "ArrayLike"

    meta_args = []
    if dtype is not None:
        meta_args.append(f"dtype={dtype.__name__}")
    if ndim is not None:
        meta_args.append(f"ndim={ndim}")
    if shape is not None:
        meta_args.append(f"shape={shape}")
    type_name += "[" + ", ".join(meta_args) + "]"

    return type(type_name, (ArrayLike,), dict(dtype=dtype, ndim=ndim, shape=shape))


# pre-define a set of commonly used array like instances for import and use in type hints
ArrayInt1D = constrained_array(dtype=int, ndim=1)
ArrayFloat1D = constrained_array(dtype=float, ndim=1)
ArrayFloat2D = constrained_array(dtype=float, ndim=2)
ArrayFloat3D = constrained_array(dtype=float, ndim=3)
ArrayFloat4D = constrained_array(dtype=float, ndim=4)
ArrayComplex1D = constrained_array(dtype=complex, ndim=1)
ArrayComplex2D = constrained_array(dtype=complex, ndim=2)
ArrayComplex3D = constrained_array(dtype=complex, ndim=3)
ArrayComplex4D = constrained_array(dtype=complex, ndim=4)

TensorReal = constrained_array(dtype=float, ndim=2, shape=(3, 3))
MatrixReal4x4 = constrained_array(dtype=float, ndim=2, shape=(4, 4))

""" Complex Values """


class ComplexNumber(pydantic.BaseModel):
    """Complex number with a well defined schema."""

    real: float
    imag: float

    @property
    def as_complex(self):
        """return complex representation of ComplexNumber."""
        return self.real + 1j * self.imag


class tidycomplex(complex):
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
ScalarSymmetry = Literal[0, 1]

""" geometric """

Size1D = pydantic.NonNegativeFloat
Size = Tuple[Size1D, Size1D, Size1D]
Coordinate = Tuple[float, float, float]
CoordinateOptional = Tuple[Optional[float], Optional[float], Optional[float]]
Coordinate2D = Tuple[float, float]
Bound = Tuple[Coordinate, Coordinate]
GridSize = Union[pydantic.PositiveFloat, Tuple[pydantic.PositiveFloat, ...]]
Axis = Literal[0, 1, 2]
Axis2D = Literal[0, 1]
Shapely = BaseGeometry
PlanePosition = Literal["bottom", "middle", "top"]
ClipOperationType = Literal["union", "intersection", "difference", "symmetric_difference"]
BoxSurface = Literal["x-", "x+", "y-", "y+", "z-", "z+"]
LengthUnit = Literal["nm", "μm", "um", "mm", "cm", "m"]

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

PermittivityComponent = Literal["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]

""" sources """

Polarization = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
Direction = Literal["+", "-"]

""" monitors """

EMField = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
FieldType = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
FreqArray = Union[Tuple[float, ...], ArrayFloat1D]
ObsGridArray = Union[Tuple[float, ...], ArrayFloat1D]
PolarizationBasis = Literal["linear", "circular"]
AuxField = Literal["Nfx", "Nfy", "Nfz"]

""" plotting """

Ax = Axes
PlotVal = Literal["real", "imag", "abs"]
FieldVal = Literal["real", "imag", "abs", "abs^2", "phase"]
RealFieldVal = Literal["real", "abs", "abs^2"]
PlotScale = Literal["lin", "dB"]
ColormapType = Literal["divergent", "sequential", "cyclic"]

""" mode solver """

ModeSolverType = Literal["tensorial", "diagonal"]
EpsSpecType = Literal["diagonal", "tensorial_real", "tensorial_complex"]

""" mode tracking """

TrackFreq = Literal["central", "lowest", "highest"]

""" lumped elements"""

LumpDistType = Literal["off", "laterally_only", "on"]

""" dataset """

xyz = Literal["x", "y", "z"]
UnitsZBF = Literal["mm", "cm", "in", "m"]
