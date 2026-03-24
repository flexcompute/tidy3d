"""Defines 'types' that various fields can be"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, Union

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PlainValidator,
    PositiveFloat,
)
from pydantic.functional_serializers import PlainSerializer
from pydantic.json_schema import WithJsonSchema

from tidy3d.exceptions import format_chained_exception_message

if TYPE_CHECKING:
    from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    # At runtime, Axes is just Any to avoid importing matplotlib
    Axes = None

from shapely.geometry.base import BaseGeometry

# type tag default name
TYPE_TAG_STR = "type"


def discriminated_union(union: type, discriminator: str = TYPE_TAG_STR) -> type:
    return Annotated[union, Field(discriminator=discriminator)]


""" Numpy Arrays """


def _dtype2python(value: Any) -> Any:
    """Converts numpy scalar types to their python equivalents."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.complexfloating):
        return complex(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _from_complex_dict(v: Any) -> Any:
    if isinstance(v, dict) and "real" in v and "imag" in v:
        return np.asarray(v["real"]) + 1j * np.asarray(v["imag"])
    return v


def _auto_serializer(a: Any, _: Any) -> Any:
    """Serializes numpy arrays and scalars for JSON."""
    if isinstance(a, complex) or (
        hasattr(np, "complexfloating") and isinstance(a, np.complexfloating)
    ):
        return {"real": float(a.real), "imag": float(a.imag)}
    if isinstance(a, np.ndarray):
        if np.iscomplexobj(a):
            return {"real": a.real.tolist(), "imag": a.imag.tolist()}
        else:
            return a.tolist()
    if isinstance(a, float) or (hasattr(np, "floating") and isinstance(a, np.floating)):
        return float(a)  # Ensure basic Python float
    if isinstance(a, int) or (hasattr(np, "integer") and isinstance(a, np.integer)):
        return int(a)  # Ensure basic Python int
    if hasattr(np, "number") and isinstance(a, np.number):
        return a.item()
    return a


DTypeLike = Annotated[np.dtype, PlainValidator(np.dtype), WithJsonSchema({"type": "np.dtype"})]


class ArrayConstraints(BaseModel):
    """Container for array constraints."""

    model_config = ConfigDict(frozen=True)

    dtype: Optional[DTypeLike] = None
    ndim: Optional[int] = None
    shape: Optional[tuple[int, ...]] = None
    forbid_nan: bool = True
    scalar_to_1d: bool = False
    strict: bool = False


def _coerce(v: Any, *, constraints: ArrayConstraints) -> NDArray:
    """Convert input to a NumPy array with constraints.

    Raises
    ------
    ValueError
        - If conversion to an array fails.
        - If the array ends up with dtype=object (unsupported element type).
        - If the number of dimensions or shape does not match the expectations.
        - If ``forbid_nan`` is ``True`` and the array contains NaN values.
    """
    if constraints.strict and np.isscalar(v):
        raise ValueError(
            f"strict mode: scalar value {type(v).__name__!r} cannot be coerced to a NumPy array. "
        )

    try:
        # constraints.dtype is already an np.dtype object or None
        arr = np.asarray(v) if constraints.dtype is None else np.asarray(v, dtype=constraints.dtype)
    except Exception as e:
        raise ValueError(
            format_chained_exception_message(
                f"cannot convert {type(v).__name__!r} to a NumPy array", e
            )
        ) from e

    if arr.dtype == np.dtype("object"):
        raise ValueError(f"unsupported element type {type(v).__name__!r} for array coercion")

    if (
        arr.ndim == 0
        and (constraints.ndim == 1 or constraints.ndim is None)
        and constraints.scalar_to_1d
    ):
        arr = arr.reshape(1)
    if constraints.ndim is not None and arr.ndim != constraints.ndim:
        raise ValueError(f"expected {constraints.ndim}-D, got {arr.ndim}-D")
    if constraints.shape is not None and tuple(arr.shape) != constraints.shape:
        raise ValueError(f"expected shape {constraints.shape}, got {tuple(arr.shape)}")
    if constraints.forbid_nan and np.any(np.isnan(arr)):
        raise ValueError("array contains NaN")

    # enforce immutability of our Pydantic models
    arr.flags.writeable = False

    return arr


def array_alias(
    *,
    dtype: Optional[Any] = None,
    ndim: Optional[int] = None,
    shape: Optional[tuple[int, ...]] = None,
    forbid_nan: bool = True,
    scalar_to_1d: bool = False,
    strict: bool = False,
) -> Any:
    constraints = ArrayConstraints(
        dtype=dtype,
        ndim=ndim,
        shape=shape,
        forbid_nan=forbid_nan,
        scalar_to_1d=scalar_to_1d,
        strict=strict,
    )
    serializer = PlainSerializer(_auto_serializer, when_used="json")

    base_schema = {
        "type": "ArrayLike",
        "x-array-dtype": getattr(constraints.dtype, "str", None),
        "x-array-ndim": constraints.ndim,
        "x-array-shape": constraints.shape,
        "x-array-forbid_nan": constraints.forbid_nan,
        "x-array-scalar_to_1d": constraints.scalar_to_1d,
        "x-array-strict": constraints.strict,
    }

    return Annotated[
        np.ndarray,
        BeforeValidator(_from_complex_dict),
        BeforeValidator(lambda v: _coerce(v, constraints=constraints)),
        serializer,
        WithJsonSchema(base_schema),
    ]


ArrayLike = array_alias()
ArrayLikeStrict = array_alias(strict=True)

ArrayInt1D = array_alias(dtype=int, ndim=1, scalar_to_1d=True)

ArrayFloat = array_alias(dtype=float)
ArrayFloat1D = array_alias(dtype=float, ndim=1, scalar_to_1d=True)
ArrayFloat2D = array_alias(dtype=float, ndim=2)
ArrayFloat3D = array_alias(dtype=float, ndim=3)
ArrayFloat4D = array_alias(dtype=float, ndim=4)

ArrayComplex = array_alias(dtype=complex)
ArrayComplex1D = array_alias(dtype=complex, ndim=1, scalar_to_1d=True)
ArrayComplex2D = array_alias(dtype=complex, ndim=2)
ArrayComplex3D = array_alias(dtype=complex, ndim=3)
ArrayComplex4D = array_alias(dtype=complex, ndim=4)

TensorReal = array_alias(dtype=float, ndim=2, shape=(3, 3))
MatrixReal4x4 = array_alias(dtype=float, ndim=2, shape=(4, 4))

""" Complex Values """


def _parse_complex(v: Any) -> complex:
    if isinstance(v, complex):
        return v

    if isinstance(v, dict) and "real" in v and "imag" in v:
        return complex(v["real"], v["imag"])

    if isinstance(v, numbers.Number):
        return complex(v)

    if hasattr(v, "__complex__"):
        try:
            return complex(v.__complex__())
        except Exception:
            pass

    if isinstance(v, (list, tuple)) and len(v) == 2:
        return complex(v[0], v[1])

    return v


_COMPLEX_JSON_SCHEMA_OBJECT = {
    "type": "object",
    "properties": {"real": {"type": "number"}, "imag": {"type": "number"}},
    "required": ["real", "imag"],
    "additionalProperties": False,
}

_COMPLEX_JSON_SCHEMA_VALIDATION = {
    "anyOf": [
        {"type": "number"},
        {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "prefixItems": [{"type": "number"}, {"type": "number"}],
        },
        _COMPLEX_JSON_SCHEMA_OBJECT,
    ]
}


Complex = Annotated[
    complex,
    BeforeValidator(_parse_complex),
    PlainSerializer(
        lambda z, _: {"real": z.real, "imag": z.imag},
        when_used="json",
        return_type=dict,
    ),
    WithJsonSchema(_COMPLEX_JSON_SCHEMA_VALIDATION, mode="validation"),
    WithJsonSchema(_COMPLEX_JSON_SCHEMA_OBJECT, mode="serialization"),
]

""" symmetry """

Symmetry = Annotated[Literal[0, -1, 1], BeforeValidator(_dtype2python)]
ScalarSymmetry = Annotated[Literal[0, 1], BeforeValidator(_dtype2python)]

""" geometric """

Size1D = NonNegativeFloat
Size = tuple[Size1D, Size1D, Size1D]
Coordinate = tuple[float, float, float]
CoordinateOptional = tuple[Optional[float], Optional[float], Optional[float]]
Coordinate2D = tuple[float, float]
Bound = tuple[Coordinate, Coordinate]
GridSize = Union[PositiveFloat, tuple[PositiveFloat, ...]]
Axis = Annotated[Literal[0, 1, 2], BeforeValidator(_dtype2python)]
Axis2D = Annotated[Literal[0, 1], BeforeValidator(_dtype2python)]
Shapely = BaseGeometry
PlanePosition = Literal["bottom", "middle", "top"]
ClipOperationType = Literal["union", "intersection", "difference", "symmetric_difference"]
BoxSurface = Literal["x-", "x+", "y-", "y+", "z-", "z+"]
LengthUnit = Literal["nm", "μm", "um", "mm", "cm", "m", "mil", "in"]
PriorityMode = Literal["equal", "conductor"]

""" medium """

# custom medium
InterpMethod = Literal["nearest", "linear"]

PoleAndResidue = tuple[Complex, Complex]
PolesAndResidues = tuple[PoleAndResidue, ...]
FreqBoundMax = float
FreqBoundMin = float
FreqBound = tuple[FreqBoundMin, FreqBoundMax]

PermittivityComponent = Literal["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]

""" sources """

Polarization = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
Direction = Literal["+", "-"]

""" monitors """


DiffractionPolarization = Literal["s", "p"]
EMField = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
EMSurfaceField = Literal["E", "H"]
FieldType = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
FreqArray = ArrayFloat1D
ObsGridArray = FreqArray
PolarizationBasis = Literal["linear", "circular"]
AuxField = Literal["Nfx", "Nfy", "Nfz"]

""" plotting """

Ax = Axes
PlotVal = Literal["real", "imag", "abs"]
FieldVal = Literal["real", "imag", "abs", "abs^2", "phase"]
RealFieldVal = Literal["real", "abs", "abs^2"]
PlotScale = Literal["lin", "dB", "log", "symlog"]
ColormapType = Literal["divergent", "sequential", "cyclic"]

""" mode solver """

ModeSolverType = Literal["tensorial", "diagonal"]
EpsSpecType = Literal["diagonal", "tensorial_real", "tensorial_complex"]
ModeClassification = Literal["TEM", "quasi-TEM", "TE", "TM", "Hybrid"]

""" mode tracking """

TrackFreq = Literal["central", "lowest", "highest"]

""" lumped elements"""

LumpDistType = Literal["off", "laterally_only", "on"]

""" dataset """

xyz = Literal["x", "y", "z"]
UnitsZBF = Literal["mm", "cm", "in", "m"]

""" sentinel """
Undefined = object()
