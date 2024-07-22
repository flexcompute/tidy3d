# type information for autograd

# utilities for working with autograd

import copy
import typing

import pydantic.v1 as pd
from autograd.builtins import dict as dict_ag
from autograd.extend import Box, defvjp, primitive

from tidy3d.components.type_util import _add_schema

from ..types import ArrayFloat2D, ArrayLike, Complex, Size1D

# add schema to the Box
_add_schema(Box, title="AutogradBox", field_type_str="autograd.tracer.Box")

# make sure Boxes in tidy3d properly define VJPs for copy operations, for computational graph
_copy = primitive(copy.copy)
_deepcopy = primitive(copy.deepcopy)

defvjp(_copy, lambda ans, x: lambda g: _copy(g))
defvjp(_deepcopy, lambda ans, x, memo: lambda g: _deepcopy(g, memo))

Box.__copy__ = lambda v: _copy(v)
Box.__deepcopy__ = lambda v, memo: _deepcopy(v, memo)

# Types for floats, or collections of floats that can also be autograd tracers
TracedFloat = typing.Union[float, Box]
TracedPositiveFloat = typing.Union[pd.PositiveFloat, Box]
TracedSize1D = typing.Union[Size1D, Box]
TracedSize = typing.Union[tuple[TracedSize1D, TracedSize1D, TracedSize1D], Box]
TracedCoordinate = typing.Union[tuple[TracedFloat, TracedFloat, TracedFloat], Box]
TracedVertices = typing.Union[ArrayFloat2D, Box]

# poles
TracedComplex = typing.Union[Complex, Box]
TracedPoleAndResidue = typing.Tuple[TracedComplex, TracedComplex]

# The data type that we pass in and out of the web.run() @autograd.primitive
AutogradTraced = typing.Union[Box, ArrayLike]
PathType = tuple[typing.Union[int, str], ...]
AutogradFieldMap = dict_ag[PathType, AutogradTraced]

InterpolationType = typing.Literal["nearest", "linear"]

__all__ = [
    "TracedFloat",
    "TracedSize1D",
    "TracedSize",
    "TracedCoordinate",
    "TracedVertices",
    "AutogradTraced",
    "AutogradFieldMap",
]
