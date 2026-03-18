# type information for autograd

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, get_origin

import autograd.numpy as anp
from autograd.builtins import dict as TracedDict
from autograd.extend import Box, defvjp, primitive
from autograd.numpy.numpy_boxes import ArrayBox
from pydantic import BeforeValidator, PlainSerializer, PositiveFloat, TypeAdapter

from tidy3d.components.types import ArrayFloat2D, ArrayLike, Complex, Size1D
from tidy3d.components.types.base import _auto_serializer, _from_complex_dict
from tidy3d.components.types.utils import _add_schema

from .utils import get_static, hasbox

if TYPE_CHECKING:
    from typing import Optional

    from pydantic import SerializationInfo

    from tidy3d.compat import TypeAlias

# add schema to the Box
_add_schema(Box, title="AutogradBox", field_type_str="autograd.tracer.Box")
_add_schema(ArrayBox, title="AutogradArrayBox", field_type_str="autograd.numpy.ArrayBox")

# make sure Boxes in tidy3d properly define VJPs for copy operations, for computational graph
_copy = primitive(copy.copy)
_deepcopy = primitive(copy.deepcopy)

defvjp(_copy, lambda ans, x: lambda g: _copy(g))
defvjp(_deepcopy, lambda ans, x, memo: lambda g: _deepcopy(g, memo))

Box.__copy__ = lambda v: _copy(v)
Box.__deepcopy__ = lambda v, memo: _deepcopy(v, memo)
Box.__str__ = lambda self: f"{self._value} <{type(self).__name__}>"
Box.__repr__ = Box.__str__


def traced_alias(base_alias: Any, *, name: Optional[str] = None) -> TypeAlias:
    base_adapter = TypeAdapter(base_alias, config={"arbitrary_types_allowed": True})

    def _validate_box_or_container(v: Any) -> Any:
        # Normalize serialized complex arrays before Union validation so they are
        # treated as ndarrays rather than scalar complex values.
        complex_arr = _from_complex_dict(v)
        if isinstance(complex_arr, anp.ndarray) and complex_arr.ndim > 0:
            return base_adapter.validate_python(complex_arr)

        # If v itself is a tracer, validate its static value but leave the tracer untouched.
        if isinstance(v, Box):
            base_adapter.validate_python(get_static(v))
            return v

        # If v is a plain container that contains at least one tracer, try to coerce it
        # into an ArrayBox for one-shot validation, but return the original structure and
        # fall back to a structural walk if needed.
        if hasbox(v):
            # decide whether we must return an array
            origin = get_origin(base_alias)
            is_array_field = base_alias in (ArrayLike, ArrayFloat2D) or origin is None

            if is_array_field:
                dense = anp.array(v)
                base_adapter.validate_python(get_static(dense))
                return dense

            # otherwise it's a Python container type
            # try the fast-path array validation, but return the array so ops work
            try:
                dense = anp.array(v)
                base_adapter.validate_python(get_static(dense))
                return dense

            except Exception:
                # ragged/un-coercible -> rebuild container of Boxes
                if isinstance(v, tuple):
                    return tuple(_validate_box_or_container(x) for x in v)
                if isinstance(v, list):
                    return [_validate_box_or_container(x) for x in v]
                if isinstance(v, dict):
                    return {k: _validate_box_or_container(x) for k, x in v.items()}
                # fallback: can't handle this structure
                raise

        return base_adapter.validate_python(v)

    def _serialize_traced(a: Any, info: SerializationInfo) -> Any:
        return _auto_serializer(get_static(a), info)

    return Annotated[
        object,
        BeforeValidator(_validate_box_or_container),
        PlainSerializer(_serialize_traced, when_used="json"),
    ]


# "primitive" types that can use traced_alias
TracedArrayLike = traced_alias(ArrayLike)
TracedArrayFloat2D = traced_alias(ArrayFloat2D)
TracedFloat = traced_alias(float)
TracedPositiveFloat = traced_alias(PositiveFloat)
TracedComplex = traced_alias(Complex)
TracedSize1D = traced_alias(Size1D)

# derived traced types (these mirror the types in `components.types`)
TracedSize = tuple[TracedSize1D, TracedSize1D, TracedSize1D]
TracedCoordinate = tuple[TracedFloat, TracedFloat, TracedFloat]
TracedPoleAndResidue = tuple[TracedComplex, TracedComplex]
TracedPolesAndResidues = tuple[TracedPoleAndResidue, ...]

# The data type that we pass in and out of the web.run() @autograd.primitive
PathType = tuple[Union[int, str], ...]
AutogradFieldMap = TracedDict[PathType, Box]

InterpolationType = Literal["nearest", "linear"]

__all__ = [
    "AutogradFieldMap",
    "InterpolationType",
    "PathType",
    "TracedArrayFloat2D",
    "TracedArrayLike",
    "TracedComplex",
    "TracedCoordinate",
    "TracedDict",
    "TracedFloat",
    "TracedPoleAndResidue",
    "TracedPolesAndResidues",
    "TracedPositiveFloat",
    "TracedSize",
    "TracedSize1D",
]
