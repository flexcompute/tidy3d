"""Special types and validators used by adjoint plugin."""

from __future__ import annotations

from typing import Annotated, Any, Union

import numpy as np
from pydantic import PlainSerializer, SerializationInfo
from pydantic_core import core_schema

from tidy3d.components.type_util import _add_schema

# special handling if we cant import the JVPTracer in the future (so it doesn't break tidy3d).
try:
    from jax.interpreters.ad import JVPTracer
except ImportError:
    import tidy3d as td

    td.log.warning(
        "Could not import 'jax.interpreters.ad.JVPTracer'. "
        "As a temporary fix, 'jax'-traced floats will use 'typing.Any' in their type annotation. "
        "If you encounter this warning, please file an issue on the Tidy3D front end repository "
        "as it indicates that the 'adjoint' plugin will need to be upgraded."
    )
    JVPTracer = Any

from jax.numpy import ndarray as JaxArrayType

""" Define schema for these jax and numpy types."""


def dump_array_like(v: Any, info: SerializationInfo) -> Any:
    """
    JSON-safe serializer for:
    * NumPy ndarrays (real or complex)
    * JAX arrays / tracers (they all expose .tolist() / .item())
    * scalars or anything else you pass in
    """
    if info.mode != "json":
        return v

    if isinstance(v, np.ndarray):
        if np.iscomplexobj(v):
            return {"real": v.real.tolist(), "imag": v.imag.tolist()}
        return v.tolist()

    if hasattr(v, "tolist"):
        return v.tolist()
    if hasattr(v, "item"):
        return v.item()

    return v


class NumpyArrayType(np.ndarray):
    """Subclass of ``np.ndarray`` with a schema defined for pydantic."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return core_schema.no_info_plain_validator_function(np.asarray)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return {
            "title": "npdarray",
            "type": "numpy.ndarray",
            "items": {},
        }


_add_schema(JaxArrayType, title="JaxArray", field_type_str="jax.numpy.ndarray")

# if the ImportError didnt occur, add the schema
if JVPTracer is not Any:
    _add_schema(JVPTracer, title="JVPTracer", field_type_str="jax.interpreters.ad.JVPTracer")

# define types usable as floats including the jax tracers
JaxArrayLike = Annotated[
    Union[JaxArrayType, NumpyArrayType],
    # JaxArrayType,
    PlainSerializer(dump_array_like, when_used="json"),
]
JaxFloat = Annotated[
    Union[float, JaxArrayLike, JVPTracer, object],
    PlainSerializer(dump_array_like, when_used="json"),
]
# note: object is included here because sometimes jax passes just `object` (i think when untraced)
