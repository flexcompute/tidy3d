"""Special types and validators used by adjoint plugin."""

from typing import Any, Union

import numpy as np

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


class NumpyArrayType(np.ndarray):
    """Subclass of ``np.ndarray`` with a schema defined for pydantic."""

    @classmethod
    def __modify_schema__(cls, field_schema):
        """Sets the schema of np.ndarray object."""

        schema = dict(
            title="npdarray",
            type="numpy.ndarray",
        )
        field_schema.update(schema)


_add_schema(JaxArrayType, title="JaxArray", field_type_str="jax.numpy.ndarray")

# if the ImportError didnt occur, add the schema
if JVPTracer is not Any:
    _add_schema(JVPTracer, title="JVPTracer", field_type_str="jax.interpreters.ad.JVPTracer")

# define types usable as floats including the jax tracers
JaxArrayLike = Union[NumpyArrayType, JaxArrayType]
JaxFloat = Union[float, JaxArrayLike, JVPTracer, object]
# note: object is included here because sometimes jax passes just `object` (i think when untraced)
