"""Special types and validators used by adjoint plugin."""
from typing import Union

import numpy as np
from jax.interpreters.ad import JVPTracer
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


def _add_schema(arbitrary_type: type, title: str, field_type_str: str) -> None:
    """Adds a schema to the ``arbitrary_type`` class without subclassing."""

    @classmethod
    def mod_schema_fn(cls, field_schema: dict) -> None:
        """Function that gets set to ``arbitrary_type.__modify_schema__``."""
        field_schema.update(dict(title=title, type=field_type_str))

    arbitrary_type.__modify_schema__ = mod_schema_fn


_add_schema(JaxArrayType, title="JaxArray", field_type_str="jax.numpy.ndarray")
_add_schema(JVPTracer, title="JVPTracer", field_type_str="jax.interpreters.ad.JVPTracer")

# define types usable as floats including the jax tracers
JaxArrayLike = Union[NumpyArrayType, JaxArrayType]
JaxFloat = Union[float, JaxArrayLike, JVPTracer, object]
# note: object is included here because sometimes jax passes just `object` (i think when untraced)
