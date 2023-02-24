"""Special types and validators used by adjoint plugin."""
from typing import Union, Callable
import pydantic as pd

import numpy as np
from jax.interpreters.ad import JVPTracer
from jax.numpy import DeviceArray

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

    # pylint: disable=unused-argument
    @classmethod
    def mod_schema_fn(cls, field_schema: dict) -> None:
        """Function that gets set to ``arbitrary_type.__modify_schema__``."""
        field_schema.update(dict(title=title, type=field_type_str))

    arbitrary_type.__modify_schema__ = mod_schema_fn


_add_schema(DeviceArray, title="DeviceArray", field_type_str="jaxlib.xla_extension.DeviceArray")
_add_schema(JVPTracer, title="JVPTracer", field_type_str="jax.interpreters.ad.JVPTracer")

# define types usable as floats including the jax tracers
JaxArrayLike = Union[NumpyArrayType, DeviceArray]
JaxFloat = Union[float, JaxArrayLike, JVPTracer]

"""
Note, currently set up for jax 0.3.x, which is the only installable version for windows.
To get Array like in 0.3:
# from jax.experimental.array import ArrayLike

for jax 0.4.x, need to use
# from jax._src.typing import ArrayLike

and can make JaxFloat like
# JaxFloat = Union[float, ArrayLike]
"""

# pylint: disable= unused-argument
def sanitize_validator_fn(cls, val):
    """if val is an object (untracable) return 0.0."""
    # pylint:disable=unidiomatic-typecheck
    if type(val) == object:
        return 0.0
    return val


def validate_jax_float(field_name: str) -> Callable:
    """Return validator that ignores any `class object` types that will break pipeline."""
    return pd.validator(field_name, pre=True, allow_reuse=True)(sanitize_validator_fn)


def validate_jax_tuple(field_name: str) -> Callable:
    """Return validator that ignores any `class object` types in a tuple."""
    return pd.validator(field_name, pre=True, allow_reuse=True, each_item=True)(
        sanitize_validator_fn
    )


def validate_jax_tuple_tuple(field_name: str) -> Callable:
    """Return validator that ignores any `class object` types in a tuple of tuples."""

    @pd.validator(field_name, pre=True, allow_reuse=True, each_item=True)
    def validator(cls, val):
        val_list = list(val)
        for i, value in enumerate(val_list):
            val_list[i] = sanitize_validator_fn(cls, value)
        return tuple(val_list)

    return validator
