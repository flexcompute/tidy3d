"""Special types and validators used by adjoint plugin."""
from typing import Union, Callable
import pydantic as pd

# from jax._src.typing import ArrayLike

from jax.experimental.array import ArrayLike
from jax.interpreters.ad import JVPTracer

JaxFloat = Union[float, ArrayLike, JVPTracer]

"""
Note, currently set up for jax 0.3.x, which is the only installable version for windows.
for jax 0.4.x, need to use

# from jax._src.typing import ArrayLike
# JaxFloat = Union[float, ArrayLike]
"""

# pylint: disable= unused-argument
def sanitize_validator_fn(cls, val):
    """if val is an object (untracable) return 0.0."""
    # pylint:disable=unidiomatic-typecheck
    return 0.0 if type(val) == object else val


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
