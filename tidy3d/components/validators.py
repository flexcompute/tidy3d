""" Defines various validation functions that get used to ensure inputs are legit """

import pydantic

from ..log import ValidationError


def assert_plane():
    """makes sure a field's `size` attribute has exactly 1 zero"""

    @pydantic.validator("size", allow_reuse=True, always=True)
    def is_plane(cls, val):
        if val.count(0.0) != 1:
            raise ValidationError(f"'{cls.__name__}' object must be planar, given size={val}")
        return val

    return is_plane
