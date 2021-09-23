""" Defines specification for mode solver """

import pydantic

from .base import Tidy3dBaseModel


class Mode(Tidy3dBaseModel):
    """defines mode specification"""

    mode_index: pydantic.NonNegativeInt
