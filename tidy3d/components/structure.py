""" defines Geometric objects with Medium properties """

from .base import Tidy3dBaseModel
from .geometry import GeometryType
from .medium import MediumType


class Structure(Tidy3dBaseModel):
    """An object that interacts with the electromagnetic fields"""

    geometry: GeometryType
    medium: MediumType
