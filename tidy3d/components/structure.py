""" defines Geometric objects with Medium properties """

from .base import Tidy3dBaseModel
from .geometry import GeometryType
from .medium import MediumType
from .types import Axis, AxesSubplot


class Structure(Tidy3dBaseModel):
    """An object that interacts with the electromagnetic fields"""

    geometry: GeometryType
    medium: MediumType

    def plot(self, position: float, axis: Axis, ax=None) -> AxesSubplot:
        """plot geometry"""
        return self.geometry.plot(position=position, axis=axis, ax=ax)
