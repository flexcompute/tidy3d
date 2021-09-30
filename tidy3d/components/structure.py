""" defines Geometric objects with Medium properties """

from .base import Tidy3dBaseModel
from .geometry import GeometryType
from .medium import MediumType
from .types import Axis, AxesSubplot


class Structure(Tidy3dBaseModel):
    """An object that interacts with the electromagnetic fields"""

    geometry: GeometryType
    medium: MediumType

    def plot(self, position: float, axis: Axis, facecolor=None, ax=None) -> AxesSubplot:
        """plot just plots self.geometry"""
        return self.geometry.plot(ax=ax, position=position, facecolor=facecolor, axis=axis)
