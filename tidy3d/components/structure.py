""" defines Geometric objects with Medium properties """

from .base import Tidy3dBaseModel
from .geometry import GeometryType
from .medium import MediumType
from .types import Axis, AxesSubplot


class Structure(Tidy3dBaseModel):
    """An object that interacts with the electromagnetic fields"""

    geometry: GeometryType
    medium: MediumType

    def plot(  # pytest: disable=invalid-name
        self, position: float, axis: Axis, facecolor=None, ax: AxesSubplot = None
    ) -> AxesSubplot:
        """plot just plots self.geometry"""
        return self.geometry.plot(position=position, facecolor=facecolor, axis=axis, ax=ax)
