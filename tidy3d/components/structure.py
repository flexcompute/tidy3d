""" defines Geometric objects with Medium properties """

from .base import Tidy3dBaseModel
from .geometry import GeometryType
from .medium import MediumType
from .types import Ax
from .viz import add_ax_if_none


class Structure(Tidy3dBaseModel):
    """An object that interacts with the electromagnetic fields"""

    geometry: GeometryType
    medium: MediumType

    @add_ax_if_none
    def plot(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """plot geometry"""
        return self.geometry.plot(ax=ax, x=x, y=y, z=z, **kwargs)
