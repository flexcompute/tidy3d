""" defines Geometric objects with Medium properties """

from .base import Tidy3dBaseModel
from .geometry import GeometryType
from .medium import MediumType
from .types import Axis, AxesSubplot
from .viz import add_ax_if_none


class Structure(Tidy3dBaseModel):
    """An object that interacts with the electromagnetic fields"""

    geometry: GeometryType
    medium: MediumType

    @add_ax_if_none
    def plot(self, position: float, axis: Axis, ax=None, **plot_params: dict) -> AxesSubplot:
        """plot geometry"""
        return self.geometry.plot(position=position, axis=axis, ax=ax, **plot_params)
