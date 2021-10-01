""" defines Geometric objects with Medium properties """

from typing import List, Tuple

from .base import Tidy3dBaseModel
from .geometry import GeometryType
from .medium import MediumType
from .types import Axis, AxesSubplot


class Structure(Tidy3dBaseModel):
    """An object that interacts with the electromagnetic fields"""

    geometry: GeometryType
    medium: MediumType

    def plot(
        self, position: float, axis: Axis, freqs: List[float] = None, ax=None
    ) -> Tuple[AxesSubplot, AxesSubplot]:
        """plot geometry with inset of medium"""
        ax = self.geometry.plot(position=position, axis=axis, ax=ax)
        if freqs is None:
            return ax, None
        axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
        axins = self.medium.plot(freqs=freqs, ax=axins)
        return ax, axins
