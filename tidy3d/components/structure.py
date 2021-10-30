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
    name: str = None

    @add_ax_if_none
    def plot(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **patch_kwargs
    ) -> Ax:
        """Plot structure geometry cross section at single (x,y,z) coordinate.

        Parameters
        ----------
        x : ``float``, optional
            Position of plane in x direction.
        y : ``float``, optional
            Position of plane in y direction.
        z : ``float``, optional
            Position of plane in z direction.
        ax : ``matplotlib.axes._subplots.Axes``, optional
            matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to ``add_artist(patch, **patch_kwargs)``.

        Returns
        -------
        ``matplotlib.axes._subplots.Axes``
            The supplied or created matplotlib axes.
        """
        return self.geometry.plot(x=x, y=y, z=z, ax=ax, **patch_kwargs)
