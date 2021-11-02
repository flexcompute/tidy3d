""" defines Geometric objects with Medium properties """

from .base import Tidy3dBaseModel
from .validators import validate_name_str
from .geometry import GeometryType, Box # pylint: disable=unused-import
from .medium import MediumType, Medium # pylint: disable=unused-import
from .types import Ax
from .viz import add_ax_if_none


class Structure(Tidy3dBaseModel):
    """Defines a physical object that interacts with the electromagnetic fields.
       A :class:`Structure` is a combination of a material property (:class:`AbstractMedium`)
       and a :class:`Geometry`.

    Parameters
    ----------
    geometry : :class:`Geometry`
        Defines spatial extent of the :class:`Structure`.
    medium : :class:`AbstractMedium`
        Defines the electromagnetic properties of the structure material.
    name : str = None
        Optional name for the structure, used for plotting and logging.

    Example
    -------
    >>> box = Box(center=(0,0,1), size=(2, 2, 2))
    >>> glass = Medium(permittivity=3.9)
    >>> struct = Structure(geometry=box, medium=glass, name='glass_box')
    """

    geometry: GeometryType
    medium: MediumType
    name: str = None

    _name_validator = validate_name_str()

    @add_ax_if_none
    def plot(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **patch_kwargs
    ) -> Ax:
        """Plot structure geometry cross section.
            Note: only one of x, y, or z must be specified to define cross section.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction.
        y : float = None
            Position of plane in y direction.
        z : float = None
            Position of plane in z direction.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        return self.geometry.plot(x=x, y=y, z=z, ax=ax, **patch_kwargs)
