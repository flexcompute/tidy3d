"""Defines Geometric objects with Medium properties."""
from typing import Union, Tuple, Optional
import pydantic

from .base import Tidy3dBaseModel
from .validators import validate_name_str
from .geometry import GeometryType
from .medium import MediumType, AbstractCustomMedium, Medium2D
from .types import Ax, TYPE_TAG_STR, Axis
from .viz import add_ax_if_none, equal_aspect
from .grid.grid import Coords
from ..constants import MICROMETER
from ..exceptions import SetupError


class AbstractStructure(Tidy3dBaseModel):
    """
    A basic structure object.
    """

    geometry: GeometryType = pydantic.Field(
        ...,
        title="Geometry",
        description="Defines geometric properties of the structure.",
        discriminator=TYPE_TAG_STR,
    )

    name: str = pydantic.Field(None, title="Name", description="Optional name for the structure.")

    _name_validator = validate_name_str()

    @equal_aspect
    @add_ax_if_none
    def plot(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **patch_kwargs
    ) -> Ax:
        """Plot structure's geometric cross section at single (x,y,z) coordinate.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2nf5c2fk>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        return self.geometry.plot(x=x, y=y, z=z, ax=ax, **patch_kwargs)


class Structure(AbstractStructure):
    """Defines a physical object that interacts with the electromagnetic fields.
    A :class:`Structure` is a combination of a material property (:class:`AbstractMedium`)
    and a :class:`Geometry`.

    Example
    -------
    >>> from tidy3d import Box, Medium
    >>> box = Box(center=(0,0,1), size=(2, 2, 2))
    >>> glass = Medium(permittivity=3.9)
    >>> struct = Structure(geometry=box, medium=glass, name='glass_box')
    """

    medium: MediumType = pydantic.Field(
        ...,
        title="Medium",
        description="Defines the electromagnetic properties of the structure's medium.",
        discriminator=TYPE_TAG_STR,
    )

    def eps_diagonal(self, frequency: float, coords: Coords) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
            The diagonal elements of the relative permittivity tensor evaluated at ``frequency``.
        """
        if isinstance(self.medium, AbstractCustomMedium):
            return self.medium.eps_diagonal_on_grid(frequency=frequency, coords=coords)
        return self.medium.eps_diagonal(frequency=frequency)

    @pydantic.validator("medium", always=True)
    def _check_2d_geometry(cls, val, values):
        """Medium2D is only consistent with certain geometry types"""
        if isinstance(val, Medium2D):
            # validate that the geometry is actually 2d
            geom = values.get("geometry")
            if not geom:
                raise SetupError(
                    "Can't validate 2D structure because its geometry did not pass validation."
                )
            _ = geom._normal_2dmaterial  # pylint: disable=protected-access
        return val

    def eps_comp(self, row: Axis, col: Axis, frequency: float, coords: Coords) -> complex:
        """Single component of the complex-valued permittivity tensor as a function of frequency.

        Parameters
        ----------
        row : int
            Component's row in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        col : int
            Component's column in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
           Element of the relative permittivity tensor evaluated at ``frequency``.
        """
        if isinstance(self.medium, AbstractCustomMedium):
            return self.medium.eps_comp_on_grid(
                row=row, col=col, frequency=frequency, coords=coords
            )
        return self.medium.eps_comp(row=row, col=col, frequency=frequency)


class MeshOverrideStructure(AbstractStructure):
    """Defines an object that is only used in the process of generating the mesh.
    A :class:`MeshOverrideStructure` is a combination of geometry :class:`Geometry`,
    grid size along x,y,z directions, and a boolean on whether the override
    will be enforced.

    Example
    -------
    >>> from tidy3d import Box
    >>> box = Box(center=(0,0,1), size=(2, 2, 2))
    >>> struct_override = MeshOverrideStructure(geometry=box, dl=(0.1,0.2,0.3), name='override_box')
    """

    dl: Tuple[
        Optional[pydantic.PositiveFloat],
        Optional[pydantic.PositiveFloat],
        Optional[pydantic.PositiveFloat],
    ] = pydantic.Field(
        ...,
        title="Grid Size",
        description="Grid size along x, y, z directions.",
        units=MICROMETER,
    )

    enforce: bool = pydantic.Field(
        False,
        title="Enforce grid size",
        description="If ``True``, enforce the grid size setup inside the structure "
        "even if the structure is inside a structure of smaller grid size. In the intersection "
        "region of multiple structures of ``enforce=True``, grid size is decided by "
        "the last added structure of ``enforce=True``.",
    )


StructureType = Union[Structure, MeshOverrideStructure]
