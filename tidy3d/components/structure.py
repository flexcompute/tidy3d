"""Defines Geometric objects with Medium properties."""
from typing import Union, Tuple, Optional
import pydantic.v1 as pydantic
import numpy as np

from .base import Tidy3dBaseModel
from .validators import validate_name_str
from .geometry.utils import GeometryType, validate_no_transformed_polyslabs
from .medium import MediumType, AbstractCustomMedium, Medium2D
from .types import Ax, TYPE_TAG_STR, Axis
from .viz import add_ax_if_none, equal_aspect
from .grid.grid import Coords
from ..constants import MICROMETER
from ..exceptions import SetupError, Tidy3dError, Tidy3dImportError

try:
    gdstk_available = True
    import gdstk
except ImportError:
    gdstk_available = False

try:
    gdspy_available = True
    import gdspy
except ImportError:
    gdspy_available = False


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

    @pydantic.validator("geometry")
    def _transformed_slanted_polyslabs_not_allowed(cls, val):
        """Prevents the creation of slanted polyslabs rotated out of plane."""
        validate_no_transformed_polyslabs(val)
        return val

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
        geom = values.get("geometry")

        if isinstance(val, Medium2D):
            # the geometry needs to be supported by 2d materials
            if not geom:
                raise SetupError(
                    "Found a 'Structure' with a 'Medium2D' medium, "
                    "but the geometry already did not pass validation."
                )
            # _normal_2dmaterial checks that the geometry is supported by 2d materials
            # and gives helpful error messages depending on the geometry details
            # if the geometry is not supported / not 2d
            _ = geom._normal_2dmaterial

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

    def to_gdstk(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        permittivity_threshold: pydantic.NonNegativeFloat = 1,
        frequency: pydantic.PositiveFloat = 0,
        gds_layer: pydantic.NonNegativeInt = 0,
        gds_dtype: pydantic.NonNegativeInt = 0,
    ) -> None:
        """Convert a structure's planar slice to a .gds type polygon.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        permittivity_threshold : float = 1.1
            Permitivitty value used to define the shape boundaries for structures with custom
            medim
        frequency : float = 0
            Frequency for permittivity evaluaiton in case of custom medium (Hz).
        gds_layer : int = 0
            Layer index to use for the shapes stored in the .gds file.
        gds_dtype : int = 0
            Data-type index to use for the shapes stored in the .gds file.

        Return
        ------
        List
            List of `gdstk.Polygon`
        """

        polygons = self.geometry.to_gdstk(x=x, y=y, z=z, gds_layer=gds_layer, gds_dtype=gds_dtype)

        if isinstance(self.medium, AbstractCustomMedium):
            axis, _ = self.geometry.parse_xyz_kwargs(x=x, y=y, z=z)

            eps, _, _ = self.medium.eps_dataarray_freq(frequency=frequency)
            scale = min(np.diff(eps.x).min(), np.diff(eps.y).min(), np.diff(eps.z).min())

            bb_min, bb_max = self.geometry.bounds
            coords = Coords(
                x=np.arange(bb_min[0], bb_max[0] + scale * 0.9, scale) if x is None else x,
                y=np.arange(bb_min[1], bb_max[1] + scale * 0.9, scale) if y is None else y,
                z=np.arange(bb_min[2], bb_max[2] + scale * 0.9, scale) if z is None else z,
            )
            eps = self.medium.eps_diagonal_on_grid(frequency=frequency, coords=coords)
            eps = np.stack((eps[0].real, eps[1].real, eps[2].real), axis=3).max(axis=3).squeeze()
            contours = gdstk.contour(eps.T, permittivity_threshold, scale, precision=scale * 1e-3)

            _, (dx, dy) = self.geometry.pop_axis(bb_min, axis)
            for polygon in contours:
                polygon.translate(dx, dy)

            polygons = gdstk.boolean(polygons, contours, "and", layer=gds_layer, datatype=gds_dtype)

        return polygons

    def to_gdspy(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        gds_layer: pydantic.NonNegativeInt = 0,
        gds_dtype: pydantic.NonNegativeInt = 0,
    ) -> None:
        """Convert a structure's planar slice to a .gds type polygon.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        gds_layer : int = 0
            Layer index to use for the shapes stored in the .gds file.
        gds_dtype : int = 0
            Data-type index to use for the shapes stored in the .gds file.

        Return
        ------
        List
            List of `gdspy.Polygon` and `gdspy.PolygonSet`.
        """

        if isinstance(self.medium, AbstractCustomMedium):
            raise Tidy3dError(
                "Structures with custom medium are not supported by 'gdspy'. They can only be "
                "exported using 'to_gdstk'."
            )

        return self.geometry.to_gdspy(x=x, y=y, z=z, gds_layer=gds_layer, gds_dtype=gds_dtype)

    def to_gds(
        self,
        cell,
        x: float = None,
        y: float = None,
        z: float = None,
        permittivity_threshold: pydantic.NonNegativeFloat = 1,
        frequency: pydantic.PositiveFloat = 0,
        gds_layer: pydantic.NonNegativeInt = 0,
        gds_dtype: pydantic.NonNegativeInt = 0,
    ) -> None:
        """Append a structure's planar slice to a .gds cell.

        Parameters
        ----------
        cell : ``gdstk.Cell`` or ``gdspy.Cell``
            Cell object to which the generated polygons are added.
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        permittivity_threshold : float = 1.1
            Permitivitty value used to define the shape boundaries for structures with custom
            medim
        frequency : float = 0
            Frequency for permittivity evaluaiton in case of custom medium (Hz).
        gds_layer : int = 0
            Layer index to use for the shapes stored in the .gds file.
        gds_dtype : int = 0
            Data-type index to use for the shapes stored in the .gds file.
        """
        if gdstk_available and isinstance(cell, gdstk.Cell):
            polygons = self.to_gdstk(
                x=x,
                y=y,
                z=z,
                permittivity_threshold=permittivity_threshold,
                frequency=frequency,
                gds_layer=gds_layer,
                gds_dtype=gds_dtype,
            )
            if len(polygons) > 0:
                cell.add(*polygons)

        elif gdspy_available and isinstance(cell, gdspy.Cell):
            polygons = self.to_gdspy(x=x, y=y, z=z, gds_layer=gds_layer, gds_dtype=gds_dtype)
            if len(polygons) > 0:
                cell.add(polygons)

        elif "gdstk" in cell.__class__ and not gdstk_available:
            raise Tidy3dImportError(
                "Module 'gdstk' not found. It is required to export shapes to gdstk cells."
            )
        elif "gdspy" in cell.__class__ and not gdspy_available:
            raise Tidy3dImportError(
                "Module 'gdspy' not found. It is required to export shapes to gdspy cells."
            )
        else:
            raise Tidy3dError(
                "Argument 'cell' must be an instance of 'gdstk.Cell' or 'gdspy.Cell'."
            )

    def to_gds_file(
        self,
        fname: str,
        x: float = None,
        y: float = None,
        z: float = None,
        permittivity_threshold: pydantic.NonNegativeFloat = 1,
        frequency: pydantic.PositiveFloat = 0,
        gds_layer: pydantic.NonNegativeInt = 0,
        gds_dtype: pydantic.NonNegativeInt = 0,
        gds_cell_name: str = "MAIN",
    ) -> None:
        """Export a structure's planar slice to a .gds file.

        Parameters
        ----------
        fname : str
            Full path to the .gds file to save the :class:`Structure` slice to.
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        permittivity_threshold : float = 1.1
            Permitivitty value used to define the shape boundaries for structures with custom
            medim
        frequency : float = 0
            Frequency for permittivity evaluaiton in case of custom medium (Hz).
        gds_layer : int = 0
            Layer index to use for the shapes stored in the .gds file.
        gds_dtype : int = 0
            Data-type index to use for the shapes stored in the .gds file.
        gds_cell_name : str = 'MAIN'
            Name of the cell created in the .gds file to store the geometry.
        """
        if gdstk_available:
            library = gdstk.Library()
        elif gdspy_available:
            library = gdspy.GdsLibrary()
        else:
            raise Tidy3dImportError(
                "Python modules 'gdspy' and 'gdstk' not found. To export geometries to .gds "
                "files, please install one of those those modules."
            )
        cell = library.new_cell(gds_cell_name)
        self.to_gds(
            cell,
            x=x,
            y=y,
            z=z,
            permittivity_threshold=permittivity_threshold,
            frequency=frequency,
            gds_layer=gds_layer,
            gds_dtype=gds_dtype,
        )
        library.write_gds(fname)


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
