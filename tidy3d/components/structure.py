"""Defines Geometric objects with Medium properties."""

from __future__ import annotations

import pathlib
from collections import defaultdict
from typing import Optional, Tuple, Union

import autograd.numpy as anp
import numpy as np
import pydantic.v1 as pydantic

from ..constants import MICROMETER
from ..exceptions import SetupError, Tidy3dError, Tidy3dImportError
from .autograd.derivative_utils import DerivativeInfo
from .autograd.types import AutogradFieldMap, Box
from .autograd.utils import get_static
from .base import Tidy3dBaseModel, skip_if_fields_missing
from .data.data_array import ScalarFieldDataArray
from .geometry.polyslab import PolySlab
from .geometry.utils import GeometryType, validate_no_transformed_polyslabs
from .grid.grid import Coords
from .medium import AbstractCustomMedium, CustomMedium, Medium2D, MediumType
from .monitor import FieldMonitor, PermittivityMonitor
from .types import TYPE_TAG_STR, Ax, Axis
from .validators import validate_name_str
from .viz import add_ax_if_none, equal_aspect

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

    background_permittivity: float = pydantic.Field(
        None,
        ge=1.0,
        title="Background Permittivity",
        description="Relative permittivity used for the background of this structure "
        "when performing shape optimization with autograd.",
    )

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

    Notes
    ------

        Structures can indeed be larger than the simulation domain in ``tidy3d``. In such cases, ``tidy3d`` will
        automatically truncate the geometry that goes beyond the domain boundaries. For best results, structures that
        intersect with absorbing boundaries or simulation edges should extend all the way through. In many such
        cases, an “infinite” size :class:`td.inf` can be used to define the size along that dimension.

    Example
    -------
    >>> from tidy3d import Box, Medium
    >>> box = Box(center=(0,0,1), size=(2, 2, 2))
    >>> glass = Medium(permittivity=3.9)
    >>> struct = Structure(geometry=box, medium=glass, name='glass_box')

    See Also
    --------

    **Notebooks:**

    * `Quickstart <../../notebooks/StartHere.html>`_: Usage in a basic simulation flow.
    * `First walkthrough <../../notebooks/Simulation.html>`_: Usage in a basic simulation flow.
    * `Visualizing geometries in Tidy3D <../../notebooks/VizSimulation.html>`_

    **Lectures:**

    * `Using FDTD to Compute a Transmission Spectrum <https://www.flexcompute.com/fdtd101/Lecture-2-Using-FDTD-to-Compute-a-Transmission-Spectrum/>`_

    **GUI:**

    * `Structures <https://www.flexcompute.com/tidy3d/learning-center/tidy3d-gui/Lecture-3-Structures/#presentation-slides>`_
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
    @skip_if_fields_missing(["geometry"])
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

    def _compatible_with(self, other: Structure) -> bool:
        """Whether these two structures are compatible."""
        # note: if the first condition fails, the second won't get triggered
        if not self.medium._compatible_with(other.medium) and self.geometry.intersects(
            other.geometry
        ):
            return False
        return True

    """ Begin autograd code."""

    @staticmethod
    def get_monitor_name(index: int, data_type: str) -> str:
        """Get the monitor name for either a field or permittivity monitor at given index."""

        monitor_name_map = dict(
            fld=f"adjoint_fld_{index}",
            eps=f"adjoint_eps_{index}",
        )

        if data_type not in monitor_name_map:
            raise KeyError(f"'data_type' must be in {monitor_name_map.keys()}")

        return monitor_name_map[data_type]

    def make_adjoint_monitors(
        self, freqs: list[float], index: int
    ) -> (FieldMonitor, PermittivityMonitor):
        """Generate the field and permittivity monitor for this structure."""

        box = self.geometry.bounding_box

        # we dont want these fields getting traced by autograd, otherwise it messes stuff up

        size = [get_static(x) for x in box.size]  # TODO: expand slightly?
        center = [get_static(x) for x in box.center]

        # polyslab only needs fields at the midpoint along axis
        if isinstance(self.geometry, PolySlab):
            size[self.geometry.axis] = 0

        mnt_fld = FieldMonitor(
            size=size,
            center=center,
            freqs=freqs,
            fields=("Ex", "Ey", "Ez"),
            name=self.get_monitor_name(index=index, data_type="fld"),
            colocate=False,
        )

        mnt_eps = PermittivityMonitor(
            size=size,
            center=center,
            freqs=freqs,
            name=self.get_monitor_name(index=index, data_type="eps"),
            colocate=False,
        )

        return mnt_fld, mnt_eps

    def compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute adjoint gradients given the forward and adjoint fields"""

        # generate a mapping from the 'medium', or 'geometry' tag to the list of fields for VJP
        structure_fields_map = defaultdict(list)
        for structure_path in derivative_info.paths:
            med_or_geo, *field_path = structure_path
            field_path = tuple(field_path)
            if med_or_geo not in ("geometry", "medium"):
                raise ValueError(
                    f"Something went wrong in the structure VJP calculation, "
                    f"got a 'structure_path: {structure_path}' with first element '{med_or_geo}', "
                    "which should be 'medium' or 'geometry. "
                    "If you encounter this error, please raise an issue on the tidy3d GitHub "
                    "repository so we can investigate."
                )
            structure_fields_map[med_or_geo].append(field_path)

        # loop through sub fields, compute VJPs, and store in the derivative map {path -> vjp_value}
        derivative_map = {}
        for med_or_geo, field_paths in structure_fields_map.items():
            # grab derivative values {field_name -> vjp_value}
            med_or_geo_field = self.medium if med_or_geo == "medium" else self.geometry
            info = derivative_info.updated_copy(paths=field_paths, deep=False)
            derivative_values_map = med_or_geo_field.compute_derivatives(derivative_info=info)

            # construct map of {field path -> derivative value}
            for field_path, derivative_value in derivative_values_map.items():
                path = tuple([med_or_geo] + list(field_path))
                derivative_map[path] = derivative_value

        return derivative_map

    """ End autograd code."""

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
            List of ``gdstk.Polygon``
        """

        polygons = self.geometry.to_gdstk(x=x, y=y, z=z, gds_layer=gds_layer, gds_dtype=gds_dtype)

        if isinstance(self.medium, AbstractCustomMedium):
            axis, _ = self.geometry.parse_xyz_kwargs(x=x, y=y, z=z)
            bb_min, bb_max = self.geometry.bounds

            # Set the contour scale to be the minimal cooridante step size w.r.t. the 3 main axes,
            # skipping those with a single coordniate. In case all axes have only a single coordinate,
            # use the largest bounding box dimension.
            eps, _, _ = self.medium.eps_dataarray_freq(frequency=frequency)
            scale = max(b - a for a, b in zip(bb_min, bb_max))
            for coord in (eps.x, eps.y, eps.z):
                if len(coord) > 1:
                    scale = min(scale, np.diff(coord).min())

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
            List of ``gdspy.Polygon`` and ``gdspy.PolygonSet``.
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
        pathlib.Path(fname).parent.mkdir(parents=True, exist_ok=True)
        library.write_gds(fname)

    @classmethod
    def from_permittivity_array(
        cls, geometry: GeometryType, eps_data: np.ndarray, **kwargs
    ) -> Structure:
        """Create ``Structure`` with ``geometry`` and ``CustomMedium`` containing ``eps_data`` for
        The ``permittivity`` field.   Extra keyword arguments are passed to ``td.Structure()``.
        """

        rmin, rmax = geometry.bounds

        if not isinstance(eps_data, (np.ndarray, Box, list, tuple)):
            raise ValueError("Must supply array-like object for 'eps_data'.")

        eps_data = anp.array(eps_data)
        shape = eps_data.shape

        if len(shape) != 3:
            raise ValueError(
                "'Structure.from_permittivity_array' method only accepts 'eps_data' with 3 dimensions, "
                f"corresponding to (x,y,z). Got array with {len(shape)} dimensions."
            )

        coords = {}
        for key, pt_min, pt_max, num_pts in zip("xyz", rmin, rmax, shape):
            if np.isinf(pt_min) and np.isinf(pt_max):
                pt_min = 0.0
                pt_max = 0.0

            coords_2x = np.linspace(pt_min, pt_max, 2 * num_pts + 1)
            coords_centers = coords_2x[1:-1:2]

            if len(coords_centers) != num_pts:
                raise ValueError(
                    "something went wrong, different number of coordinate values and data values. "
                    "Check your 'geometry', 'eps_data', and file a bug report."
                )

            # handle infinite size dimension edge case
            coords_centers = np.nan_to_num(coords_centers, 0.0)

            _, count = np.unique(coords_centers, return_counts=True)
            if np.any(count > 1):
                raise ValueError(
                    "Found duplicates in the coordinates constructed from the supplied "
                    "'geometry' and 'eps_data'. This is likely due to having a geometry with an "
                    "infinite size in one dimension and a 'eps_data' with a 'shape' > 1 in that "
                    "dimension. "
                )

            coords[key] = coords_centers

        eps_data_array = ScalarFieldDataArray(eps_data, coords=coords)
        custom_med = CustomMedium(permittivity=eps_data_array)

        return Structure(
            geometry=geometry,
            medium=custom_med,
            **kwargs,
        )


class MeshOverrideStructure(AbstractStructure):
    """Defines an object that is only used in the process of generating the mesh.

    Notes
    -----

        A :class:`MeshOverrideStructure` is a combination of geometry :class:`Geometry`,
        grid size along ``x``, ``y``, ``z`` directions, and a boolean on whether the override
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
