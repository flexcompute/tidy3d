"""Defines Geometric objects with Medium properties."""
from __future__ import annotations

from typing import Union, Tuple, Optional, Callable, Any
import pathlib
import pydantic.v1 as pydantic
import numpy as np

from .autograd import get_static
from .base import Tidy3dBaseModel, skip_if_fields_missing
from .validators import validate_name_str
from .geometry.utils import GeometryType, validate_no_transformed_polyslabs
from .medium import MediumType, AbstractCustomMedium, Medium2D
from .monitor import FieldMonitor, PermittivityMonitor
from .types import Ax, TYPE_TAG_STR, Axis, Bound
from .viz import add_ax_if_none, equal_aspect
from .grid.grid import Coords
from ..constants import MICROMETER
from ..exceptions import SetupError, Tidy3dError, Tidy3dImportError
from .data.monitor_data import PermittivityData, FieldData

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
        size = tuple(get_static(x) for x in box.size)  # TODO: expand slightly?
        center = tuple(get_static(x) for x in box.center)

        mnt_fld = FieldMonitor(
            size=size,
            center=center,
            freqs=freqs,
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

    @property
    def derivative_function_map(self) -> dict[tuple[str, str], Callable]:
        """Map path to the right derivative function function."""
        return {
            ("medium", "permittivity"): self.derivative_medium_permittivity,
            ("medium", "conductivity"): self.derivative_medium_conductivity,
            ("geometry", "size"): self.derivative_geometry_size,
            ("geometry", "center"): self.derivative_geometry_center,
        }

    def get_derivative_function(self, path: tuple[str, ...]) -> Callable:
        """Get the derivative function function."""

        derivative_map = self.derivative_function_map
        if path not in derivative_map:
            raise NotImplementedError(f"Can't compute derivative for structure field path: {path}.")
        return derivative_map[path]

    # def derivative_eps_complex_volume(self, E_der_map: FieldData) -> xr.DataArray:
    #     """Get the derivative w.r.t complex permittivity in the volume vs frequency."""

    #     vjp_value = 0.0
    #     for field_name in ("Ex", "Ey", "Ez"):
    #         fld = E_der_map.field_components[field_name]
    #         vjp_value_fld = self.integrate_within_bounds(
    #             arr=fld,
    #             dims=("x", "y", "z"),
    #             bounds=self.geometry.bounds,
    #         )
    #         vjp_value += vjp_value_fld

    #     return vjp_value

    # def derivative_eps_sigma_volume(
    #     self, E_der_map: FieldData
    # ) -> tuple[xr.DataArray, xr.DataArray]:
    #     """Get the derivative w.r.t permittivity and conductivity in the volume vs frequency."""

    #     vjp_eps_complex = self.derivative_eps_complex_volume(E_der_map=E_der_map)

    #     freqs = get_static(vjp_eps_complex.coords["f"].values)
    #     values = get_static(vjp_eps_complex.values)

    #     eps_vjp, sigma_vjp = self.medium.eps_complex_to_eps_sigma(eps_complex=values, freq=freqs)

    #     return eps_vjp, sigma_vjp

    # def derivative_medium_permittivity(
    #     self,
    #     E_der_map: FieldData,
    #     D_der_map: FieldData,
    #     eps_structure: PermittivityData,
    #     eps_sim: float,
    #     **kwargs,
    # ) -> float:
    #     """Compute the derivative for the medium.permittivity given forward and adjoint fields."""

    #     eps_vjp, _ = self.derivative_eps_sigma_volume(E_der_map=E_der_map)

    #     return npa.sum(eps_vjp)

    # def derivative_medium_conductivity(
    #     self,
    #     E_der_map: FieldData,
    #     D_der_map: FieldData,
    #     eps_structure: PermittivityData,
    #     eps_sim: float,
    #     **kwargs,
    # ) -> float:
    #     """Compute the derivative for the medium.conductivity given forward and adjoint fields."""

    #     _, sigma_vjp = self.derivative_eps_sigma_volume(E_der_map=E_der_map)

    #     return npa.sum(sigma_vjp)

    # def derivative_geometry_box_faces(
    #     self,
    #     E_der_map: FieldData,
    #     D_der_map: FieldData,
    #     eps_structure: PermittivityData,
    #     eps_sim: float,
    #     **kwargs,
    # ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    #     """Derivative with respect to positions of min and max faces of ``Box`` along all 3 dims."""

    #     # change in permittivity between inside and outside
    #     # TODO: assumes non-dispersive here and eps_sim, generalize
    #     eps_in = get_static(self.medium.permittivity)
    #     eps_out = eps_sim
    #     delta_eps = eps_in - eps_out
    #     delta_eps_inv = 1.0 / eps_in - 1.0 / eps_out

    #     vjp_faces = np.zeros((2, 3))

    #     for min_max_index, _ in enumerate((0, -1)):
    #         for axis, dim_normal in enumerate("xyz"):
    #             # get normal D and tangential Es
    #             fld_normal, flds_tangential = self.geometry.pop_axis(("Ex", "Ey", "Ez"), axis=axis)
    #             D_normal = D_der_map.field_components[fld_normal]
    #             Es_tangential = [E_der_map.field_components[key] for key in flds_tangential]

    #             # evaluate all fields at the face location in this min/max and axis
    #             bounds_normal, bounds_tangential = self.geometry.pop_axis(
    #                 np.array(self.geometry.bounds).T, axis=axis
    #             )
    #             bounds_tangential = np.array(bounds_tangential).T
    #             pos_normal = get_static(
    #                 bounds_normal[min_max_index]
    #             )  # TODO: no need for autograd here
    #             interp_kwargs_face = {dim_normal: pos_normal}
    #             D_normal = D_normal.interp(**interp_kwargs_face)

    #             Es_tangential = [E.interp(**interp_kwargs_face) for E in Es_tangential]
    #             _, dims_tangential = self.geometry.pop_axis("xyz", axis=axis)

    #             # start recording VJP at this surface
    #             vjp_value_face = 0.0

    #             # compute normal contribution
    #             D_integrated = self.integrate_within_bounds(
    #                 arr=D_normal,
    #                 dims=dims_tangential,
    #                 bounds=bounds_tangential,
    #             )
    #             D_integrated = get_static(D_integrated.values)

    #             # D_integrated = get_static(D_normal.integrate(coord=dims_tangential).values)
    #             D_contribution = -delta_eps_inv * D_integrated
    #             vjp_value_face += D_contribution

    #             # compute tangential contributions
    #             for E_tangential in Es_tangential:
    #                 E_integrated = self.integrate_within_bounds(
    #                     arr=E_tangential,
    #                     dims=dims_tangential,
    #                     bounds=bounds_tangential,
    #                 )
    #                 E_integrated = get_static(E_integrated.values)
    #                 # E_integrated = get_static(E_tangential.integrate(coord=dims_tangential).values)

    #                 E_contribution = delta_eps * E_integrated
    #                 vjp_value_face += E_contribution

    #             # record vjp for this face
    #             vjp_faces[min_max_index, axis] = float(np.real(vjp_value_face).astype(float))

    #     return vjp_faces

    # def derivative_geometry_size(
    #     self,
    #     E_der_map: FieldData,
    #     D_der_map: FieldData,
    #     eps_structure: PermittivityData,
    #     eps_sim: float,
    # ) -> tuple[float, float, float]:
    #     """Derivative of self.geometry w.r.t. ``size``."""

    #     derivative_faces = self.derivative_geometry_box_faces(
    #         E_der_map=E_der_map,
    #         D_der_map=D_der_map,
    #         eps_structure=eps_structure,
    #         eps_sim=eps_sim,
    #     )

    #     (xmin, ymin, zmin), (xmax, ymax, zmax) = derivative_faces

    #     return (0.5 * (xmax + xmin), 0.5 * (ymax + ymin), 0.5 * (zmax + zmin))

    # def derivative_geometry_center(
    #     self,
    #     E_der_map: FieldData,
    #     D_der_map: FieldData,
    #     eps_structure: PermittivityData,
    #     eps_sim: float,
    # ) -> tuple[float, float, float]:
    #     """Derivative of self.geometry w.r.t. ``center``."""

    #     derivative_faces = self.derivative_geometry_box_faces(
    #         E_der_map=E_der_map,
    #         D_der_map=D_der_map,
    #         eps_structure=eps_structure,
    #         eps_sim=eps_sim,
    #     )

    #     (xmin, ymin, zmin), (xmax, ymax, zmax) = derivative_faces

    #     return (xmax - xmin, ymax - ymin, zmax - zmin)

    def compute_derivatives(
        self,
        structure_paths: list[tuple[str, ...]],
        E_der_map: FieldData,
        D_der_map: FieldData,
        eps_structure: PermittivityData,
        eps_sim: float,
        bounds: Bound,
    ) -> dict[tuple[str, ...], Any]:
        """Compute adjoint gradients given the forward and adjoint fields"""

        derivative_map = {}
        for structure_path in structure_paths:
            # TODO: maybe construct E_der and D_der and pass them to derivative functions?

            med_or_geo, field_name = structure_path

            if med_or_geo not in ("geometry", "medium"):
                raise ValueError(
                    f"Something went wrong in the structure VJP calculation, "
                    f"got a 'structure_path: {structure_path}' with first element '{med_or_geo}', "
                    "which should be 'medium' or 'geometry. "
                    "If you encounter this error, please raise an issue on the tidy3d GitHub "
                    "repository so we can investigate."
                )

            med_or_geo_field = self.medium if med_or_geo == "medium" else self.geometry

            derivative_value = med_or_geo_field.compute_derivative(
                field_name=field_name,  # TODO: consolidate and do these in one shot later
                E_der_map=E_der_map,
                D_der_map=D_der_map,
                eps_structure=eps_structure,
                eps_sim=eps_sim,
                bounds=bounds,  # TODO: intersecting with the sim maybe?
            )

            # old way
            # derivative_function = self.get_derivative_function(structure_path)
            # derivative_value = derivative_function(
            #     E_der_map=E_der_map,
            #     D_der_map=D_der_map,
            #     eps_structure=eps_structure,
            #     eps_sim=eps_sim,
            # )

            derivative_map[structure_path] = derivative_value

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
