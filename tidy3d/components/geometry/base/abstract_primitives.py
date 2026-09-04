"""Abstract and primitive geometry base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import autograd.numpy as np
import shapely
from pydantic import Field, NonNegativeFloat, field_validator

from tidy3d.components.autograd import TracedCoordinate, TracedFloat
from tidy3d.components.base import cached_property
from tidy3d.components.types import (  # noqa: TC
    Axis,
    Coordinate,  # noqa: TC
    MatrixReal4x4,  # noqa: TC
    PlanePosition,
)
from tidy3d.constants import LARGE_NUMBER, MICROMETER, RADIAN, fp_eps
from tidy3d.exceptions import (
    ValidationError,
)

from .core import Geometry

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tidy3d.components.types import Shapely


class Centered(Geometry, ABC):
    """Geometry with a well defined center."""

    center: TracedCoordinate = Field(
        (0.0, 0.0, 0.0),
        title="Center",
        description="Center of object in x, y, and z.",
        json_schema_extra={"units": MICROMETER},
    )

    @field_validator("center")
    @classmethod
    def _center_not_inf(cls, val: tuple[float, float, float]) -> tuple[float, float, float]:
        """Make sure center is not infinitiy."""
        if any(np.isinf(v) for v in val):
            raise ValidationError("center can not contain td.inf terms.")
        return val


class SimplePlaneIntersection(Geometry, ABC):
    """A geometry where intersections with an axis aligned plane may be computed efficiently."""

    def intersections_tilted_plane(
        self,
        normal: Coordinate,
        origin: Coordinate,
        to_2D: MatrixReal4x4,
        cleanup: bool = True,
        quad_segs: int | None = None,
        section_tolerance_2d: bool = False,
    ) -> list[Shapely]:
        """Return a list of shapely geometries at the plane specified by normal and origin.
        Checks special cases before relying on the complete computation.

        Parameters
        ----------
        normal : Coordinate
            Vector defining the normal direction to the plane.
        origin : Coordinate
            Vector defining the plane origin.
        to_2D : MatrixReal4x4
            Transformation matrix to apply to resulting shapes.
        cleanup : bool = True
            If True, removes extremely small features from each polygon's boundary.
        quad_segs : Optional[int] = None
            Number of segments used to discretize circular shapes. If ``None``, uses
            high-quality visualization settings.
        section_tolerance_2d : bool = False
            See :meth:`Geometry.intersections_tilted_plane`.

        Returns
        -------
        list[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

        # Check if normal is a special case, where the normal is aligned with an axis.
        if np.sum(np.isclose(normal, 0.0)) == 2:
            axis = np.argmax(np.abs(normal)).item()
            coord = "xyz"[axis]
            kwargs = {coord: origin[axis]}
            section = self.intersections_plane(
                cleanup=cleanup,
                quad_segs=quad_segs,
                section_tolerance_2d=section_tolerance_2d,
                **kwargs,
            )
            # Apply transformation in the plane by removing row and column
            to_2D_in_plane = np.delete(np.delete(to_2D, 2, 0), axis, 1)

            def transform(p_array: NDArray) -> NDArray:
                x_coord, y_coord = p_array.T
                x_transformed = (
                    to_2D_in_plane[0, 0] * x_coord
                    + to_2D_in_plane[0, 1] * y_coord
                    + to_2D_in_plane[0, 2]
                )
                y_transformed = (
                    to_2D_in_plane[1, 0] * x_coord
                    + to_2D_in_plane[1, 1] * y_coord
                    + to_2D_in_plane[1, 2]
                )
                return np.stack((x_transformed, y_transformed), axis=-1)

            transformed_section = shapely.transform(section, transformation=transform)
            return transformed_section
        # Otherwise compute the arbitrary intersection
        return self._do_intersections_tilted_plane(
            normal=normal, origin=origin, to_2D=to_2D, quad_segs=quad_segs
        )

    @abstractmethod
    def _do_intersections_tilted_plane(
        self,
        normal: Coordinate,
        origin: Coordinate,
        to_2D: MatrixReal4x4,
        quad_segs: int | None = None,
    ) -> list[Shapely]:
        """Return a list of shapely geometries at the plane specified by normal and origin.

        Parameters
        ----------
        normal : Coordinate
            Vector defining the normal direction to the plane.
        origin : Coordinate
            Vector defining the plane origin.
        to_2D : MatrixReal4x4
            Transformation matrix to apply to resulting shapes.
        quad_segs : Optional[int] = None
            Number of segments used to discretize circular shapes.

        Returns
        -------
        list[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """


class Planar(SimplePlaneIntersection, Geometry, ABC):
    """Geometry with one ``axis`` that is slab-like with thickness ``height``."""

    axis: Axis = Field(
        2,
        title="Axis",
        description="Specifies dimension of the planar axis (0,1,2) -> (x,y,z).",
    )

    sidewall_angle: TracedFloat = Field(
        0.0,
        title="Sidewall angle",
        description="Angle of the sidewall. "
        "``sidewall_angle=0`` (default) specifies a vertical wall; "
        "``0<sidewall_angle<np.pi/2`` specifies a shrinking cross section "
        "along the ``axis`` direction; "
        "and ``-np.pi/2<sidewall_angle<0`` specifies an expanding cross section "
        "along the ``axis`` direction.",
        json_schema_extra={"units": RADIAN},
    )

    reference_plane: PlanePosition = Field(
        "middle",
        title="Reference plane for cross section",
        description="The position of the plane where the supplied cross section are "
        "defined. The plane is perpendicular to the ``axis``. "
        "The plane is located at the ``bottom``, ``middle``, or ``top`` of the "
        "geometry with respect to the axis. "
        "E.g. if ``axis=1``, ``bottom`` refers to the negative side of the y-axis, and "
        "``top`` refers to the positive side of the y-axis.",
    )

    @field_validator("sidewall_angle")
    @classmethod
    def validate_angle(cls, val: float) -> float:
        lower_bound = -np.pi / 2
        upper_bound = np.pi / 2
        if (val <= lower_bound) or (val >= upper_bound):
            # u03C0 is unicode for pi
            raise ValidationError(f"Sidewall angle ({val}) must be between -π/2 and π/2 rad.")
        return val

    @property
    @abstractmethod
    def center_axis(self) -> float:
        """Gets the position of the center of the geometry in the out of plane dimension."""

    @property
    @abstractmethod
    def length_axis(self) -> float:
        """Gets the length of the geometry along the out of plane dimension."""

    @property
    def finite_length_axis(self) -> float:
        """Gets the length of the geometry along the out of plane dimension.
        If the length is td.inf, return ``LARGE_NUMBER``
        """
        return min(self.length_axis, LARGE_NUMBER)

    @property
    def reference_axis_pos(self) -> float:
        """Coordinate along the slab axis at the reference plane.

        Returns the axis coordinate corresponding to the selected
        reference_plane:
        - "bottom": lower bound of slab_bounds
        - "middle": center_axis
        - "top": upper bound of slab_bounds
        """
        if self.reference_plane == "bottom":
            return self.slab_bounds[0]
        if self.reference_plane == "top":
            return self.slab_bounds[1]
        # default to middle
        return self.center_axis

    def intersections_plane(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        cleanup: bool = True,
        quad_segs: int | None = None,
        section_tolerance_2d: bool = False,
    ) -> list[Shapely]:
        """Returns shapely geometry at plane specified by one non None value of x,y,z.

        Parameters
        ----------
        x : float
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        cleanup : bool = True
            If True, removes extremely small features from each polygon's boundary.
        quad_segs : Optional[int] = None
            Number of segments used to discretize circular shapes. If ``None``, uses
            high-quality visualization settings.
        section_tolerance_2d : bool = False
            See :meth:`Geometry.intersections_plane`.

        Returns
        -------
        list[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
        `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>``.
        """
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        use_2d_tolerance = (
            section_tolerance_2d
            and axis == self.axis
            and np.isclose(self.length_axis, 0.0, rtol=fp_eps, atol=fp_eps)
        )
        if not self.intersects_axis_position(axis, position, section_tolerance_2d=use_2d_tolerance):
            return []
        if use_2d_tolerance and np.isclose(
            position, self.reference_axis_pos, rtol=fp_eps, atol=fp_eps
        ):
            position = self.reference_axis_pos

        if axis == self.axis:
            return self._intersections_normal(position, quad_segs=quad_segs)
        return self._intersections_side(position, axis)

    @abstractmethod
    def _intersections_normal(self, z: float, quad_segs: int | None = None) -> list:
        """Find shapely geometries intersecting planar geometry with axis normal to slab.

        Parameters
        ----------
        z : float
            Position along the axis normal to slab
        quad_segs : Optional[int] = None
            Number of segments used to discretize circular shapes. If ``None``, uses
            high-quality visualization settings.

        Returns
        -------
        list[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

    @abstractmethod
    def _intersections_side(self, position: float, axis: Axis) -> list[Shapely]:
        """Find shapely geometries intersecting planar geometry with axis orthogonal to plane.

        Parameters
        ----------
        position : float
            Position along axis.
        axis : int
            Integer index into 'xyz' (0,1,2).

        Returns
        -------
        list[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

    def _order_axis(self, axis: int) -> int:
        """Order the axis as if self.axis is along z-direction.

        Parameters
        ----------
        axis : int
            Integer index into the structure's planar axis.

        Returns
        -------
        int
            New index of axis.
        """
        axis_index = [0, 1]
        axis_index.insert(self.axis, 2)
        return axis_index[axis]

    def _order_by_axis(self, plane_val: Any, axis_val: Any, axis: int) -> tuple[Any, Any]:
        """Orders a value in the plane and value along axis in correct (x,y) order for plotting.
           Note: sometimes if axis=1 and we compute cross section values orthogonal to axis,
           they can either be x or y in the plots.
           This function allows one to figure out the ordering.

        Parameters
        ----------
        plane_val : Any
            The value in the planar coordinate.
        axis_val : Any
            The value in the ``axis`` coordinate.
        axis : int
            Integer index into the structure's planar axis.

        Returns
        -------
        ``(Any, Any)``
            The two planar coordinates in this new coordinate system.
        """
        vals = 3 * [plane_val]
        vals[self.axis] = axis_val
        _, (val_x, val_y) = self.pop_axis(vals, axis=axis)
        return val_x, val_y

    @cached_property
    def _tanq(self) -> float:
        """Value of ``tan(sidewall_angle)``.

        The (possibliy infinite) geometry offset is given by ``_tanq * length_axis``.
        """
        return np.tan(self.sidewall_angle)


class Circular(Geometry):
    """Geometry with circular characteristics (specified by a radius)."""

    radius: NonNegativeFloat = Field(
        title="Radius",
        description="Radius of geometry.",
        json_schema_extra={"units": MICROMETER},
    )

    @field_validator("radius")
    @classmethod
    def _radius_not_inf(cls, val: float) -> float:
        """Make sure center is not infinitiy."""
        if np.isinf(val):
            raise ValidationError("radius can not be 'td.inf'.")
        return val

    def _intersect_dist(self, position: float, z0: float) -> float:
        """Distance between points on circle at z=position where center of circle at z=z0.

        Parameters
        ----------
        position : float
            position along z.
        z0 : float
            center of circle in z.

        Returns
        -------
        float
            Distance between points on the circle intersecting z=z, if no points, ``None``.
        """
        dz = np.abs(z0 - position)
        if dz > self.radius:
            return None
        return 2 * np.sqrt(self.radius**2 - dz**2)


"""Primitive classes"""
