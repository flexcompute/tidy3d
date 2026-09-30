"""Transformed geometry base class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np
from pydantic import Field, field_validator, model_validator

from tidy3d._runtime import always_validate
from tidy3d.components.base import cached_property
from tidy3d.components.transformation import ReflectionFromPlane, RotationAroundAxis
from tidy3d.components.types import (
    MatrixReal4x4,
)
from tidy3d.components.types.base import discriminated_union
from tidy3d.exceptions import (
    Tidy3dError,
    ValidationError,
)
from tidy3d.log import log

from .core import Geometry, assert_geometry_finite, check_transform_invertible

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tidy3d.components.geometry.utils import GeometryType
    from tidy3d.components.types import (
        ArrayFloat2D,
        Axis,
        Bound,
        Coordinate,
        Shapely,
    )


class Transformed(Geometry):
    """Class representing a transformed geometry."""

    geometry: discriminated_union(GeometryType) = Field(
        title="Geometry",
        description="Base geometry to be transformed.",
    )

    transform: MatrixReal4x4 = Field(
        default_factory=lambda: np.eye(4).tolist(),
        title="Transform",
        description="Transform matrix applied to the base geometry.",
    )

    _geometry_is_finite = assert_geometry_finite("geometry")

    @field_validator("transform")
    @classmethod
    def _transform_is_invertible(cls, val: MatrixReal4x4) -> MatrixReal4x4:
        """Raise validation error if transform is not invertible."""
        check_transform_invertible(val)
        return val

    @model_validator(mode="after")
    @always_validate
    def _apply_transforms(self: dict[str, Any]) -> dict[str, Any]:
        while isinstance(self.geometry, Transformed):
            inner = self.geometry
            object.__setattr__(self, "geometry", inner.geometry)
            object.__setattr__(self, "transform", np.dot(self.transform, inner.transform))
        return self

    @cached_property
    def inverse(self) -> MatrixReal4x4:
        """Inverse of this transform."""
        return np.linalg.inv(self.transform)

    @staticmethod
    def _vertices_from_bounds(bounds: Bound) -> ArrayFloat2D:
        """Return the 8 vertices derived from bounds.

        The vertices are returned as homogeneous coordinates (with 4 components).

        Parameters
        ----------
        bounds : Bound
            Bounds from which to derive the vertices.

        Returns
        -------
        ArrayFloat2D
            Array with shape (4, 8) with all vertices from ``bounds``.
        """
        (x0, y0, z0), (x1, y1, z1) = bounds
        return np.array(
            (
                (x0, x0, x0, x0, x1, x1, x1, x1),
                (y0, y0, y1, y1, y0, y0, y1, y1),
                (z0, z1, z0, z1, z0, z1, z0, z1),
                (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            )
        )

    @cached_property
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates.

        Returns
        -------
        tuple[float, float, float], tuple[float, float, float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """
        # NOTE (Lucas): The bounds are overestimated because we don't want to calculate
        # precise TriangleMesh representations for GeometryGroup or ClipOperation.
        vertices = np.dot(self.transform, self._vertices_from_bounds(self.geometry.bounds))[:3]
        return (tuple(vertices.min(axis=1)), tuple(vertices.max(axis=1)))

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
        return self.geometry.intersections_tilted_plane(
            tuple(np.dot((normal[0], normal[1], normal[2], 0.0), self.transform)[:3]),
            tuple(np.dot(self.inverse, (origin[0], origin[1], origin[2], 1.0))[:3]),
            np.dot(to_2D, self.transform),
            cleanup=cleanup,
            quad_segs=quad_segs,
            section_tolerance_2d=section_tolerance_2d,
        )

    def inside(self, x: NDArray[float], y: NDArray[float], z: NDArray[float]) -> NDArray[bool]:
        """For input arrays ``x``, ``y``, ``z`` of arbitrary but identical shape, return an array
        with the same shape which is ``True`` for every point in zip(x, y, z) that is inside the
        volume of the :class:`~tidy3d.Geometry`, and ``False`` otherwise.

        Parameters
        ----------
        x : np.ndarray[float]
            Array of point positions in x direction.
        y : np.ndarray[float]
            Array of point positions in y direction.
        z : np.ndarray[float]
            Array of point positions in z direction.

        Returns
        -------
        np.ndarray[bool]
            ``True`` for every point that is inside the geometry.
        """
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        xyz = np.dot(self.inverse, np.vstack((x.flat, y.flat, z.flat, np.ones(x.size))))
        return self.geometry.inside(xyz[0], xyz[1], xyz[2]).reshape(x.shape)

    def inside_meshgrid(
        self, x: NDArray[float], y: NDArray[float], z: NDArray[float]
    ) -> NDArray[bool]:
        """Perform ``self.inside`` on a set of sorted 1D coordinates."""
        from tidy3d.components.geometry.polyslab import PolySlab

        if isinstance(self.geometry, PolySlab):
            result = self.geometry._inside_transformed_meshgrid(self, x, y, z)
            if result is not NotImplemented:
                return result

        return super().inside_meshgrid(x, y, z)

    def _volume(self, bounds: Bound) -> float:
        """Returns object's volume within given bounds."""
        # NOTE (Lucas): Bounds are overestimated.
        vertices = np.dot(self.inverse, self._vertices_from_bounds(bounds))[:3]
        inverse_bounds = (tuple(vertices.min(axis=1)), tuple(vertices.max(axis=1)))
        return abs(np.linalg.det(self.transform)) * self.geometry.volume(inverse_bounds)

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area within given bounds."""
        log.warning("Surface area of transformed elements cannot be calculated.")
        return None

    @staticmethod
    def identity() -> MatrixReal4x4:
        """Return an identity matrix where no transform is applied.

        Returns
        -------
        numpy.ndarray
            Identity transform matrix with shape (4, 4).
        """
        return np.eye(4)

    @staticmethod
    def translation(x: float, y: float, z: float) -> MatrixReal4x4:
        """Return a translation matrix.

        Parameters
        ----------
        x : float
            Translation along x.
        y : float
            Translation along y.
        z : float
            Translation along z.

        Returns
        -------
        numpy.ndarray
            Transform matrix with shape (4, 4).
        """
        return np.array(
            [
                (1.0, 0.0, 0.0, x),
                (0.0, 1.0, 0.0, y),
                (0.0, 0.0, 1.0, z),
                (0.0, 0.0, 0.0, 1.0),
            ],
            dtype=float,
        )

    @staticmethod
    def scaling(x: float = 1.0, y: float = 1.0, z: float = 1.0) -> MatrixReal4x4:
        """Return a scaling matrix.

        Parameters
        ----------
        x : float = 1.0
            Scaling factor along x.
        y : float = 1.0
            Scaling factor along y.
        z : float = 1.0
            Scaling factor along z.

        Returns
        -------
        numpy.ndarray
            Transform matrix with shape (4, 4).
        """
        if np.isclose((x, y, z), 0.0).any():
            raise Tidy3dError("Scaling factors cannot be zero in any dimensions.")
        return np.array(
            [
                (x, 0.0, 0.0, 0.0),
                (0.0, y, 0.0, 0.0),
                (0.0, 0.0, z, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ],
            dtype=float,
        )

    @staticmethod
    def rotation(angle: float, axis: Axis | Coordinate) -> MatrixReal4x4:
        """Return a rotation matrix.

        Parameters
        ----------
        angle : float
            Rotation angle (in radians).
        axis : Union[int, tuple[float, float, float]]
            Axis of rotation: 0, 1, or 2 for x, y, and z, respectively, or a 3D vector.

        Returns
        -------
        numpy.ndarray
            Transform matrix with shape (4, 4).
        """
        transform = np.eye(4)
        transform[:3, :3] = RotationAroundAxis(angle=angle, axis=axis).matrix
        return transform

    @staticmethod
    def reflection(normal: Coordinate) -> MatrixReal4x4:
        """Return a reflection matrix.

        Parameters
        ----------
        normal : tuple[float, float, float]
            Normal of the plane of reflection.

        Returns
        -------
        numpy.ndarray
            Transform matrix with shape (4, 4).
        """

        transform = np.eye(4)
        transform[:3, :3] = ReflectionFromPlane(normal=normal).matrix
        return transform

    @staticmethod
    def preserves_axis(transform: MatrixReal4x4, axis: Axis) -> bool:
        """Indicate if the transform preserves the orientation of a given axis.

        Parameters
        ----------
        transform : MatrixReal4x4
            Transform matrix to check.
        axis : int
            Axis to check. Values 0, 1, or 2, to check x, y, or z, respectively.

        Returns
        -------
        bool
            ``True`` if the transformation preserves the axis orientation, ``False`` otherwise.
        """
        i = (axis + 1) % 3
        j = (axis + 2) % 3
        return np.isclose(transform[i, axis], 0) and np.isclose(transform[j, axis], 0)

    @cached_property
    def _normal_2dmaterial(self) -> Axis:
        """Get the normal to the given geometry, checking that it is a 2D geometry."""
        normal = self.geometry._normal_2dmaterial
        preserves_axis = Transformed.preserves_axis(self.transform, normal)

        if not preserves_axis:
            raise ValidationError(
                "'Medium2D' requires geometries of type 'Transformed' to "
                "perserve the axis normal to the 'Medium2D'."
            )

        return normal

    def _update_from_bounds(self, bounds: tuple[float, float], axis: Axis) -> Transformed:
        """Returns an updated geometry which has been transformed to fit within ``bounds``
        along the ``axis`` direction."""
        min_bound = np.array([0, 0, 0, 1.0])
        min_bound[axis] = bounds[0]
        max_bound = np.array([0, 0, 0, 1.0])
        max_bound[axis] = bounds[1]
        new_bounds = []
        new_bounds.append(np.dot(self.inverse, min_bound)[axis])
        new_bounds.append(np.dot(self.inverse, max_bound)[axis])
        new_geometry = self.geometry._update_from_bounds(bounds=new_bounds, axis=axis)
        return self.updated_copy(geometry=new_geometry)
