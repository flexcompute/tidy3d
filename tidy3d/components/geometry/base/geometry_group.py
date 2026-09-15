"""Geometry group."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator

from tidy3d.components.autograd.path_utils import (
    AutogradRoute,
    raise_with_traced_path_context,
)
from tidy3d.components.base import cached_property
from tidy3d.components.types.base import discriminated_union
from tidy3d.exceptions import AdjointError, ValidationError

from .core import Geometry, _raise_unsupported_traced_geometry_path

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.autograd.types import PathType
    from tidy3d.components.geometry.utils import GeometryType
    from tidy3d.components.types import Axis, Bound, Coordinate, MatrixReal4x4, Shapely
    from tidy3d.em.translate.sample_sets import SamplingContext, SurfaceSampleSet


class GeometryGroup(Geometry):
    """A collection of Geometry objects that can be called as a single geometry object."""

    geometries: tuple[discriminated_union(GeometryType), ...] = Field(
        title="Geometries",
        description="Tuple of geometries in a single grouping. "
        "Can provide significant performance enhancement in ``Structure`` when all geometries are "
        "assigned the same medium.",
    )

    @field_validator("geometries")
    @classmethod
    def _geometries_not_empty(cls, val: tuple[GeometryType, ...]) -> tuple[GeometryType, ...]:
        """make sure geometries are not empty."""
        if not len(val) > 0:
            raise ValidationError("GeometryGroup.geometries must not be empty.")
        return val

    @cached_property
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates.

        Returns
        -------
        tuple[float, float, float], tuple[float, float, float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """

        bounds = tuple(geometry.bounds for geometry in self.geometries)
        return (
            tuple(min(b[i] for b, _ in bounds) for i in range(3)),
            tuple(max(b[i] for _, b in bounds) for i in range(3)),
        )

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

        Returns
        -------
        list[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        return [
            intersection
            for geometry in self.geometries
            for intersection in geometry.intersections_tilted_plane(
                normal,
                origin,
                to_2D,
                cleanup=cleanup,
                quad_segs=quad_segs,
                section_tolerance_2d=section_tolerance_2d,
            )
        ]

    def intersections_plane(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        cleanup: bool = True,
        quad_segs: int | None = None,
        section_tolerance_2d: bool = False,
    ) -> list[Shapely]:
        """Returns list of shapely geometries at plane specified by one non-None value of x,y,z.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        cleanup : bool = True
            If True, removes extremely small features from each polygon's boundary.
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
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        return [
            intersection
            for geometry in self.geometries
            if geometry.intersects_axis_position(
                axis, position, section_tolerance_2d=section_tolerance_2d
            )
            for intersection in geometry.intersections_plane(
                x=x,
                y=y,
                z=z,
                cleanup=cleanup,
                quad_segs=quad_segs,
                section_tolerance_2d=section_tolerance_2d,
            )
        ]

    def intersects_axis_position(
        self, axis: float, position: float, section_tolerance_2d: bool = False
    ) -> bool:
        """Whether self intersects plane specified by a given position along a normal axis.

        Parameters
        ----------
        axis : int = None
            Axis normal to the plane.
        position : float = None
            Position of plane along the normal axis.
        section_tolerance_2d : bool = False
            See :meth:`Geometry.intersects_axis_position`.

        Returns
        -------
        bool
            Whether this geometry intersects the plane.
        """
        return any(
            geom.intersects_axis_position(axis, position, section_tolerance_2d=section_tolerance_2d)
            for geom in self.geometries
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
        individual_insides = (geometry.inside(x, y, z) for geometry in self.geometries)
        return functools.reduce(lambda a, b: a | b, individual_insides)

    def inside_meshgrid(
        self, x: NDArray[float], y: NDArray[float], z: NDArray[float]
    ) -> NDArray[bool]:
        """Faster way to check ``self.inside`` on a meshgrid. The input arrays are assumed sorted.

        Parameters
        ----------
        x : np.ndarray[float]
            1D array of point positions in x direction.
        y : np.ndarray[float]
            1D array of point positions in y direction.
        z : np.ndarray[float]
            1D array of point positions in z direction.

        Returns
        -------
        np.ndarray[bool]
            Array with shape ``(x.size, y.size, z.size)``, which is ``True`` for every
            point that is inside the geometry.
        """
        individual_insides = (geom.inside_meshgrid(x, y, z) for geom in self.geometries)
        return functools.reduce(lambda a, b: a | b, individual_insides)

    def _volume(self, bounds: Bound) -> float:
        """Returns object's volume within given bounds."""
        return sum(geometry.volume(bounds) for geometry in self.geometries)

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area within given bounds."""
        return sum(geometry.surface_area(bounds) for geometry in self.geometries)

    @cached_property
    def _normal_2dmaterial(self) -> Axis:
        """Get the normal to the given geometry, checking that it is a 2D geometry."""

        normals = {geom._normal_2dmaterial for geom in self.geometries}

        if len(normals) != 1:
            raise ValidationError(
                "'Medium2D' requires all geometries in the 'GeometryGroup' to "
                "share exactly one dimension with zero size."
            )
        normal = list(normals)[0]
        positions = {geom.bounds[0][normal] for geom in self.geometries}
        if len(positions) != 1:
            raise ValidationError(
                "'Medium2D' requires all geometries in the 'GeometryGroup' to be co-planar."
            )
        return normal

    def _update_from_bounds(self, bounds: tuple[float, float], axis: Axis) -> GeometryGroup:
        """Returns an updated geometry which has been transformed to fit within ``bounds``
        along the ``axis`` direction."""
        new_geometries = tuple(
            geometry._update_from_bounds(bounds=bounds, axis=axis) for geometry in self.geometries
        )
        return self.updated_copy(geometries=new_geometries)

    def _resolve_autograd_route(self, field_path: tuple[Any, ...]) -> AutogradRoute:
        """Resolve and validate one traced GeometryGroup path for adjoint routing."""
        if len(field_path) < 2 or field_path[0] != "geometries":
            _raise_unsupported_traced_geometry_path(
                type(self).__name__,
                field_path,
                supported_parameters=("geometries[index].<parameter>",),
            )

        index = field_path[1]
        sub_path = field_path[2:]
        try:
            self.geometries[index]._resolve_autograd_route(sub_path)
        except AdjointError as err:
            raise_with_traced_path_context(
                err,
                parameter_kind="geometry",
                local_path=sub_path,
                full_path=field_path,
            )
        return AutogradRoute(local_path=field_path)

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute the adjoint derivatives for this object."""

        grad_vjps = {}

        # create interpolators once for all geometries to avoid redundant field data conversions
        interpolators = derivative_info.interpolators or derivative_info.create_interpolators()

        for field_path in derivative_info.paths:
            _, index, *geo_path = field_path

            geo = self.geometries[index]
            # pass pre-computed interpolators if available
            geo_info = derivative_info.updated_copy(
                paths=[tuple(geo_path)],
                bounds=geo.bounds,
                bounds_intersect=self.bounds_intersection(
                    geo.bounds, derivative_info.simulation_bounds
                ),
                deep=False,
                interpolators=interpolators,
            )

            vjp_dict_geo = geo._compute_derivatives(geo_info)

            if len(vjp_dict_geo) != 1:
                raise AssertionError("Got multiple gradients for single geometry field.")

            grad_vjps[field_path] = vjp_dict_geo.popitem()[1]

        return grad_vjps

    @staticmethod
    def _sub_paths_by_child(paths: list[PathType]) -> dict[int, list[PathType]]:
        """Group child-relative sub-paths by child geometry index."""
        grouped: dict[int, list[PathType]] = {}
        for field_path in paths:
            _, index, *geo_path = field_path
            grouped.setdefault(index, []).append(tuple(geo_path))
        return grouped

    def _make_adjoint_sample_sets(
        self, paths: list[PathType], ctx: SamplingContext
    ) -> dict[PathType, SurfaceSampleSet]:
        """Route generation to child geometries, prefixing their canonical keys.

        All of a child's requested paths are generated in one call so canonical sets
        shared across derivative paths (e.g. box faces) are produced exactly once.
        """
        sample_sets = {}
        for index, child_paths in self._sub_paths_by_child(paths).items():
            child = self.geometries[index]
            ctx_child = ctx.updated_copy(
                bounds=child.bounds,
                bounds_intersect=self.bounds_intersection(child.bounds, ctx.simulation_bounds),
            )
            child_sets = child._make_adjoint_sample_sets(paths=child_paths, ctx=ctx_child)
            for key, key_sample_sets in child_sets.items():
                sample_sets[("geometries", index, *key)] = key_sample_sets
        return sample_sets

    def _compute_derivatives_from_sample_sets(
        self,
        sample_sets: dict[PathType, SurfaceSampleSet],
        paths: list[PathType],
        derivative_info: DerivativeInfo,
    ) -> AutogradFieldMap:
        """Route consumption to child geometries, mapping their vjps to full paths."""
        grad_vjps = {}

        # volumetric route only: the point-cloud route reads precomputed integrands
        # and needs no interpolation setup
        interpolators = derivative_info.interpolators
        if interpolators is None and derivative_info.point_integrands is None:
            interpolators = derivative_info.create_interpolators()

        for index, child_paths in self._sub_paths_by_child(paths).items():
            child = self.geometries[index]

            prefix = ("geometries", index)
            child_sets = {key[2:]: value for key, value in sample_sets.items() if key[:2] == prefix}

            # pass pre-computed interpolators if available
            child_info = derivative_info.updated_copy(
                paths=child_paths,
                bounds=child.bounds,
                bounds_intersect=self.bounds_intersection(
                    child.bounds, derivative_info.simulation_bounds
                ),
                deep=False,
                interpolators=interpolators,
                # point-cloud integrands are keyed like sample sets: strip the routing
                # prefix so the child's local canonical keys resolve
                point_integrands=(
                    {
                        key[2:]: value
                        for key, value in derivative_info.point_integrands.items()
                        if key[:2] == prefix
                    }
                    if derivative_info.point_integrands is not None
                    else None
                ),
            )

            child_vjps = child._compute_derivatives_from_sample_sets(
                sample_sets=child_sets, paths=child_paths, derivative_info=child_info
            )

            for geo_path, geo_vjp in child_vjps.items():
                grad_vjps[("geometries", index, *geo_path)] = geo_vjp

        return grad_vjps
