"""Clip geometry operation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import shapely
from pydantic import Field

from tidy3d.components.autograd.path_utils import (
    AutogradRoute,  # noqa: TC
    resolve_delegated_autograd_route,
)
from tidy3d.components.base import cached_property
from tidy3d.components.types import ClipOperationType
from tidy3d.components.types.base import discriminated_union
from tidy3d.exceptions import ValidationError

from .core import (
    Geometry,
    _bit_operations,
    _shapely_operations,
)
from .utils import cleanup_shapely_object

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.autograd.types import PathType
    from tidy3d.components.geometry.utils import GeometryType
    from tidy3d.components.types import Axis, Bound, Coordinate, MatrixReal4x4, Shapely
    from tidy3d.flex_em.translate.sample_sets import SamplingContext, SurfaceSampleSet


class ClipOperation(Geometry):
    """Class representing the result of a set operation between geometries."""

    operation: ClipOperationType = Field(
        title="Operation Type",
        description="Operation to be performed between geometries.",
    )

    geometry_a: discriminated_union(GeometryType) = Field(
        title="Geometry A",
        description="First operand for the set operation. It can be any geometry type, including "
        ":class:`GeometryGroup`.",
    )

    geometry_b: discriminated_union(GeometryType) = Field(
        title="Geometry B",
        description="Second operand for the set operation. It can also be any geometry type.",
    )

    @staticmethod
    def to_polygon_list(base_geometry: Shapely, cleanup: bool = False) -> list[Shapely]:
        """Return a list of valid polygons from a shapely geometry, discarding points, lines, and
        empty polygons, and empty triangles within polygons.

        Parameters
        ----------
        base_geometry : shapely.geometry.base.BaseGeometry
            Base geometry for inspection.
        cleanup: bool = False
            If True, removes extremely small features from each polygon's boundary.
            This is useful for removing artifacts from 2D plots displayed to the user.

        Returns
        -------
        list[shapely.geometry.base.BaseGeometry]
            Valid polygons retrieved from ``base geometry``.
        """
        unfiltered_geoms = []
        if base_geometry.geom_type == "GeometryCollection":
            unfiltered_geoms = [
                p
                for geom in base_geometry.geoms
                for p in ClipOperation.to_polygon_list(geom, cleanup)
            ]
        if base_geometry.geom_type == "MultiPolygon":
            unfiltered_geoms = [p for p in base_geometry.geoms if not p.is_empty]
        if base_geometry.geom_type == "Polygon" and not base_geometry.is_empty:
            unfiltered_geoms = [base_geometry]
        geoms = []
        if cleanup:
            # Optional: "clean" each of the polygons (by removing extremely small or thin features).
            for geom in unfiltered_geoms:
                geom_clean = cleanup_shapely_object(geom)
                if geom_clean.geom_type == "Polygon":
                    geoms.append(geom_clean)
                if geom_clean.geom_type == "MultiPolygon":
                    geoms += [p for p in geom_clean.geoms if not p.is_empty]
                # Ignore other types of shapely objects (points and lines)
        else:
            geoms = unfiltered_geoms
        return geoms

    @property
    def _shapely_operation(self) -> Callable[[Shapely, Shapely], Shapely]:
        """Return a Shapely function equivalent to this operation."""
        result = _shapely_operations.get(self.operation, None)
        if not result:
            raise ValueError(
                "'operation' must be one of 'union', 'intersection', 'difference', or "
                "'symmetric_difference'."
            )
        return result

    @property
    def _bit_operation(self) -> Callable[[Any, Any], Any]:
        """Return a function equivalent to this operation using bit operators."""
        result = _bit_operations.get(self.operation, None)
        if not result:
            raise ValueError(
                "'operation' must be one of 'union', 'intersection', 'difference', or "
                "'symmetric_difference'."
            )
        return result

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
        a = self.geometry_a.intersections_tilted_plane(
            normal,
            origin,
            to_2D,
            cleanup=cleanup,
            quad_segs=quad_segs,
            section_tolerance_2d=section_tolerance_2d,
        )
        b = self.geometry_b.intersections_tilted_plane(
            normal,
            origin,
            to_2D,
            cleanup=cleanup,
            quad_segs=quad_segs,
            section_tolerance_2d=section_tolerance_2d,
        )
        geom_a = shapely.unary_union([Geometry.evaluate_inf_shape(g) for g in a])
        geom_b = shapely.unary_union([Geometry.evaluate_inf_shape(g) for g in b])
        return ClipOperation.to_polygon_list(
            self._shapely_operation(geom_a, geom_b),
            cleanup=cleanup,
        )

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
        section_tolerance_2d : bool = False
            See :meth:`Geometry.intersections_plane`.

        Returns
        -------
        list[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        a = self.geometry_a.intersections_plane(
            x,
            y,
            z,
            cleanup=cleanup,
            quad_segs=quad_segs,
            section_tolerance_2d=section_tolerance_2d,
        )
        b = self.geometry_b.intersections_plane(
            x,
            y,
            z,
            cleanup=cleanup,
            quad_segs=quad_segs,
            section_tolerance_2d=section_tolerance_2d,
        )
        geom_a = shapely.unary_union([Geometry.evaluate_inf_shape(g) for g in a])
        geom_b = shapely.unary_union([Geometry.evaluate_inf_shape(g) for g in b])
        return ClipOperation.to_polygon_list(
            self._shapely_operation(geom_a, geom_b),
            cleanup=cleanup,
        )

    @cached_property
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates.

        Returns
        -------
        tuple[float, float, float], tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """
        # Overestimates
        if self.operation == "difference":
            result = self.geometry_a.bounds
        elif self.operation == "intersection":
            bounds = (self.geometry_a.bounds, self.geometry_b.bounds)
            result = (
                tuple(max(b[i] for b, _ in bounds) for i in range(3)),
                tuple(min(b[i] for _, b in bounds) for i in range(3)),
            )
            if any(result[0][i] > result[1][i] for i in range(3)):
                result = ((0, 0, 0), (0, 0, 0))
        else:
            bounds = (self.geometry_a.bounds, self.geometry_b.bounds)
            result = (
                tuple(min(b[i] for b, _ in bounds) for i in range(3)),
                tuple(max(b[i] for _, b in bounds) for i in range(3)),
            )
        return result

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
        inside_a = self.geometry_a.inside(x, y, z)
        inside_b = self.geometry_b.inside(x, y, z)
        return self._bit_operation(inside_a, inside_b)

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
        inside_a = self.geometry_a.inside_meshgrid(x, y, z)
        inside_b = self.geometry_b.inside_meshgrid(x, y, z)
        return self._bit_operation(inside_a, inside_b)

    def _volume(self, bounds: Bound) -> float:
        """Returns object's volume within given bounds."""
        # Overestimates
        if self.operation == "intersection":
            return min(self.geometry_a.volume(bounds), self.geometry_b.volume(bounds))
        if self.operation == "difference":
            return self.geometry_a.volume(bounds)
        return self.geometry_a.volume(bounds) + self.geometry_b.volume(bounds)

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area within given bounds."""
        # Overestimates
        return self.geometry_a.surface_area(bounds) + self.geometry_b.surface_area(bounds)

    @cached_property
    def _normal_2dmaterial(self) -> Axis:
        """Get the normal to the given geometry, checking that it is a 2D geometry."""
        normal_a = self.geometry_a._normal_2dmaterial
        normal_b = self.geometry_b._normal_2dmaterial

        if normal_a != normal_b:
            raise ValidationError(
                "'Medium2D' requires both geometries in the 'ClipOperation' to "
                "have exactly one dimension with zero size in common."
            )

        plane_position_a = self.geometry_a.bounds[0][normal_a]
        plane_position_b = self.geometry_b.bounds[0][normal_b]

        if plane_position_a != plane_position_b:
            raise ValidationError(
                "'Medium2D' requires both geometries in the 'ClipOperation' to be co-planar."
            )
        return normal_a

    def _update_from_bounds(self, bounds: tuple[float, float], axis: Axis) -> ClipOperation:
        """Returns an updated geometry which has been transformed to fit within ``bounds``
        along the ``axis`` direction."""
        new_geom_a = self.geometry_a._update_from_bounds(bounds=bounds, axis=axis)
        new_geom_b = self.geometry_b._update_from_bounds(bounds=bounds, axis=axis)
        return self.updated_copy(geometry_a=new_geom_a, geometry_b=new_geom_b)

    def _resolve_autograd_route(self, field_path: tuple[Any, ...]) -> AutogradRoute:
        """Resolve and validate one traced ClipOperation path for adjoint routing."""
        return resolve_delegated_autograd_route(
            parameter_kind="geometry",
            owner_kind="geometry type",
            owner_name=type(self).__name__,
            field_path=field_path,
            delegates={"geometry_a": self.geometry_a, "geometry_b": self.geometry_b},
            supported_parameters=(
                "geometry_a.<parameter>",
                "geometry_b.<parameter>",
            ),
        )

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute adjoint derivatives by accumulating contributions from both operands."""
        geometry_paths = {"geometry_a": [], "geometry_b": []}
        for path in derivative_info.paths:
            geometry_key, *sub_path = path
            geometry_paths[geometry_key].append(tuple(sub_path))

        if derivative_info.clipped_geometry is None:
            raise ValidationError(
                "ClipOperation derivative evaluation requires `clipped_geometry`."
            )

        geometry_map = {
            "geometry_a": self.geometry_a,
            "geometry_b": self.geometry_b,
        }
        grad_vjps = {}

        # Reuse interpolation data for both operands to avoid duplicate setup.
        interpolators = derivative_info.interpolators or derivative_info.create_interpolators()

        for geometry_key, geometry in geometry_map.items():
            paths = geometry_paths[geometry_key]
            if not paths:
                continue

            geometry_info = derivative_info.updated_copy(
                paths=paths,
                bounds=geometry.bounds,
                bounds_intersect=self.bounds_intersection(
                    geometry.bounds, derivative_info.simulation_bounds
                ),
                deep=False,
                interpolators=interpolators,
            )

            vjp_dict_geometry = geometry._compute_derivatives(geometry_info)

            for geo_path, geo_vjp in vjp_dict_geometry.items():
                full_path = (geometry_key, *geo_path)
                if full_path in grad_vjps:
                    existing = grad_vjps[full_path]
                    if isinstance(existing, (list, tuple)) and isinstance(geo_vjp, (list, tuple)):
                        grad_vjps[full_path] = type(existing)(
                            x + y for x, y in zip(existing, geo_vjp)
                        )
                    else:
                        grad_vjps[full_path] = existing + geo_vjp
                else:
                    grad_vjps[full_path] = geo_vjp

        return grad_vjps

    @staticmethod
    def _operand_paths(paths: list[PathType]) -> dict[str, list[PathType]]:
        """Group operand-relative sub-paths by operand key."""
        geometry_paths = {"geometry_a": [], "geometry_b": []}
        for path in paths:
            geometry_key, *sub_path = path
            geometry_paths[geometry_key].append(tuple(sub_path))
        return geometry_paths

    def _make_adjoint_sample_sets(
        self, paths: list[PathType], ctx: SamplingContext
    ) -> dict[PathType, SurfaceSampleSet]:
        """Route generation to both operands, prefixing their canonical keys.

        Operands generate their full (unmasked) surfaces; the clip-activity masking
        stays at evaluation time via ``derivative_info.clipped_geometry``.
        """
        geometry_map = {
            "geometry_a": self.geometry_a,
            "geometry_b": self.geometry_b,
        }
        sample_sets = {}
        for geometry_key, operand_paths in self._operand_paths(paths).items():
            if not operand_paths:
                continue
            geometry = geometry_map[geometry_key]
            ctx_operand = ctx.updated_copy(
                bounds=geometry.bounds,
                bounds_intersect=self.bounds_intersection(geometry.bounds, ctx.simulation_bounds),
            )
            operand_sets = geometry._make_adjoint_sample_sets(paths=operand_paths, ctx=ctx_operand)
            for key, key_sample_sets in operand_sets.items():
                sample_sets[(geometry_key, *key)] = key_sample_sets
        return sample_sets

    def _compute_derivatives_from_sample_sets(
        self,
        sample_sets: dict[PathType, SurfaceSampleSet],
        paths: list[PathType],
        derivative_info: DerivativeInfo,
    ) -> AutogradFieldMap:
        """Route consumption to both operands, accumulating their contributions."""
        if derivative_info.clipped_geometry is None:
            raise ValidationError(
                "ClipOperation derivative evaluation requires `clipped_geometry`."
            )

        geometry_map = {
            "geometry_a": self.geometry_a,
            "geometry_b": self.geometry_b,
        }
        grad_vjps = {}

        # Reuse interpolation data for both operands to avoid duplicate setup.
        interpolators = derivative_info.interpolators or derivative_info.create_interpolators()

        for geometry_key, operand_paths in self._operand_paths(paths).items():
            if not operand_paths:
                continue
            geometry = geometry_map[geometry_key]

            prefix = (geometry_key,)
            operand_sets = {
                key[1:]: value for key, value in sample_sets.items() if key[:1] == prefix
            }

            geometry_info = derivative_info.updated_copy(
                paths=operand_paths,
                bounds=geometry.bounds,
                bounds_intersect=self.bounds_intersection(
                    geometry.bounds, derivative_info.simulation_bounds
                ),
                deep=False,
                interpolators=interpolators,
            )

            vjp_dict_geometry = geometry._compute_derivatives_from_sample_sets(
                sample_sets=operand_sets,
                paths=operand_paths,
                derivative_info=geometry_info,
            )

            for geo_path, geo_vjp in vjp_dict_geometry.items():
                full_path = (geometry_key, *geo_path)
                if full_path in grad_vjps:
                    existing = grad_vjps[full_path]
                    if isinstance(existing, (list, tuple)) and isinstance(geo_vjp, (list, tuple)):
                        grad_vjps[full_path] = type(existing)(
                            x + y for x, y in zip(existing, geo_vjp)
                        )
                    else:
                        grad_vjps[full_path] = existing + geo_vjp
                else:
                    grad_vjps[full_path] = geo_vjp

        return grad_vjps
