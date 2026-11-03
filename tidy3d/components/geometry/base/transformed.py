"""Transformed geometry base class."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import autograd.numpy as np
from pydantic import Field, model_validator

from tidy3d._runtime import always_validate
from tidy3d.components.autograd import TracedMatrixReal4x4, get_static
from tidy3d.components.autograd.path_utils import resolve_delegated_autograd_route
from tidy3d.components.base import cached_property
from tidy3d.components.transformation import ReflectionFromPlane, RotationAroundAxis
from tidy3d.components.types.base import discriminated_union
from tidy3d.constants import fp_eps
from tidy3d.exceptions import (
    AdjointError,
    Tidy3dError,
    ValidationError,
)
from tidy3d.log import log

from .core import (
    _TRANSLATION_PATH,
    Geometry,
    check_transform_affine,
)

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray

    from tidy3d.components.autograd.derivative_terms import ShapeDerivativeContext
    from tidy3d.components.autograd.types import GeometryAdjointSettings, PathType
    from tidy3d.components.geometry.utils import GeometryType
    from tidy3d.components.types import (
        ArrayFloat2D,
        Axis,
        Bound,
        Coordinate,
        MatrixReal4x4,
        Shapely,
    )
    from tidy3d.em.translate.sample_sets import SamplingContext, SurfaceSampleSet


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to an ``(N, 3)`` point array."""
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    return np.dot(homogeneous, np.asarray(transform).T)[:, :3]


def _transformed_simulation_bounds(
    transform: np.ndarray, simulation_bounds: tuple[tuple[float, ...], tuple[float, ...]]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return an axis-aligned local bound containing the inverse-transformed domain."""
    corners = Transformed._vertices_from_bounds(simulation_bounds)
    local_corners = np.dot(np.linalg.inv(transform), corners)[:3].T
    return tuple(local_corners.min(axis=0)), tuple(local_corners.max(axis=0))


def _check_flat_box_transform(geometry: Geometry, linear: np.ndarray) -> None:
    """Require flat boxes' transformed zero-thickness directions to stay normal to their planes."""
    from tidy3d.components.geometry.utils import flatten_groups

    from .box import Box

    for child in flatten_groups(geometry, flatten_nonunion_type=True):
        if isinstance(child, Transformed):
            _check_flat_box_transform(child.geometry, linear @ np.asarray(child.transform)[:3, :3])
        elif isinstance(child, Box) and child.size.count(0.0) == 1:
            flat_axis = child.size.index(0.0)
            in_plane = [index for index in range(3) if index != flat_axis]
            directions = linear / np.linalg.norm(linear, axis=0, keepdims=True)
            if not np.allclose(
                np.dot(directions[:, flat_axis], directions[:, in_plane]), 0.0, atol=1e-12
            ):
                raise AdjointError(
                    "Geometry gradients for flat boxes in 3D require the transformed flat "
                    "direction to stay perpendicular to the transformed plane. Shears coupling "
                    "that direction to the plane are not supported; use rotations, reflections, "
                    "scaling, or in-plane shear."
                )


def _local_sampling_context(
    geometry: Geometry, transform: np.ndarray, ctx: SamplingContext
) -> SamplingContext:
    """Express a transformed geometry's sampling context in its local coordinates."""
    from tidy3d.components.bound_ops import bounds_intersection

    check_transform_affine(get_static(transform))
    sim_min, sim_max = map(np.asarray, ctx.simulation_bounds)
    collapsed = np.flatnonzero(np.isclose(sim_max - sim_min, 0.0))
    linear = np.asarray(transform)[:3, :3]
    sampling_linear = linear
    if len(collapsed) == 1:
        axis = collapsed[0]
        in_plane = [index for index in range(3) if index != axis]
        if not (
            np.allclose(linear[axis, in_plane], 0.0, atol=1e-12)
            and np.allclose(linear[in_plane, axis], 0.0, atol=1e-12)
        ):
            axis_name = "xyz"[axis]
            plane = "-".join("xyz"[index] for index in in_plane)
            entries = ", ".join(
                f"transform[{i}][{j}]={linear[i, j]:.3g}"
                for i, j in [(axis, p) for p in in_plane] + [(p, axis) for p in in_plane]
                if not np.isclose(linear[i, j], 0.0, atol=1e-12)
            )
            raise AdjointError(
                f"Geometry gradients in a 2D simulation (zero size along '{axis_name}') require "
                f"'transform' to keep the '{axis_name}' axis separate from the {plane} plane, "
                f"but these entries are nonzero: {entries}. Supported: rotations about "
                f"'{axis_name}'; scaling, shear, or reflection within the {plane} plane; "
                f"scaling along '{axis_name}'; and translations. For other transforms, "
                "use a 3D simulation."
            )
        sampling_linear = linear[np.ix_(in_plane, in_plane)]
    elif len(collapsed) == 0:
        # Validate the geometry, not the requested sample sets: a flat box's
        # normal-axis paths may request only area faces, with no edge samples.
        _check_flat_box_transform(geometry, linear)
    # A scalar local resolution must account for the largest possible stretch,
    # including shear. Perpendicular scaling does not refine a 2D cross-section.
    stretch = np.linalg.norm(sampling_linear, ord=2)
    local_simulation_bounds = _transformed_simulation_bounds(transform, ctx.simulation_bounds)
    return ctx.updated_copy(
        bounds=geometry.bounds,
        bounds_intersect=bounds_intersection(geometry.bounds, local_simulation_bounds),
        simulation_bounds=local_simulation_bounds,
        spacing=ctx.spacing / stretch,
        material_wavelength=(
            ctx.material_wavelength / stretch if ctx.material_wavelength is not None else None
        ),
    )


def _transform_sample_set(
    sample_set: SurfaceSampleSet, transform: np.ndarray, simulation_bounds: Bound
) -> SurfaceSampleSet:
    """Map a local surface sample set into world coordinates for field sampling."""
    from tidy3d.em.translate.sample_sets import IndexedDataArray, PointDataArray, SurfaceSampleSet

    if sample_set.num_points == 0:
        return sample_set

    linear = np.asarray(transform)[:3, :3]
    points = np.asarray(sample_set.points.values)
    normals = np.asarray(sample_set.normals.values)
    perps1 = np.asarray(sample_set.perps1.values)
    det = np.linalg.det(linear)

    normal_raw = np.dot(normals, np.linalg.inv(linear))
    normal_norm = np.linalg.norm(normal_raw, axis=1, keepdims=True)
    world_normals = normal_raw / normal_norm
    world_perps1 = np.dot(perps1, linear.T)
    world_perps1 /= np.linalg.norm(world_perps1, axis=1, keepdims=True)
    world_perps2 = np.cross(world_normals, world_perps1)

    area_scale = np.abs(det) * normal_norm[:, 0]
    sim_min, sim_max = map(np.asarray, simulation_bounds)
    collapsed = np.flatnonzero(np.isclose(sim_max - sim_min, 0.0))
    if len(collapsed) == 1:
        # 2D weights are per unit length of the collapsed axis, not surface areas.
        # The inverse-normal pullback then carries only the in-plane Jacobian.
        area_scale /= np.abs(linear[collapsed[0], collapsed[0]])
    elif len(collapsed) == 0 and any(sample_set.pec_flat_perp_dims):
        # Flat PEC boxes also supply line samples in otherwise 3D simulations.
        tangent = sample_set.perps2 if sample_set.pec_flat_perp_dims[0] else sample_set.perps1
        area_scale = np.linalg.norm(np.dot(tangent.values, linear.T), axis=1)
    world_points = _transform_points(points, transform)
    inside_by_axis = (world_points >= sim_min - fp_eps) & (world_points <= sim_max + fp_eps)
    # Some native 2D samplers use a narrow strip around the collapsed plane;
    # staging projects its queries onto the plane. Clip only the simulated axes.
    inside_by_axis[:, collapsed] = True
    inside = np.all(inside_by_axis, axis=1)
    if not np.any(inside):
        return SurfaceSampleSet.empty()
    metadata = sample_set.metadata
    if metadata is not None and not np.all(inside):
        metadata = metadata.updated_copy(
            **{
                name: value.isel(index=inside).assign_coords(
                    index=np.arange(np.count_nonzero(inside))
                )
                for name, value in metadata
                if isinstance(value, (IndexedDataArray, PointDataArray))
            }
        )
    return SurfaceSampleSet.from_arrays(
        points=world_points[inside],
        normals=world_normals[inside],
        perps1=world_perps1[inside],
        perps2=world_perps2[inside],
        weights=(np.asarray(sample_set.weights.values) * area_scale)[inside],
        pec_flat_perp_dims=sample_set.pec_flat_perp_dims,
        metadata=metadata,
    )


def _localize_sample_set(sample_set: SurfaceSampleSet, transform: np.ndarray) -> SurfaceSampleSet:
    """Map a world sample set back to local coordinates for geometry VJP scatter."""
    from tidy3d.em.translate.sample_sets import SurfaceSampleSet

    if sample_set.num_points == 0:
        return sample_set

    linear = np.asarray(transform)[:3, :3]
    inverse = np.linalg.inv(linear)
    world_normals = np.asarray(sample_set.normals.values)
    # Normals are stored as row vectors.  Pulling a world normal back through
    # ``y = A @ x`` therefore applies ``A.T`` to the corresponding column, or
    # ``@ A`` in row-vector form.
    local_normal_raw = np.dot(world_normals, linear)
    local_normal_norm = np.linalg.norm(local_normal_raw, axis=1, keepdims=True)
    local_normals = local_normal_raw / local_normal_norm
    local_perps1 = np.dot(np.asarray(sample_set.perps1.values), inverse.T)
    local_perps1 /= np.linalg.norm(local_perps1, axis=1, keepdims=True)
    local_perps2 = np.cross(local_normals, local_perps1)
    # A local displacement is mapped through the linear transform before the
    # world-coordinate boundary integral is reduced.  The effective local weight
    # therefore includes the transform's area-volume Jacobian.
    local_weights = np.asarray(sample_set.weights.values) * local_normal_norm[:, 0]

    inverse_transform = np.linalg.inv(np.asarray(transform))
    return SurfaceSampleSet.from_arrays(
        points=_transform_points(np.asarray(sample_set.points.values), inverse_transform),
        normals=local_normals,
        perps1=local_perps1,
        perps2=local_perps2,
        weights=local_weights,
        pec_flat_perp_dims=sample_set.pec_flat_perp_dims,
        metadata=sample_set.metadata,
    )


def _affine_transform_vjp(
    geometry: Geometry,
    sample_sets: dict[PathType, SurfaceSampleSet],
    derivative_info: ShapeDerivativeContext,
    transform: np.ndarray,
) -> np.ndarray:
    """Return the VJP with respect to a local-to-world affine transform.

    Each column is a translation VJP weighted by the local homogeneous point.
    Reuse the geometry consumer's slice-motion and composite chain rules instead
    of assuming that its sample normals contain every displacement sensitivity.
    Samples and integrands are local; boolean masks are already in the integrands.
    The consumer selects its translation partition, ignoring other parameter sets.
    """
    gradient = np.zeros((4, 4))
    for column in range(4):
        weighted_info = replace(
            derivative_info,
            point_integrands={
                key: (
                    Geometry._sample_set_integrand(derivative_info, key)
                    * (sample_set.points.values[:, column] if column < 3 else 1.0)
                )[:, None]
                for key, sample_set in sample_sets.items()
                if sample_set.num_points
            },
        )
        translations = geometry._compute_derivatives_from_sample_sets(
            sample_sets=sample_sets, paths=[_TRANSLATION_PATH], derivative_info=weighted_info
        )
        gradient[:3, column] = translations[_TRANSLATION_PATH]

    gradient[:3] = np.dot(np.linalg.inv(np.asarray(transform)[:3, :3]).T, gradient[:3])
    return gradient


class Transformed(Geometry):
    """Class representing a transformed geometry.

    Parameters of ``geometry`` and the invertible affine ``transform`` support automatic
    differentiation. In 2D simulations the transform must preserve the simulation plane.
    """

    geometry: discriminated_union(GeometryType) = Field(
        title="Geometry",
        description="Base geometry to be transformed.",
    )

    transform: TracedMatrixReal4x4 = Field(
        default_factory=lambda: np.eye(4).tolist(),
        title="Transform",
        description="Transform matrix applied to the base geometry.",
    )

    @model_validator(mode="after")
    @always_validate
    def _apply_transforms(self: dict[str, Any]) -> dict[str, Any]:
        while isinstance(self.geometry, Transformed):
            inner = self.geometry
            object.__setattr__(self, "geometry", inner.geometry)
            object.__setattr__(self, "transform", np.dot(self.transform, inner.transform))
        return self

    @classmethod
    def of(cls, geometry: GeometryType, transform: MatrixReal4x4) -> Transformed:
        """Return ``geometry`` placed by ``transform``, as one transformed geometry.

        A geometry that is already transformed keeps its own geometry, and the two transforms
        compose into one matrix, as the adjoint of a ``GeometryArray`` copy expects.
        """
        if isinstance(geometry, Transformed):
            return cls(geometry=geometry.geometry, transform=np.dot(transform, geometry.transform))
        return cls(geometry=geometry, transform=transform)

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
        # The core composes a nested transform into one matrix, so its bounds stay as tight as
        # those of the composed transform.
        return self._core.bounds

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
        rotation = RotationAroundAxis(angle=angle, axis=axis).matrix
        return np.pad(rotation, ((0, 1), (0, 1)), mode="constant") + np.diag((0.0, 0.0, 0.0, 1.0))

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

        reflection = ReflectionFromPlane(normal=normal).matrix
        return np.pad(reflection, ((0, 1), (0, 1)), mode="constant") + np.diag((0.0, 0.0, 0.0, 1.0))

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

    # Autograd delegation for parameters of the wrapped geometry.

    def _resolve_autograd_route(self, field_path: tuple[Any, ...]) -> None:
        """Validate a traced path owned by the wrapped geometry."""
        check_transform_affine(get_static(self.transform))
        if field_path in (_TRANSLATION_PATH, ("transform",)):
            return self.geometry._resolve_autograd_route(_TRANSLATION_PATH)
        resolve_delegated_autograd_route(
            parameter_kind="geometry",
            owner_kind="geometry type",
            owner_name=type(self).__name__,
            field_path=field_path,
            delegates={"geometry": self.geometry},
            supported_parameters=("transform", "geometry.<parameter>"),
        )

    @staticmethod
    def _geometry_paths(paths: list[PathType]) -> list[PathType]:
        """Strip this wrapper's prefix from paths delegated to the geometry."""
        child_paths = []
        for path in paths:
            if path == _TRANSLATION_PATH:
                child_paths.append(path)
            elif len(path) >= 2 and path[0] == "geometry":
                child_paths.append(path[1:])
            else:
                raise AdjointError("Transformed geometry paths must start with 'geometry'.")
        return list(dict.fromkeys(child_paths))

    def _adjoint_sampling_geometry(self, settings: GeometryAdjointSettings) -> Self:
        """Preserve the transform while using the child's sampled boundary."""
        child = self.geometry._adjoint_sampling_geometry(settings)
        if child is self.geometry:
            return self
        return self.updated_copy(deep=False, validate=False, geometry=child)

    def _make_adjoint_sample_sets(
        self, paths: list[PathType], ctx: SamplingContext, settings: GeometryAdjointSettings
    ) -> dict[PathType, SurfaceSampleSet]:
        """Generate world-coordinate samples by delegating to the local geometry."""
        child_paths = self._geometry_paths(
            [_TRANSLATION_PATH if path == ("transform",) else path for path in paths]
        )

        local_ctx = _local_sampling_context(self.geometry, self.transform, ctx)
        local_sets = self.geometry._make_adjoint_sample_sets(
            paths=child_paths, ctx=local_ctx, settings=settings
        )
        return {
            ("geometry", *key): _transform_sample_set(
                sample_set, self.transform, ctx.simulation_bounds
            )
            for key, sample_set in local_sets.items()
        }

    def _compute_derivatives_from_sample_sets(
        self,
        sample_sets: dict[PathType, SurfaceSampleSet],
        paths: list[PathType],
        derivative_info: ShapeDerivativeContext,
    ) -> dict[PathType, Any]:
        """Consume world-coordinate samples using the wrapped geometry's VJP."""
        geometry_paths = [path for path in paths if path != ("transform",)]
        child_paths = self._geometry_paths(geometry_paths)

        local_sample_sets = {
            key[1:]: _localize_sample_set(sample_set, self.transform)
            for key, sample_set in sample_sets.items()
            if key[:1] == ("geometry",)
        }
        local_point_integrands = {
            key[1:]: value
            for key, value in derivative_info.point_integrands.items()
            if key[:1] == ("geometry",)
        }
        local_info = replace(
            derivative_info,
            simulation_bounds=_transformed_simulation_bounds(
                self.transform, derivative_info.simulation_bounds
            ),
            point_integrands=local_point_integrands,
        )
        result = {}
        if ("transform",) in paths:
            result[("transform",)] = _affine_transform_vjp(
                geometry=self.geometry,
                sample_sets=local_sample_sets,
                derivative_info=local_info,
                transform=self.transform,
            )
        if child_paths:
            child_vjps = self.geometry._compute_derivatives_from_sample_sets(
                sample_sets=local_sample_sets,
                paths=child_paths,
                derivative_info=local_info,
            )
            linear = np.asarray(self.transform)[:3, :3]
            inverse_transpose = np.linalg.inv(linear).T
            for path in geometry_paths:
                child_path = path if path == _TRANSLATION_PATH else path[1:]
                value = child_vjps[child_path]
                if child_path[-len(_TRANSLATION_PATH) :] == _TRANSLATION_PATH:
                    # The wrapped geometry reports dJ/du in local coordinates, while
                    # this wrapper's translation is a world-space displacement t with
                    # u = A^-1 t. Apply the fixed-transform chain rule dJ/dt=A^-T dJ/du.
                    value = tuple(np.dot(inverse_transpose, np.asarray(value)).tolist())
                result[path] = value
        return result

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
