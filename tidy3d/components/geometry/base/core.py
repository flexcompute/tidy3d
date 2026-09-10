"""Abstract base classes for geometry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import autograd.numpy as np
import shapely
from pydantic import field_validator

from tidy3d.components.autograd import get_static
from tidy3d.components.autograd.path_utils import (
    format_traced_paths,
    raise_unsupported_traced_path,
)
from tidy3d.components.autograd.types import PathType
from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.geometry.bound_ops import bounds_intersection, bounds_union
from tidy3d.components.viz import plot_params_geometry
from tidy3d.constants import fp_eps
from tidy3d.exceptions import SetupError, ValidationError, format_chained_exception_message
from tidy3d.log import log

from . import gds as _gds_methods
from . import operators as _operator_methods
from . import plotting as _plot_methods
from . import transforms as _transform_methods

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.types import (
        Axis,
        Bound,
        Coordinate,
        Coordinate2D,
        MatrixReal4x4,
        Shapely,
    )
    from tidy3d.components.viz import PlotParams
    from tidy3d.em.translate.sample_sets import SamplingContext, SurfaceSampleSet

    from .box import Box


_BOX_FACES = "faces"


def _raise_unsupported_traced_geometry_path(
    geometry_name: str,
    field_path: tuple[Any, ...],
    *,
    supported_parameters: tuple[str, ...] = (),
) -> None:
    """Raise a user-facing validation error for an unsupported geometry trace."""
    raise_unsupported_traced_path(
        parameter_kind="geometry",
        owner_kind="geometry type",
        owner_name=geometry_name,
        field_path=field_path,
        supported_parameters=supported_parameters,
    )


_shapely_operations = {
    "union": shapely.union,
    "intersection": shapely.intersection,
    "difference": shapely.difference,
    "symmetric_difference": shapely.symmetric_difference,
}

_bit_operations = {
    "union": lambda a, b: a | b,
    "intersection": lambda a, b: a & b,
    "difference": lambda a, b: a & ~b,
    "symmetric_difference": lambda a, b: a != b,
}


# Validators for geometry classes (defined here instead of validators.py to avoid circular imports)
def assert_geometry_finite(field_name: str = "geometry") -> Callable[[type, Geometry], Geometry]:
    """Validator that ensures a geometry field has finite bounds."""

    @field_validator(field_name)
    @classmethod
    def geometry_has_finite_bounds(cls: type, val: Geometry) -> Geometry:
        """Raise validation error if geometry has non-finite bounds."""
        if not np.isfinite(val.bounds).all():
            raise ValidationError(
                f"'{cls.__name__}' requires a geometry with finite dimensions. "
                "Try using a large value instead of 'inf' when creating geometries."
            )
        return val

    return geometry_has_finite_bounds


def check_transform_invertible(transform: MatrixReal4x4, index: int | None = None) -> None:
    """Check if a transform matrix is invertible.

    Parameters
    ----------
    transform : MatrixReal4x4
        The 4x4 transformation matrix to check.
    index : Optional[int]
        If provided, includes the index in the error message (for array of transforms).

    Raises
    ------
    ValidationError
        If the transform matrix is not invertible.
    """
    try:
        _ = np.linalg.inv(transform)
    except np.linalg.LinAlgError as err:
        if index is not None:
            raise ValidationError(
                format_chained_exception_message(
                    f"Transform at index {index} is not invertible", err
                )
            ) from err
        raise ValidationError(
            format_chained_exception_message("Transform matrix is not invertible", err)
        ) from err


class Geometry(Tidy3dBaseModel, ABC):
    """Abstract base class, defines where something exists in space."""

    _traced_supported_paths: ClassVar[tuple[PathType, ...]] = ()

    @classmethod
    def _traced_autograd_supported_parameters(cls) -> tuple[str, ...]:
        """Return user-facing supported parameter names for setup validation."""
        return format_traced_paths(cls._traced_supported_paths)

    @cached_property
    def plot_params(self) -> PlotParams:
        """Default parameters for plotting a Geometry object."""
        return plot_params_geometry

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

        def point_inside(x: float, y: float, z: float) -> bool:
            """Returns ``True`` if a single point ``(x, y, z)`` is inside."""
            shapes_intersect = self.intersections_plane(z=z)
            loc = self.make_shapely_point(x, y)
            return any(shape.contains(loc) for shape in shapes_intersect)

        arrays = tuple(map(np.array, (x, y, z)))
        self._ensure_equal_shape(*arrays)
        inside = np.zeros((arrays[0].size,), dtype=bool)
        arrays_flat = map(np.ravel, arrays)
        for ipt, args in enumerate(zip(*arrays_flat)):
            inside[ipt] = point_inside(*args)
        return inside.reshape(arrays[0].shape)

    @staticmethod
    def _ensure_equal_shape(*arrays: Any) -> None:
        """Ensure all input arrays have the same shape."""
        shapes = {np.array(arr).shape for arr in arrays}
        if len(shapes) > 1:
            raise ValueError("All coordinate inputs (x, y, z) must have the same shape.")

    @staticmethod
    def make_shapely_box(minx: float, miny: float, maxx: float, maxy: float) -> shapely.box:
        """Make a shapely box ensuring everything untraced."""

        minx = get_static(minx)
        miny = get_static(miny)
        maxx = get_static(maxx)
        maxy = get_static(maxy)

        return shapely.box(minx, miny, maxx, maxy)

    @staticmethod
    def make_shapely_point(minx: float, miny: float) -> shapely.Point:
        """Make a shapely Point ensuring everything untraced."""

        minx = get_static(minx)
        miny = get_static(miny)

        return shapely.Point(minx, miny)

    def _inds_inside_bounds(
        self, x: NDArray[float], y: NDArray[float], z: NDArray[float]
    ) -> tuple[slice, slice, slice]:
        """Return slices into the sorted input arrays that are inside the geometry bounds.

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
        tuple[slice, slice, slice]
            Slices into each of the three arrays that are inside the geometry bounds.
        """
        bounds = self.bounds
        inds_in = []
        for dim, coords in enumerate([x, y, z]):
            inds = np.nonzero((bounds[0][dim] <= coords) * (coords <= bounds[1][dim]))[0]
            inds_in.append(slice(0, 0) if inds.size == 0 else slice(inds[0], inds[-1] + 1))

        return tuple(inds_in)

    def inside_meshgrid(
        self, x: NDArray[float], y: NDArray[float], z: NDArray[float]
    ) -> NDArray[bool]:
        """Perform ``self.inside`` on a set of sorted 1D coordinates. Applies meshgrid to the
        supplied coordinates before checking inside.

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

        arrays = tuple(map(np.array, (x, y, z)))
        if any(arr.ndim != 1 for arr in arrays):
            raise ValueError("Each of the supplied coordinates (x, y, z) must be 1D.")
        shape = tuple(arr.size for arr in arrays)
        is_inside = np.zeros(shape, dtype=bool)
        inds_inside = self._inds_inside_bounds(*arrays)
        coords_inside = tuple(arr[ind] for ind, arr in zip(inds_inside, arrays))
        coords_3d = np.meshgrid(*coords_inside, indexing="ij")
        is_inside[inds_inside] = self.inside(*coords_3d)
        return is_inside

    @abstractmethod
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
            If ``True``, allow effectively zero-thickness geometries to contribute a section
            when the requested plane is within ``fp_eps`` of the geometry bounds along that
            axis. Intended for plotting paths where small transform or snap offsets should not
            hide 2D structures; does not affect strictly 3D geometries.

        Returns
        -------
        list[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

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
            If ``True``, allow effectively zero-thickness geometries to contribute a section
            when the requested plane is within ``fp_eps`` of the geometry bounds along that
            axis. Intended for plotting paths where small transform or snap offsets should not
            hide 2D structures; does not affect strictly 3D geometries.

        Returns
        -------
        list[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        origin = self.unpop_axis(position, (0, 0), axis=axis)
        normal = self.unpop_axis(1, (0, 0), axis=axis)
        to_2D = np.eye(4)
        if axis != 2:
            last, indices = self.pop_axis((0, 1, 2), axis)
            to_2D = to_2D[[*list(indices), last, 3]]
        return self.intersections_tilted_plane(
            normal,
            origin,
            to_2D,
            cleanup=cleanup,
            quad_segs=quad_segs,
            section_tolerance_2d=section_tolerance_2d,
        )

    def intersections_2dbox(self, plane: Box) -> list[Shapely]:
        """Returns list of shapely geometries representing the intersections of the geometry with
        a 2D box.

        Returns
        -------
        list[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane. For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        log.warning(
            "'intersections_2dbox()' is deprecated and will be removed in the future. "
            "Use 'plane.intersections_with(...)' for the same functionality."
        )
        return plane.intersections_with(self)

    def intersects(
        self, other: Geometry, strict_inequality: tuple[bool, bool, bool] = [False, False, False]
    ) -> bool:
        """Returns ``True`` if two :class:`~tidy3d.Geometry` have intersecting `.bounds`.

        Parameters
        ----------
        other : :class:`~tidy3d.Geometry`
            Geometry to check intersection with.
        strict_inequality : tuple[bool, bool, bool] = [False, False, False]
            For each dimension, defines whether to include equality in the boundaries comparison.
            If ``False``, equality is included, and two geometries that only intersect at their
            boundaries will evaluate as ``True``. If ``True``, such geometries will evaluate as
            ``False``.

        Returns
        -------
        bool
            Whether the rectangular bounding boxes of the two geometries intersect.
        """

        self_bmin, self_bmax = self.bounds
        other_bmin, other_bmax = other.bounds

        for smin, omin, smax, omax, strict in zip(
            self_bmin, other_bmin, self_bmax, other_bmax, strict_inequality
        ):
            # are all of other's minimum coordinates less than self's maximum coordinate?
            in_minus = omin < smax if strict else omin <= smax
            # are all of other's maximum coordinates greater than self's minimum coordinate?
            in_plus = omax > smin if strict else omax >= smin

            # if either failed, return False
            if not all((in_minus, in_plus)):
                return False

        return True

    def contains(
        self, other: Geometry, strict_inequality: tuple[bool, bool, bool] = [False, False, False]
    ) -> bool:
        """Returns ``True`` if the `.bounds` of  ``other`` are contained within the
        `.bounds` of ``self``.

        Parameters
        ----------
        other : :class:`~tidy3d.Geometry`
            Geometry to check containment with.
        strict_inequality : tuple[bool, bool, bool] = [False, False, False]
            For each dimension, defines whether to include equality in the boundaries comparison.
            If ``False``, equality will be considered as contained. If ``True``, ``other``'s
            bounds must be strictly within the bounds of ``self``.

        Returns
        -------
        bool
            Whether the rectangular bounding box of ``other`` is contained within the bounding
            box of ``self``.
        """

        self_bmin, self_bmax = self.bounds
        other_bmin, other_bmax = other.bounds

        for smin, omin, smax, omax, strict in zip(
            self_bmin, other_bmin, self_bmax, other_bmax, strict_inequality
        ):
            # are all of other's minimum coordinates greater than self's minimim coordinate?
            in_minus = omin > smin if strict else omin >= smin
            # are all of other's maximum coordinates less than self's maximum coordinate?
            in_plus = omax < smax if strict else omax <= smax

            # if either failed, return False
            if not all((in_minus, in_plus)):
                return False

        return True

    def intersects_plane(
        self, x: float | None = None, y: float | None = None, z: float | None = None
    ) -> bool:
        """Whether self intersects plane specified by one non-None value of x,y,z.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.

        Returns
        -------
        bool
            Whether this geometry intersects the plane.
        """

        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        return self.intersects_axis_position(axis, position)

    def intersects_axis_position(
        self, axis: int, position: float, section_tolerance_2d: bool = False
    ) -> bool:
        """Whether self intersects plane specified by a given position along a normal axis.

        Parameters
        ----------
        axis : int = None
            Axis normal to the plane.
        position : float = None
            Position of plane along the normal axis.
        section_tolerance_2d : bool = False
            If ``True``, allow effectively zero-thickness geometries to intersect a plane when
            the plane is within ``fp_eps`` of the geometry bounds along that axis.

        Returns
        -------
        bool
            Whether this geometry intersects the plane.
        """
        min_bound = self.bounds[0][axis]
        max_bound = self.bounds[1][axis]
        if min_bound <= position <= max_bound:
            return True
        if section_tolerance_2d and np.isclose(min_bound, max_bound, rtol=fp_eps, atol=fp_eps):
            return bool(
                np.isclose(position, min_bound, rtol=fp_eps, atol=fp_eps)
                or np.isclose(position, max_bound, rtol=fp_eps, atol=fp_eps)
            )
        return False

    @cached_property
    @abstractmethod
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates.

        Returns
        -------
        tuple[float, float, float], tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """

    @staticmethod
    def bounds_intersection(bounds1: Bound, bounds2: Bound) -> Bound:
        """Return the bounds that are the intersection of two bounds."""
        return bounds_intersection(bounds1, bounds2)

    @staticmethod
    def bounds_union(bounds1: Bound, bounds2: Bound) -> Bound:
        """Return the bounds that are the union of two bounds."""
        return bounds_union(bounds1, bounds2)

    @cached_property
    def bounding_box(self) -> Box:
        """Returns :class:`~tidy3d.Box` representation of the bounding box of a :class:`~tidy3d.Geometry`.

        Returns
        -------
        :class:`~tidy3d.Box`
            Geometric object representing bounding box.
        """
        from .box import Box

        return Box.from_bounds(*self.bounds)

    @cached_property
    def zero_dims(self) -> list[Axis]:
        """A list of axes along which the :class:`~tidy3d.Geometry` is zero-sized based on its bounds."""
        zero_dims = []
        for dim in range(3):
            if self.bounds[1][dim] == self.bounds[0][dim]:
                zero_dims.append(dim)
        return zero_dims

    def _pop_bounds(self, axis: Axis) -> tuple[Coordinate2D, tuple[Coordinate2D, Coordinate2D]]:
        """Returns min and max bounds in plane normal to and tangential to ``axis``.

        Parameters
        ----------
        axis : int
            Integer index into 'xyz' (0,1,2).

        Returns
        -------
        tuple[float, float], tuple[tuple[float, float], tuple[float, float]]
            Bounds along axis and a tuple of bounds in the ordered planar coordinates.
            Packed as ``(zmin, zmax), ((xmin, ymin), (xmax, ymax))``.
        """
        b_min, b_max = self.bounds
        zmin, (xmin, ymin) = self.pop_axis(b_min, axis=axis)
        zmax, (xmax, ymax) = self.pop_axis(b_max, axis=axis)
        return (zmin, zmax), ((xmin, ymin), (xmax, ymax))

    @staticmethod
    def _get_center(pt_min: float, pt_max: float) -> float:
        """Returns center point based on bounds along dimension."""
        if np.isneginf(pt_min) and np.isposinf(pt_max):
            return 0.0
        if np.isneginf(pt_min) or np.isposinf(pt_max):
            raise SetupError(
                f"Bounds of ({pt_min}, {pt_max}) supplied along one dimension. "
                "We currently don't support a single ``inf`` value in bounds for ``Box``. "
                "To construct a semi-infinite ``Box``, "
                "please supply a large enough number instead of ``inf``. "
                "For example, a location extending outside of the "
                "Simulation domain (including PML)."
            )
        return (pt_min + pt_max) / 2.0

    @cached_property
    def _normal_2dmaterial(self) -> Axis:
        """Get the normal to the given geometry, checking that it is a 2D geometry."""
        raise ValidationError("'Medium2D' is not compatible with this geometry class.")

    def _update_from_bounds(self, bounds: tuple[float, float], axis: Axis) -> Geometry:
        """Returns an updated geometry which has been transformed to fit within ``bounds``
        along the ``axis`` direction."""
        raise NotImplementedError(
            "'_update_from_bounds' is not compatible with this geometry class."
        )

    def _make_adjoint_sample_sets(
        self, paths: list[PathType], ctx: SamplingContext
    ) -> dict[PathType, SurfaceSampleSet]:
        """Generate the surface sample sets for the requested shape-derivative paths.

        Returns exactly one sample set per geometry-owned canonical key, where the key
        fully identifies its sampling unit (e.g. ``("faces", 0, 1)`` names one box
        face). One canonical set may serve several derivative paths (box face sets
        serve both ``center`` and ``size``).

        Every canonical key the requested paths imply is emitted: a sampling unit that
        legitimately contributes nothing (outside the simulation domain, degenerate)
        appears as an explicit empty set, never as a missing key.

        Pure and deterministic in ``paths`` and ``ctx`` — pre-simulation collection
        and late generation must produce identical sets. Each requested path's samples
        must not depend on which other paths are co-requested (grouped router dispatch
        relies on this).
        """
        raise NotImplementedError(
            f"Can't generate adjoint sample sets for 'Geometry': '{type(self)}'."
        )

    def _compute_derivatives_from_sample_sets(
        self,
        sample_sets: dict[PathType, SurfaceSampleSet],
        paths: list[PathType],
        derivative_info: DerivativeInfo,
    ) -> AutogradFieldMap:
        """Compute adjoint derivatives for ``paths`` by consuming pre-generated sample sets.

        ``sample_sets`` must come from this geometry's ``_make_adjoint_sample_sets``
        (canonical keys and metadata are a contract between the two methods). Every
        canonical key the requested paths imply must be present — an empty set is a
        legitimate zero contribution, a missing key means data was lost and raises.
        Each requested path's vjp must not depend on which other paths are
        co-requested (grouped router dispatch relies on this).
        """
        raise NotImplementedError(
            f"Can't compute derivative from sample sets for 'Geometry': '{type(self)}'."
        )

    # Transformations
    translated = _transform_methods.translated
    scaled = _transform_methods.scaled
    rotated = _transform_methods.rotated
    reflected = _transform_methods.reflected
    array = _transform_methods.array
    car_2_sph = staticmethod(_transform_methods.car_2_sph)
    sph_2_car = staticmethod(_transform_methods.sph_2_car)
    sph_2_car_field = staticmethod(_transform_methods.sph_2_car_field)
    car_2_sph_field = staticmethod(_transform_methods.car_2_sph_field)
    kspace_2_sph = staticmethod(_transform_methods.kspace_2_sph)

    # Plotting and measures
    plot = _plot_methods.plot
    plot_shape = _plot_methods.plot_shape
    _do_not_intersect = staticmethod(_plot_methods._do_not_intersect)
    _get_plot_labels = staticmethod(_plot_methods._get_plot_labels)
    _get_plot_limits = _plot_methods._get_plot_limits
    add_ax_lims = _plot_methods.add_ax_lims
    add_ax_labels_and_title = staticmethod(_plot_methods.add_ax_labels_and_title)
    _evaluate_inf = staticmethod(_plot_methods._evaluate_inf)
    evaluate_inf_shape = staticmethod(_plot_methods.evaluate_inf_shape)
    pop_axis = staticmethod(_plot_methods.pop_axis)
    unpop_axis = staticmethod(_plot_methods.unpop_axis)
    parse_xyz_kwargs = staticmethod(_plot_methods.parse_xyz_kwargs)
    _validate_gds_precision = staticmethod(_plot_methods._validate_gds_precision)
    parse_two_xyz_kwargs = staticmethod(_plot_methods.parse_two_xyz_kwargs)
    rotate_points = staticmethod(_plot_methods.rotate_points)
    reflect_points = _plot_methods.reflect_points
    volume = _plot_methods.volume
    _volume = _plot_methods._volume
    surface_area = _plot_methods.surface_area
    _surface_area = _plot_methods._surface_area

    # GDS
    load_gds_vertices_gdstk = staticmethod(_gds_methods.load_gds_vertices_gdstk)
    from_gds = staticmethod(_gds_methods.from_gds)
    from_shapely = staticmethod(_gds_methods.from_shapely)
    to_gdstk = _gds_methods.to_gdstk
    to_gds = _gds_methods.to_gds
    to_gds_file = _gds_methods.to_gds_file

    # Operators and autograd
    _compute_derivatives = _operator_methods._compute_derivatives
    _resolve_autograd_route = _operator_methods._resolve_autograd_route
    _as_union = _operator_methods._as_union
    __add__ = _operator_methods.__add__
    __radd__ = _operator_methods.__radd__
    __or__ = _operator_methods.__or__
    __mul__ = _operator_methods.__mul__
    __and__ = _operator_methods.__and__
    __sub__ = _operator_methods.__sub__
    __xor__ = _operator_methods.__xor__
    __pos__ = _operator_methods.__pos__
    __neg__ = _operator_methods.__neg__
    __invert__ = _operator_methods.__invert__
