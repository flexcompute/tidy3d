"""Shared contour-to-polyslab conversion helpers used by structures and invdes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import autograd.numpy as np
import numpy as npo
from shapely.geometry import LinearRing, MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from tidy3d.components.autograd import get_static
from tidy3d.components.grid.grid import Coords
from tidy3d.exceptions import Tidy3dImportError
from tidy3d.log import log

from .polyslab import PolySlab

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tidy3d.components.data.data_array import SpatialDataArray
    from tidy3d.components.data.utils import CustomSpatialDataType
    from tidy3d.components.medium import AbstractCustomMedium
    from tidy3d.components.types import ArrayFloat1D, ArrayFloat2D, Bound2D, InterpMethod


RingRef = tuple[str, int]


@dataclass(frozen=True)
class ContourPolyslabData:
    """Shared geometry result used by structure and invdes helpers."""

    solid_polyslabs: tuple[PolySlab, ...]
    hole_polyslabs: tuple[PolySlab, ...]
    solid_frame_boundary_vertex_mask: tuple[npo.ndarray, ...]
    hole_frame_boundary_vertex_mask: tuple[npo.ndarray, ...]
    frame_bounds: Bound2D
    in_plane_step: float


def validate_optional_finite_float(value: float | None, *, name: str) -> float | None:
    """Validate an optional finite scalar value."""
    if value is None:
        return None
    value_f = float(value)
    if not np.isfinite(value_f):
        raise ValueError(f"'{name}' must be finite.")
    return value_f


def validate_sigma(value: float, *, name: str = "sigma") -> float:
    """Validate a smoothing sigma."""
    value_f = float(value)
    if not np.isfinite(value_f):
        raise ValueError(f"'{name}' must be finite.")
    if value_f < 0:
        raise ValueError(f"'{name}' must be >= 0.")
    return value_f


def _orient_ring(vertices: ArrayFloat2D, ccw: bool) -> ArrayFloat2D:
    """Force ring orientation to CCW or CW."""
    ring = vertices
    if ring.shape[0] < 3:
        raise ValueError("A polygon ring requires at least 3 vertices.")
    if LinearRing(ring).is_ccw != ccw:
        ring = ring[::-1]
    return ring


def _densify_ring(vertices: ArrayFloat2D, step: float) -> ArrayFloat2D:
    """Densify ring edges so each segment is at most ``step`` long."""
    if vertices.shape[0] < 3:
        raise ValueError("A polygon ring requires at least 3 vertices.")
    if step <= 0:
        raise ValueError("boundary_step must be > 0.")

    output = np.asarray(LinearRing(vertices).segmentize(step).coords[:-1], dtype=float)
    if output.shape[0] < 3:
        raise ValueError("Densification collapsed polygon below 3 vertices.")
    return output


def _coord_step_min(values: ArrayFloat1D) -> float:
    """Smallest strictly positive spacing in one coordinate array."""
    if values.size <= 1:
        return np.inf
    diffs = np.abs(np.diff(values))
    diffs = diffs[diffs > 0]
    return float(np.min(diffs)) if diffs.size > 0 else np.inf


def _resolve_axis(
    coord_arrays: tuple[ArrayFloat1D, ArrayFloat1D, ArrayFloat1D], axis: int | None
) -> int:
    """Resolve the singleton axis for a structured 2D data slice."""
    if axis is not None and axis not in (0, 1, 2):
        raise ValueError("'axis' must be one of {0, 1, 2} when provided.")

    singleton_axes = [idx for idx, coord in enumerate(coord_arrays) if coord.size == 1]
    if len(singleton_axes) != 1:
        raise ValueError(
            "Contour conversion supports only 2D media/data: exactly one coordinate dimension "
            "must have exactly one sample."
        )

    if axis is None:
        return singleton_axes[0]
    if axis not in singleton_axes:
        raise ValueError(
            "For 2D contour conversion, 'axis' must correspond to a coordinate dimension "
            "with exactly one sample."
        )
    return axis


def _resolve_boundary_step(
    coord_arrays: tuple[ArrayFloat1D, ArrayFloat1D, ArrayFloat1D],
    *,
    axis: int,
    boundary_step: float | None,
) -> float:
    """Resolve an in-plane densification step from coordinates or override."""
    if boundary_step is None:
        in_plane_axes = tuple(idx for idx in (0, 1, 2) if idx != axis)
        step_candidates = [
            _coord_step_min(np.asarray(coord_arrays[in_plane_axis], dtype=float))
            for in_plane_axis in in_plane_axes
        ]
        step_candidates = [step for step in step_candidates if np.isfinite(step)]
        if not step_candidates:
            raise ValueError(
                "Could not infer a default 'boundary_step' from in-plane grid coordinates. "
                "Provide 'boundary_step' explicitly."
            )
        boundary_step_use = float(min(step_candidates))
    else:
        boundary_step_use = float(boundary_step)
    return boundary_step_use


def _geometry_to_polygons(geometry: Any) -> list[Any]:
    """Flatten polygon containers from contour cleanup / union into plain polygons."""
    if geometry.is_empty:
        log.warning("Contour cleanup produced an empty geometry; skipping it.")
        return []
    if isinstance(geometry, ShapelyPolygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    raise TypeError(f"Expected polygonal geometry, got {type(geometry).__name__}.")


def smooth_polygon_vertices(vertices: ArrayFloat2D, sigma: float) -> ArrayFloat2D:
    """Smooth closed polygon-ring vertices with cyclic Gaussian averaging."""
    ring = vertices
    if ring.ndim != 2 or ring.shape[1] != 2:
        raise ValueError(f"'vertices' must have shape (N, 2), got {ring.shape}.")
    if ring.shape[0] < 3:
        raise ValueError("A polygon ring requires at least 3 vertices.")

    sigma_val = validate_sigma(sigma)
    if sigma_val == 0:
        # Keep a traced no-op instead of ``copy()`` so ArrayBox inputs stay differentiable.
        return ring + 0.0

    radius = max(int(np.ceil(3.0 * sigma_val)), 1)
    offsets = np.arange(-radius, radius + 1, dtype=int)
    kernel = np.exp(-0.5 * (offsets / sigma_val) ** 2)
    kernel = kernel / np.sum(kernel)

    smoothed = np.zeros_like(ring, dtype=float)
    for weight, shift in zip(kernel, offsets):
        smoothed = smoothed + float(weight) * np.roll(ring, int(shift), axis=0)
    return smoothed


def smooth_polyslabs(polyslabs: Sequence[PolySlab], sigma: float) -> tuple[PolySlab, ...]:
    """Return polyslabs with all vertex rings Gaussian-smoothed."""
    sigma_val = validate_sigma(sigma)
    if sigma_val == 0:
        return tuple(polyslabs)

    return tuple(
        polyslab.updated_copy(
            vertices=smooth_polygon_vertices(np.array(polyslab.vertices), sigma_val)
        )
        for polyslab in polyslabs
    )


def ordered_ring_refs_by_area(
    solid_polyslabs: Sequence[PolySlab], hole_polyslabs: Sequence[PolySlab]
) -> tuple[RingRef, ...]:
    """Order solid/hole rings by gross polygon area, largest first."""
    ring_records: list[tuple[float, int, str, int]] = []
    for ring_type, polyslabs in (("solid", solid_polyslabs), ("hole", hole_polyslabs)):
        for idx, polyslab in enumerate(polyslabs):
            polygon = ShapelyPolygon(npo.asarray(get_static(polyslab.vertices), dtype=float))
            ring_records.append(
                (float(polygon.area), 0 if ring_type == "solid" else 1, ring_type, idx)
            )
    ring_records.sort(key=lambda record: (-record[0], record[1], record[3]))
    return tuple((ring_type, ring_idx) for _, _, ring_type, ring_idx in ring_records)


def gdstk_contours_from_custom_medium(
    medium: AbstractCustomMedium,
    *,
    axis: int,
    plane_position: float,
    bounds_xyz: tuple[tuple[float, float, float], tuple[float, float, float]],
    permittivity_threshold: float | None,
    frequency: float,
    pixel_exact: bool,
    eps_components: tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]
    | None = None,
) -> tuple[list[Any], Bound2D, float, float, float, float]:
    """Create GDS contour polygons from one planar slice of a custom medium."""
    try:
        import gdstk
    except ImportError as exc:
        raise Tidy3dImportError(
            "Module 'gdstk' not found. It is required to extract custom-medium contours. "
            f"Original import error: {exc}"
        ) from exc

    data, contour_scale = _custom_medium_slice_dataarray(
        medium,
        axis=axis,
        plane_position=plane_position,
        frequency=frequency,
        pixel_exact=pixel_exact,
        bounds_xyz=bounds_xyz,
        eps_components=eps_components,
    )
    bb_min, bb_max = bounds_xyz
    w_axis, h_axis = tuple(idx for idx in (0, 1, 2) if idx != axis)
    target_frame_bounds = ((bb_min[w_axis], bb_min[h_axis]), (bb_max[w_axis], bb_max[h_axis]))
    contours, frame_bounds_data, in_plane_step, eps_min, eps_max, threshold_use = (
        gdstk_contours_from_dataarray(
            data,
            axis=axis,
            permittivity_threshold=permittivity_threshold,
            pixel_exact=pixel_exact,
            interp_method="nearest",
            bounds_xyz=bounds_xyz,
            contour_scale=contour_scale,
        )
    )
    frame_bounds = target_frame_bounds
    if pixel_exact and frame_bounds != frame_bounds_data:
        clip_box = gdstk.rectangle(frame_bounds[0], frame_bounds[1])
        contours = gdstk.boolean(contours, [clip_box], "and")
        if contours is None:
            contours = []
    return contours, frame_bounds, in_plane_step, eps_min, eps_max, threshold_use


def contours_to_polyslab_data(
    *,
    contours: list[Any],
    slab_bounds: tuple[float, float],
    axis: int,
    boundary_step: float,
    frame_bounds: Bound2D,
    in_plane_step: float,
    min_hole_area: float = 0.0,
    min_island_area: float = 0.0,
) -> ContourPolyslabData:
    """Convert contour polygons to shared contour polyslab geometry."""
    if boundary_step <= 0:
        raise ValueError("'boundary_step' must be > 0.")
    if not np.isfinite(boundary_step):
        raise ValueError("'boundary_step' must be finite.")
    if min_hole_area < 0:
        raise ValueError("'min_hole_area' must be >= 0.")
    if min_island_area < 0:
        raise ValueError("'min_island_area' must be >= 0.")

    if not contours:
        return ContourPolyslabData(
            solid_polyslabs=(),
            hole_polyslabs=(),
            solid_frame_boundary_vertex_mask=(),
            hole_frame_boundary_vertex_mask=(),
            frame_bounds=frame_bounds,
            in_plane_step=float(in_plane_step),
        )

    shapely_polygons = []
    for contour in contours:
        vertices = np.asarray(contour.points, dtype=float)
        if vertices.shape[0] < 3:
            continue
        polygon = ShapelyPolygon(vertices)
        if polygon.is_empty:
            continue
        polygon = polygon.buffer(0)
        shapely_polygons.extend(_geometry_to_polygons(polygon))

    if not shapely_polygons:
        return ContourPolyslabData(
            solid_polyslabs=(),
            hole_polyslabs=(),
            solid_frame_boundary_vertex_mask=(),
            hole_frame_boundary_vertex_mask=(),
            frame_bounds=frame_bounds,
            in_plane_step=float(in_plane_step),
        )

    merged = unary_union(shapely_polygons)
    polygons = sorted(
        [poly for poly in _geometry_to_polygons(merged) if (not poly.is_empty) and poly.area > 0],
        key=lambda poly: poly.area,
        reverse=True,
    )

    tol = max(in_plane_step * 1e-3, 1e-12)
    (wmin, hmin), (wmax, hmax) = frame_bounds

    solid_polyslabs = []
    hole_polyslabs = []
    solid_masks = []
    hole_masks = []

    for polygon in polygons:
        if float(polygon.area) < float(min_island_area):
            continue

        exterior = np.asarray(polygon.exterior.coords[:-1], dtype=float)
        exterior = _orient_ring(exterior, ccw=True)
        exterior = _densify_ring(exterior, step=boundary_step)
        solid_polyslabs.append(PolySlab(vertices=exterior, slab_bounds=slab_bounds, axis=axis))
        solid_masks.append(
            (
                (np.abs(exterior[:, 0] - wmin) <= tol)
                | (np.abs(exterior[:, 0] - wmax) <= tol)
                | (np.abs(exterior[:, 1] - hmin) <= tol)
                | (np.abs(exterior[:, 1] - hmax) <= tol)
            ).astype(bool)
        )

        for interior in polygon.interiors:
            hole = np.asarray(interior.coords[:-1], dtype=float)
            if hole.shape[0] < 3:
                continue
            if float(ShapelyPolygon(hole).area) < float(min_hole_area):
                continue
            hole = _orient_ring(hole, ccw=True)
            hole = _densify_ring(hole, step=boundary_step)
            hole_polyslabs.append(PolySlab(vertices=hole, slab_bounds=slab_bounds, axis=axis))
            hole_masks.append(
                (
                    (np.abs(hole[:, 0] - wmin) <= tol)
                    | (np.abs(hole[:, 0] - wmax) <= tol)
                    | (np.abs(hole[:, 1] - hmin) <= tol)
                    | (np.abs(hole[:, 1] - hmax) <= tol)
                ).astype(bool)
            )

    return ContourPolyslabData(
        solid_polyslabs=tuple(solid_polyslabs),
        hole_polyslabs=tuple(hole_polyslabs),
        solid_frame_boundary_vertex_mask=tuple(solid_masks),
        hole_frame_boundary_vertex_mask=tuple(hole_masks),
        frame_bounds=frame_bounds,
        in_plane_step=float(in_plane_step),
    )


def _dataarray_to_polyslab_data_and_permittivity_bounds(
    data: SpatialDataArray,
    *,
    slab_bounds: tuple[float, float] | None,
    axis: int | None = None,
    threshold: float | None = None,
    pixel_exact: bool = False,
    boundary_step: float | None = None,
    interp_method: InterpMethod = "nearest",
    contour_scale: float | None = None,
    min_hole_area: float = 0.0,
    min_island_area: float = 0.0,
) -> tuple[ContourPolyslabData, float, float]:
    """Convert a structured 2D permittivity data array and return derived min/max values."""
    if slab_bounds is None:
        raise ValueError("'slab_bounds' must be provided for 2D permittivity contour conversion.")

    coord_arrays = tuple(np.asarray(getattr(data, dim), dtype=float) for dim in "xyz")
    if any(coord.size == 0 for coord in coord_arrays):
        raise ValueError("Permittivity data coordinates must be non-empty on all axes.")

    axis_use = _resolve_axis(coord_arrays, axis)
    boundary_step_use = _resolve_boundary_step(
        coord_arrays,
        axis=axis_use,
        boundary_step=boundary_step,
    )

    (
        contours,
        frame_bounds,
        in_plane_step,
        permittivity_min,
        permittivity_max,
        _threshold_use,
    ) = gdstk_contours_from_dataarray(
        data,
        axis=axis_use,
        permittivity_threshold=threshold,
        pixel_exact=pixel_exact,
        interp_method=interp_method,
        contour_scale=contour_scale,
    )

    contour_data = contours_to_polyslab_data(
        contours=contours,
        slab_bounds=slab_bounds,
        axis=axis_use,
        boundary_step=boundary_step_use,
        frame_bounds=frame_bounds,
        in_plane_step=in_plane_step,
        min_hole_area=min_hole_area,
        min_island_area=min_island_area,
    )
    return contour_data, permittivity_min, permittivity_max


def gdstk_contours_from_dataarray(
    data: SpatialDataArray,
    *,
    axis: int,
    permittivity_threshold: float | None,
    pixel_exact: bool,
    interp_method: InterpMethod = "nearest",
    bounds_xyz: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    plane_position: float | None = None,
    contour_scale: float | None = None,
) -> tuple[list[Any], Bound2D, float, float, float, float]:
    """Create GDS contour polygons from one structured 2D permittivity data slice."""
    try:
        import gdstk
    except ImportError as exc:
        raise Tidy3dImportError(
            "Module 'gdstk' not found. It is required to extract permittivity contours. "
            f"Original import error: {exc}"
        ) from exc

    if axis not in (0, 1, 2):
        raise ValueError("'axis' must be one of {0, 1, 2}.")
    permittivity_threshold = validate_optional_finite_float(
        permittivity_threshold, name="permittivity_threshold"
    )

    coord_arrays = tuple(np.asarray(getattr(data, dim), dtype=float) for dim in "xyz")
    if any(coord.size == 0 for coord in coord_arrays):
        raise ValueError("Permittivity data coordinates must be non-empty on all axes.")
    axis_is_singleton = coord_arrays[axis].size == 1
    if not axis_is_singleton and plane_position is None:
        raise ValueError(
            "The low-level dataarray contour helper expects either a 2D slice or an explicit "
            f"'plane_position' for axis {axis}."
        )

    data_bb_min = tuple(float(coord[0]) for coord in coord_arrays)
    data_bb_max = tuple(float(coord[-1]) for coord in coord_arrays)
    if bounds_xyz is None:
        bb_min, bb_max = data_bb_min, data_bb_max
    else:
        bb_min, bb_max = bounds_xyz
    in_plane_axes = tuple(idx for idx in (0, 1, 2) if idx != axis)
    w_axis, h_axis = in_plane_axes
    frame_bounds = ((bb_min[w_axis], bb_min[h_axis]), (bb_max[w_axis], bb_max[h_axis]))

    if pixel_exact:
        if axis_is_singleton:
            eps_slice = np.real(np.asarray(data)).squeeze(axis=axis)
        else:
            plane_position_use = float(plane_position)
            coords = Coords(
                x=np.asarray(coord_arrays[0], dtype=float) if axis != 0 else plane_position_use,
                y=np.asarray(coord_arrays[1], dtype=float) if axis != 1 else plane_position_use,
                z=np.asarray(coord_arrays[2], dtype=float) if axis != 2 else plane_position_use,
            )
            eps_slice = np.real(np.asarray(coords.spatial_interp(data, interp_method))).squeeze(
                axis=axis
            )
    else:
        plane_position_use = (
            float(coord_arrays[axis][0]) if axis_is_singleton else float(plane_position)
        )
        if contour_scale is not None:
            if not axis_is_singleton:
                raise ValueError("'contour_scale' requires a 2D singleton-axis data slice.")
            scale = float(contour_scale)
            if scale <= 0:
                raise ValueError("'contour_scale' must be > 0.")
            eps_slice = np.real(np.asarray(data)).squeeze(axis=axis)
        else:
            scale = max(abs(b - a) for a, b in zip(bb_min, bb_max))
            step_candidates = [
                _coord_step_min(np.asarray(coord, dtype=float)) for coord in coord_arrays
            ]
            step_candidates = [step for step in step_candidates if np.isfinite(step)]
            if step_candidates:
                scale = min(scale, *step_candidates)
            if scale <= 0:
                raise ValueError("Failed to determine a positive contour sampling scale.")
            coords = Coords(
                x=np.arange(bb_min[0], bb_max[0] + scale * 0.9, scale)
                if axis != 0
                else plane_position_use,
                y=np.arange(bb_min[1], bb_max[1] + scale * 0.9, scale)
                if axis != 1
                else plane_position_use,
                z=np.arange(bb_min[2], bb_max[2] + scale * 0.9, scale)
                if axis != 2
                else plane_position_use,
            )
            eps_slice = np.real(np.asarray(coords.spatial_interp(data, interp_method))).squeeze(
                axis=axis
            )

    permittivity_min = float(np.min(eps_slice))
    permittivity_max = float(np.max(eps_slice))
    if permittivity_threshold is None:
        permittivity_threshold_use = 0.5 * (permittivity_min + permittivity_max)
    else:
        permittivity_threshold_use = permittivity_threshold

    if pixel_exact:
        w = np.asarray(coord_arrays[w_axis], dtype=float)
        h = np.asarray(coord_arrays[h_axis], dtype=float)
        (wmin, hmin), (wmax, hmax) = frame_bounds

        if w.size > 1:
            dw = np.diff(w) * 0.5
            wb = np.concatenate(([wmin], w[:-1] + dw, [wmax]))
        else:
            wb = np.array([wmin, wmax])

        if h.size > 1:
            dh = np.diff(h) * 0.5
            hb = np.concatenate(([hmin], h[:-1] + dh, [hmax]))
        else:
            hb = np.array([hmin, hmax])

        mask = eps_slice > permittivity_threshold_use
        w_idxs, h_idxs = np.where(mask)
        contours = [
            gdstk.rectangle((wb[wi], hb[hi]), (wb[wi + 1], hb[hi + 1]))
            for wi, hi in zip(w_idxs, h_idxs)
        ]
        step_candidates = [_coord_step_min(w), _coord_step_min(h)]
        step_candidates = [step for step in step_candidates if np.isfinite(step)]
        in_plane_step = float(min(step_candidates)) if step_candidates else 1.0
        return (
            contours,
            frame_bounds,
            in_plane_step,
            permittivity_min,
            permittivity_max,
            permittivity_threshold_use,
        )

    contours = gdstk.contour(
        eps_slice.T,
        permittivity_threshold_use,
        scale,
        precision=scale * 1e-3,
    )
    for polygon in contours:
        polygon.translate(bb_min[w_axis], bb_min[h_axis])
    return (
        contours,
        frame_bounds,
        float(scale),
        permittivity_min,
        permittivity_max,
        permittivity_threshold_use,
    )


def _structured_coord_arrays_from_component(
    component: CustomSpatialDataType,
) -> tuple[ArrayFloat1D, ArrayFloat1D, ArrayFloat1D]:
    """Extract structured x/y/z coordinate arrays from one custom-medium component."""
    from tidy3d.components.data.unstructured.base import UnstructuredGridDataset

    if isinstance(component, UnstructuredGridDataset):
        raise NotImplementedError(
            "Custom-medium contour conversion to polyslabs does not support unstructured datasets."
        )
    return tuple(np.asarray(getattr(component, dim), dtype=float) for dim in "xyz")


def _scalar_permittivity_dataarray_from_eps_diagonal(
    eps_diagonal: tuple[Any, Any, Any], coords: Coords
) -> SpatialDataArray:
    """Collapse interpolated diagonal permittivity components to one real scalar data array."""
    from tidy3d.components.data.data_array import SpatialDataArray

    coord_map = {}
    for dim, coord in zip("xyz", (coords.x, coords.y, coords.z)):
        coord_arr = np.asarray(coord, dtype=float)
        if coord_arr.ndim == 0:
            coord_arr = coord_arr.reshape((1,))
        coord_map[dim] = coord_arr

    eps_values = np.stack([np.real(np.asarray(component)) for component in eps_diagonal], axis=0)
    return SpatialDataArray(np.max(eps_values, axis=0), coords=coord_map)


def _custom_medium_slice_dataarray(
    medium: AbstractCustomMedium,
    *,
    axis: int,
    plane_position: float,
    frequency: float,
    pixel_exact: bool,
    bounds_xyz: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    eps_components: tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]
    | None = None,
) -> tuple[SpatialDataArray, float | None]:
    """Evaluate one planar custom-medium slice to a structured scalar data array."""
    if eps_components is None:
        eps_components = medium.eps_dataarray_freq(frequency=frequency)

    coord_arrays = _structured_coord_arrays_from_component(eps_components[0])
    if any(coord.size == 0 for coord in coord_arrays):
        raise ValueError("Permittivity data coordinates must be non-empty on all axes.")

    plane_position_use = float(plane_position)
    if pixel_exact:
        coords = Coords(
            x=coord_arrays[0] if axis != 0 else plane_position_use,
            y=coord_arrays[1] if axis != 1 else plane_position_use,
            z=coord_arrays[2] if axis != 2 else plane_position_use,
        )
        contour_scale = None
    else:
        if bounds_xyz is None:
            bb_min = tuple(float(coord[0]) for coord in coord_arrays)
            bb_max = tuple(float(coord[-1]) for coord in coord_arrays)
        else:
            bb_min, bb_max = bounds_xyz

        scale = max(abs(b - a) for a, b in zip(bb_min, bb_max))
        for coord in coord_arrays:
            if len(coord) > 1:
                scale = min(scale, np.diff(coord).min())
        if scale <= 0:
            raise ValueError("Failed to determine a positive contour sampling scale.")

        coords = Coords(
            x=np.arange(bb_min[0], bb_max[0] + scale * 0.9, scale)
            if axis != 0
            else plane_position_use,
            y=np.arange(bb_min[1], bb_max[1] + scale * 0.9, scale)
            if axis != 1
            else plane_position_use,
            z=np.arange(bb_min[2], bb_max[2] + scale * 0.9, scale)
            if axis != 2
            else plane_position_use,
        )
        contour_scale = float(scale)

    eps_diagonal = medium._interp_eps_diagonal_on_grid(eps_spatial=eps_components, coords=coords)
    return _scalar_permittivity_dataarray_from_eps_diagonal(eps_diagonal, coords), contour_scale


def dataarray_to_polyslab_data(
    data: SpatialDataArray,
    *,
    slab_bounds: tuple[float, float] | None,
    axis: int | None = None,
    threshold: float | None = None,
    pixel_exact: bool = False,
    boundary_step: float | None = None,
    interp_method: InterpMethod = "nearest",
    contour_scale: float | None = None,
    min_hole_area: float = 0.0,
    min_island_area: float = 0.0,
) -> ContourPolyslabData:
    """Convert a structured 2D permittivity data array to contour polyslab geometry."""
    contour_data, _permittivity_min, _permittivity_max = (
        _dataarray_to_polyslab_data_and_permittivity_bounds(
            data,
            slab_bounds=slab_bounds,
            axis=axis,
            threshold=threshold,
            pixel_exact=pixel_exact,
            boundary_step=boundary_step,
            interp_method=interp_method,
            contour_scale=contour_scale,
            min_hole_area=min_hole_area,
            min_island_area=min_island_area,
        )
    )
    return contour_data


def _custom_medium_to_polyslab_data_and_permittivity_bounds(
    medium: AbstractCustomMedium,
    *,
    slab_bounds: tuple[float, float] | None,
    axis: int | None = None,
    frequency: float | None = None,
    threshold: float | None = None,
    pixel_exact: bool = False,
    boundary_step: float | None = None,
    min_hole_area: float = 0.0,
    min_island_area: float = 0.0,
) -> tuple[ContourPolyslabData, float, float]:
    """Convert a 2D custom medium and return derived min/max values from the sampled slice."""

    if slab_bounds is None:
        raise ValueError("'slab_bounds' must be provided for 2D custom-medium conversion.")

    if frequency is None:
        frequency_eval = float("inf")
    else:
        frequency_eval = float(frequency)
        if np.isnan(frequency_eval) or np.isneginf(frequency_eval):
            raise ValueError("'frequency' must be finite or +inf.")

    eps_components = medium.eps_dataarray_freq(frequency=frequency_eval)
    coord_arrays = _structured_coord_arrays_from_component(eps_components[0])
    if any(coord.size == 0 for coord in coord_arrays):
        raise ValueError("Permittivity data coordinates must be non-empty on all axes.")

    axis_use = _resolve_axis(coord_arrays, axis)
    plane_position = float(coord_arrays[axis_use][0])
    data, contour_scale = _custom_medium_slice_dataarray(
        medium,
        axis=axis_use,
        plane_position=plane_position,
        frequency=frequency_eval,
        pixel_exact=pixel_exact,
        eps_components=eps_components,
    )
    return _dataarray_to_polyslab_data_and_permittivity_bounds(
        data,
        slab_bounds=slab_bounds,
        axis=axis_use,
        threshold=threshold,
        pixel_exact=pixel_exact,
        boundary_step=boundary_step,
        interp_method="nearest",
        contour_scale=contour_scale,
        min_hole_area=min_hole_area,
        min_island_area=min_island_area,
    )


def custom_medium_to_polyslab_data(
    medium: AbstractCustomMedium,
    *,
    slab_bounds: tuple[float, float] | None,
    axis: int | None = None,
    frequency: float | None = None,
    threshold: float | None = None,
    pixel_exact: bool = False,
    boundary_step: float | None = None,
    min_hole_area: float = 0.0,
    min_island_area: float = 0.0,
) -> ContourPolyslabData:
    """Convert a 2D custom medium to contour polyslab geometry."""
    contour_data, _permittivity_min, _permittivity_max = (
        _custom_medium_to_polyslab_data_and_permittivity_bounds(
            medium,
            slab_bounds=slab_bounds,
            axis=axis,
            frequency=frequency,
            threshold=threshold,
            pixel_exact=pixel_exact,
            boundary_step=boundary_step,
            min_hole_area=min_hole_area,
            min_island_area=min_island_area,
        )
    )
    return contour_data
