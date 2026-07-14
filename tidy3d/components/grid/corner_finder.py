"""Find corners of structures on a 2D plane."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
from pydantic import Field, PositiveFloat, PositiveInt

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.geometry.base import Box, ClipOperation
from tidy3d.components.geometry.float_utils import increment_float
from tidy3d.components.geometry.utils import merging_geometries_on_plane
from tidy3d.components.geometry.vertex_utils import remove_adjacent_duplicate_vertices
from tidy3d.components.medium import PEC, LossyMetalMedium
from tidy3d.constants import fp_eps, inf

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from tidy3d.components.structure import Structure
    from tidy3d.components.types import ArrayFloat1D, ArrayFloat2D, Axis, Shapely

CORNER_ANGLE_THRESOLD = 0.25 * np.pi
# Default tolerance for treating an in-plane segment as axis-aligned (~1 degree, in radians).
AXIS_ALIGNED_ANGLE_THRESHOLD = np.pi / 180.0
# For shapely circular shapes discretization.
N_SHAPELY_QUAD_SEGS = 8
# whether to clean tiny features that sometimes occurs in shapely operations
SHAPELY_CLEANUP = False
# Compare normalized directions with a small multiple of machine epsilon.
ROUND_DIRECTION_TOL = fp_eps * 64
# Use the next representable float at at least this scale so tiny geometries still get a
# stable absolute tolerance instead of falling back to raw machine epsilon.
ROUND_GEOMETRY_TOL_SCALE_FLOOR = 1024.0
# Single-edge supports still need to be clearly above numerical noise to be meaningful.
MIN_SUPPORT_RUN_TOL_FACTOR = 8.0
# A support edge should stand out clearly from the discretized rounded-chain segments around it.
MIN_SUPPORT_CHAIN_RATIO = 2.0


class _SupportRun(NamedTuple):
    """Contiguous support edges that lie on the same straight line."""

    start_edge: int
    end_edge: int
    start_vertex: int
    end_vertex: int
    direction: ArrayFloat1D
    length: float


class CornerFinderSpec(Tidy3dBaseModel):
    """Specification for corner detection on a 2D plane."""

    medium: Literal["metal", "dielectric", "all"] = Field(
        "metal",
        title="Material Type For Corner Identification",
        description="Find corners of structures made of :class:`.Medium`, "
        "which can take value ``metal`` for PEC and lossy metal, ``dielectric`` "
        "for non-metallic materials, and ``all`` for all materials.",
    )

    angle_threshold: float = Field(
        CORNER_ANGLE_THRESOLD,
        title="Angle Threshold In Corner Identification",
        description="A vertex is qualified as a corner if the angle spanned by its two edges "
        "is larger than the supplementary angle of "
        "this threshold value.",
        ge=0,
        lt=np.pi,
    )

    distance_threshold: PositiveFloat | None = Field(
        None,
        title="Distance Threshold In Corner Identification",
        description="If not ``None`` and the distance of the vertex to its neighboring vertices "
        "is below the threshold value based on Douglas-Peucker algorithm, the vertex is disqualified as a corner.",
    )

    axis_aligned_angle_threshold: float = Field(
        AXIS_ALIGNED_ANGLE_THRESHOLD,
        title="Axis-Alignment Angle Threshold For Edge Refinement",
        description="An in-plane edge is treated as axis-aligned "
        "when its direction is within this angle of the nearest in-plane axis. "
        "Given in radians, like ``angle_threshold``.",
        ge=0,
        lt=np.pi / 4,
    )

    corner_rounding_collapse_extent: PositiveFloat | None = Field(
        None,
        title="Corner Rounding Collapse Extent",
        description="If not ``None``, collapse a small rounded or chamfered corner chain bounded "
        "by straight support edges, convex or concave, back to the support-line intersection "
        "when its local in-plane extent stays below this value and the recovered sharp corner "
        "would pass ``angle_threshold``. Useful for imported GDS/STL polygons whose sharp "
        "corners were discretized into short segments.",
    )

    concave_resolution: PositiveInt | None = Field(
        None,
        title="Concave Region Resolution.",
        description="Specifies number of steps to use for determining `dl_min` based on concave featues."
        "If set to ``None``, then the corresponding `dl_min` reduction is not applied.",
    )

    convex_resolution: PositiveInt | None = Field(
        None,
        title="Convex Region Resolution.",
        description="Specifies number of steps to use for determining `dl_min` based on convex featues."
        "If set to ``None``, then the corresponding `dl_min` reduction is not applied.",
    )

    mixed_resolution: PositiveInt | None = Field(
        None,
        title="Mixed Region Resolution.",
        description="Specifies number of steps to use for determining `dl_min` based on mixed featues."
        "If set to ``None``, then the corresponding `dl_min` reduction is not applied.",
    )

    @cached_property
    def _no_min_dl_override(self) -> bool:
        return all(
            (
                self.concave_resolution is None,
                self.convex_resolution is None,
                self.mixed_resolution is None,
            )
        )

    def _segment_axis_aligned(self, segment_vectors: ArrayFloat2D) -> NDArray[np.bool_]:
        """Per-segment mask: ``True`` where a segment lies within ``axis_aligned_angle_threshold``
        of an in-plane axis.

        Degenerate (zero-length) segments should not reach here.
        """
        threshold = self.axis_aligned_angle_threshold
        # angle from the nearer in-plane axis, in [0, pi/2]
        angle_from_x = np.arctan2(np.abs(segment_vectors[:, 1]), np.abs(segment_vectors[:, 0]))
        return (angle_from_x <= threshold) | (angle_from_x >= np.pi / 2 - threshold)

    @staticmethod
    def _group_unaligned_segments(
        axis_aligned_mask: NDArray[np.bool_],
    ) -> list[NDArray[np.intp]]:
        """Group maximal runs of consecutive axis-unaligned segments, wrapping the ring closure.

        Returns arrays of segment indices in walk order. A ring with no axis-aligned segment is a
        single run over the whole ring; a ring that is entirely axis-aligned yields no runs.
        """
        num_segments = len(axis_aligned_mask)
        if np.all(axis_aligned_mask):
            return []
        if not np.any(axis_aligned_mask):
            return [np.arange(num_segments)]
        # Rotate the walk to start at an axis-aligned segment so no run spans the array boundary;
        # a run can still wrap the ring-closure index once mapped back to the original numbering.
        start = int(np.argmax(axis_aligned_mask))
        walk_pos = np.flatnonzero(~np.roll(axis_aligned_mask, -start))
        breaks = np.flatnonzero(np.diff(walk_pos) > 1) + 1
        return [(start + run) % num_segments for run in np.split(walk_pos, breaks)]

    def _axis_unaligned_runs(self, vertices: ArrayFloat2D) -> list[ArrayFloat2D]:
        """In-plane vertices of each maximal axis-unaligned run on one polygon ring.

        ``vertices`` is the open ring (no repeated closing vertex). Each returned array holds the
        vertices spanned by one run of consecutive axis-unaligned segments (run length + 1
        vertices).
        """
        vertices = np.asarray(vertices, dtype=float)
        num_vertices = len(vertices)
        if num_vertices < 2:
            return []
        segment_vectors = np.roll(vertices, axis=0, shift=-1) - vertices
        axis_aligned_mask = self._segment_axis_aligned(segment_vectors)
        runs = []
        for segment_inds in self._group_unaligned_segments(axis_aligned_mask):
            # the run's vertices are its segments' tails plus the head of the next segment
            vertex_inds = (segment_inds[0] + np.arange(len(segment_inds) + 1)) % num_vertices
            runs.append(vertices[vertex_inds])
        return runs

    @classmethod
    def _merged_pec_on_plane(
        cls,
        normal_axis: Axis,
        coord: float,
        structure_list: list[Structure],
        center: tuple[float, float] = [0, 0, 0],
        size: tuple[float, float, float] = [inf, inf, inf],
        interior_disjoint_geometries: bool = False,
        keep_metal_only: bool = False,
    ) -> list[tuple[Any, Shapely]]:
        """On a 2D plane specified by axis = `normal_axis` and coordinate `coord`, merge geometries made of PEC.

        Parameters
        ----------
        normal_axis : Axis
            Axis normal to the 2D plane.
        coord : float
            Position of plane along the normal axis.
        structure_list : list[Structure]
            list of structures present in simulation.
        center : tuple[float, float] = [0, 0, 0]
            Center of the 2D plane (coordinate along ``axis`` is ignored)
        size : tuple[float, float, float] = [inf, inf, inf]
            Size of the 2D plane (size along ``axis`` is ignored)
        interior_disjoint_geometries: bool = False
            If ``True``, geometries of different properties on the plane must be interior disjoint.
        keep_metal_only: bool = False
            If ``True``, drop all other structures that are not made of metal.
        Returns
        -------
        list[tuple[Any, Shapely]]
            list of shapes and their property value on the plane after merging.
        """

        # Construct plane
        slice_center = list(center)
        slice_size = list(size)
        slice_center[normal_axis] = coord
        slice_size[normal_axis] = 0
        plane = Box(center=slice_center, size=slice_size)

        # prepare geometry and medium list
        geometry_list = [structure.geometry for structure in structure_list]
        # For metal, we don't distinguish between LossyMetal and PEC,
        # so they'll be merged to PEC. Other materials are considered as dielectric.
        medium_list = (structure.medium for structure in structure_list)
        medium_list = [
            PEC if (mat.is_pec or isinstance(mat, LossyMetalMedium)) else mat for mat in medium_list
        ]
        if keep_metal_only:
            geometry_list = [geo for geo, mat in zip(geometry_list, medium_list) if mat.is_pec]
            medium_list = [PEC for _ in geometry_list]
        # merge geometries
        merged_geos = merging_geometries_on_plane(
            geometry_list,
            plane,
            medium_list,
            interior_disjoint_geometries,
            cleanup=SHAPELY_CLEANUP,
            quad_segs=N_SHAPELY_QUAD_SEGS,
        )

        return merged_geos

    def _polygons_from_merged_geos(self, merged_geos: list[tuple[Any, Shapely]]) -> list[Shapely]:
        """Disjoint polygons from a merged-geometry list, keeping only those whose material
        matches ``medium``. Zero-area shapes (lines and points) are dropped."""
        polygons = []
        for mat, shapes in merged_geos:
            if self.medium != "all" and mat.is_pec != (self.medium == "metal"):
                continue
            polygons.extend(ClipOperation.to_polygon_list(shapes))
        return polygons

    def _corners_and_convexity(
        self,
        merged_geos: list[tuple[Any, Shapely]],
        ravel: bool,
    ) -> tuple[ArrayFloat2D, ArrayFloat1D]:
        """On a 2D plane specified by axis = `normal_axis` and coordinate `coord`, find out corners of merged
        geometries made of PEC.


        Parameters
        ----------
        merged_geos : list[tuple[Any, Shapely]]
            list of shapes and their property value on the plane after merging.
        ravel : bool
            Whether to put the resulting corners in a single list or per polygon.

        Returns
        -------
        tuple[ArrayFloat2D, ArrayFloat1D]
            Corner coordinates and their convexity.
        """
        # corner finder
        corner_list = []
        convexity_list = []
        for mat, shapes in merged_geos:
            if self.medium != "all" and mat.is_pec != (self.medium == "metal"):
                continue
            polygon_list = ClipOperation.to_polygon_list(shapes)
            for poly in polygon_list:
                poly = poly.normalize().buffer(0)
                if self.distance_threshold is not None:
                    poly = poly.simplify(self.distance_threshold, preserve_topology=True)
                corners_xy, corners_convexity = self._filter_collinear_vertices(
                    self._collapse_rounded_corners(list(poly.exterior.coords))
                )
                corner_list.append(corners_xy)
                convexity_list.append(corners_convexity)
                # in case the polygon has holes
                for poly_inner in poly.interiors:
                    corners_xy, corners_convexity = self._filter_collinear_vertices(
                        self._collapse_rounded_corners(list(poly_inner.coords))
                    )
                    corner_list.append(corners_xy)
                    convexity_list.append(corners_convexity)
        return self._ravel_corners_and_convexity(ravel, corner_list, convexity_list)

    @staticmethod
    def _close_ring(points: ArrayFloat2D) -> ArrayFloat2D:
        """Append the first point to the end of an open ring."""
        return np.vstack([points, points[0]])

    def _collapse_rounded_corners(self, vertices: ArrayFloat2D) -> ArrayFloat2D:
        """Collapse short rounded runs between straight support edges.

        The detector scans each polygon ring for a short intermediate chain between two
        straight support edges. The missing sharp corner is inferred from the intersection of
        those two support lines, then the chain is collapsed only if the recovered sharp corner
        would pass the legacy angle threshold and the chain remains local and monotone in the
        support-edge basis.
        """

        max_extent = self.corner_rounding_collapse_extent
        if max_extent is None:
            return vertices

        vertices = np.asarray(vertices, dtype=float)
        points = vertices[:-1]
        if len(points) < 4:
            return vertices

        tol = self._corner_geometry_tolerance(points, max_extent)
        dir_tol = ROUND_DIRECTION_TOL
        points = remove_adjacent_duplicate_vertices(points, tol=tol, drop_closing_duplicate=True)
        if len(points) < 4:
            return self._close_ring(points)

        # Normalize all polygon edges together so the later support-run scan can stay vectorized.
        edge_dirs, edge_lengths = self._edge_directions(points)

        start_ind = self._first_direction_change(edge_dirs, dir_tol)
        points = np.roll(points, axis=0, shift=-start_ind)
        edge_dirs = np.roll(edge_dirs, axis=0, shift=-start_ind)
        edge_lengths = np.roll(edge_lengths, axis=0, shift=-start_ind)

        # Open the cyclic ring at a direction change so the run builder never has to merge
        # across the array boundary.
        support_runs, edge_to_run = self._support_runs(edge_dirs, edge_lengths, dir_tol)
        if len(support_runs) < 2:
            return self._close_ring(points)

        best_chain_cache: dict[int, tuple[int, ArrayFloat1D, int] | None] = {}

        def get_best_rounded_chain(start_run_ind: int) -> tuple[int, ArrayFloat1D, int] | None:
            """Cache rounded-chain candidates for one collapse pass."""
            if start_run_ind not in best_chain_cache:
                best_chain_cache[start_run_ind] = self._best_rounded_chain(
                    points=points,
                    support_runs=support_runs,
                    start_run_ind=start_run_ind,
                    max_extent=max_extent,
                    tol=tol,
                    dir_tol=dir_tol,
                )
            return best_chain_cache[start_run_ind]

        start_run_ind = self._scan_start_run(support_runs, get_best_rounded_chain)

        collapsed = []
        num_points = len(points)
        # Start the cyclic walk at the chosen anchor run's start vertex so the array boundary
        # stays before that run without physically rotating the vertex or support-run arrays.
        ind = support_runs[start_run_ind].start_vertex
        consumed = 0

        while consumed < num_points:
            current_run_ind = edge_to_run[(ind - 1) % num_points]
            if support_runs[current_run_ind].end_vertex == ind:
                candidate = get_best_rounded_chain(current_run_ind)
                if candidate is not None:
                    end_run_ind, corner, chain_size = candidate
                    collapsed.append(corner)
                    consumed += chain_size
                    ind = (support_runs[end_run_ind].start_edge + 1) % num_points
                    continue

            collapsed.append(points[ind])
            ind = (ind + 1) % num_points
            consumed += 1

        collapsed = remove_adjacent_duplicate_vertices(
            np.array(collapsed, dtype=float), tol=tol, drop_closing_duplicate=True
        )
        if len(collapsed) < 3:
            return self._close_ring(points)
        return self._close_ring(collapsed)

    @staticmethod
    def _cyclic_vertex_slice(points: ArrayFloat2D, start_ind: int, end_ind: int) -> ArrayFloat2D:
        """Vertices from ``start_ind`` to ``end_ind`` on a cyclic ring, inclusive."""
        if start_ind <= end_ind:
            return points[start_ind : end_ind + 1]
        return np.concatenate((points[start_ind:], points[: end_ind + 1]), axis=0)

    @staticmethod
    def _corner_geometry_tolerance(points: ArrayFloat2D, max_extent: float) -> float:
        """Position tolerance from scale-based floating-point spacing."""
        span = float(np.max(np.ptp(points, axis=0)))
        scale = max(span, max_extent, ROUND_GEOMETRY_TOL_SCALE_FLOOR)
        return increment_float(scale, 1) - scale

    @staticmethod
    def _edge_directions(points: ArrayFloat2D) -> tuple[ArrayFloat2D, ArrayFloat1D]:
        """Return normalized edge directions and lengths for one polygon ring."""
        deltas = np.roll(points, axis=0, shift=-1) - points
        lengths = np.linalg.norm(deltas, axis=1)
        return deltas / lengths[:, np.newaxis], lengths

    @staticmethod
    def _cross_2d(
        vec1: ArrayFloat1D | ArrayFloat2D, vec2: ArrayFloat1D | ArrayFloat2D
    ) -> float | ArrayFloat1D:
        """Scalar 2D cross product for one vector pair or an array of vector pairs."""
        return vec1[..., 0] * vec2[..., 1] - vec1[..., 1] * vec2[..., 0]

    @staticmethod
    def _directions_form_supported_corner(
        dir1: ArrayFloat1D, dir2: ArrayFloat1D, angle_threshold: float, dir_tol: float
    ) -> bool:
        """Whether the support-edge turn would be kept by the legacy sharp-corner filter."""
        cross = float(CornerFinderSpec._cross_2d(dir1, dir2))
        if abs(cross) <= dir_tol:
            return False
        # ``dir1`` follows the polygon into the corner, while ``dir2`` leaves it. The legacy
        # detector compares the outgoing rays from the sharp corner, so its angle test becomes a
        # minimum turn-angle test on the traversal directions used here.
        return float(np.dot(dir1, dir2)) <= np.cos(angle_threshold)

    def _first_direction_change(self, edge_dirs: ArrayFloat2D, dir_tol: float) -> int:
        """Pick a deterministic edge-direction change to open the cyclic ring."""
        changed = np.abs(self._cross_2d(np.roll(edge_dirs, axis=0, shift=1), edge_dirs)) > dir_tol
        change_inds = np.flatnonzero(changed)
        return int(change_inds[0]) if len(change_inds) > 0 else 0

    @staticmethod
    def _support_runs(
        edge_dirs: ArrayFloat2D,
        edge_lengths: ArrayFloat1D,
        dir_tol: float,
    ) -> tuple[list[_SupportRun], list[int]]:
        """Merge adjacent collinear edges into straight support runs.

        The ring has already been opened at a direction change, so this only has to segment the
        linearized edge array and never merge across the array boundary.
        """
        num_edges = len(edge_dirs)
        changed = np.abs(CornerFinderSpec._cross_2d(edge_dirs[:-1], edge_dirs[1:])) > dir_tol
        run_starts = np.concatenate(([0], np.flatnonzero(changed) + 1))
        run_ends = np.concatenate((run_starts[1:] - 1, [num_edges - 1]))
        run_sizes = run_ends - run_starts + 1
        cumulative_lengths = np.concatenate(([0.0], np.cumsum(edge_lengths)))
        run_lengths = cumulative_lengths[run_ends + 1] - cumulative_lengths[run_starts]
        edge_to_run = np.repeat(np.arange(len(run_starts)), run_sizes).tolist()

        runs = [
            _SupportRun(
                start_edge=int(start_edge),
                end_edge=int(end_edge),
                start_vertex=int(start_edge),
                end_vertex=int((end_edge + 1) % num_edges),
                direction=edge_dirs[start_edge],
                length=float(run_length),
            )
            for start_edge, end_edge, run_length in zip(run_starts, run_ends, run_lengths)
        ]
        return runs, edge_to_run

    def _scan_start_run(
        self,
        support_runs: list[_SupportRun],
        get_best_rounded_chain: Callable[[int], tuple[int, ArrayFloat1D, int] | None],
    ) -> int:
        """Pick a support run that begins a valid rounded-chain collapse when possible."""
        for run_ind in range(len(support_runs)):
            if get_best_rounded_chain(run_ind) is not None:
                return run_ind
        return 0

    @classmethod
    def _support_line_corner(
        cls,
        start_point: ArrayFloat1D,
        end_point: ArrayFloat1D,
        start_dir: ArrayFloat1D,
        end_dir: ArrayFloat1D,
    ) -> ArrayFloat1D:
        """Return the intersection of two non-parallel support lines."""
        # Callers only pass normalized directions that already passed the legacy corner-angle
        # gate and are non-parallel, so the support-line intersection is well-defined here.
        det = cls._cross_2d(start_dir, end_dir)
        offset = end_point - start_point
        factor = cls._cross_2d(offset, end_dir) / det
        return start_point + factor * start_dir

    @classmethod
    def _support_basis(cls, start_dir: ArrayFloat1D, end_dir: ArrayFloat1D) -> ArrayFloat2D:
        """Return an orthonormal local basis aligned with the two support directions."""
        basis0 = start_dir
        basis1 = end_dir - float(np.dot(end_dir, basis0)) * basis0
        # Callers only pass normalized directions that already passed the corner-angle gate and
        # are non-parallel, so the second basis vector cannot collapse here.
        basis1_norm = np.linalg.norm(basis1)
        basis1 = basis1 / basis1_norm
        return np.vstack((basis0, basis1))

    def _best_rounded_chain(
        self,
        points: ArrayFloat2D,
        support_runs: list[_SupportRun],
        start_run_ind: int,
        max_extent: float,
        tol: float,
        dir_tol: float,
    ) -> tuple[int, ArrayFloat1D, int] | None:
        """Find the longest valid rounded chain after ``start_run_ind`` if one exists."""
        num_runs = len(support_runs)
        if num_runs < 3:
            return None

        start_run = support_runs[start_run_ind]
        start_ind = start_run.end_vertex
        start_dir = start_run.direction
        best_candidate = None

        for step in range(2, num_runs):
            end_run_ind = (start_run_ind + step) % num_runs
            end_run = support_runs[end_run_ind]
            end_dir = end_run.direction
            if not self._directions_form_supported_corner(
                start_dir, end_dir, self.angle_threshold, dir_tol
            ):
                continue

            end_vertex = end_run.start_vertex
            chain = self._cyclic_vertex_slice(points, start_ind, end_vertex)
            # The supports define the missing sharp corner; the intermediate chain is what we
            # test as a rounded or chamfered approximation to that corner.
            if not self._has_viable_support_lengths(start_run, end_run, chain, tol):
                continue
            corner = self._support_line_corner(chain[0], chain[-1], start_dir, end_dir)
            if self._is_valid_rounded_component(chain, corner, start_dir, end_dir, max_extent, tol):
                best_candidate = (end_run_ind, corner, len(chain))

        return best_candidate

    @staticmethod
    def _has_viable_support_lengths(
        start_run: _SupportRun, end_run: _SupportRun, chain: ArrayFloat2D, tol: float
    ) -> bool:
        """Require meaningful straight supports and reject local arc fragments.

        Rounded-chain fragments can also look locally monotone, so the flanking supports must be
        clearly longer than the chain segments they bracket. That keeps the detector focused on
        real straight-edge / rounded-corner / straight-edge patterns instead of partial arcs.
        """
        chain_edge_lengths = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        min_support = min(start_run.length, end_run.length)
        if min_support <= MIN_SUPPORT_RUN_TOL_FACTOR * tol:
            return False
        return min_support > MIN_SUPPORT_CHAIN_RATIO * float(chain_edge_lengths.max()) + tol

    @classmethod
    def _is_valid_rounded_component(
        cls,
        chain: ArrayFloat2D,
        corner: ArrayFloat1D,
        start_dir: ArrayFloat1D,
        end_dir: ArrayFloat1D,
        max_extent: float,
        tol: float,
    ) -> bool:
        """Whether the rounded run stays local and monotone in the support basis."""
        basis = cls._support_basis(start_dir, end_dir)

        # Work in the local support basis so the same checks apply to axis-aligned and rotated
        # geometry. A valid rounded/chamfered chain stays inside one local quadrant and moves
        # monotonically from one support line to the other.
        local_chain = np.matmul(chain - corner, basis.T)
        max_deviation = np.max(np.abs(local_chain), axis=0)
        if np.any(max_deviation > max_extent + tol):
            return False

        local_diffs = np.diff(local_chain, axis=0)
        nonzero_motion = np.abs(local_diffs) > tol
        if not np.all(np.any(nonzero_motion, axis=0)):
            return False

        positive_motion = np.all(np.where(nonzero_motion, local_diffs > 0, True), axis=0)
        negative_motion = np.all(np.where(nonzero_motion, local_diffs < 0, True), axis=0)
        return bool(np.all(positive_motion | negative_motion))

    def _ravel_corners_and_convexity(
        self, ravel: bool, corner_list: list[ArrayFloat2D], convexity_list: list[ArrayFloat1D]
    ) -> tuple[ArrayFloat2D, ArrayFloat1D]:
        """Whether to put the resulting corners in a single list or per polygon."""
        if ravel and len(corner_list) > 0:
            return np.concatenate(corner_list), np.concatenate(convexity_list)
        return corner_list, convexity_list

    def corners(
        self,
        normal_axis: Axis,
        coord: float,
        structure_list: list[Structure],
        center: tuple[float, float, float] = [0, 0, 0],
        size: tuple[float, float, float] = [inf, inf, inf],
        interior_disjoint_geometries: bool = False,
        keep_metal_only: bool = False,
    ) -> ArrayFloat2D:
        """On a 2D plane specified by axis = `normal_axis` and coordinate `coord`, find out corners of merged
        geometries made of `medium`.


        Parameters
        ----------
        normal_axis : Axis
            Axis normal to the 2D plane.
        coord : float
            Position of plane along the normal axis.
        structure_list : list[Structure]
            list of structures present in simulation.
        interior_disjoint_geometries: bool = False
            If ``True``, geometries made of different materials on the plane must not be overlapping.
        center : tuple[float, float, float]=[0, 0, 0]
            Center of the 2D plane (coordinate along ``axis`` is ignored).
        size : tuple[float, float, float]=[inf, inf, inf]
            Size of the 2D plane (size along ``axis`` is ignored).
        keep_metal_only: bool = False
            If ``True``, drop all other structures that are not made of metal.
        Returns
        -------
        ArrayFloat2D
            Corner coordinates.
        """
        merged_geos = self._merged_pec_on_plane(
            normal_axis=normal_axis,
            coord=coord,
            structure_list=structure_list,
            center=center,
            size=size,
            interior_disjoint_geometries=interior_disjoint_geometries,
            keep_metal_only=keep_metal_only,
        )
        corner_list, _ = self._corners_and_convexity(
            merged_geos=merged_geos,
            ravel=True,
        )
        return corner_list

    def _filter_collinear_vertices(
        self, vertices: ArrayFloat2D
    ) -> tuple[ArrayFloat2D, ArrayFloat1D]:
        """Filter collinear vertices of a polygon, and return corners locations and their convexity.

        Parameters
        ----------
        vertices : ArrayFloat2D
            Polygon vertices from shapely.Polygon. The last vertex is identical to the 1st
            vertex to make a valid polygon.

        Returns
        -------
        ArrayFloat2D
            Corner coordinates.
        ArrayFloat1D
            Convexity of corners: True for outer corners, False for inner corners.
        """

        def normalize(v: NDArray) -> NDArray:
            return v / np.linalg.norm(v, axis=-1)[:, np.newaxis]

        # drop the last vertex, which is identical to the 1st one.
        vs_orig = np.array(vertices[:-1])
        # compute unit vector to next and previous vertex
        vs_next = np.roll(vs_orig, axis=0, shift=-1)
        vs_previous = np.roll(vs_orig, axis=0, shift=+1)
        unit_next = normalize(vs_next - vs_orig)
        unit_previous = normalize(vs_previous - vs_orig)
        # angle
        inner_product = np.sum(unit_next * unit_previous, axis=-1)
        inner_product = np.where(inner_product > 1, 1, inner_product)
        inner_product = np.where(inner_product < -1, -1, inner_product)
        angle = np.arccos(inner_product)
        num_vs = len(vs_orig)
        cross_product = np.cross(
            np.hstack([unit_next, np.zeros((num_vs, 1))]),
            np.hstack([unit_previous, np.zeros((num_vs, 1))]),
            axis=-1,
        )
        convexity = cross_product[:, 2] < 0
        ind_filter = angle <= np.pi - self.angle_threshold
        return vs_orig[ind_filter], convexity[ind_filter]
