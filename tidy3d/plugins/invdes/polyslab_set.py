from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import autograd.numpy as np
import numpy as npo

from tidy3d.components.geometry.contour_conversion import (
    dataarray_to_polyslab_data,
    ordered_ring_refs_by_area,
    smooth_polygon_vertices,
    smooth_polyslabs,
)
from tidy3d.components.geometry.polyslab import PolySlab
from tidy3d.components.structure import polyslabs_to_structures
from tidy3d.plugins.invdes.self_intersection import (
    aggregate_self_intersection_statuses,
    repair_self_intersecting_ring,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tidy3d.components.data.data_array import SpatialDataArray
    from tidy3d.components.geometry.contour_conversion import ContourPolyslabData
    from tidy3d.components.material.types import StructureMediumType
    from tidy3d.components.medium import AbstractCustomMedium
    from tidy3d.components.structure import Structure
    from tidy3d.components.types import ArrayFloat1D, ArrayFloat2D, Bound2D
    from tidy3d.plugins.invdes.self_intersection import SelfIntersectionStatus


@dataclass(frozen=True)
class PolySlabSet:
    """Inverse-design contour representation enriched with frame metadata."""

    solid_polyslabs: tuple[PolySlab, ...]
    hole_polyslabs: tuple[PolySlab, ...]
    solid_frame_boundary_vertex_mask: tuple[npo.ndarray, ...]
    hole_frame_boundary_vertex_mask: tuple[npo.ndarray, ...]
    frame_bounds: Bound2D
    in_plane_step: float
    _stable_ring_refs: tuple[tuple[str, int], ...] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        expected_ring_refs = tuple(
            [("solid", idx) for idx in range(len(self.solid_polyslabs))]
            + [("hole", idx) for idx in range(len(self.hole_polyslabs))]
        )
        stable_ring_refs = self._stable_ring_refs
        if stable_ring_refs is None:
            stable_ring_refs = ordered_ring_refs_by_area(self.solid_polyslabs, self.hole_polyslabs)
        else:
            if len(stable_ring_refs) != len(expected_ring_refs):
                raise ValueError("stable_ring_refs length must match total number of rings.")
            if set(stable_ring_refs) != set(expected_ring_refs):
                raise ValueError(
                    "stable_ring_refs must be a permutation of all solid/hole ring refs."
                )
        object.__setattr__(self, "_stable_ring_refs", stable_ring_refs)

    @property
    def stable_ring_refs(self) -> tuple[tuple[str, int], ...]:
        """Read-only stable internal ring order preserved across updates."""
        stable_ring_refs = self._stable_ring_refs
        if stable_ring_refs is None:
            raise RuntimeError("stable ring refs were not initialized.")
        return stable_ring_refs

    @staticmethod
    def from_contour_data(
        contour_data: ContourPolyslabData,
        *,
        smooth_sigma: float = 0.0,
    ) -> PolySlabSet:
        """Create a polyslab set from shared contour-conversion output."""
        polyslab_set = PolySlabSet(
            solid_polyslabs=tuple(contour_data.solid_polyslabs),
            hole_polyslabs=tuple(contour_data.hole_polyslabs),
            solid_frame_boundary_vertex_mask=tuple(contour_data.solid_frame_boundary_vertex_mask),
            hole_frame_boundary_vertex_mask=tuple(contour_data.hole_frame_boundary_vertex_mask),
            frame_bounds=contour_data.frame_bounds,
            in_plane_step=contour_data.in_plane_step,
        )
        return polyslab_set.smooth(smooth_sigma)

    @staticmethod
    def from_custom_medium(
        medium: AbstractCustomMedium,
        *,
        slab_bounds: tuple[float, float] | None = None,
        axis: int | None = None,
        frequency: float | None = None,
        threshold: float | None = None,
        pixel_exact: bool = False,
        boundary_step: float | None = None,
        smooth_sigma: float = 0.0,
        min_hole_area: float = 0.0,
        min_island_area: float = 0.0,
    ) -> PolySlabSet:
        """Create a polyslab set from a custom medium slice."""
        from tidy3d.components.geometry.contour_conversion import custom_medium_to_polyslab_data

        contour_data = custom_medium_to_polyslab_data(
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
        return PolySlabSet.from_contour_data(contour_data, smooth_sigma=smooth_sigma)

    @staticmethod
    def from_dataarray(
        data: SpatialDataArray,
        *,
        slab_bounds: tuple[float, float] | None = None,
        axis: int | None = None,
        threshold: float | None = None,
        pixel_exact: bool = False,
        boundary_step: float | None = None,
        smooth_sigma: float = 0.0,
        min_hole_area: float = 0.0,
        min_island_area: float = 0.0,
    ) -> PolySlabSet:
        """Create a polyslab set from a permittivity data array."""
        contour_data = dataarray_to_polyslab_data(
            data,
            slab_bounds=slab_bounds,
            axis=axis,
            threshold=threshold,
            pixel_exact=pixel_exact,
            boundary_step=boundary_step,
            min_hole_area=min_hole_area,
            min_island_area=min_island_area,
        )
        return PolySlabSet.from_contour_data(contour_data, smooth_sigma=smooth_sigma)

    @staticmethod
    def from_polyslab(polyslab: PolySlab) -> PolySlabSet:
        """Create a polyslab set from one solid polyslab with no frozen boundary vertices."""
        vertices = np.asarray(polyslab.vertices, dtype=float)
        x_min = float(np.min(vertices[:, 0]))
        x_max = float(np.max(vertices[:, 0]))
        y_min = float(np.min(vertices[:, 1]))
        y_max = float(np.max(vertices[:, 1]))
        n_vertices = int(vertices.shape[0])
        return PolySlabSet(
            solid_polyslabs=(polyslab,),
            hole_polyslabs=(),
            solid_frame_boundary_vertex_mask=(npo.zeros((n_vertices,), dtype=bool),),
            hole_frame_boundary_vertex_mask=(),
            frame_bounds=((x_min, y_min), (x_max, y_max)),
            in_plane_step=float("inf"),
        )

    def _lookup_polyslab(self, ring_ref: tuple[str, int]) -> PolySlab:
        ring_type, ring_idx = ring_ref
        if ring_type == "solid":
            return self.solid_polyslabs[ring_idx]
        if ring_type == "hole":
            return self.hole_polyslabs[ring_idx]
        raise ValueError(f"Unsupported ring type {ring_type!r}.")

    def _lookup_mask(self, ring_ref: tuple[str, int]) -> npo.ndarray:
        ring_type, ring_idx = ring_ref
        if ring_type == "solid":
            return npo.asarray(self.solid_frame_boundary_vertex_mask[ring_idx], dtype=bool)
        if ring_type == "hole":
            return npo.asarray(self.hole_frame_boundary_vertex_mask[ring_idx], dtype=bool)
        raise ValueError(f"Unsupported ring type {ring_type!r}.")

    @property
    def ring_types(self) -> tuple[str, ...]:
        """Ring types aligned with ``polyslabs`` order."""
        return tuple(ring_type for ring_type, _ in self.stable_ring_refs)

    @property
    def polyslabs(self) -> tuple[PolySlab, ...]:
        """All polyslabs in ordered ring sequence."""
        return tuple(self._lookup_polyslab(ring_ref) for ring_ref in self.stable_ring_refs)

    @property
    def ring_vertices(self) -> tuple[ArrayFloat2D, ...]:
        """Ring vertices aligned with ``polyslabs`` order."""
        return tuple(np.array(polyslab.vertices) for polyslab in self.polyslabs)

    @property
    def ring_vertex_counts(self) -> tuple[int, ...]:
        """Number of vertices for each ring in ``polyslabs`` order."""
        return tuple(int(np.array(polyslab.vertices).shape[0]) for polyslab in self.polyslabs)

    @property
    def frame_boundary_vertex_mask(self) -> tuple[npo.ndarray, ...]:
        """Boolean per-vertex frame mask aligned with ``polyslabs`` order."""
        return tuple(self._lookup_mask(ring_ref) for ring_ref in self.stable_ring_refs)

    def flatten_ring_vertices(self) -> npo.ndarray:
        """Flatten all ring vertices into a ``(2 * sum_i N_i,)`` vector."""
        rings = self.ring_vertices
        if not rings:
            return npo.zeros((0,), dtype=float)
        return np.concatenate([np.reshape(ring, (-1,)) for ring in rings], axis=0)

    def frame_boundary_mask_flat(self, *, repeat_xy: bool = False) -> npo.ndarray:
        """Flatten frame-boundary masks into one boolean vector."""
        masks = self.frame_boundary_vertex_mask
        if not masks:
            return npo.zeros((0,), dtype=bool)

        chunks = []
        for mask in masks:
            mask_arr = npo.asarray(mask, dtype=bool).reshape(-1)
            if repeat_xy:
                mask_arr = npo.repeat(mask_arr[:, None], 2, axis=1).reshape(-1)
            chunks.append(mask_arr)
        return npo.concatenate(chunks, axis=0).astype(bool)

    def with_ring_vertices(self, ring_vertices: Sequence[ArrayFloat2D]) -> PolySlabSet:
        """Return a copy with updated ring vertices, preserving split + masks + metadata."""
        ring_vertices_list = list(ring_vertices)
        ring_refs = self.stable_ring_refs
        if len(ring_vertices_list) != len(ring_refs):
            raise ValueError("ring_vertices length must match number of polyslabs.")

        solid_polyslabs = list(self.solid_polyslabs)
        hole_polyslabs = list(self.hole_polyslabs)
        solid_masks = [
            npo.asarray(mask, dtype=bool).reshape(-1)
            for mask in self.solid_frame_boundary_vertex_mask
        ]
        hole_masks = [
            npo.asarray(mask, dtype=bool).reshape(-1)
            for mask in self.hole_frame_boundary_vertex_mask
        ]

        for ring_ref, vertices in zip(ring_refs, ring_vertices_list):
            template = self._lookup_polyslab(ring_ref)
            frame_mask = self._lookup_mask(ring_ref)
            count_expected = int(np.array(template.vertices).shape[0])

            vertices_arr = np.array(vertices)
            if int(vertices_arr.shape[0]) != count_expected:
                raise ValueError("Updated ring vertex count must match template ring count.")
            if frame_mask.reshape(-1).shape[0] != count_expected:
                raise ValueError("Frame-boundary mask length must match template ring count.")

            updated = template.updated_copy(vertices=vertices_arr)
            ring_type, ring_idx = ring_ref
            if ring_type == "solid":
                solid_polyslabs[ring_idx] = updated
                solid_masks[ring_idx] = frame_mask.reshape(-1)
            else:
                hole_polyslabs[ring_idx] = updated
                hole_masks[ring_idx] = frame_mask.reshape(-1)

        return PolySlabSet(
            solid_polyslabs=tuple(solid_polyslabs),
            hole_polyslabs=tuple(hole_polyslabs),
            solid_frame_boundary_vertex_mask=tuple(solid_masks),
            hole_frame_boundary_vertex_mask=tuple(hole_masks),
            frame_bounds=self.frame_bounds,
            in_plane_step=self.in_plane_step,
            _stable_ring_refs=self.stable_ring_refs,
        )

    def with_flat_ring_vertices(self, flat_vertices: ArrayFloat1D) -> PolySlabSet:
        """Return a copy from flattened ring-vertex vector."""
        rings = self._split_flat_ring_vertices(flat_vertices)
        return self.with_ring_vertices(rings)

    def _split_flat_ring_vertices(self, flat_vertices: ArrayFloat1D) -> list[ArrayFloat2D]:
        """Split a flat vertex vector into per-ring ``(N_i, 2)`` arrays."""
        flat = np.array(flat_vertices)
        counts = self.ring_vertex_counts
        expected = 2 * sum(counts)
        if int(flat.shape[0]) != expected:
            raise ValueError(
                f"Expected flat vertex vector of size {expected}, got {int(flat.shape[0])}."
            )

        rings = []
        start = 0
        for count in counts:
            stop = start + 2 * count
            rings.append(np.reshape(flat[start:stop], (count, 2)))
            start = stop
        return rings

    def clip_to_bounds(self, bounds_xy: Bound2D | None = None) -> PolySlabSet:
        """Hard clip ring vertices to ``((x_min, y_min), (x_max, y_max))``."""
        (x_min, y_min), (x_max, y_max) = self.frame_bounds if bounds_xy is None else bounds_xy
        clipped_rings = []
        for ring in self.ring_vertices:
            ring_arr = np.array(ring)
            clipped_x = np.clip(ring_arr[:, 0], x_min, x_max)
            clipped_y = np.clip(ring_arr[:, 1], y_min, y_max)
            clipped_rings.append(np.stack([clipped_x, clipped_y], axis=1))
        return self.with_ring_vertices(clipped_rings)

    def smooth(self, sigma: float) -> PolySlabSet:
        """Return a copy with all ring vertices Gaussian-smoothed."""
        sigma_val = sigma
        if sigma_val == 0:
            return self
        return PolySlabSet(
            solid_polyslabs=smooth_polyslabs(self.solid_polyslabs, sigma_val),
            hole_polyslabs=smooth_polyslabs(self.hole_polyslabs, sigma_val),
            solid_frame_boundary_vertex_mask=self.solid_frame_boundary_vertex_mask,
            hole_frame_boundary_vertex_mask=self.hole_frame_boundary_vertex_mask,
            frame_bounds=self.frame_bounds,
            in_plane_step=self.in_plane_step,
            _stable_ring_refs=self.stable_ring_refs,
        )

    def _prepare_updated_rings(
        self,
        new_flat_vertices: ArrayFloat1D,
        *,
        freeze_boundary: bool,
        respect_bounds: bool,
        smooth_sigma: float | None,
    ) -> list[ArrayFloat2D]:
        """Split and preprocess updated rings before validation or repair."""
        updated_rings = self._split_flat_ring_vertices(new_flat_vertices)

        if smooth_sigma is not None:
            updated_rings = [
                smooth_polygon_vertices(np.array(ring_vertices), smooth_sigma)
                for ring_vertices in updated_rings
            ]

        if freeze_boundary:
            updated_rings = [
                np.where(mask.reshape((-1, 1)), current_ring, updated_ring)
                for updated_ring, current_ring, mask in zip(
                    updated_rings,
                    self.ring_vertices,
                    self.frame_boundary_vertex_mask,
                    strict=True,
                )
            ]

        if respect_bounds:
            (x_min, y_min), (x_max, y_max) = self.frame_bounds
            updated_rings = [
                np.stack(
                    [
                        np.clip(np.array(ring)[:, 0], x_min, x_max),
                        np.clip(np.array(ring)[:, 1], y_min, y_max),
                    ],
                    axis=1,
                )
                for ring in updated_rings
            ]

        return updated_rings

    def update(
        self,
        new_flat_vertices: ArrayFloat1D,
        *,
        freeze_boundary: bool = True,
        respect_bounds: bool = True,
        smooth_sigma: float | None = None,
    ) -> PolySlabSet:
        """Apply flattened vertex updates and return a new ``PolySlabSet``.

        Parameters
        ----------
        new_flat_vertices : ArrayFloat1D
            Replacement flattened vertex coordinates in the order returned by
            :meth:`flatten_ring_vertices`.
        freeze_boundary : bool = True
            If ``True``, coordinates marked by the frame-boundary mask stay fixed at their
            current values instead of taking the proposed update.
        respect_bounds : bool = True
            If ``True``, clip every updated vertex back into ``frame_bounds`` after applying
            the update.
        smooth_sigma : float | None = None
            Optional cyclic Gaussian smoothing applied to each updated ring before frozen
            boundary coordinates are restored and bounds clipping is applied. ``None`` skips
            smoothing; ``0.0`` is a no-op.

        Notes
        -----
        This is the raw update path and preserves the current strict behavior for invalid
        polygon proposals. Use :meth:`safe_update` for best-effort self-intersection repair
        plus an explicit repair status.
        """
        flat_vertices = np.array(new_flat_vertices)
        updated = self.with_flat_ring_vertices(flat_vertices)
        if smooth_sigma is not None:
            updated = updated.smooth(smooth_sigma)
        if freeze_boundary:
            frozen_mask = self.frame_boundary_mask_flat(repeat_xy=True)
            current_flat = self.flatten_ring_vertices()
            updated_flat = updated.flatten_ring_vertices()
            updated = updated.with_flat_ring_vertices(
                np.where(frozen_mask, current_flat, updated_flat)
            )
        if respect_bounds:
            updated = updated.clip_to_bounds()
        return updated

    def safe_update(
        self,
        new_flat_vertices: ArrayFloat1D,
        *,
        freeze_boundary: bool = True,
        respect_bounds: bool = True,
        smooth_sigma: float | None = None,
    ) -> tuple[PolySlabSet, SelfIntersectionStatus]:
        """Apply flattened vertex updates with best-effort self-intersection repair.

        Parameters
        ----------
        new_flat_vertices : ArrayFloat1D
            Replacement flattened vertex coordinates in the order returned by
            :meth:`flatten_ring_vertices`.
        freeze_boundary : bool = True
            If ``True``, coordinates marked by the frame-boundary mask stay fixed at their
            current values instead of taking the proposed update.
        respect_bounds : bool = True
            If ``True``, clip every updated vertex back into ``frame_bounds`` after applying
            the update.
        smooth_sigma : float | None = None
            Optional cyclic Gaussian smoothing applied to each updated ring before frozen
            boundary coordinates are restored and bounds clipping is applied. ``None`` skips
            smoothing; ``0.0`` is a no-op.

        Returns
        -------
        tuple[PolySlabSet, SelfIntersectionStatus]
            The updated polyslab set and a status describing whether the full proposal was
            applied, a corrected update was found, or no valid corrected update could be found.
        """
        updated_rings = self._prepare_updated_rings(
            new_flat_vertices,
            freeze_boundary=freeze_boundary,
            respect_bounds=respect_bounds,
            smooth_sigma=smooth_sigma,
        )

        repaired_ring_results = [
            repair_self_intersecting_ring(
                current_vertices=current_ring,
                proposed_vertices=updated_ring,
                return_status=True,
            )
            for current_ring, updated_ring in zip(self.ring_vertices, updated_rings, strict=True)
        ]
        repaired_rings = [ring for ring, _ in repaired_ring_results]
        status = aggregate_self_intersection_statuses(
            ring_status for _, ring_status in repaired_ring_results
        )

        updated = self.with_ring_vertices(repaired_rings)
        return updated, status

    def max_edge_length(self) -> float:
        """Maximum edge length over all rings."""
        max_edge = 0.0
        for ring in self.ring_vertices:
            if int(ring.shape[0]) < 2:
                continue
            edges = np.roll(ring, -1, axis=0) - ring
            lengths = np.linalg.norm(edges, axis=1)
            if int(lengths.size) > 0:
                max_edge = max(max_edge, float(np.max(lengths)))
        return max_edge

    def to_structures(
        self,
        *,
        foreground_medium: StructureMediumType,
        background_medium: StructureMediumType,
        name_prefix: str = "design_polyslab",
    ) -> tuple[Structure, ...]:
        """Convert the polyslab set to foreground/background structures."""
        return polyslabs_to_structures(
            solid_polyslabs=self.solid_polyslabs,
            hole_polyslabs=self.hole_polyslabs,
            foreground_medium=foreground_medium,
            background_medium=background_medium,
            ring_refs=ordered_ring_refs_by_area(self.solid_polyslabs, self.hole_polyslabs),
            name_prefix=name_prefix,
        )
