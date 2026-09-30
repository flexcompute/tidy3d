"""Shape-gradient integrand computation from point-cloud monitor data.

This is the consumption half of the point-cloud adjoint path: for one traced
structure, walk the point-cloud chunk monitors one at a time (the artifact's entry
order is the deterministic index contract shared with monitor construction) and
evaluate the shape-gradient integrand through the same pure kernels the legacy
volumetric path uses (``dielectric_gradient_from_samples``,
``pec_gradient_from_samples``, ``blend_pec_dielectric_gradient``,
``clip_active_from_inside_check``), so the two paths share the arithmetic by
construction. Each chunk is evaluated against the sample-set segments it covers and
written into preallocated per-set outputs, so the extraction working set stays
bounded by the per-monitor point cap no matter how many chunks a structure's
surface sampling was split into.

The output is one ``(N_key, F)`` integrand array per canonical sample-set key,
attached to ``DerivativeInfo.point_integrands`` (frequency-chunk-sliced) and read by
the ``sample_set_integrand`` consumption seam in place of interpolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tidy3d.config import config
from tidy3d.exceptions import AdjointError

from .derivative_utils import (
    blend_pec_dielectric_gradient,
    clip_active_from_inside_check,
    dielectric_gradient_from_samples,
    pec_gradient_from_samples,
)
from .monitor_names import adjoint_monitor_name

if TYPE_CHECKING:
    from tidy3d.components.data.data_array import FreqDataArray
    from tidy3d.components.data.monitor_data import (
        PointCloudFieldData,
        PointCloudPermittivityData,
    )
    from tidy3d.components.data.sim_data import SimulationData
    from tidy3d.components.geometry.utils import GeometryType
    from tidy3d.em.translate.sample_sets import StructureSampleSets

    from .types import PathType

    PointCloudMonitorData = PointCloudFieldData | PointCloudPermittivityData

_E_COMPONENTS = ("Ex", "Ey", "Ez")
_D_COMPONENTS = ("Dx", "Dy", "Dz")
_H_COMPONENTS = ("Hx", "Hy", "Hz")
_EPS_COMPONENTS = ("xx", "yy", "zz")


def _adjoint_point_cloud_chunk_size() -> int:
    """Maximum rows per staged point-cloud monitor (import deferred: cycle)."""
    from tidy3d.components.monitor import (
        MAX_POINT_CLOUD_FIELD_MONITOR_POINTS,
        MAX_POINT_CLOUD_PERMITTIVITY_MONITOR_POINTS,
    )

    return min(MAX_POINT_CLOUD_FIELD_MONITOR_POINTS, MAX_POINT_CLOUD_PERMITTIVITY_MONITOR_POINTS)


def adjoint_point_cloud_chunk_bounds(num_points: int) -> list[tuple[int, int]]:
    """Row ranges splitting one structure's concatenated point stream across monitors.

    The two-sided deterministic chunking contract: construction slices each staged
    stream (field query points, per-component PEC points, per-component material
    points — all ``num_points`` rows in artifact entry order) into these ranges, one
    monitor per range, and consumption processes the recorded chunks against these
    same ranges. Structures whose surface sampling exceeds a single monitor's point
    cap therefore split transparently, with no stored payload and no change to the
    per-set offset bookkeeping downstream.
    """
    chunk_size = _adjoint_point_cloud_chunk_size()
    return [
        (start, min(start + chunk_size, num_points)) for start in range(0, num_points, chunk_size)
    ]


def adjoint_point_cloud_chunk_tag(monitor_tag: str, chunk_index: int) -> str:
    """Monitor monitor tag of one chunk; chunk 0 keeps the unsplit base tag."""
    return monitor_tag if chunk_index == 0 else f"{monitor_tag}_c{chunk_index}"


def _chunked_monitor_names(structure_index: int, monitor_tag: str, num_chunks: int) -> list[str]:
    """Monitor names of one staged stream's chunks, in row order."""
    return [
        adjoint_monitor_name(
            structure_index, adjoint_point_cloud_chunk_tag(monitor_tag, chunk_index)
        )
        for chunk_index in range(num_chunks)
    ]


def _normalized_components(
    data: PointCloudMonitorData,
    components: tuple[str, ...],
    frequencies: np.ndarray,
    scale: float | FreqDataArray | None = None,
) -> dict[str, np.ndarray]:
    """Select the frequency subset once, normalize once, and convert to numpy.

    The returned arrays are row-sliced per sample set with plain numpy views, so
    the frequency selection and adjoint post-normalization run once per monitor
    component instead of once per canonical set. Values are cast to the configured
    gradient precision (``config.adjoint.gradient_dtype_complex``), matching the
    legacy volumetric path's interpolator casting.
    """
    complex_dtype = config.adjoint.gradient_dtype_complex
    sampled = {}
    for component in components:
        arr = data.field_components[component].sel(f=frequencies)
        if scale is not None:
            arr = arr * scale
        sampled[component] = np.asarray(arr.values, dtype=complex_dtype)
    return sampled


def _component_rows(
    all_components: dict[str, np.ndarray], index_slice: slice
) -> dict[str, np.ndarray]:
    """One row segment of pre-extracted per-component arrays (numpy views)."""
    return {name: values[index_slice] for name, values in all_components.items()}


def compute_point_integrands(
    sample_sets: StructureSampleSets,
    structure_index: int,
    sim_data_fwd: SimulationData,
    sim_data_adj: SimulationData,
    frequencies: np.ndarray,
    adjoint_post_norm: float | FreqDataArray,
    is_medium_pec: bool,
    background_medium_is_pec: bool,
    clipped_geometry: GeometryType | None = None,
    material_length_scale: float | None = None,
) -> dict[PathType, np.ndarray]:
    """Per-set shape-gradient integrands from this structure's point-cloud data.

    ``sample_sets`` is the structure's staged ``StructureSampleSets`` (uniform staging
    and PEC presence are validated model invariants); entries are consumed in artifact
    entry order, the monitor concatenation order. ``sim_data_fwd`` / ``sim_data_adj``
    hold the forward and adjoint point-cloud monitor data (``adjoint_post_norm`` is
    applied to the adjoint samples, matching the volumetric path's
    post-normalization). ``frequencies`` may be any subset of the recorded adjoint
    frequencies: callers chunk it under the adjoint frequency memory budget, and each
    call touches only that chunk's data. Returns ``{canonical key: (N_key, F_chunk)
    integrand}`` for every non-empty set; empty sets are omitted (consumption skips
    them).

    PEC-vs-dielectric blending is driven by the artifact's per-side masks; a
    dielectric structure whose masks detect PEC outside takes the background-PEC
    branch even without an explicit ``background_medium`` hint (automatic detection,
    over-inclusion-safe).

    Chunk monitors are consumed one at a time against the sample-set segments they
    cover (all kernels are elementwise along the point axis, so segment evaluation
    is exact), keeping the extraction working set bounded by the per-monitor point
    cap rather than the structure's total point count.
    """
    staged_entries = [
        (entry.key, entry.sample_set)
        for entry in sample_sets.entries
        if entry.sample_set.num_points > 0
    ]
    if not staged_entries:
        return {}

    # global row ranges of each set in the concatenated stream (artifact entry order)
    set_ranges = []
    total_points = 0
    for key, sample_set in staged_entries:
        set_ranges.append((key, sample_set, total_points, total_points + sample_set.num_points))
        total_points += sample_set.num_points

    chunk_bounds = adjoint_point_cloud_chunk_bounds(total_points)
    num_chunks = len(chunk_bounds)
    fld_pc_names = _chunked_monitor_names(structure_index, "fld_pc", num_chunks)

    fwd_monitor_names = {monitor.name for monitor in sim_data_fwd.simulation.monitors}
    missing_names = [name for name in fld_pc_names if name not in fwd_monitor_names]
    if missing_names:
        raise AdjointError(
            "Rerun the forward simulation to regenerate its adjoint data: the "
            f"shape-derivative sample data for structure {structure_index} expects "
            f"point-cloud monitor data ('{missing_names[0]}') that the forward data does "
            "not contain, so the two are out of sync."
        )

    pec_staged = sample_sets.pec_staged
    # geometry and payload arrays follow the configured gradient precision so kernel
    # promotion never silently upcasts single-precision monitor data to double
    float_dtype = config.adjoint.gradient_dtype_float
    integrands: dict[PathType, np.ndarray] = {
        tuple(key): np.empty(
            (sample_set.num_points, len(frequencies)),
            dtype=config.adjoint.gradient_dtype_complex,
        )
        for key, sample_set in staged_entries
    }

    for chunk_index, (chunk_start, chunk_end) in enumerate(chunk_bounds):
        # extract this chunk's monitor data to numpy once (frequency selection and
        # adjoint post-normalization applied here, never per set); working memory
        # stays bounded by the chunk row range regardless of total_points
        fld_name = fld_pc_names[chunk_index]
        E_fwd_chunk = _normalized_components(sim_data_fwd[fld_name], _E_COMPONENTS, frequencies)
        D_fwd_chunk = _normalized_components(sim_data_fwd[fld_name], _D_COMPONENTS, frequencies)
        E_adj_chunk = _normalized_components(
            sim_data_adj[fld_name], _E_COMPONENTS, frequencies, scale=adjoint_post_norm
        )
        D_adj_chunk = _normalized_components(
            sim_data_adj[fld_name], _D_COMPONENTS, frequencies, scale=adjoint_post_norm
        )
        eps_chunk = {
            (side, component): _normalized_components(
                sim_data_fwd[
                    adjoint_monitor_name(
                        structure_index,
                        adjoint_point_cloud_chunk_tag(f"eps_pc_{side}_{component}", chunk_index),
                    )
                ],
                (f"eps_{component}",),
                frequencies,
            )[f"eps_{component}"]
            for side in ("in", "out")
            for component in _EPS_COMPONENTS
        }

        pec_fwd_chunk: dict[tuple[str, str], np.ndarray] = {}
        pec_adj_chunk: dict[tuple[str, str], np.ndarray] = {}
        if pec_staged:
            for side in ("out", "in"):
                for component in (*_E_COMPONENTS, *_H_COMPONENTS):
                    name = adjoint_monitor_name(
                        structure_index,
                        adjoint_point_cloud_chunk_tag(
                            f"fld_pc_{component.lower()}_{side}", chunk_index
                        ),
                    )
                    pec_fwd_chunk[(side, component)] = _normalized_components(
                        sim_data_fwd[name], (component,), frequencies
                    )[component]
                    pec_adj_chunk[(side, component)] = _normalized_components(
                        sim_data_adj[name], (component,), frequencies, scale=adjoint_post_norm
                    )[component]

        for key, sample_set, set_start, set_end in set_ranges:
            segment_start = max(set_start, chunk_start)
            segment_end = min(set_end, chunk_end)
            if segment_start >= segment_end:
                continue
            # the same rows addressed in the chunk arrays and in the set's own arrays
            chunk_slice = slice(segment_start - chunk_start, segment_end - chunk_start)
            set_slice = slice(segment_start - set_start, segment_end - set_start)

            points = np.asarray(sample_set.points.values, dtype=float_dtype)[set_slice]
            normals = np.asarray(sample_set.normals.values, dtype=float_dtype)[set_slice]
            perps1 = np.asarray(sample_set.perps1.values, dtype=float_dtype)[set_slice]
            perps2 = np.asarray(sample_set.perps2.values, dtype=float_dtype)[set_slice]

            eps_in = {
                f"eps_{component}": eps_chunk[("in", component)][chunk_slice]
                for component in _EPS_COMPONENTS
            }
            eps_out = {
                f"eps_{component}": eps_chunk[("out", component)][chunk_slice]
                for component in _EPS_COMPONENTS
            }

            vjps_dielectric = dielectric_gradient_from_samples(
                E_fwd=_component_rows(E_fwd_chunk, chunk_slice),
                E_adj=_component_rows(E_adj_chunk, chunk_slice),
                D_fwd=_component_rows(D_fwd_chunk, chunk_slice),
                D_adj=_component_rows(D_adj_chunk, chunk_slice),
                eps_in=eps_in,
                eps_out=eps_out,
                normals=normals,
                perps1=perps1,
                perps2=perps2,
            )

            mask_pec_outside = None
            mask_pec_inside = None
            vjps_pec_fields_outside = None
            vjps_pec_fields_inside = None
            background_pec_effective = background_medium_is_pec

            if pec_staged:
                pec = sample_set.staging.pec_sampling

                def _pec_side_vjps(
                    side: str,
                    eps_side: dict[str, np.ndarray],
                    pec: Any = pec,
                    chunk_slice: slice = chunk_slice,
                    set_slice: slice = set_slice,
                    pec_fwd_chunk: dict = pec_fwd_chunk,
                    pec_adj_chunk: dict = pec_adj_chunk,
                    normals: np.ndarray = normals,
                    perps1: np.ndarray = perps1,
                    perps2: np.ndarray = perps2,
                ) -> np.ndarray:
                    side_payload = pec.outside if side == "out" else pec.inside
                    field_kwargs = {}
                    for group, components, source in (
                        ("E_fwd", _E_COMPONENTS, pec_fwd_chunk),
                        ("E_adj", _E_COMPONENTS, pec_adj_chunk),
                        ("H_fwd", _H_COMPONENTS, pec_fwd_chunk),
                        ("H_adj", _H_COMPONENTS, pec_adj_chunk),
                    ):
                        field_kwargs[group] = {
                            component: source[(side, component)][chunk_slice]
                            for component in components
                        }
                    return pec_gradient_from_samples(
                        **field_kwargs,
                        eps_dielectric=eps_side,
                        normals=normals,
                        perps1=perps1,
                        perps2=perps2,
                        edge_distance_e=np.asarray(
                            side_payload.edge_distance_e.values, dtype=float_dtype
                        )[set_slice],
                        edge_distance_h=np.asarray(
                            side_payload.edge_distance_h.values, dtype=float_dtype
                        )[set_slice],
                        line_integration=any(pec.flat_perp_dims),
                        flat_perp_dims=tuple(pec.flat_perp_dims),
                    )

                # masks are frequency-independent at generation; broadcast against (N, F)
                mask_pec_outside = np.asarray(pec.outside.pec_mask.values, dtype=float_dtype)[
                    set_slice, None
                ]
                # segment-local branch guard: exact, because every guarded blend branch
                # coincides with the unguarded formula wherever the mask is zero
                has_pec_outside = bool(np.any(mask_pec_outside > 0))
                # automatic background detection: a dielectric structure with PEC found
                # outside takes the background-PEC branch without an explicit hint
                background_pec_effective = background_medium_is_pec or (
                    not is_medium_pec and has_pec_outside
                )

                if is_medium_pec:
                    # the staged mask classifies the inside node against the analytic
                    # geometry, which is ambiguous for nodes lying on a face (grid/bound
                    # round-off); a PEC-like recorded eps_in also marks the inside as PEC,
                    # as the legacy value-based detection did, so the dielectric integrand
                    # never sees metal permittivity
                    # reduced over components and frequencies: (N, 1) like the staged mask
                    eps_in_is_pec = np.any(
                        [
                            eps.real < config.adjoint.pec_detection_threshold
                            for eps in eps_in.values()
                        ],
                        axis=(0, 2),
                    )[:, None]
                    mask_pec_inside = np.maximum(
                        np.asarray(pec.inside.pec_mask.values, dtype=float_dtype)[set_slice, None],
                        eps_in_is_pec,
                    )
                    # fields pulled outside the boundary use the outside-side eps samples
                    vjps_pec_fields_outside = _pec_side_vjps("out", eps_out)
                if has_pec_outside:
                    # fields pulled inside the boundary use the inside-side eps samples
                    vjps_pec_fields_inside = _pec_side_vjps("in", eps_in)

            vjps = blend_pec_dielectric_gradient(
                vjps_dielectric=vjps_dielectric,
                vjps_pec_fields_outside=vjps_pec_fields_outside,
                vjps_pec_fields_inside=vjps_pec_fields_inside,
                mask_pec_outside=mask_pec_outside,
                mask_pec_inside=mask_pec_inside,
                is_medium_pec=is_medium_pec,
                background_medium_is_pec=background_pec_effective,
            )

            if clipped_geometry is not None:
                if material_length_scale is None:
                    raise AdjointError(
                        "Clip-context shape gradients require 'material_length_scale' for the "
                        "inside-probe offsets, but none was resolved for this structure."
                    )
                clip_active = clip_active_from_inside_check(
                    clipped_geometry=clipped_geometry,
                    material_length_scale=material_length_scale,
                    spatial_coords=points,
                    normals=normals,
                )
                invalid_active = ~np.isfinite(vjps[clip_active])
                if np.any(invalid_active):
                    num_invalid = np.count_nonzero(invalid_active)
                    raise AdjointError(
                        "Detected non-finite clip-context gradient values inside occupied clip "
                        f"regions ({num_invalid} points)."
                    )
                vjps = np.where(clip_active[:, None], vjps, 0.0)

            integrands[tuple(key)][set_slice] = vjps

    return integrands
