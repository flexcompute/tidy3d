"""Structure-level monitor staging of adjoint surface sample sets.

Geometries generate pure surface quadrature (points, normals, tangents, weights,
scatter metadata) and stay medium-agnostic. This module is the structure-level step
that enriches those sets with a single ``PointCloudSamplingData`` bundle holding
everything point-cloud monitor construction and consumption need beyond the
quadrature itself:

- ``field_query_points`` for the dielectric-path field monitor;
- per-component inside/outside permittivity query points (``MaterialQueryPoints``),
  snapped against each ``eps_ii`` component's native Yee grid;
- PEC staging data (``PECSamplingData``) when the structure-level PEC trigger fires:
  symmetric outside- and inside-snapped per-field-component query points, mean edge
  distances for the singularity correction, per-side per-point PEC masks, and the
  line-integration facts.

Everything is definition-derived. Snapping uses the same kernels as the legacy
volumetric consumption path (``autograd.snapping``), against the simulation grid the
monitors will sample on (the grid-parity contract), so generation-time placement and
the previous data-coordinate snapping are the same arithmetic on the same grid.

PEC detection is deliberately over-inclusive and safe: the structure-level trigger
(own medium PEC, explicit PEC ``background_medium``, PEC simulation medium, or any
AABB-overlapping structure with a PEC medium) only decides whether the payload is
staged and H is recorded; the per-side per-point masks — the effective medium at each
snapped point, precedence-aware against the simulation definition — are what gate
PEC-vs-dielectric integration at consumption.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tidy3d.config import config
from tidy3d.em.translate.sample_sets import (
    IndexedDataArray,
    MaterialQueryPoints,
    PECSamplingData,
    PECSideSamplingData,
    PointCloudSamplingData,
    PointDataArray,
)

from .snapping import (
    edge_distance_after_snapping,
    nearest_grid_coords,
    snap_coords_to_boundary,
)

if TYPE_CHECKING:
    from tidy3d.components.grid.grid import Grid
    from tidy3d.components.simulation import Simulation
    from tidy3d.components.structure import Structure
    from tidy3d.em.translate.sample_sets import SurfaceSampleSet

    from .spacing import SamplingScanIndex
    from .types import PathType

# permittivity diagonal components live on the corresponding E-component Yee locations
_EPS_COMPONENT_GRID_KEYS = {"xx": "Ex", "yy": "Ey", "zz": "Ez"}
_E_COMPONENT_KEYS = ("Ex", "Ey", "Ez")
_H_COMPONENT_KEYS = ("Hx", "Hy", "Hz")


def _component_grid_centers(grid: Grid, component_key: str) -> dict[str, np.ndarray]:
    """Native Yee grid centers of one field/permittivity component, keyed by dimension."""
    coords = grid[component_key]
    return {dim: np.asarray(getattr(coords, dim)) for dim in "xyz"}


def _point_data_array(values: np.ndarray) -> PointDataArray:
    """Wrap an ``(N, 3)`` array as a coordinate-carrying ``PointDataArray``."""
    return PointDataArray(
        values,
        coords={"index": np.arange(values.shape[0]), "axis": np.arange(values.shape[1])},
    )


def _clamp_to_bounds(points: np.ndarray, simulation: Simulation) -> np.ndarray:
    """Clamp query points into the simulation domain.

    Sample sets may legitimately contain surface points on (or, for snapped queries,
    just beyond) the simulation boundary — structures often extend past the domain.
    The legacy volumetric path sampled these by interpolator edge extrapolation;
    point-cloud monitors reject points outside the domain, so query points clamp to
    the boundary and record the nearest available data instead. Quadrature points,
    weights, and normals are never modified.
    """
    sim_min = np.asarray(simulation.bounds[0], dtype=float)
    sim_max = np.asarray(simulation.bounds[1], dtype=float)
    # non-collapsed dimensions require strict interiority (the simulation-level
    # point-cloud bounds validator uses strict inequalities there); a relative inset
    # of 1e-9 of the domain extent is physically negligible at any sensible scale
    extent = sim_max - sim_min
    inset = np.where(extent > 0, 1e-9 * extent, 0.0)
    return np.clip(points, sim_min + inset, sim_max - inset)


def _pec_trigger_is_unconditional(structure: Structure, simulation: Simulation) -> bool:
    """PEC triggers that keep the payload even when every sampled mask is zero.

    The structure's own PEC medium, an explicit PEC ``background_medium`` hint, and a
    PEC simulation medium all assert PEC contact independently of the mask samples
    (the hint exists precisely for exactly-touching ambiguity that node-sampled masks
    can miss), so their payload is never pruned.
    """
    if structure.medium.is_pec:
        return True
    if structure.background_medium is not None and structure.background_medium.is_pec:
        return True
    return simulation.medium.is_pec


def _overlapping_structures(structure: Structure, scan_index: SamplingScanIndex) -> list[Structure]:
    """Structures whose bounding box overlaps this structure's (cheap AABB prefilter)."""
    bounds_min, bounds_max = np.asarray(structure.geometry.bounds)
    overlaps = np.all(scan_index.bounds_min <= bounds_max, axis=1) & np.all(
        scan_index.bounds_max >= bounds_min, axis=1
    )
    return [
        candidate for candidate, overlapping in zip(scan_index.structures, overlaps) if overlapping
    ]


def _pec_trigger_from_neighbors(structure: Structure, scan_index: SamplingScanIndex) -> bool:
    """Whether any PEC structure's bounding box overlaps this structure's.

    A cheap prefilter rather than exact geometry intersection: over-inclusion here is
    resolved after mask computation (``stage_sample_sets`` prunes the payload when
    every sampled mask is zero), so a false positive costs generation-time work,
    never extra recorded data.
    """
    return any(
        candidate.medium.is_pec for candidate in _overlapping_structures(structure, scan_index)
    )


def _fully_anisotropic_trigger(
    structure: Structure, simulation: Simulation, scan_index: SamplingScanIndex
) -> bool:
    """Whether a fully anisotropic medium can sit on either side of this interface.

    Mirrors the PEC trigger's over-inclusive shape: the structure's own medium, an
    explicit ``background_medium`` hint, the simulation medium, or any
    AABB-overlapping structure. Shape gradients evaluate the interface formula with
    diagonal permittivity components only (both the point-cloud and the legacy
    volumetric path), so a hit here surfaces the diagonal-approximation warning at
    collection; over-inclusion costs a warning, never gradients.
    """
    media = [structure.medium, simulation.medium]
    if structure.background_medium is not None:
        media.append(structure.background_medium)
    media.extend(candidate.medium for candidate in _overlapping_structures(structure, scan_index))
    return any(medium.is_fully_anisotropic for medium in media)


def _metal_like_trigger(
    structure: Structure,
    simulation: Simulation,
    scan_index: SamplingScanIndex,
    frequencies: list[float],
) -> bool:
    """Whether an undeclared metal-like medium can sit at this interface.

    Same over-inclusive candidate set as the PEC and anisotropy triggers. A medium
    whose real permittivity at the adjoint frequencies falls below
    ``config.adjoint.pec_detection_threshold`` behaves PEC-like at the boundary,
    but only declared ``is_pec`` media receive PEC handling — so a hit here (on a
    structure with no PEC payload staged) surfaces the dielectric-only-approximation
    warning at collection, matching the legacy consumption-time warning.
    """
    media = [structure.medium, simulation.medium]
    if structure.background_medium is not None:
        media.append(structure.background_medium)
    media.extend(candidate.medium for candidate in _overlapping_structures(structure, scan_index))
    freqs = np.asarray(frequencies)
    threshold = config.adjoint.pec_detection_threshold
    for medium in media:
        if medium.is_pec:
            continue
        if np.min(np.asarray(medium.eps_model(freqs)).real) < threshold:
            return True
    return False


def effective_is_pec_at_points(
    points: np.ndarray,
    simulation: Simulation,
    scan_index: SamplingScanIndex,
) -> np.ndarray:
    """Per-point PEC mask of the effective medium at ``points`` (``(N, 3)``).

    Precedence-aware against the simulation definition: the last volumetric structure
    containing a point decides its medium, falling back to the simulation medium.
    Returns a float array (``1.0`` where PEC) matching the legacy detection's dtype.
    """
    mask = np.full(points.shape[0], float(simulation.medium.is_pec))
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    points_min, points_max = points.min(axis=0), points.max(axis=0)
    for candidate, cand_min, cand_max in zip(
        scan_index.structures, scan_index.bounds_min, scan_index.bounds_max
    ):
        if np.any(cand_min > points_max) or np.any(cand_max < points_min):
            continue
        inside = candidate.geometry.inside(x, y, z)
        if np.any(inside):
            mask[inside] = float(candidate.medium.is_pec)
    return mask


def _material_query_points(
    points: np.ndarray,
    normals: np.ndarray,
    grid: Grid,
    snapping_fraction: float,
    simulation: Simulation,
) -> MaterialQueryPoints:
    """Per-component inside/outside permittivity query points for one sample set."""
    query_points = {}
    for component, grid_key in _EPS_COMPONENT_GRID_KEYS.items():
        grid_centers = _component_grid_centers(grid, grid_key)
        for side, is_outside in (("in", False), ("out", True)):
            query_points[f"{side}_{component}"] = _point_data_array(
                _clamp_to_bounds(
                    snap_coords_to_boundary(
                        spatial_coords=points,
                        normals=normals,
                        grid_centers=grid_centers,
                        is_outside=is_outside,
                        snapping_fraction=snapping_fraction,
                    ),
                    simulation,
                )
            )
    return MaterialQueryPoints(**query_points)


def _pec_side_sampling_data(
    points: np.ndarray,
    normals: np.ndarray,
    is_outside: bool,
    simulation: Simulation,
    scan_index: SamplingScanIndex,
    grid: Grid,
    snapping_fraction: float,
    material_query_points: MaterialQueryPoints,
) -> PECSideSamplingData:
    """One-sided PEC staging payload: per-component points, edge distances, mask."""
    field_points: dict[str, PointDataArray] = {}
    edge_distances: dict[str, list[np.ndarray]] = {"e": [], "h": []}
    for component_key in (*_E_COMPONENT_KEYS, *_H_COMPONENT_KEYS):
        grid_centers = _component_grid_centers(grid, component_key)
        snapped = snap_coords_to_boundary(
            spatial_coords=points,
            normals=normals,
            grid_centers=grid_centers,
            is_outside=is_outside,
            snapping_fraction=snapping_fraction,
        )
        # place the query point ON the grid node the legacy nearest lookup resolves
        # to: the point-cloud monitor records trilinear interpolation, and trilinear
        # at a node is exactly that node's raw value (nearest semantics, bit-exact)
        field_points[component_key.lower()] = _point_data_array(
            _clamp_to_bounds(nearest_grid_coords(grid_centers, snapped), simulation)
        )
        edge_distances[component_key[0].lower()].append(
            edge_distance_after_snapping(
                spatial_coords=points, grid_centers=grid_centers, adjusted_coords=snapped
            )
        )

    # the mask reproduces the legacy detection exactly: legacy reads the eps VALUE at
    # the grid node its nearest lookup samples, so the effective-medium query must be
    # made at that node's coordinates (not the biased query point, whose containment
    # can differ near edges); per eps component, max-reduced. The three components'
    # node points are classified in one concatenated batch: the classification is
    # per-point, so batching preserves the masks exactly while running the
    # candidate-structure traversal once per side instead of once per component
    side = "out" if is_outside else "in"
    component_node_points = [
        nearest_grid_coords(
            _component_grid_centers(grid, grid_key),
            np.asarray(getattr(material_query_points, f"{side}_{component}").values, dtype=float),
        )
        for component, grid_key in _EPS_COMPONENT_GRID_KEYS.items()
    ]
    stacked_mask = effective_is_pec_at_points(
        np.concatenate(component_node_points, axis=0), simulation, scan_index
    )
    pec_mask = np.maximum.reduce(np.split(stacked_mask, len(component_node_points)))

    index_coords = {"index": np.arange(points.shape[0])}
    return PECSideSamplingData(
        **field_points,
        edge_distance_e=IndexedDataArray(np.mean(edge_distances["e"], axis=0), coords=index_coords),
        edge_distance_h=IndexedDataArray(np.mean(edge_distances["h"], axis=0), coords=index_coords),
        pec_mask=IndexedDataArray(pec_mask, coords=index_coords),
    )


def _pec_sampling_data(
    sample_set: SurfaceSampleSet,
    structure: Structure,
    simulation: Simulation,
    scan_index: SamplingScanIndex,
    grid: Grid,
    snapping_fraction: float,
    material_query_points: MaterialQueryPoints,
) -> PECSamplingData:
    """PEC staging payload for one non-empty sample set of a PEC-triggered structure.

    Both sides are staged symmetrically: PEC consumption samples outside-snapped for a
    PEC structure's own gradient and inside-snapped when PEC surrounds a dielectric
    structure or seams PEC-to-PEC.
    """
    points = np.asarray(sample_set.points.values, dtype=float)
    normals = np.asarray(sample_set.normals.values, dtype=float)

    side_kwargs = {
        "points": points,
        "normals": normals,
        "simulation": simulation,
        "scan_index": scan_index,
        "grid": grid,
        "snapping_fraction": snapping_fraction,
        "material_query_points": material_query_points,
    }

    return PECSamplingData(
        outside=_pec_side_sampling_data(is_outside=True, **side_kwargs),
        inside=_pec_side_sampling_data(is_outside=False, **side_kwargs),
        flat_perp_dims=sample_set.pec_flat_perp_dims,
    )


def stage_sample_sets(
    sample_sets: dict[PathType, SurfaceSampleSet],
    structure: Structure,
    simulation: Simulation,
    scan_index: SamplingScanIndex,
    grid: Grid,
) -> dict[PathType, SurfaceSampleSet]:
    """Enrich one structure's sample sets with the monitor staging payload.

    Empty sets pass through untouched (they stage no monitors and consumption skips
    them); every non-empty set gains a ``PointCloudSamplingData`` bundle: field and
    material query points, plus PEC sampling data when the structure-level PEC
    trigger fires. A payload staged only because of the bounding-box neighbor
    prefilter is pruned again when every sampled mask is zero: the masks gate
    PEC-vs-dielectric integration, so all-zero masks make the PEC contribution
    exactly zero and pruning changes recorded data volume, never gradients.
    """
    snapping_fraction = config.adjoint.boundary_snapping_fraction
    pec_unconditional = _pec_trigger_is_unconditional(structure, simulation)
    pec_triggered = pec_unconditional or _pec_trigger_from_neighbors(structure, scan_index)

    staged = {}
    for key, sample_set in sample_sets.items():
        if sample_set.num_points == 0:
            staged[key] = sample_set
            continue

        points = np.asarray(sample_set.points.values, dtype=float)
        normals = np.asarray(sample_set.normals.values, dtype=float)
        material_query_points = _material_query_points(
            points=points,
            normals=normals,
            grid=grid,
            snapping_fraction=snapping_fraction,
            simulation=simulation,
        )
        pec_sampling = None
        if pec_triggered:
            pec_sampling = _pec_sampling_data(
                sample_set=sample_set,
                structure=structure,
                simulation=simulation,
                scan_index=scan_index,
                grid=grid,
                snapping_fraction=snapping_fraction,
                material_query_points=material_query_points,
            )

        staged[key] = sample_set.updated_copy(
            staging=PointCloudSamplingData(
                field_query_points=_point_data_array(_clamp_to_bounds(points, simulation)),
                material_query_points=material_query_points,
                pec_sampling=pec_sampling,
            ),
        )

    if pec_triggered and not pec_unconditional:
        any_pec_mask = any(
            np.any(sample_set.staging.pec_sampling.outside.pec_mask.values > 0)
            or np.any(sample_set.staging.pec_sampling.inside.pec_mask.values > 0)
            for sample_set in staged.values()
            if sample_set.staging is not None and sample_set.staging.pec_sampling is not None
        )
        if not any_pec_mask:
            # the bounding-box prefilter over-included: no sampled point ever sees
            # PEC, so drop the payload (and with it the twelve per-component PEC
            # monitors this structure would otherwise stage)
            staged = {
                key: (
                    sample_set
                    if sample_set.staging is None
                    else sample_set.updated_copy(
                        staging=sample_set.staging.updated_copy(pec_sampling=None)
                    )
                )
                for key, sample_set in staged.items()
            }
    return staged
