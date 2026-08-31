"""Pre-simulation collection of adjoint surface sample sets.

This is the single generation entry point of the adjoint sample-set pipeline
(the alignment invariant): forward setup calls it to pre-collect sample sets before
any simulation runs — onto the in-process context for local gradients and into the
uploaded sidecar artifact for remote gradients — and future point-cloud monitor
construction will consume the same sets. There is never a second generation
implementation that can drift.

Everything here is definition-derived: sampling resolutions come from the simulation
definition (``adjoint_sampling_resolution``), never from monitor data, so collection
is a pure, deterministic function of the simulation and the traced keys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tidy3d.components.geometry.bound_ops import bounds_intersection
from tidy3d.flex_em.translate.sample_sets import (
    GeometrySampleSets,
    SampleSetEntry,
    SamplingContext,
    StructureSampleSets,
    StructureSampleSetsEntry,
    encode_traced_keys,
    shape_paths_by_structure,
)
from tidy3d.log import log

from .spacing import SamplingScanIndex, adjoint_sampling_resolution

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping

    from tidy3d.components.simulation import Simulation

    from .types import PathType

# maximum number of per-structure debug log lines emitted by one collection; large
# simulations (e.g. thousands of traced structures) are summarized past this cap
_MAX_STRUCTURE_LOG_LINES = 20


def collect_adjoint_sample_sets(
    simulation: Simulation,
    sim_fields_keys: Iterable[PathType],
    exclude: Mapping[int, Collection[PathType]] | None = None,
) -> GeometrySampleSets:
    """Collect the adjoint surface sample sets for all traced shape-derivative paths.

    Only shape (geometry) paths produce sample sets; medium, source, and numerical
    paths are ignored. ``exclude`` removes (structure, path-prefix) combinations
    owned by other derivative mechanisms (custom vjps), exactly as in
    :func:`shape_paths_by_structure`.

    The per-structure sampling context mirrors the backward-pass ``DerivativeInfo``
    geometry fields (structure geometry bounds, their intersection with the
    simulation domain, the full simulation bounds) and carries definition-derived
    resolutions computed at the full adjoint frequency set.
    """
    sim_fields_keys = list(sim_fields_keys)
    paths_by_structure = shape_paths_by_structure(sim_fields_keys, exclude=exclude)

    scan_index = SamplingScanIndex.from_simulation(simulation) if paths_by_structure else None
    structure_entries = []
    for structure_index, geometry_paths in paths_by_structure.items():
        structure = simulation.structures[structure_index]
        resolution = adjoint_sampling_resolution(simulation, structure_index, scan_index=scan_index)

        geometry = structure.geometry
        bounds = geometry.bounds
        ctx = SamplingContext(
            bounds=bounds,
            bounds_intersect=bounds_intersection(simulation.bounds, bounds),
            simulation_bounds=simulation.bounds,
            spacing=resolution.spacing,
            material_wavelength=resolution.material_wavelength,
        )

        sample_sets = geometry._make_adjoint_sample_sets(paths=list(geometry_paths), ctx=ctx)

        structure_sample_sets = StructureSampleSets(
            entries=tuple(
                SampleSetEntry(key=key, sample_set=sample_set)
                for key, sample_set in sample_sets.items()
            ),
            requested_paths=geometry_paths,
            spacing=resolution.spacing,
            material_wavelength=resolution.material_wavelength,
            material_length_scale=resolution.material_length_scale,
        )
        structure_entries.append(
            StructureSampleSetsEntry(
                structure_index=structure_index,
                sample_sets=structure_sample_sets,
            )
        )

        # per-structure detail is aggregated across canonical keys (composite geometries
        # can imply thousands of keys) and capped across structures
        if len(structure_entries) <= _MAX_STRUCTURE_LOG_LINES:
            log.debug(
                f"Adjoint sample sets: structure {structure_index}: "
                f"{len(structure_sample_sets.entries)} keys, "
                f"{structure_sample_sets.num_points} points, "
                f"spacing {resolution.spacing:.3e} um, "
                f"material wavelength {resolution.material_wavelength:.3e} um."
            )
        elif len(structure_entries) == _MAX_STRUCTURE_LOG_LINES + 1:
            log.debug(
                "Adjoint sample sets: per-structure detail capped at "
                f"{_MAX_STRUCTURE_LOG_LINES} structures; remaining structures summarized "
                "in the aggregate line."
            )

    artifact = GeometrySampleSets(
        entries=tuple(structure_entries),
        traced_keys=encode_traced_keys(sim_fields_keys),
    )
    log.debug(
        f"Adjoint sample sets: collected {artifact.num_points} points across "
        f"{len(artifact.entries)} traced structure(s)."
    )
    return artifact
