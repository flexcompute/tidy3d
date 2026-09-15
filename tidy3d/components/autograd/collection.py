"""Pre-simulation collection of adjoint surface sample sets.

This is the single generation entry point of the adjoint sample-set pipeline
(the alignment invariant): forward setup calls it to pre-collect sample sets before
any simulation runs — onto the in-process context for local gradients and into the
uploaded sidecar artifact for remote gradients — and point-cloud monitor
construction consumes the same sets. There is never a second generation
implementation that can drift.

Everything here is definition-derived: sampling resolutions come from the simulation
definition (``adjoint_sampling_resolution``), never from monitor data, so collection
is a pure, deterministic function of the simulation and the traced keys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tidy3d.components.geometry.bound_ops import bounds_intersection
from tidy3d.em.translate.sample_sets import (
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
from .staging import _fully_anisotropic_trigger, _metal_like_trigger, stage_sample_sets

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
        # structure-level staging: geometries are medium-agnostic, so the monitor
        # query points (and PEC payload, when triggered) are attached here where the
        # medium, grid, and full simulation are in scope. Downstream monitor
        # construction (including the solver's, through the uploaded artifact) keys
        # off payload presence, so the artifact is the single authority and every
        # construction site stages identical monitors.
        sample_sets = stage_sample_sets(
            sample_sets=sample_sets,
            structure=structure,
            simulation=simulation,
            scan_index=scan_index,
            grid=simulation.grid,
        )

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

        if structure_sample_sets.num_points > 0 and _fully_anisotropic_trigger(
            structure, simulation, scan_index
        ):
            # not an error: this matches the legacy volumetric path's treatment, but
            # the approximation deserves surfacing at collection time
            log.warning(
                f"Adjoint sample sets: structure {structure_index} interfaces fully "
                "anisotropic media; its shape gradient uses only the diagonal "
                "permittivity components, so off-diagonal tensor coupling is not "
                "included in the surface-gradient computation."
            )

        # parity with the legacy consumption-time warning: metals must be declared
        # PEC to receive PEC boundary handling; undeclared metal-like media take the
        # dielectric-only path, which is a poor approximation at a metal boundary
        if (
            structure_sample_sets.num_points > 0
            and not structure_sample_sets.pec_staged
            and _metal_like_trigger(structure, simulation, scan_index, simulation._freqs_adjoint)
        ):
            log.warning(
                f"Adjoint sample sets: structure {structure_index} interfaces a medium "
                "with metal-like permittivity (real part below "
                "'config.adjoint.pec_detection_threshold') that is not declared PEC, so "
                "its shape gradient uses dielectric-only surface integration. If PEC "
                "behavior is intended, use a PEC medium or set the structure's "
                "'background_medium' to PEC so the PEC correction is applied."
            )

        from .point_consumption import _adjoint_point_cloud_chunk_size

        chunk_size = _adjoint_point_cloud_chunk_size()
        if structure_sample_sets.num_points > chunk_size:
            # not an error: monitors split transparently across chunks, but the
            # recorded adjoint data volume scales with the point count, so surface
            # the cost and the resolution lever
            log.warning(
                f"Adjoint sample sets: structure {structure_index} stages "
                f"{structure_sample_sets.num_points:,} surface points, above the "
                f"{chunk_size:,}-point per-monitor cap; its point-cloud adjoint "
                "monitors are split across multiple monitors and the recorded adjoint "
                "data will be correspondingly large. Coarsen the adjoint sampling "
                "resolution (e.g. 'config.adjoint.default_wavelength_fraction') if "
                "this is unintended."
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
