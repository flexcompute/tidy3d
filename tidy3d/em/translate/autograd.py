"""Public Tidy3D autograd file preparation at the translator edge."""

from __future__ import annotations


def prepare_forward_task_file(task_file: str, tracer_keys_file: str, sample_sets_file: str) -> str:
    """Add public adjoint monitors to a public task file before schema conversion.

    ``sample_sets_file`` is the client-uploaded sample-set artifact; shape-derivative
    monitors are staged as point-cloud monitors from its query points (the artifact
    is the authority — see ``Simulation._with_adjoint_monitors``). Every supported
    client uploads it with the autograd forward task. An artifact that does not
    cover every traced shape-derivative path indicates something went wrong
    upstream and is rejected here, before any solver resources are consumed.
    """

    from tidy3d import Simulation
    from tidy3d.components.autograd.field_map import TracerKeys
    from tidy3d.em.translate.sample_sets import (
        GeometrySampleSets,
        shape_paths_by_structure,
        validate_sample_sets_coverage,
    )

    sim = Simulation.from_file(task_file)
    sim_fields_keys = TracerKeys.from_file(tracer_keys_file).keys
    # combined shape+custom-medium tracing is unsupported: reject before staging
    sim._check_custom_medium_geometry_overlap(sim_fields_keys)
    sample_sets = GeometrySampleSets.from_file(sample_sets_file)
    # coverage is required with no exclusions: only remote-gradient tasks reach this
    # seam, and the exclusion sources (custom vjps, numerical structures) force local
    # gradients client-side (see engine._autograd_forward_sidecar_artifacts)
    validate_sample_sets_coverage(sample_sets, shape_paths_by_structure(sim_fields_keys))
    sim = sim._with_adjoint_monitors(sim_fields_keys, sample_sets=sample_sets)
    sim.to_file(task_file)
    return task_file
