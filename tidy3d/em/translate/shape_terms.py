"""Derivative-program pieces for structure geometries: the boundary shape term and its bindings.

A structure's staged surface sample sets become one ``BoundaryShapeKernel`` term, whose outputs
are real per-point sensitivities per sample set. Each traced path binds through the geometry's
own chain rule, ``_compute_derivatives_from_sample_sets``.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from flexcompute.core._migration.em.derivatives import (
    BoundaryShapeKernel,
    PECSide,
    PECSides,
    ShapeSampleSet,
    SumFrequencies,
)

from tidy3d.components.autograd.derivative_terms import Binding, Compiled, ShapeDerivativeContext
from tidy3d.components.autograd.derivative_utils import (
    boundary_field_frame,
    clip_active_from_inside_check,
    contains_clip_operation,
)
from tidy3d.config import config

if TYPE_CHECKING:
    from tidy3d.components.autograd.derivative_terms import PathType
    from tidy3d.components.autograd.types import GeometryAdjointSettings
    from tidy3d.components.types import Bound


def shape_terms(
    simulation: Any,
    index: int,
    paths: list[PathType],
    structure_sets: Any,
    frequencies: Any,
    *,
    settings: GeometryAdjointSettings,
    key: tuple | None = None,
    token: int | str | None = None,
) -> Compiled:
    """One boundary-shape term over every sample set of a structure, and one binding for its paths.

    The point-cloud monitors are the ones the structure stages for its sample sets. A structure
    whose sample sets hold no points stages none, and every path binds the geometry's zero.
    ``settings`` are the adjoint settings ``structure_sets`` were generated with; the binding
    carries them to the chain rule, so compiling and binding read one snapshot.
    ``key`` and the monitor-name ``token`` default to the structure's own; a numerical query
    sets both so that its terms and monitors stay distinct.
    """

    structure = simulation.structures[index]
    key = ("structures", index, "geometry") if key is None else key
    token = index if token is None else token
    vjps = partial(
        _geometry_vjps,
        simulation=simulation,
        structure=structure,
        structure_sets=structure_sets,
        key=key,
        paths=tuple(paths),
        settings=settings,
    )
    traced = tuple((*key, *p) for p in paths)
    fields, permittivity, point_chunks = structure._make_adjoint_point_cloud_monitors(
        list(frequencies), token, structure_sets
    )
    if not fields:
        return Compiled(bindings=(Binding(traced, None, vjps),))
    clip = contains_clip_operation(structure.geometry)
    kernel = BoundaryShapeKernel(
        sample_sets=tuple(
            _shape_sample_set(entry, structure, structure_sets, clip, simulation.bounds)
            for entry in structure_sets.entries
        ),
        point_chunks=point_chunks,
        is_medium_pec=structure.medium.is_pec,
        pec_detection_threshold=config.adjoint.pec_detection_threshold,
        reductions=(SumFrequencies(real=True),),
    )
    return Compiled(
        terms=((key, kernel),),
        monitors=(*fields, *permittivity),
        bindings=(Binding(traced, key, vjps),),
    )


def _shape_sample_set(
    entry: Any, structure: Any, structure_sets: Any, clip: bool, simulation_bounds: Bound
) -> ShapeSampleSet:
    """What the kernel reads from one staged sample set: normals, the clip mask, PEC sampling."""

    sample_set = entry.sample_set
    normals, perps1, perps2 = boundary_field_frame(sample_set, simulation_bounds)
    active = None
    if clip and sample_set.num_points:
        active = clip_active_from_inside_check(
            clipped_geometry=structure.geometry,
            material_length_scale=structure_sets.material_length_scale,
            spatial_coords=np.asarray(sample_set.points.values, dtype=float),
            normals=normals,
        )
    pec = None
    if structure_sets.pec_staged and sample_set.num_points:
        sampling = sample_set.staging.pec_sampling

        def side(data: Any) -> PECSide:
            return PECSide(
                pec_mask=data.pec_mask.values,
                edge_distance_e=data.edge_distance_e.values,
                edge_distance_h=data.edge_distance_h.values,
            )

        pec = PECSides(
            perps1=perps1,
            perps2=perps2,
            flat_perp_dims=tuple(sampling.flat_perp_dims),
            inside=side(sampling.inside),
            outside=side(sampling.outside),
        )
    return ShapeSampleSet(key=tuple(entry.key), normals=normals, active=active, pec=pec)


def _geometry_vjps(
    outputs: Any,
    frequencies: Any,
    *,
    simulation: Any,
    structure: Any,
    structure_sets: Any,
    key: tuple,
    paths: tuple[PathType, ...],
    settings: GeometryAdjointSettings,
) -> dict[PathType, np.ndarray]:
    """Every traced geometry path, through one call of the geometry's chain rule."""

    geometry = structure.geometry
    integrands = {
        tuple(entry.key): np.asarray(
            outputs.get(tuple(entry.key), np.zeros(entry.sample_set.num_points)), dtype=float
        )[:, None]
        for entry in structure_sets.entries
    }
    info = ShapeDerivativeContext(
        simulation_bounds=simulation.bounds,
        point_integrands=integrands,
        resolved_material_wavelength=structure_sets.material_wavelength,
        settings=settings,
    )
    vjps = geometry._compute_derivatives_from_sample_sets(
        sample_sets=structure_sets.by_key(), paths=list(paths), derivative_info=info
    )
    return {(*key, *path): np.asarray(vjps[path]) for path in paths}


__all__ = ["shape_terms"]
