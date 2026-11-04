"""Derivative-program pieces for structure geometries: the shape terms and their bindings.

A structure's staged surface sample sets become a ``BoundaryShapeKernel`` term, whose outputs
are real per-point sensitivities per sample set, and, where the structure moves a PEC surface,
a ``CollarKernel`` term, whose outputs are per-sample force vectors from the stress in the cells
around it. A PEC structure takes every force from the collar and needs no boundary term. The
binding turns both into one force vector per sample, and each traced path binds through the
geometry's own chain rule, ``_compute_derivatives_from_sample_sets``.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from flexcompute.core._migration.em.derivatives import (
    BoundaryShapeKernel,
    CollarKernel,
    CollarSampleSet,
    ForeignPEC,
    OwnInterface,
    PECSide,
    PECSides,
    ShapeSampleSet,
    SumFrequencies,
)
from flexcompute.core._migration.em.numerical.raw.collar import mirror_images

from tidy3d.components.autograd.collar_diagnostics import structure_medium, warn_near_cut
from tidy3d.components.autograd.collar_staging import invariant_axes
from tidy3d.components.autograd.derivative_terms import Binding, Compiled, ShapeDerivativeContext
from tidy3d.components.autograd.derivative_utils import (
    boundary_field_frame,
    clip_active_from_inside_check,
    contains_clip_operation,
)
from tidy3d.components.autograd.staging import is_pec_surface
from tidy3d.components.medium.abstract_custom import AbstractCustomMedium
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
    """The shape terms over every sample set of a structure, and one binding for its paths.

    The point-cloud monitors are the ones the structure stages for its sample sets and its
    collar. A structure whose sample sets hold no points stages none, and every path binds the
    geometry's zero.
    ``settings`` are the adjoint settings ``structure_sets`` were generated with; the binding
    carries them to the chain rule, so compiling and binding read one snapshot.
    ``key`` and the monitor-name ``token`` default to the structure's own; a numerical query
    sets both so that its terms and monitors stay distinct.
    """

    structure = simulation.structures[index]
    key = ("structures", index, "geometry") if key is None else key
    token = index if token is None else token
    is_pec = is_pec_surface(structure.medium, simulation)
    collar = structure_sets.collar if structure_sets.collar_staged else None
    vjps = partial(
        _geometry_vjps,
        simulation=simulation,
        structure=structure,
        structure_sets=structure_sets,
        collar=collar,
        key=key,
        paths=tuple(paths),
        settings=settings,
    )
    traced = tuple((*key, *p) for p in paths)
    terms = []
    monitors = []
    if collar is None or not is_pec:
        fields, permittivity, point_chunks = structure._make_adjoint_point_cloud_monitors(
            list(frequencies), token, structure_sets
        )
        if fields:
            clipping_geometry = (
                structure.geometry._adjoint_sampling_geometry(settings)
                if contains_clip_operation(structure.geometry)
                else None
            )
            kernel = BoundaryShapeKernel(
                sample_sets=tuple(
                    _shape_sample_set(entry, clipping_geometry, structure_sets, simulation.bounds)
                    for entry in structure_sets.entries
                ),
                point_chunks=point_chunks,
                is_medium_pec=is_pec,
                pec_detection_threshold=config.adjoint.pec_detection_threshold,
                reductions=(SumFrequencies(real=True),),
            )
            terms.append((key, kernel))
            monitors += [*fields, *permittivity]
    if collar is not None:
        collar_monitors = structure._make_adjoint_collar_monitors(
            list(frequencies), token, structure_sets
        )
        kernel = _collar_kernel(
            simulation, index, structure_sets, collar, collar_monitors, frequencies
        )
        terms.append(((*key, "collar"), kernel))
        monitors += collar_monitors
    return Compiled(
        terms=tuple(terms),
        monitors=tuple(monitors),
        bindings=(Binding(traced, tuple(term for term, _ in terms), vjps),),
    )


def _medium_eps_comp(medium: Any, row: int, col: int, frequency: float) -> complex:
    """One permittivity tensor component of a medium. A custom medium's own ``eps_comp`` averages
    its diagonal, so it gives the spatial average of the requested component instead."""
    if not isinstance(medium, AbstractCustomMedium):
        return medium.eps_comp(row, col, frequency)
    if row != col:
        return 0j
    return medium._spatial_average(medium.eps_dataarray_freq(frequency)[row], frequency)


def _collar_kernel(
    simulation: Any,
    index: int,
    structure_sets: Any,
    collar: Any,
    monitors: list[Any],
    frequencies: Any,
) -> CollarKernel:
    """The collar term of one structure, from its staged collar and surface samples.

    The permittivity of each medium holding a collar cell is taken from the simulation
    definition rather than recorded: within a cell of a PEC surface the solver's subpixel
    averaging leaks the PEC into the recorded ``D`` and permittivity (in vacuum the recorded
    ``D`` was 1.4x ``E`` in the first cell layer), and the electric stress of exactly those
    cells is the PEC face traction. Anisotropic media enter with their full tensor, spatially
    varying ones with the spatial average of each diagonal component.
    """

    is_pec = is_pec_surface(simulation.structures[index].medium, simulation)
    structure_indices = np.rint(np.asarray(collar.medium_structure_index.values)).astype(int)
    media, cell_media = np.unique(structure_indices, return_inverse=True)
    permittivity = np.array(
        [
            [
                [
                    [
                        _medium_eps_comp(structure_medium(simulation, m), row, col, float(f))
                        for col in range(3)
                    ]
                    for row in range(3)
                ]
                for m in media
            ]
            for f in frequencies
        ],
        dtype=complex,
    )

    def values(data: Any) -> np.ndarray:
        return np.asarray(data.values, dtype=float)

    staged = [entry for entry in structure_sets.entries if entry.sample_set.num_points > 0]
    on_surface, owned, removed = _collar_masks(collar)
    own_interface = None
    if not is_pec:
        # a dielectric carving a PEC: its own interface moves in the collar next to the metal
        own_interface = OwnInterface(
            medium=int(np.searchsorted(media, index)) if index in media else -1,
            samples=~removed & ~on_surface,
        )
    foreign_pec = None
    if collar.pec_neighbor_distance is not None:
        foreign_pec = ForeignPEC(
            distance=values(collar.pec_neighbor_distance),
            normal=values(collar.pec_neighbor_normal),
        )

    return CollarKernel(
        monitors=tuple(monitor.name for monitor in monitors),
        cell_indices=np.rint(np.asarray(collar.cell_indices.values)).astype(np.int64),
        cell_centers=values(collar.points),
        cell_sizes=values(collar.cell_sizes),
        cell_media=cell_media.astype(np.int64),
        permittivity=permittivity,
        foreign_pec=foreign_pec,
        sample_sets=tuple(
            CollarSampleSet(
                key=tuple(entry.key),
                points=values(entry.sample_set.points),
                normals=values(entry.sample_set.normals),
            )
            for entry in staged
        ),
        active=owned,
        # a PEC structure's buried samples stay targets so the velocity vanishes above them;
        # a non-PEC structure's interface away from the metal does not stop the velocity, as
        # it would not in the collar of the PEC it borders
        targets=~removed if is_pec else owned,
        sample_offset=values(collar.sample_offset),
        sample_taper=values(collar.sample_taper),
        own_interface=own_interface,
        flat_interface=None
        if collar.flat_interface is None
        else values(collar.flat_interface) > 0.5,
        invariant_axes=invariant_axes(simulation),
        reductions=(SumFrequencies(real=True),),
    )


def _merge_collar(
    values: Any, normals: np.ndarray, owned: np.ndarray, interface: np.ndarray, fallback: Any
) -> np.ndarray:
    """Collar values ``(N, 3)`` where the collar owns the sample, normal part only on the
    structure's own interface, and ``fallback`` elsewhere."""

    values = np.asarray(values, dtype=float)
    normal_part = np.sum(values * normals, axis=1, keepdims=True) * normals
    return np.where(owned[:, None], np.where(interface[:, None], normal_part, values), fallback)


def _collar_masks(collar: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per sample: on the effective PEC surface, collar owned, and on no surface."""

    return tuple(
        np.asarray(mask.values, dtype=bool)
        for mask in (collar.on_surface, collar.collar_owned, collar.removed)
    )


def warn_collar_source_cuts(program: Any, adjoint_simulations: Any) -> None:
    """Warn where an adjoint source shortens the collar of a traced PEC surface.

    The collar kernel cuts the cells at each adjoint source (an objective monitor), since a
    source breaks the stress identity; the identity holds on the rest, but the part of the
    surface whose collar column a source shortens loses accuracy.
    """

    collars = [(key, kernel) for key, kernel in program.terms if isinstance(kernel, CollarKernel)]
    for simulation in adjoint_simulations if collars else ():
        bounds = [
            image
            for source in simulation.sources
            for image in mirror_images(source.bounds, simulation.symmetry, simulation.center)
        ]
        for key, kernel in collars:
            cut = kernel.source_cut(bounds)
            if not np.any(cut):
                continue
            active = kernel.active
            warn_near_cut(
                f"structure {key[1]}" if key[:1] == ("structures",) else f"term {key}",
                np.concatenate([s.points for s in kernel.sample_sets])[active],
                np.concatenate([s.normals for s in kernel.sample_sets])[active],
                (kernel.sample_offset + kernel.sample_taper)[active],
                kernel.cell_centers[cut],
                kernel.cell_sizes[cut],
                "an adjoint source (an objective monitor)",
            )


def _shape_sample_set(
    entry: Any, clipping_geometry: Any, structure_sets: Any, simulation_bounds: Bound
) -> ShapeSampleSet:
    """What the kernel reads from one staged sample set: normals, the clip mask, PEC sampling."""

    sample_set = entry.sample_set
    normals, perps1, perps2 = boundary_field_frame(sample_set, simulation_bounds)
    active = None
    if clipping_geometry is not None and sample_set.num_points:
        active = clip_active_from_inside_check(
            clipped_geometry=clipping_geometry,
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
    collar: Any,
    key: tuple,
    paths: tuple[PathType, ...],
    settings: GeometryAdjointSettings,
) -> dict[PathType, np.ndarray]:
    """Every traced geometry path, through one call of the geometry's chain rule.

    Each sample's force is the boundary integrand's ``w Re(g) n`` or, where the collar owns
    the sample, the collar force. A dielectric carving a PEC keeps only the normal part of the
    collar force on its own interface, since a dielectric interface carries no tangential
    stress jump.
    """

    geometry = structure.geometry
    masks = None if collar is None else _collar_masks(collar)
    forces = {}
    moments = {}
    start = 0  # the collar's masks run over the nonempty sets' samples, in order
    for entry in structure_sets.entries:
        set_key = tuple(entry.key)
        sample_set = entry.sample_set
        weights = np.asarray(sample_set.weights.values, dtype=float)
        normals = np.asarray(sample_set.normals.values, dtype=float).reshape(-1, 3)
        integrand = np.asarray(outputs.get(set_key, np.zeros(sample_set.num_points)), dtype=float)
        forces[set_key] = (weights * integrand)[:, None] * normals
        if masks is None or sample_set.num_points == 0:
            continue
        rows = slice(start, start + sample_set.num_points)
        start = rows.stop
        on_surface, owned, _ = (mask[rows] for mask in masks)
        # a collar-owned sample off the metal surface is the structure's own interface, which
        # keeps the normal part of the collar force; a sample the collar does not own keeps its
        # boundary force
        interface = owned & ~on_surface
        forces[set_key] = _merge_collar(
            outputs[("force", *set_key)], normals, owned, interface, forces[set_key]
        )
        moments[set_key] = _merge_collar(
            outputs[("moment", *set_key)], normals, owned, interface, 0.0
        )
    info = ShapeDerivativeContext(
        simulation_bounds=simulation.bounds,
        point_forces=forces,
        point_force_moments=moments,
        resolved_material_wavelength=structure_sets.material_wavelength,
        settings=settings,
    )
    vjps = geometry._compute_derivatives_from_sample_sets(
        sample_sets=structure_sets.by_key(), paths=list(paths), derivative_info=info
    )
    return {(*key, *path): np.asarray(vjps[path]) for path in paths}


__all__ = ["shape_terms", "warn_collar_source_cuts"]
