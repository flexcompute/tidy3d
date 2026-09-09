"""Grid-spec generation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tidy3d.components.geometry.base import Box
from tidy3d.components.grid.grid import Coords, Grid
from tidy3d.components.structure import Structure
from tidy3d.constants import inf
from tidy3d.log import log

if TYPE_CHECKING:
    from pydantic import NonNegativeInt, PositiveFloat

    from tidy3d.components.lumped_element import LumpedElementType
    from tidy3d.components.source.utils import SourceType
    from tidy3d.components.structure import MeshOverrideStructure, StructureType
    from tidy3d.components.types import CoordinateOptional, PriorityMode, Shapely, Symmetry

    from .model import GridSpec


from tidy3d.components.grid.grid_spec.constants import (
    DL_MIN_FROM_GAPS_FRACTION,
    GAP_REFINEMENT_WARNING_THRESH,
    MIN_GRID_SPACING,
    UNITS_HELP_URL,
    _GeneratedGridSizeError,
)
from tidy3d.components.grid.grid_spec.grid_1d import (
    AbstractAutoGrid,
    AutoGrid,
)


def get_wavelength(self: GridSpec, sources: list[SourceType]) -> float:
    """Get wavelength for automatic mesh generation if needed."""
    wavelength = self.wavelength
    if wavelength is None and self.auto_grid_used:
        wavelength = self.wavelength_from_sources(sources)
        log.info(f"Auto meshing using wavelength {wavelength:1.4f} defined from sources.")
    return wavelength


def make_grid(
    self: GridSpec,
    structures: list[Structure],
    symmetry: tuple[Symmetry, Symmetry, Symmetry],
    periodic: tuple[bool, bool, bool],
    sources: list[SourceType],
    num_pml_layers: list[tuple[NonNegativeInt, NonNegativeInt]],
    lumped_elements: list[LumpedElementType] = (),
    internal_override_structures: list[MeshOverrideStructure] | None = None,
    internal_snapping_points: list[CoordinateOptional] | None = None,
    boundary_types: tuple[tuple[str, str], tuple[str, str], tuple[str, str]] = [
        [None, None],
        [None, None],
        [None, None],
    ],
    structure_priority_mode: PriorityMode = "equal",
) -> Grid:
    """Make the entire simulation grid based on some simulation parameters.

    Parameters
    ----------
    structures : list[Structure]
        List of structures present in the simulation. The first structure must be the
        simulation geometry with the simulation background medium.
    symmetry : tuple[Symmetry, Symmetry, Symmetry]
        Reflection symmetry across a plane bisecting the simulation domain
        normal to each of the three axes.
    periodic: tuple[bool, bool, bool]
        Apply periodic boundary condition or not along each of the dimensions.
        Only relevant for autogrids.
    sources : list[SourceType]
        List of sources.
    num_pml_layers : list[tuple[float, float]]
        List containing the number of absorber layers in - and + boundaries.
    lumped_elements : list[LumpedElementType]
        List of lumped elements.
    internal_override_structures : list[MeshOverrideStructure]
        If ``None``, recomputes internal override structures.
    internal_snapping_points : list[CoordinateOptional]
        If ``None``, recomputes internal snapping points.
    boundary_types : tuple[tuple[str, str], tuple[str, str], tuple[str, str]] = [[None, None], [None, None], [None, None]]
        Type of boundary conditions along each dimension: "pec/pmc", "periodic", or
        None for any other. This is relevant only for gap meshing.
    structure_priority_mode : PriorityMode
        Structure priority setting.

    Returns
    -------
    Grid:
        Entire simulation grid.
    """

    grid, _ = self._make_grid_and_snapping_lines(
        structures=structures,
        symmetry=symmetry,
        periodic=periodic,
        sources=sources,
        num_pml_layers=num_pml_layers,
        lumped_elements=lumped_elements,
        internal_override_structures=internal_override_structures,
        internal_snapping_points=internal_snapping_points,
        structure_priority_mode=structure_priority_mode,
    )

    return grid


def _generated_grid_size_error_message(grid_name: str, axis_name: str, min_size: float) -> str:
    """Error message for generated grid cells below the supported minimum."""
    return (
        f"{grid_name} generated a minimum grid spacing of {min_size:.2e} µm "
        f"along the '{axis_name}' axis, which is below the supported minimum "
        f"of {MIN_GRID_SPACING:.1e} µm. Please check your units! For more info "
        f"on Tidy3D units, see: {UNITS_HELP_URL}"
    )


def _raise_generated_grid_size_error(
    self: GridSpec, grid_name: str, axis_name: str, min_size: float
) -> None:
    """Raise a generated-grid spacing error with axis metadata."""
    raise _GeneratedGridSizeError(
        self._generated_grid_size_error_message(grid_name, axis_name, min_size),
        grid_name,
        axis_name,
        min_size,
    )


def _grid_spec_size_estimate_violation(
    self: GridSpec,
    wavelength: float,
    structures: list[StructureType],
    sim_bounds: tuple,
) -> tuple[str, float, str] | None:
    """Return the first generated-grid axis whose grid-spec estimate is too small."""
    if not self.snapped_grid_used:
        return None

    sim_size = tuple(bmax - bmin for bmin, bmax in zip(*sim_bounds))
    medium_structures = [structure for structure in structures if isinstance(structure, Structure)]
    min_dl_by_axis = [
        grid.estimated_min_dl(wavelength, medium_structures, sim_size)
        for grid in (self.grid_x, self.grid_y, self.grid_z)
    ]

    for axis, (grid_spec, min_size) in enumerate(
        zip((self.grid_x, self.grid_y, self.grid_z), min_dl_by_axis)
    ):
        if not isinstance(grid_spec, AbstractAutoGrid):
            continue
        if sim_size[axis] == 0:
            continue
        if min_size >= MIN_GRID_SPACING:
            continue
        return "xyz"[axis], min_size, type(grid_spec).__name__

    return None


def _generated_grid_size_violation(
    self: GridSpec, grid: Grid, sim_size: tuple[float, float, float]
) -> tuple[str, float, str] | None:
    """Return the first generated-grid axis whose actual cell size is too small."""
    if not self.snapped_grid_used:
        return None

    for axis, (grid_spec, sizes) in enumerate(
        zip((self.grid_x, self.grid_y, self.grid_z), grid.sizes.to_list)
    ):
        if not isinstance(grid_spec, AbstractAutoGrid):
            continue
        if sim_size[axis] == 0:
            continue

        min_size = float(np.min(sizes))
        if min_size >= MIN_GRID_SPACING:
            continue
        return "xyz"[axis], min_size, type(grid_spec).__name__

    return None


def _validate_generated_grid_size(
    self: GridSpec, grid: Grid, sim_size: tuple[float, float, float]
) -> None:
    """Error if a generated grid cell size is below the supported minimum."""
    violation = self._generated_grid_size_violation(grid, sim_size)
    if violation is None:
        return

    axis_name, min_size, grid_name = violation
    self._raise_generated_grid_size_error(grid_name, axis_name, min_size)


def _make_grid_and_snapping_lines(
    self: GridSpec,
    structures: list[Structure],
    symmetry: tuple[Symmetry, Symmetry, Symmetry],
    periodic: tuple[bool, bool, bool],
    sources: list[SourceType],
    num_pml_layers: list[tuple[NonNegativeInt, NonNegativeInt]],
    lumped_elements: list[LumpedElementType] = (),
    internal_override_structures: list[MeshOverrideStructure] | None = None,
    internal_snapping_points: list[CoordinateOptional] | None = None,
    boundary_types: tuple[tuple[str, str], tuple[str, str], tuple[str, str]] = [
        [None, None],
        [None, None],
        [None, None],
    ],
    structure_priority_mode: PriorityMode = "equal",
    cached_merged_geos: list[list[tuple[Any, Shapely]]] | None = None,
) -> tuple[Grid, list[CoordinateOptional]]:
    """Make the entire simulation grid based on some simulation parameters.
    Also return snapping point resulted from iterative gap meshing.

    Parameters
    ----------
    structures : list[Structure]
        List of structures present in the simulation. The first structure must be the
        simulation geometry with the simulation background medium.
    symmetry : tuple[Symmetry, Symmetry, Symmetry]
        Reflection symmetry across a plane bisecting the simulation domain
        normal to each of the three axes.
    periodic: tuple[bool, bool, bool]
        Apply periodic boundary condition or not along each of the dimensions.
        Only relevant for autogrids.
    sources : list[SourceType]
        List of sources.
    num_pml_layers : list[tuple[float, float]]
        List containing the number of absorber layers in - and + boundaries.
    lumped_elements : list[LumpedElementType]
        List of lumped elements.
    internal_override_structures : list[MeshOverrideStructure]
        If `None`, recomputes internal override structures.
    internal_snapping_points : list[CoordinateOptional]
        If `None`, recomputes internal snapping points.
    boundary_types : tuple[tuple[str, str], tuple[str, str], tuple[str, str]] = [[None, None], [None, None], [None, None]]
        Type of boundary conditions along each dimension: "pec/pmc", "periodic", or
        None for any other. This is relevant only for gap meshing.
    structure_priority_mode : PriorityMode
        Structure priority setting.
    cached_merged_geos : Optional[list[list[tuple[Any, Shapely]]]]
        Cached merged geometries for each layer. If None, will be computed.

    Returns
    -------
    tuple[Grid, list[CoordinateOptional]]:
        Entire simulation grid and snapping points generated during iterative gap meshing.
    """

    # Pre-compute results from parse_structures
    wavelength = self.get_wavelength(sources)
    sim_bounds = structures[0].geometry.bounds
    all_structures = self._get_all_structures_affecting_grid(
        structures,
        wavelength,
        lumped_elements,
        boundary_types,
        sim_bounds,
        structure_priority_mode,
        internal_override_structures,
    )

    parse_structures_interval_coords = []
    parse_structures_max_dl_list = []
    grids_1d = [self.grid_x, self.grid_y, self.grid_z]

    for idim, grid_1d in enumerate(grids_1d):
        if isinstance(grid_1d, AbstractAutoGrid):
            interval_coords, max_dl_list = grid_1d._parse_structures(
                axis=idim,
                structures=all_structures,
                wavelength=wavelength,
                symmetry=symmetry,
                snapping_points=self.all_snapping_points(
                    structures,
                    lumped_elements,
                    boundary_types,
                    sim_bounds,
                    internal_snapping_points,
                ),
            )
            parse_structures_interval_coords.append(interval_coords)
            parse_structures_max_dl_list.append(max_dl_list)
        else:
            # For non-AutoGrid, append None since they don't use _parse_structures
            parse_structures_interval_coords.append(None)
            parse_structures_max_dl_list.append(None)

    old_grid = self._make_grid_one_iteration(
        structures=structures,
        symmetry=symmetry,
        periodic=periodic,
        sources=sources,
        num_pml_layers=num_pml_layers,
        lumped_elements=lumped_elements,
        internal_override_structures=internal_override_structures,
        internal_snapping_points=internal_snapping_points,
        structure_priority_mode=structure_priority_mode,
        boundary_types=boundary_types,
        parse_structures_interval_coords=parse_structures_interval_coords,
        parse_structures_max_dl_list=parse_structures_max_dl_list,
        all_structures=all_structures,
    )
    self._validate_generated_grid_size(old_grid, sim_size=structures[0].geometry.size)

    # gap refinement only place snapping points, and decrease dl_min that only affects
    # snapping points insertion.
    snapping_lines = []
    if len(self.layer_refinement_specs) > 0:
        num_iters = max(layer_spec.gap_meshing_iters for layer_spec in self.layer_refinement_specs)

        min_gap_width = inf
        for ind in range(num_iters):
            new_snapping_lines = []
            sim_bounds = structures[0].geometry.bounds
            for ind_layer, layer_spec in enumerate(self.layer_refinement_specs):
                if layer_spec.gap_meshing_iters > ind:
                    # use cached merged geometries if available, otherwise compute
                    if cached_merged_geos is not None:
                        merged_geos = cached_merged_geos[ind_layer]
                    else:
                        merged_geos = layer_spec._merged_geos(
                            structures,
                            sim_bounds,
                            boundary_types,
                        )
                    one_layer_snapping_lines, gap_width = layer_spec._resolve_gaps(
                        old_grid,
                        merged_geos,
                        boundary_types,
                    )
                    new_snapping_lines = new_snapping_lines + one_layer_snapping_lines
                    if layer_spec.dl_min_from_gap_width:
                        min_gap_width = min(min_gap_width, gap_width)

                        # Warn if dl_min_from_gaps would be very small relative to lateral grid size
                        if gap_width < inf:
                            # Get lateral dimensions (perpendicular to layer axis)
                            _, tan_dims = Box.pop_axis([0, 1, 2], layer_spec.axis)
                            dim_names = ["x", "y", "z"]

                            # Get grid sizes along lateral dimensions
                            grid_sizes = old_grid.sizes.to_dict
                            lateral_grid_sizes = [
                                grid_sizes[dim_names[tan_dims[0]]],
                                grid_sizes[dim_names[tan_dims[1]]],
                            ]

                            # Find minimum grid size along lateral dimensions
                            min_lateral_grid_size = min(
                                min(sizes) if len(sizes) > 0 else inf
                                for sizes in lateral_grid_sizes
                            )

                            # Calculate dl_min_from_gaps for this layer spec
                            dl_min_from_gaps = DL_MIN_FROM_GAPS_FRACTION * gap_width

                            # Warn if dl_min_from_gaps is too small relative to lateral grid size
                            if (
                                min_lateral_grid_size < inf
                                and dl_min_from_gaps
                                < GAP_REFINEMENT_WARNING_THRESH * min_lateral_grid_size
                            ):
                                log.warning(
                                    f"'LayerRefinementSpec' (axis={layer_spec.axis}) detected a very small gap width "
                                    f"({gap_width:.2e}), resulting in 'dl_min_from_gaps'={dl_min_from_gaps:.2e}. "
                                    f"This is less than {GAP_REFINEMENT_WARNING_THRESH * 100:.0f}% of the smallest "
                                    f"lateral grid size ({min_lateral_grid_size:.2e}). This may lead to "
                                    "excessive grid refinement. Consider adjusting the geometry or grid "
                                    "specification.",
                                    log_once=True,
                                )

            if len(new_snapping_lines) == 0:
                log.info(
                    "Grid is no longer changing. "
                    f"Stopping iterative gap meshing after {ind + 1}/{num_iters} iterations."
                )
                break

            snapping_lines = snapping_lines + new_snapping_lines
            dl_min_from_gaps = DL_MIN_FROM_GAPS_FRACTION * min_gap_width

            new_grid = self._make_grid_one_iteration(
                structures=structures,
                symmetry=symmetry,
                periodic=periodic,
                sources=sources,
                num_pml_layers=num_pml_layers,
                lumped_elements=lumped_elements,
                internal_override_structures=internal_override_structures,
                internal_snapping_points=snapping_lines + (internal_snapping_points or []),
                dl_min_from_gaps=dl_min_from_gaps,
                structure_priority_mode=structure_priority_mode,
                boundary_types=boundary_types,
                parse_structures_interval_coords=parse_structures_interval_coords,
                parse_structures_max_dl_list=parse_structures_max_dl_list,
                all_structures=all_structures,
            )
            self._validate_generated_grid_size(new_grid, sim_size=structures[0].geometry.size)

            same = old_grid == new_grid

            if same:
                log.info(
                    "Grid is no longer changing. "
                    f"Stopping iterative gap meshing after {ind + 1}/{num_iters} iterations."
                )
                break

            old_grid = new_grid

        # Small-geometry resolution: measure the final gap-meshed grid and refine any disjoint
        # metal geometry it leaves under-resolved, in one extra rebuild.
        # Unlike corner/edge refinement these overrides are emitted post-mesh, not pre-mesh.
        small_geometry_overrides = []
        dl_min_from_small_geometry = inf
        for ind_layer, layer_spec in enumerate(self.layer_refinement_specs):
            if layer_spec.min_steps_per_geometry is None:
                continue
            # reuse the merge gap meshing already consumed; never rerun it, no corner detection
            if cached_merged_geos is not None:
                merged_geos = cached_merged_geos[ind_layer]
            else:
                merged_geos = layer_spec._merged_geos(structures, sim_bounds, boundary_types)
            layer_overrides, layer_dl_min = layer_spec._small_geometry_measurement_overrides(
                old_grid, merged_geos
            )
            small_geometry_overrides += layer_overrides
            dl_min_from_small_geometry = min(dl_min_from_small_geometry, layer_dl_min)

        if small_geometry_overrides:
            # Lower dl_min as gap meshing does (via min) so sub-dl_min overrides are not
            # clamped. The override set changed, so let parse/structures recompute (pass None).
            dl_min_from_gaps = min(
                DL_MIN_FROM_GAPS_FRACTION * min_gap_width, dl_min_from_small_geometry
            )
            # Recompute the internal (corner/edge) overrides when the caller passed None, so the
            # rebuild keeps them: passing a non-None list suppresses their recomputation in
            # all_override_structures, which would drop in-plane refinement from the final mesh.
            base_override_structures = internal_override_structures
            if base_override_structures is None:
                base_override_structures = self.internal_override_structures(
                    structures,
                    wavelength,
                    sim_bounds,
                    lumped_elements,
                    boundary_types,
                )
            old_grid = self._make_grid_one_iteration(
                structures=structures,
                symmetry=symmetry,
                periodic=periodic,
                sources=sources,
                num_pml_layers=num_pml_layers,
                lumped_elements=lumped_elements,
                internal_override_structures=base_override_structures + small_geometry_overrides,
                internal_snapping_points=snapping_lines + (internal_snapping_points or []),
                dl_min_from_gaps=dl_min_from_gaps,
                structure_priority_mode=structure_priority_mode,
                boundary_types=boundary_types,
            )
            self._validate_generated_grid_size(old_grid, sim_size=structures[0].geometry.size)

    return old_grid, snapping_lines


def _make_grid_one_iteration(
    self: GridSpec,
    structures: list[Structure],
    symmetry: tuple[Symmetry, Symmetry, Symmetry],
    periodic: tuple[bool, bool, bool],
    sources: list[SourceType],
    num_pml_layers: list[tuple[NonNegativeInt, NonNegativeInt]],
    boundary_types: tuple[tuple[str, str], tuple[str, str], tuple[str, str]],
    lumped_elements: list[LumpedElementType] = (),
    internal_override_structures: list[MeshOverrideStructure] | None = None,
    internal_snapping_points: list[CoordinateOptional] | None = None,
    dl_min_from_gaps: PositiveFloat = inf,
    structure_priority_mode: PriorityMode = "equal",
    parse_structures_interval_coords: list[np.ndarray] | None = None,
    parse_structures_max_dl_list: list[np.ndarray] | None = None,
    all_structures: list[StructureType] | None = None,
) -> Grid:
    """Make the entire simulation grid based on some simulation parameters.

    Parameters
    ----------
    structures : list[Structure]
        List of structures present in the simulation. The first structure must be the
        simulation geometry with the simulation background medium.
    symmetry : tuple[Symmetry, Symmetry, Symmetry]
        Reflection symmetry across a plane bisecting the simulation domain
        normal to each of the three axes.
    periodic: tuple[bool, bool, bool]
        Apply periodic boundary condition or not along each of the dimensions.
        Only relevant for autogrids.
    sources : list[SourceType]
        List of sources.
    num_pml_layers : list[tuple[float, float]]
        List containing the number of absorber layers in - and + boundaries.
    lumped_elements : list[LumpedElementType]
        List of lumped elements.
    internal_override_structures : list[MeshOverrideStructure]
        If `None`, recomputes internal override structures.
    internal_snapping_points : list[CoordinateOptional]
        If `None`, recomputes internal snapping points.
    dl_min_from_gaps : PositiveFloat
        Minimal grid size computed based on autodetected gaps.
    structure_priority_mode : PriorityMode
        Structure priority setting.
    parse_structures_interval_coords : Optional[List[np.ndarray]]
        If not None, pre-computed interval coordinates from parsing structures for each dimension.
        List of length 3, one for each axis (x, y, z).
    parse_structures_max_dl_list : Optional[List[np.ndarray]]
        If not None, pre-computed maximum grid spacing list from parsing structures for each dimension.
        List of length 3, one for each axis (x, y, z).
    all_structures : Optional[List[StructureType]]
        If not None, pre-computed original and override structures affecting the grid.

    Returns
    -------
    Grid:
        Entire simulation grid.
    """

    # Set up wavelength for automatic mesh generation if needed.
    wavelength = self.get_wavelength(sources)

    # Warn user if ``GridType`` along some axis is not ``AutoGrid`` and
    # ``override_structures`` is not empty. The override structures
    # are not effective along those axes.
    for axis_ind, override_used_axis, snapping_used_axis, grid_axis in zip(
        ["x", "y", "z"],
        self.override_structures_used,
        self.snapping_points_used,
        [self.grid_x, self.grid_y, self.grid_z],
    ):
        if not isinstance(grid_axis, AbstractAutoGrid):
            if override_used_axis:
                log.warning(
                    f"Override structures take no effect along {axis_ind}-axis. "
                    "If intending to apply override structures to this axis, "
                    "use 'AutoGrid' or 'QuasiUniformGrid'.",
                    capture=False,
                )
            if snapping_used_axis:
                log.warning(
                    f"Snapping points take no effect along {axis_ind}-axis. "
                    "If intending to apply snapping points to this axis, "
                    "use 'AutoGrid' or 'QuasiUniformGrid'.",
                    capture=False,
                )

        if self.layer_refinement_used and not isinstance(grid_axis, AutoGrid):
            log.warning(
                f"layer_refinement_specs take no effect along {axis_ind}-axis. "
                "If intending to apply automatic refinement to this axis, "
                "use 'AutoGrid'.",
                capture=False,
            )

    grids_1d = [self.grid_x, self.grid_y, self.grid_z]

    if any(s._strip_traced_fields() for s in self.override_structures):
        log.warning(
            "The override structures were detected as having a dependence on the objective "
            "function parameters. This is not supported by our automatic differentiation "
            "framework. The derivative will be un-traced through the override structures. "
            "To make this explicit and remove this warning, use 'y = autograd.tracer.getval(x)'"
            " to remove any derivative information from values being passed to create "
            "override structures. Alternatively, 'obj = obj.to_static()' will create a copy of "
            "an instance without any autograd tracers."
        )

    sim_bounds = structures[0].geometry.bounds
    if all_structures is None:
        all_structures = self._get_all_structures_affecting_grid(
            structures,
            wavelength,
            lumped_elements,
            boundary_types,
            sim_bounds,
            structure_priority_mode,
            internal_override_structures,
        )

    violation = self._grid_spec_size_estimate_violation(
        wavelength=wavelength,
        structures=all_structures,
        sim_bounds=sim_bounds,
    )
    if violation is not None:
        axis_name, min_size, grid_name = violation
        self._raise_generated_grid_size_error(grid_name, axis_name, min_size)

    # apply internal `dl_min` if any AutoGrid has unset `dl_min`
    update_dl_min = False
    for grid in grids_1d:
        if isinstance(grid, AutoGrid) and grid._undefined_dl_min:
            update_dl_min = True
            break
    if update_dl_min:
        new_dl_min = self._dl_min(
            wavelength,
            list(structures) + self.external_override_structures,
            sim_bounds,
            lumped_elements,
            boundary_types,
        )
        new_dl_min = min(new_dl_min, dl_min_from_gaps)
        for ind, grid in enumerate(grids_1d):
            if isinstance(grid, AutoGrid) and grid._undefined_dl_min:
                grids_1d[ind] = grid.updated_copy(dl_min=new_dl_min)

    coords_dict = {}
    for idim, (dim, grid_1d) in enumerate(zip("xyz", grids_1d)):
        coords_dict[dim] = grid_1d.make_coords(
            axis=idim,
            structures=all_structures,
            symmetry=symmetry,
            periodic=periodic[idim],
            wavelength=wavelength,
            num_pml_layers=num_pml_layers[idim],
            snapping_points=self.all_snapping_points(
                structures,
                lumped_elements,
                boundary_types,
                sim_bounds,
                internal_snapping_points,
            ),
            parse_structures_interval_coords=(
                parse_structures_interval_coords[idim]
                if parse_structures_interval_coords is not None
                else None
            ),
            parse_structures_max_dl_list=(
                parse_structures_max_dl_list[idim]
                if parse_structures_max_dl_list is not None
                else None
            ),
        )

    coords = Coords(**coords_dict)
    return Grid(boundaries=coords)
