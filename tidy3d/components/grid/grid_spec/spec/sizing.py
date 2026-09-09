"""Grid-spec sizing helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tidy3d.components.structure import MeshOverrideStructure, Structure
from tidy3d.constants import inf

if TYPE_CHECKING:
    from tidy3d.components.lumped_element import LumpedElementType
    from tidy3d.components.structure import StructureType
    from tidy3d.components.types import Shapely

    from .model import GridSpec


from tidy3d.components.grid.grid_spec.constants import (
    MIN_STEP_BOUND_SCALE,
)
from tidy3d.components.grid.grid_spec.grid_1d import (
    AutoGrid,
)


def _min_vacuum_dl_in_autogrid(
    self: GridSpec, wavelength: float, sim_size: tuple[float, 3]
) -> float:
    """Compute grid step size in vacuum for Autogrd. If AutoGrid is applied along more than 1 dimension,
    return the minimal.
    """
    dl = inf
    for grid in [self.grid_x, self.grid_y, self.grid_z]:
        if isinstance(grid, AutoGrid):
            dl = min(dl, grid._vacuum_dl(wavelength, sim_size))
    return dl


def _dl_min(
    self: GridSpec,
    wavelength: float,
    structure_list: list[StructureType],
    sim_bounds: tuple,
    lumped_elements: list[LumpedElementType],
    boundary_types: tuple[tuple[str, str], tuple[str, str], tuple[str, str]],
    cached_merged_geos: list[list[tuple[Any, Shapely]]] | None = None,
) -> float:
    """Lower bound of grid size to be applied to dimensions where AutoGrid with unset
    `dl_min` (0 or None) is applied.
    """

    return (
        min(
            self._estimated_min_dl_by_axis(
                wavelength=wavelength,
                structure_list=structure_list,
                sim_bounds=sim_bounds,
                boundary_types=boundary_types,
                cached_merged_geos=cached_merged_geos,
                lumped_elements=lumped_elements,
            )
        )
        * MIN_STEP_BOUND_SCALE
    )


def _estimated_min_dl_by_axis(
    self: GridSpec,
    wavelength: float,
    structure_list: list[StructureType],
    sim_bounds: tuple,
    boundary_types: tuple[tuple[str, str], tuple[str, str], tuple[str, str]],
    cached_merged_geos: list[list[tuple[Any, Shapely]]] | None = None,
    lumped_elements: list[LumpedElementType] = (),
    layer_refinement_scale: float = 1.0,
) -> tuple[float, float, float]:
    """Estimate minimum grid size per axis before full grid generation."""
    # split structure list into `Structure` and `MeshOverrideStructure`
    structures = [medium_str for medium_str in structure_list if isinstance(medium_str, Structure)]
    mesh_structures = [
        mesh_str for mesh_str in structure_list if isinstance(mesh_str, MeshOverrideStructure)
    ]
    for lumped_element in lumped_elements:
        mesh_structures.extend(lumped_element.to_mesh_overrides())

    # local simulation size derived from bounds
    sim_size_local = tuple(bmax - bmin for bmin, bmax in zip(*sim_bounds))
    grids = (self.grid_x, self.grid_y, self.grid_z)
    undefined_auto_axes = [
        axis
        for axis, grid in enumerate(grids)
        if isinstance(grid, AutoGrid) and grid._undefined_dl_min
    ]

    # from mesh specification
    min_dl_by_axis = [
        grid.estimated_min_dl(wavelength, structures, sim_size_local) for grid in grids
    ]

    # minimal grid size from MeshOverrideStructure
    for structure in mesh_structures:
        for axis, dl in enumerate(structure._dl):
            if dl is not None:
                min_dl_by_axis[axis] = min(min_dl_by_axis[axis], dl)

    # from layer refinement specifications
    if self.layer_refinement_used and undefined_auto_axes:
        min_vacuum_dl = self._min_vacuum_dl_in_autogrid(wavelength, sim_size_local)
        for ind, layer in enumerate(self.layer_refinement_specs):
            cached_merged = cached_merged_geos[ind] if cached_merged_geos is not None else None
            layer_dl_min = layer_refinement_scale * layer.suggested_dl_min(
                min_vacuum_dl,
                structures,
                sim_bounds,
                boundary_types,
                cached_merged,
            )
            for axis in undefined_auto_axes:
                min_dl_by_axis[axis] = min(min_dl_by_axis[axis], layer_dl_min)

    return tuple(min_dl_by_axis)
