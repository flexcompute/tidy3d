"""Grid-spec factories helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tidy3d.components.grid.mesher import GradedMesher
from tidy3d.components.types import Undefined

if TYPE_CHECKING:
    from pydantic import NonNegativeFloat, PositiveFloat

    from tidy3d.components.grid.grid import Grid
    from tidy3d.components.grid.grid_spec.refinement import LayerRefinementSpec
    from tidy3d.components.grid.mesher import MesherType
    from tidy3d.components.structure import StructureType
    from tidy3d.components.types import CoordinateOptional

    from .model import GridSpec


from tidy3d.components.grid.grid_spec.grid_1d import (
    AutoGrid,
    CustomGridBoundaries,
    QuasiUniformGrid,
    UniformGrid,
)


def from_grid(cls: type[GridSpec], grid: Grid) -> GridSpec:
    """Import grid directly from another simulation, e.g. ``grid_spec = GridSpec.from_grid(sim.grid)``."""
    grid_dict = {}
    for dim in "xyz":
        grid_dict["grid_" + dim] = CustomGridBoundaries(coords=grid.boundaries.to_dict[dim])
    return cls(**grid_dict)


def auto(
    cls: type[GridSpec],
    wavelength: PositiveFloat = None,
    min_steps_per_wvl: PositiveFloat = 10.0,
    max_scale: PositiveFloat = 1.4,
    override_structures: list[StructureType] = (),
    snapping_points: tuple[CoordinateOptional, ...] = (),
    layer_refinement_specs: list[LayerRefinementSpec] = (),
    dl_min: NonNegativeFloat = 0.0,
    min_steps_per_sim_size: PositiveFloat = 10.0,
    mesher: MesherType = Undefined,
) -> GridSpec:
    """Use the same :class:`.AutoGrid` along each of the three directions.

    Parameters
    ----------
    wavelength : PositiveFloat, optional
        Free-space wavelength for automatic nonuniform grid. It can be 'None'
        if there is at least one source in the simulation, in which case it is defined by
        the source central frequency.
    min_steps_per_wvl : PositiveFloat, optional
        Minimal number of steps per wavelength in each medium.
    max_scale : PositiveFloat, optional
        Sets the maximum ratio between any two consecutive grid steps.
    override_structures : list[StructureType]
        A list of structures that is added on top of the simulation structures in
        the process of generating the grid. This can be used to refine the grid or make it
        coarser depending than the expected need for higher/lower resolution regions.
    snapping_points : tuple[CoordinateOptional, ...]
        A set of points that enforce grid boundaries to pass through them.
    layer_refinement_specs: list[LayerRefinementSpec]
        Mesh refinement according to layer specifications.
    dl_min: NonNegativeFloat
        Lower bound of grid size.
    min_steps_per_sim_size : PositiveFloat, optional
        Minimal number of steps per longest edge length of simulation domain.
    mesher : MesherType = GradedMesher()
        The type of mesher to use to generate the grid automatically.

    Returns
    -------
    GridSpec
        :class:`.GridSpec` with the same automatic nonuniform grid settings in each direction.
    """
    if mesher is Undefined:
        mesher = GradedMesher()

    grid_1d = AutoGrid(
        min_steps_per_wvl=min_steps_per_wvl,
        min_steps_per_sim_size=min_steps_per_sim_size,
        max_scale=max_scale,
        dl_min=dl_min,
        mesher=mesher,
    )
    return cls(
        wavelength=wavelength,
        grid_x=grid_1d,
        grid_y=grid_1d,
        grid_z=grid_1d,
        override_structures=override_structures,
        snapping_points=snapping_points,
        layer_refinement_specs=layer_refinement_specs,
    )


def uniform(cls: type[GridSpec], dl: float) -> GridSpec:
    """Use the same :class:`.UniformGrid` along each of the three directions.

    Parameters
    ----------
    dl : float
        Grid size for uniform grid generation.

    Returns
    -------
    GridSpec
        :class:`.GridSpec` with the same uniform grid size in each direction.
    """

    grid_1d = UniformGrid(dl=dl)
    return cls(grid_x=grid_1d, grid_y=grid_1d, grid_z=grid_1d)


def quasiuniform(
    cls: type[GridSpec],
    dl: float,
    max_scale: PositiveFloat = 1.4,
    override_structures: list[StructureType] = (),
    snapping_points: tuple[CoordinateOptional, ...] = (),
    mesher: MesherType = Undefined,
) -> GridSpec:
    """Use the same :class:`.QuasiUniformGrid` along each of the three directions.

    Parameters
    ----------
    dl : float
        Grid size for quasi-uniform grid generation.
    max_scale : PositiveFloat, optional
        Sets the maximum ratio between any two consecutive grid steps.
    override_structures : list[StructureType]
        A list of structures that is added on top of the simulation structures in
        the process of generating the grid. This can be used to snap grid points to
        the bounding box boundary.
    snapping_points : tuple[CoordinateOptional, ...]
        A set of points that enforce grid boundaries to pass through them.
    mesher : MesherType = GradedMesher()
        The type of mesher to use to generate the grid automatically.

    Returns
    -------
    GridSpec
        :class:`.GridSpec` with the same uniform grid size in each direction.
    """
    if mesher is Undefined:
        mesher = GradedMesher()

    grid_1d = QuasiUniformGrid(dl=dl, max_scale=max_scale, mesher=mesher)
    return cls(
        grid_x=grid_1d,
        grid_y=grid_1d,
        grid_z=grid_1d,
        override_structures=override_structures,
        snapping_points=snapping_points,
    )
