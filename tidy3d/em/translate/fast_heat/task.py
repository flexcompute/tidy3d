"""Fast-heat simulation translation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flexcompute.core._migration.solver.fast_heat.schema.fast_heat import (
    FastHeatSimulation as FastHeatSimulationTaskInput,
)

from tidy3d.em.translate.base import dump_for_public, dump_for_schema

if TYPE_CHECKING:
    from tidy3d.components.grid.grid import Grid


def from_task(sim: Any) -> FastHeatSimulationTaskInput:
    """Convert a public Tidy3D fast-heat simulation to the schema model."""

    return FastHeatSimulationTaskInput.model_validate(
        dump_for_schema(sim, type_name="FastHeatSimulation")
    )


def to_task(sim: Any) -> Any:
    """Convert a schema fast-heat simulation to the public Tidy3D model."""

    from tidy3d.components.fast_heat.simulation import FastHeatSimulation

    return FastHeatSimulation.model_validate(dump_for_public(sim, type_name="FastHeatSimulation"))


def grids(sim: Any) -> tuple[Grid, ...]:
    """The grids the schema model of a public fast-heat simulation generates, as public
    :class:`.Grid` objects."""

    from tidy3d.components.grid.grid import Coords, Grid

    return tuple(
        Grid(boundaries=Coords(**grid.boundaries.to_dict)) for grid in from_task(sim).grids
    )
