"""Heat simulation translation."""

from __future__ import annotations

from typing import Any

from flexcompute.core._migration.em.schema.tidy3d.components.tcad.simulation.heat import (
    HeatSimulation as HeatSimulationTaskInput,
)

from tidy3d.em.translate.base import dump_for_public, dump_for_schema


def from_task(sim: Any) -> HeatSimulationTaskInput:
    """Convert a public Tidy3D heat simulation to the schema model."""

    return HeatSimulationTaskInput.model_validate(dump_for_schema(sim, type_name="HeatSimulation"))


def to_task(sim: Any) -> Any:
    """Convert a schema heat simulation to the public Tidy3D model."""

    from tidy3d.components.tcad.simulation.heat import HeatSimulation

    return HeatSimulation.model_validate(dump_for_public(sim, type_name="HeatSimulation"))
