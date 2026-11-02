"""Mode simulation translation."""

from __future__ import annotations

from typing import Any

from flexcompute.core._migration.solver.mode.schema.simulation import (
    ModeSimulation as ModeSimulationTaskInput,
)

from tidy3d.em.translate.base import dump_for_public, dump_for_schema


def from_task(sim: Any) -> ModeSimulationTaskInput:
    """Convert a public Tidy3D mode simulation to the schema model."""

    return ModeSimulationTaskInput.model_validate(dump_for_schema(sim, type_name="ModeSimulation"))


def to_task(sim: Any) -> Any:
    """Convert a schema mode simulation to the public Tidy3D model."""

    from tidy3d.components.mode.simulation import ModeSimulation

    return ModeSimulation.model_validate(dump_for_public(sim, type_name="ModeSimulation"))
