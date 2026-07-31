"""Mode simulation translation."""

from __future__ import annotations

from typing import Any

from flex_em.schema.tidy3d.components.mode.simulation import (
    ModeSimulation as ModeSimulationTaskInput,
)

from tidy3d.flex_em.translate.base import dump_for_public, dump_for_schema


def from_task(sim: Any) -> ModeSimulationTaskInput:
    """Convert a public Tidy3D mode simulation to the schema model."""

    return ModeSimulationTaskInput.model_validate(dump_for_schema(sim, type_name="ModeSimulation"))


def to_task(sim: Any) -> Any:
    """Convert a schema mode simulation to the public Tidy3D model."""

    from tidy3d.components.mode.simulation import ModeSimulation

    data = dump_for_public(sim, type_name="ModeSimulation")
    data.pop("subpixel_scheme", None)
    return ModeSimulation.model_validate(data)
