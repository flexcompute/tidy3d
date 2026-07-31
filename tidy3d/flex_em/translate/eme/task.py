"""EME simulation translation."""

from __future__ import annotations

from typing import Any

from flex_em.schema.tidy3d.components.eme.simulation import EMESimulation as EMESimulationTaskInput

from tidy3d.flex_em.translate.base import dump_for_public, dump_for_schema


def from_task(sim: Any) -> EMESimulationTaskInput:
    """Convert a public Tidy3D EME simulation to the schema model."""

    return EMESimulationTaskInput.model_validate(dump_for_schema(sim, type_name="EMESimulation"))


def to_task(sim: Any) -> Any:
    """Convert a schema EME simulation to the public Tidy3D model."""

    from tidy3d.components.eme.simulation import EMESimulation

    return EMESimulation.model_validate(dump_for_public(sim, type_name="EMESimulation"))
