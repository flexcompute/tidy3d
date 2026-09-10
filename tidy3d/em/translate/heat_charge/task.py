"""Heat-charge simulation translation."""

from __future__ import annotations

from typing import Any

from flexcompute.core._migration.em.schema.tidy3d.components.tcad.simulation.heat_charge import (
    HeatChargeSimulation as HeatChargeSimulationTaskInput,
)

from tidy3d.em.translate.base import dump_for_public, dump_for_schema


def from_task(sim: Any) -> HeatChargeSimulationTaskInput:
    """Convert a public Tidy3D heat-charge simulation to the schema model."""

    return HeatChargeSimulationTaskInput.model_validate(
        dump_for_schema(sim, type_name="HeatChargeSimulation")
    )


def to_task(sim: Any) -> Any:
    """Convert a schema heat-charge simulation to the public Tidy3D model."""

    from tidy3d.components.tcad.simulation.heat_charge import (
        HeatChargeSimulation,
    )

    return HeatChargeSimulation.model_validate(
        dump_for_public(sim, type_name="HeatChargeSimulation")
    )
