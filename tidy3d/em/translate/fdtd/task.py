"""FDTD simulation translation."""

from __future__ import annotations

from typing import Any

from flexcompute.core._migration.solver.fdtd.schema.model import Simulation as SimulationTaskInput

from tidy3d.em.translate.base import dump_for_public, dump_for_schema


def from_task(sim: Any) -> SimulationTaskInput:
    """Convert a public Tidy3D FDTD simulation to the schema model."""

    return SimulationTaskInput.model_validate(dump_for_schema(sim, type_name="Simulation"))


def to_task(sim: Any) -> Any:
    """Convert a schema FDTD simulation to the public Tidy3D model."""

    from tidy3d.components.simulation import Simulation

    return Simulation.model_validate(dump_for_public(sim, type_name="Simulation"))
