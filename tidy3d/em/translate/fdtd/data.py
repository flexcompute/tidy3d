"""FDTD simulation result-data translation."""

from __future__ import annotations

from typing import Any

from flexcompute.core._migration.em.schema.tidy3d.components.data.sim_data import (
    SimulationData as SimulationDataOutput,
)

from tidy3d.em.translate.base import (
    dump_data_for_public,
    dump_data_for_schema,
    validate_data,
    validate_public_data,
)


def from_data(data: Any) -> SimulationDataOutput:
    """Convert public FDTD simulation data to the schema model."""

    return validate_data(SimulationDataOutput, dump_data_for_schema(data))


def to_data(data: Any) -> Any:
    """Convert schema FDTD simulation data to the public Tidy3D model."""

    from tidy3d.components.data.sim_data import SimulationData

    return validate_public_data(SimulationData, dump_data_for_public(data))
