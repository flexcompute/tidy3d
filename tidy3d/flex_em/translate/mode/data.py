"""Mode-simulation result-data translation."""

from __future__ import annotations

from typing import Any

from flex_em.schema.tidy3d.components.mode.data.sim_data import (
    ModeSimulationData as ModeSimulationDataOutput,
)

from tidy3d.flex_em.translate.base import (
    dump_data_for_public,
    dump_data_for_schema,
    validate_public_data,
)


def from_data(data: Any) -> ModeSimulationDataOutput:
    """Convert public mode-simulation data to the schema model."""

    return ModeSimulationDataOutput.model_validate(dump_data_for_schema(data))


def to_data(data: Any) -> Any:
    """Convert schema mode-simulation data to the public Tidy3D model."""

    from tidy3d.components.mode.data.sim_data import ModeSimulationData

    payload = dump_data_for_public(data)
    return validate_public_data(ModeSimulationData, payload)
