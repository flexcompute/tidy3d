"""Mode-solver result-data translation."""

from __future__ import annotations

from typing import Any

from flexcompute.core._migration.em.schema.tidy3d.components.data.monitor_data import (
    ModeSolverData as ModeSolverDataOutput,
)
from flexcompute.core._migration.em.schema.tidy3d.components.microwave.data.monitor_data import (
    MicrowaveModeSolverData as MicrowaveModeSolverDataOutput,
)

from tidy3d.em.translate.base import (
    dump_data_for_public,
    dump_data_for_schema,
    validate_public_data,
)


def from_data(data: Any) -> ModeSolverDataOutput | MicrowaveModeSolverDataOutput:
    """Convert public mode-solver data to the schema model."""

    from tidy3d.components.microwave.data.monitor_data import MicrowaveModeSolverData

    if isinstance(data, MicrowaveModeSolverData):
        return MicrowaveModeSolverDataOutput.model_validate(dump_data_for_schema(data))
    return ModeSolverDataOutput.model_validate(dump_data_for_schema(data))


def to_data(data: Any) -> Any:
    """Convert schema mode-solver data to the public Tidy3D model."""

    from tidy3d.components.data.monitor_data import ModeSolverData
    from tidy3d.components.microwave.data.monitor_data import (
        MicrowaveModeSolverData,
    )

    if isinstance(data, MicrowaveModeSolverDataOutput):
        return validate_public_data(MicrowaveModeSolverData, dump_data_for_public(data))
    return validate_public_data(ModeSolverData, dump_data_for_public(data))
