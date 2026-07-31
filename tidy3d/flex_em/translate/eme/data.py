"""EME simulation result-data translation."""

from __future__ import annotations

from typing import Any

from flex_em.schema.tidy3d.components.eme.data.sim_data import (
    EMESimulationData as EMESimulationDataOutput,
)

from tidy3d.flex_em.translate.base import (
    dump_data_for_public,
    dump_data_for_schema,
    validate_public_data,
)


def from_data(data: Any) -> EMESimulationDataOutput:
    """Convert public EME simulation data to the schema model."""

    return EMESimulationDataOutput.model_validate(dump_data_for_schema(data))


def to_data(data: Any) -> Any:
    """Convert schema EME simulation data to the public Tidy3D model."""

    from tidy3d.components.eme.data.sim_data import EMESimulationData

    return validate_public_data(EMESimulationData, dump_data_for_public(data))
