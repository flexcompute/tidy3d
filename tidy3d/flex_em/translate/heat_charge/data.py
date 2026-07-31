"""Heat-charge simulation result-data translation."""

from __future__ import annotations

from typing import Any

from flex_em.schema.tidy3d.components.tcad.data.sim_data import (
    HeatChargeSimulationData as HeatChargeSimulationDataOutput,
)

from tidy3d.flex_em.translate.base import (
    dump_data_for_public,
    dump_data_for_schema,
    validate_public_data,
)


def from_data(data: Any) -> HeatChargeSimulationDataOutput:
    """Convert public heat-charge simulation data to the schema model."""

    return HeatChargeSimulationDataOutput.model_validate(dump_data_for_schema(data))


def to_data(data: Any) -> Any:
    """Convert schema heat-charge simulation data to the public Tidy3D model."""

    from tidy3d.components.tcad.data.sim_data import (
        HeatChargeSimulationData,
    )

    return validate_public_data(HeatChargeSimulationData, dump_data_for_public(data))
