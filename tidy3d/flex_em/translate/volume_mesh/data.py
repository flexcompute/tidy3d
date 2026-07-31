"""Volume-mesh result-data translation."""

from __future__ import annotations

from typing import Any

from flex_em.schema.tidy3d.components.tcad.data.sim_data import (
    VolumeMesherData as VolumeMesherDataOutput,
)

from tidy3d.flex_em.translate.base import (
    dump_data_for_public,
    dump_data_for_schema,
    validate_public_data,
)


def from_data(data: Any) -> VolumeMesherDataOutput:
    """Convert public volume-mesh data to the schema model."""

    return VolumeMesherDataOutput.model_validate(dump_data_for_schema(data))


def to_data(data: Any) -> Any:
    """Convert schema volume-mesh data to the public Tidy3D model."""

    from tidy3d.components.tcad.data.sim_data import VolumeMesherData

    return validate_public_data(VolumeMesherData, dump_data_for_public(data))
