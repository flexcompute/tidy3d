"""FDTD simulation result-data translation."""

from __future__ import annotations

from typing import Any

from flexcompute.core._migration.em.schema.tidy3d.components.data.sim_data import (
    SimulationData as SimulationDataOutput,
)

from tidy3d.em.translate.base import (
    apply_type_field_map,
    dump_data_for_public,
    dump_data_for_schema,
    validate_data,
    validate_public_data,
)

PUBLIC_TO_SCHEMA_TYPE_FIELD_MAP: dict[str, dict[str, str]] = {}

SCHEMA_TO_PUBLIC_TYPE_FIELD_MAP: dict[str, dict[str, str]] = {}


def from_data(data: Any) -> SimulationDataOutput:
    """Convert public FDTD simulation data to the schema model."""

    payload = apply_type_field_map(dump_data_for_schema(data), PUBLIC_TO_SCHEMA_TYPE_FIELD_MAP)
    return validate_data(SimulationDataOutput, payload)


def to_data(data: Any) -> Any:
    """Convert schema FDTD simulation data to the public Tidy3D model."""

    from tidy3d.components.data.sim_data import SimulationData

    payload = apply_type_field_map(dump_data_for_public(data), SCHEMA_TO_PUBLIC_TYPE_FIELD_MAP)
    return validate_public_data(SimulationData, payload)
