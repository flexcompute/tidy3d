"""FDTD simulation result-data translation."""

from __future__ import annotations

from typing import Any

from flex_em.schema.tidy3d.components.data.sim_data import SimulationData as SimulationDataOutput

from tidy3d.flex_em.translate.base import (
    apply_type_field_map,
    dump_data_for_public,
    dump_data_for_schema,
    validate_public_data,
)

PUBLIC_TO_SCHEMA_TYPE_FIELD_MAP = {
    "SimulationData": {
        "parameter_B_tidy3d": "parameter_B_flex_em",
    },
    "Simulation": {
        "parameter_A_tidy3d": "parameter_A_flex_em",
    },
    "GaussianPulse": {
        "parameter_C_tidy3d": "parameter_C_flex_em",
    },
}

SCHEMA_TO_PUBLIC_TYPE_FIELD_MAP = {
    "SimulationData": {
        "parameter_B_flex_em": "parameter_B_tidy3d",
    },
    "Simulation": {
        "parameter_A_flex_em": "parameter_A_tidy3d",
    },
    "GaussianPulse": {
        "parameter_C_flex_em": "parameter_C_tidy3d",
    },
}


def from_data(data: Any) -> SimulationDataOutput:
    """Convert public FDTD simulation data to the schema model."""

    payload = apply_type_field_map(dump_data_for_schema(data), PUBLIC_TO_SCHEMA_TYPE_FIELD_MAP)
    return SimulationDataOutput.model_validate(payload)


def to_data(data: Any) -> Any:
    """Convert schema FDTD simulation data to the public Tidy3D model."""

    from tidy3d.components.data.sim_data import SimulationData

    payload = apply_type_field_map(dump_data_for_public(data), SCHEMA_TO_PUBLIC_TYPE_FIELD_MAP)
    return validate_public_data(SimulationData, payload)
