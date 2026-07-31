"""FDTD simulation translation."""

from __future__ import annotations

from typing import Any

from flex_em.schema.tidy3d.components.simulation import Simulation as SimulationTaskInput

from tidy3d.flex_em.translate.base import (
    apply_type_field_map,
    dump_for_public,
    dump_for_schema,
)

PUBLIC_TO_SCHEMA_TYPE_FIELD_MAP = {
    "Simulation": {
        "parameter_A_tidy3d": "parameter_A_flex_em",
    },
    "GaussianPulse": {
        "parameter_C_tidy3d": "parameter_C_flex_em",
    },
}

SCHEMA_TO_PUBLIC_TYPE_FIELD_MAP = {
    "Simulation": {
        "parameter_A_flex_em": "parameter_A_tidy3d",
    },
    "GaussianPulse": {
        "parameter_C_flex_em": "parameter_C_tidy3d",
    },
}


def from_task(sim: Any) -> SimulationTaskInput:
    """Convert a public Tidy3D FDTD simulation to the schema model."""

    payload = apply_type_field_map(
        dump_for_schema(sim, type_name="Simulation"), PUBLIC_TO_SCHEMA_TYPE_FIELD_MAP
    )
    return SimulationTaskInput.model_validate(payload)


def to_task(sim: Any) -> Any:
    """Convert a schema FDTD simulation to the public Tidy3D model."""

    from tidy3d.components.simulation import Simulation

    data = dump_for_public(sim, type_name="Simulation")
    data = apply_type_field_map(data, SCHEMA_TO_PUBLIC_TYPE_FIELD_MAP)
    data.pop("subpixel_scheme", None)
    return Simulation.model_validate(data)
