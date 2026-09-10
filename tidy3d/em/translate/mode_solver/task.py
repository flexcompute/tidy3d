"""Mode solver translation."""

from __future__ import annotations

from typing import Any

from flexcompute.core._migration.em.schema.tidy3d.components.mode.mode_solver import (
    ModeSolver as ModeSolverTaskInput,
)

from tidy3d.em.translate.base import (
    apply_type_field_map,
    dump_for_public,
    dump_for_schema,
)
from tidy3d.em.translate.fdtd.task import (
    PUBLIC_TO_SCHEMA_TYPE_FIELD_MAP,
    SCHEMA_TO_PUBLIC_TYPE_FIELD_MAP,
)


def from_task(mode_solver: Any) -> ModeSolverTaskInput:
    """Convert a public Tidy3D mode solver to the schema model."""

    payload = apply_type_field_map(
        dump_for_schema(mode_solver, type_name="ModeSolver"),
        PUBLIC_TO_SCHEMA_TYPE_FIELD_MAP,
    )
    return ModeSolverTaskInput.model_validate(payload)


def to_task(mode_solver: Any) -> Any:
    """Convert a schema mode solver to the public Tidy3D model."""

    from tidy3d.plugins.mode import ModeSolver

    payload = apply_type_field_map(
        dump_for_public(mode_solver, type_name="ModeSolver"),
        SCHEMA_TO_PUBLIC_TYPE_FIELD_MAP,
    )
    return ModeSolver.model_validate(payload)
