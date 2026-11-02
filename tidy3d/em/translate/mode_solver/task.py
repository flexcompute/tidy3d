"""Mode solver translation."""

from __future__ import annotations

from typing import Any

from flexcompute.core._migration.solver.mode.schema.mode_solver import (
    ModeSolver as ModeSolverTaskInput,
)

from tidy3d.em.translate.base import dump_for_public, dump_for_schema


def from_task(mode_solver: Any) -> ModeSolverTaskInput:
    """Convert a public Tidy3D mode solver to the schema model."""

    return ModeSolverTaskInput.model_validate(dump_for_schema(mode_solver, type_name="ModeSolver"))


def to_task(mode_solver: Any) -> Any:
    """Convert a schema mode solver to the public Tidy3D model."""

    from tidy3d.plugins.mode import ModeSolver

    return ModeSolver.model_validate(dump_for_public(mode_solver, type_name="ModeSolver"))
