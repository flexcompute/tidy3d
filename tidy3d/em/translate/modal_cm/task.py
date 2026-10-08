"""Modal component-modeler translation."""

from __future__ import annotations

from typing import Any

from flexcompute.core._migration.solver.component_modeler.schema import (
    ComponentModelerTaskInput,
)

from tidy3d.em.translate.fdtd.task import from_task as simulation_from_task


def from_task(modeler: Any) -> ComponentModelerTaskInput:
    """Lower a public Tidy3D modal component modeler to its solver task input."""

    return ComponentModelerTaskInput(
        base_sim=simulation_from_task(modeler.base_sim),
        task_names=tuple(
            modeler.task_name_from_index(index) for index in modeler.matrix_indices_run_sim
        ),
    )


def to_task(task_input: Any) -> Any:
    """Refuse the reverse conversion: a task input holds no ports to rebuild the modeler from."""

    raise TypeError(
        "A component-modeler task input cannot be converted to a public modal component modeler."
    )
