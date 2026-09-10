"""Terminal component modeler translation."""

from __future__ import annotations

from typing import Any

from flexcompute.core._migration.em.schema.tidy3d.plugins.smatrix.component_modelers.terminal import (
    TerminalComponentModeler as TerminalComponentModelerTaskInput,
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


def from_task(modeler: Any) -> TerminalComponentModelerTaskInput:
    """Convert a public Tidy3D terminal component modeler to the schema model."""

    payload = apply_type_field_map(
        dump_for_schema(modeler, type_name="TerminalComponentModeler"),
        PUBLIC_TO_SCHEMA_TYPE_FIELD_MAP,
    )
    return TerminalComponentModelerTaskInput.model_validate(payload)


def to_task(modeler: Any) -> Any:
    """Convert a schema terminal component modeler to the public Tidy3D model."""

    from tidy3d.plugins.smatrix.component_modelers.terminal import (
        TerminalComponentModeler,
    )

    payload = apply_type_field_map(
        dump_for_public(modeler, type_name="TerminalComponentModeler"),
        SCHEMA_TO_PUBLIC_TYPE_FIELD_MAP,
    )
    return TerminalComponentModeler.model_validate(payload)
