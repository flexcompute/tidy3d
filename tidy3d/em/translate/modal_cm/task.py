"""Modal component modeler translation."""

from __future__ import annotations

from typing import Any

from flexcompute.core._migration.em.schema.tidy3d.plugins.smatrix.component_modelers.modal import (
    ModalComponentModeler as ModalComponentModelerTaskInput,
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


def from_task(modeler: Any) -> ModalComponentModelerTaskInput:
    """Convert a public Tidy3D modal component modeler to the schema model."""

    payload = apply_type_field_map(
        dump_for_schema(modeler, type_name="ModalComponentModeler"),
        PUBLIC_TO_SCHEMA_TYPE_FIELD_MAP,
    )
    return ModalComponentModelerTaskInput.model_validate(payload)


def to_task(modeler: Any) -> Any:
    """Convert a schema modal component modeler to the public Tidy3D model."""

    from tidy3d.plugins.smatrix.component_modelers.modal import (
        ModalComponentModeler,
    )

    payload = apply_type_field_map(
        dump_for_public(modeler, type_name="ModalComponentModeler"),
        SCHEMA_TO_PUBLIC_TYPE_FIELD_MAP,
    )
    return ModalComponentModeler.model_validate(payload)
