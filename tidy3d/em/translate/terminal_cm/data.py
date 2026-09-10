"""Terminal component-modeler result-data translation."""

from __future__ import annotations

from typing import Any

from flexcompute.core._migration.em.schema.tidy3d.plugins.smatrix.data.terminal import (
    TerminalComponentModelerData as TerminalComponentModelerDataOutput,
)

from tidy3d.em.translate.base import (
    apply_type_field_map,
    dump_data_for_public,
    dump_data_for_schema,
    validate_public_data,
)
from tidy3d.em.translate.fdtd.data import (
    PUBLIC_TO_SCHEMA_TYPE_FIELD_MAP,
    SCHEMA_TO_PUBLIC_TYPE_FIELD_MAP,
)


def from_data(data: Any) -> TerminalComponentModelerDataOutput:
    """Convert public terminal component-modeler data to the schema model."""

    payload = apply_type_field_map(dump_data_for_schema(data), PUBLIC_TO_SCHEMA_TYPE_FIELD_MAP)
    return TerminalComponentModelerDataOutput.model_validate(payload)


def to_data(data: Any) -> Any:
    """Convert schema terminal component-modeler data to the public Tidy3D model."""

    from tidy3d.plugins.smatrix.data.terminal import (
        TerminalComponentModelerData,
    )

    payload = apply_type_field_map(dump_data_for_public(data), SCHEMA_TO_PUBLIC_TYPE_FIELD_MAP)
    return validate_public_data(TerminalComponentModelerData, payload)
