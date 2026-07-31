"""Modal component-modeler result-data translation."""

from __future__ import annotations

from typing import Any

from flex_em.schema.tidy3d.plugins.smatrix.data.modal import (
    ModalComponentModelerData as ModalComponentModelerDataOutput,
)

from tidy3d.flex_em.translate.base import (
    apply_type_field_map,
    dump_data_for_public,
    dump_data_for_schema,
    validate_public_data,
)
from tidy3d.flex_em.translate.fdtd.data import (
    PUBLIC_TO_SCHEMA_TYPE_FIELD_MAP,
    SCHEMA_TO_PUBLIC_TYPE_FIELD_MAP,
)


def from_data(data: Any) -> ModalComponentModelerDataOutput:
    """Convert public modal component-modeler data to the schema model."""

    payload = apply_type_field_map(dump_data_for_schema(data), PUBLIC_TO_SCHEMA_TYPE_FIELD_MAP)
    return ModalComponentModelerDataOutput.model_validate(payload)


def to_data(data: Any) -> Any:
    """Convert schema modal component-modeler data to the public Tidy3D model."""

    from tidy3d.plugins.smatrix.data.modal import (
        ModalComponentModelerData,
    )

    payload = apply_type_field_map(dump_data_for_public(data), SCHEMA_TO_PUBLIC_TYPE_FIELD_MAP)
    return validate_public_data(ModalComponentModelerData, payload)
