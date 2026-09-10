"""Inverse-design optimizer result-data translation."""

from __future__ import annotations

from typing import Any

from flexcompute.core._migration.em.schema.tidy3d.plugins.invdes.result import (
    InverseDesignResult as InverseDesignResultOutput,
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


def from_data(data: Any) -> InverseDesignResultOutput:
    """Convert public inverse-design result data to the schema model."""

    payload = apply_type_field_map(dump_data_for_schema(data), PUBLIC_TO_SCHEMA_TYPE_FIELD_MAP)
    return InverseDesignResultOutput.model_validate(payload)


def to_data(data: Any) -> Any:
    """Convert schema inverse-design result data to the public Tidy3D model."""

    from tidy3d.plugins.invdes.result import InverseDesignResult

    payload = apply_type_field_map(dump_data_for_public(data), SCHEMA_TO_PUBLIC_TYPE_FIELD_MAP)
    return validate_public_data(InverseDesignResult, payload)
