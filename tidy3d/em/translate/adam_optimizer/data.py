"""Inverse-design optimizer result-data translation."""

from __future__ import annotations

from typing import Any

from flexcompute.core._migration.em.schema.tidy3d.plugins.invdes.result import (
    InverseDesignResult as InverseDesignResultOutput,
)

from tidy3d.em.translate.base import (
    dump_data_for_public,
    dump_data_for_schema,
    validate_public_data,
)


def from_data(data: Any) -> InverseDesignResultOutput:
    """Convert public inverse-design result data to the schema model."""

    return InverseDesignResultOutput.model_validate(dump_data_for_schema(data))


def to_data(data: Any) -> Any:
    """Convert schema inverse-design result data to the public Tidy3D model."""

    from tidy3d.plugins.invdes.result import InverseDesignResult

    return validate_public_data(InverseDesignResult, dump_data_for_public(data))
