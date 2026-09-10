"""Inverse-design Adam optimizer translation."""

from __future__ import annotations

from typing import Any

from flexcompute.core._migration.em.schema.tidy3d.plugins.invdes.optimizer import (
    AdamOptimizer as AdamOptimizerTaskInput,
)

from tidy3d.em.translate.base import dump_for_public, dump_for_schema


def from_task(optimizer: Any) -> AdamOptimizerTaskInput:
    """Convert a public Tidy3D Adam optimizer to the schema model."""

    return AdamOptimizerTaskInput.model_validate(
        dump_for_schema(optimizer, type_name="AdamOptimizer")
    )


def to_task(optimizer: Any) -> Any:
    """Convert a schema Adam optimizer to the public Tidy3D model."""

    from tidy3d.plugins.invdes.optimizer import AdamOptimizer

    return AdamOptimizer.model_validate(dump_for_public(optimizer, type_name="AdamOptimizer"))
