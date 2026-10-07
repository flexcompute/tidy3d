"""Registered public-component translation and EM facade selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tidy3d.em.translate.map import (
    COMPONENT_CONSTRUCTION_TYPES,
    COMPONENT_ERROR_TYPES,
    COMPONENT_SCOPES,
    resolve,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from flexcompute.core.validation import ValidationReport
    from pydantic import BaseModel

    from tidy3d.components.base import Tidy3dBaseModel


def from_component(model: BaseModel) -> object | None:
    """Translate registered models; unrelated models opt out, missing family members fail."""
    cls = type(model)
    registration = COMPONENT_SCOPES.get(f"{cls.__module__}:{cls.__name__}")
    if registration is None:
        if any(f"{base.__module__}:{base.__name__}" in COMPONENT_SCOPES for base in cls.__mro__):
            raise NotImplementedError(
                f"{cls.__name__} must register its EM schema in tidy3d.em.translate.map."
            )
        return None
    from flexcompute.core._migration.em.schema.tidy3d.log import log as copied_log

    # The public model already logged its warnings; copies would repeat them past td.log.
    with copied_log.suppress_output():
        return resolve(registration[0]).model_validate(model.model_dump())


def validate_component(
    model: BaseModel, schema: object, *, context: Mapping[str, Any]
) -> ValidationReport | None:
    """Run the registered facade; current EME scopes require no external context."""
    cls = type(model)
    registration = COMPONENT_SCOPES.get(f"{cls.__module__}:{cls.__name__}")
    if registration is None:
        return None
    return resolve(registration[1])(schema)


def validate_on_construction(model: Tidy3dBaseModel) -> Tidy3dBaseModel:
    """Run registered shared semantics; leave unrelated/client-specific validation alone."""
    cls = type(model)
    key = f"{cls.__module__}:{cls.__name__}"
    if key not in COMPONENT_CONSTRUCTION_TYPES:
        if key not in COMPONENT_SCOPES and any(
            f"{base.__module__}:{base.__name__}" in COMPONENT_SCOPES for base in cls.__mro__
        ):
            from_component(model)  # Missing family registrations must fail, not silently opt out.
        return model
    return model._run_migrated_model_rules(
        error_types={code: resolve(path) for code, path in COMPONENT_ERROR_TYPES.items()}
    )
