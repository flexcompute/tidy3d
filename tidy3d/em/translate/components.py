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


def _key(cls: type) -> str:
    return f"{cls.__module__}:{cls.__name__}"


def _translating_class(cls: type) -> type | None:
    """Return the class whose registration translates ``cls``; ``None`` opts out.

    A Tidy3D class translates only through its own registration, so a family member
    that misses one fails instead of opting out. A subclass defined outside Tidy3D
    translates through its nearest ancestor with a schema.
    """
    if COMPONENT_SCOPES.get(_key(cls)) is not None:
        return cls
    family = [base for base in cls.__mro__ if _key(base) in COMPONENT_SCOPES]
    if not family:
        return None
    owner = next((base for base in family if COMPONENT_SCOPES[_key(base)] is not None), None)
    if owner is not None and cls.__module__.partition(".")[0] != "tidy3d":
        return owner
    raise NotImplementedError(
        f"{cls.__name__} must register its EM schema in tidy3d.em.translate.map."
    )


def from_component(model: BaseModel) -> object | None:
    """Translate registered models; unrelated models opt out, missing family members fail."""
    owner = _translating_class(type(model))
    if owner is None:
        return None
    from flexcompute.core._migration.em.schema.tidy3d.log import log as copied_log

    data = model.model_dump()
    if owner is not type(model):
        from tidy3d.components.types import TYPE_TAG_STR

        # A subclass validates as its ancestor, without the fields it adds; the schema
        # fills in the ancestor's type tag.
        data = {
            name: data[name] for name in owner.model_fields.keys() - {TYPE_TAG_STR} if name in data
        }
    # The public model already logged its warnings; copies would repeat them past td.log.
    with copied_log.suppress_output():
        return resolve(COMPONENT_SCOPES[_key(owner)][0]).model_validate(data)


def validate_component(
    model: BaseModel, schema: object, *, context: Mapping[str, Any]
) -> ValidationReport | None:
    """Run the registered facade; current EME scopes require no external context."""
    owner = _translating_class(type(model))
    if owner is None:
        return None
    return resolve(COMPONENT_SCOPES[_key(owner)][1])(schema)


def validate_on_construction(model: Tidy3dBaseModel) -> Tidy3dBaseModel:
    """Run registered shared semantics; leave unrelated/client-specific validation alone."""
    owner = _translating_class(type(model))  # Missing family registrations fail here.
    if owner is None or _key(owner) not in COMPONENT_CONSTRUCTION_TYPES:
        return model
    return model._run_migrated_model_rules(
        error_types={code: resolve(path) for code, path in COMPONENT_ERROR_TYPES.items()}
    )
