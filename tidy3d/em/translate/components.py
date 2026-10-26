"""Registered public-component translation, validation and presentation."""

from __future__ import annotations

from contextvars import ContextVar
from functools import cache
from operator import is_
from typing import TYPE_CHECKING, Any

from flexcompute.core.validation import prefix_paths
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from pydantic_core import InitErrorDetails

from tidy3d.em.translate.map import COMPONENT_ERROR_TYPES, COMPONENT_SCOPES, resolve
from tidy3d.log import log

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from flexcompute.core.validation import ValidationDiagnostic, ValidationPath

    from tidy3d.components.base import Tidy3dBaseModel


def _key(cls: type) -> str:
    return f"{cls.__module__}:{cls.__name__}"


def _translating_class(cls: type) -> type | None:
    """Return the class whose registration translates ``cls``; ``None`` opts out.

    A Tidy3D class translates only through its own registration, so a family member
    that misses one fails instead of opting out. A subclass defined outside Tidy3D
    translates through its nearest ancestor with a schema, and opts out when only
    abstract roots without a schema are registered above it.
    """
    if COMPONENT_SCOPES.get(_key(cls)) is not None:
        return cls
    family = [base for base in cls.__mro__ if _key(base) in COMPONENT_SCOPES]
    if not family:
        return None
    owner = next((base for base in family if COMPONENT_SCOPES[_key(base)] is not None), None)
    if cls.__module__.partition(".")[0] != "tidy3d":
        return owner
    raise NotImplementedError(
        f"{cls.__name__} must register its EM schema in tidy3d.em.translate.map."
    )


@cache
def _tidy3d_tag(cls: type) -> str | None:
    """Return the type tag of the nearest Tidy3D class above a class defined outside Tidy3D."""
    if cls.__module__.partition(".")[0] == "tidy3d":
        return None
    from tidy3d.components.types import TYPE_TAG_STR

    return next(
        (
            base.model_fields[TYPE_TAG_STR].default
            for base in cls.__mro__
            if base.__module__.partition(".")[0] == "tidy3d"
            and TYPE_TAG_STR in getattr(base, "model_fields", {})
        ),
        None,
    )


def _retag(value: Any, data: Any) -> None:
    """Give each nested subclass defined outside Tidy3D its Tidy3D ancestor's type tag.

    Pydantic dumps a nested subclass with only the fields of the declared field type, but
    keeps the subclass's own tag, which the schema's literal tags reject.
    """
    if isinstance(value, BaseModel):
        if not isinstance(data, dict):
            return
        tag = _tidy3d_tag(type(value))
        if tag is not None:
            from tidy3d.components.types import TYPE_TAG_STR

            data[TYPE_TAG_STR] = tag
        for name, field in value.__dict__.items():
            if name in data:
                _retag(field, data[name])
    elif isinstance(value, (list, tuple)) and isinstance(data, (list, tuple)):
        for item, item_data in zip(value, data):
            _retag(item, item_data)
    elif isinstance(value, dict) and isinstance(data, dict):
        for key, item in value.items():
            if key in data:
                _retag(item, data[key])


# Error lines the copied schema logs during a translation, per thread and per nested call.
_copied_errors: ContextVar[list[tuple[tuple[Any, ...], dict[str, Any]]] | None] = ContextVar(
    "_copied_errors", default=None
)


def _capture_copied_errors(copied_log: Any) -> None:
    """Route the copied log's ``error`` into ``_copied_errors`` while a translation runs."""
    if "error" in vars(copied_log):
        return
    logged = copied_log.error

    def error(*args: Any, **kwargs: Any) -> None:
        captured = _copied_errors.get()
        if captured is None:
            return logged(*args, **kwargs)
        captured.append((args, kwargs))
        return None

    copied_log.error = error


def from_component(model: BaseModel) -> object | None:
    """Translate registered models; unrelated models opt out, missing family members fail."""
    owner = _translating_class(type(model))
    if owner is None:
        return None
    from flexcompute.core.log import log as copied_log

    from tidy3d.components.autograd.utils import get_static

    data = model.model_dump()
    _retag(model, data)
    # The schema only validates, so a traced value is reduced to its number.
    data = get_static(data)
    if owner is not type(model):
        from tidy3d.components.types import TYPE_TAG_STR

        # A subclass validates as its ancestor, without the fields it adds; the schema
        # fills in the ancestor's type tag.
        data = {
            name: data[name] for name in owner.model_fields.keys() - {TYPE_TAG_STR} if name in data
        }
    # The public model already logged its warnings; copies would repeat them past td.log.
    # The copy's error lines are kept and logged only if the translation fails, because then
    # the public model's own check never ran.
    _capture_copied_errors(copied_log)
    errors: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    token = _copied_errors.set(errors)
    try:
        with copied_log.suppress_output():
            return resolve(COMPONENT_SCOPES[_key(owner)].schema).model_validate(data)
    except PydanticValidationError:
        for args, kwargs in errors:
            log.error(*args, **kwargs)
        raise
    finally:
        _copied_errors.reset(token)


def validate(
    model: BaseModel, *, path: ValidationPath = (), context: object | None = None
) -> tuple[ValidationDiagnostic, ...]:
    """Translate ``model``, run its registered facade and nest the diagnostics under ``path``.

    ``context`` is passed to the facade as ``context=`` only when it is given, so facades
    without a context parameter still work. Unrelated models report nothing. Each call
    translates afresh and keeps no translation.
    """
    owner = _translating_class(type(model))
    if owner is None:
        return ()
    facade = resolve(COMPONENT_SCOPES[_key(owner)].facade)
    schema = from_component(model)
    report = facade(schema) if context is None else facade(schema, context=context)
    return prefix_paths(path, report.diagnostics)


def _value_at(model: BaseModel, path: ValidationPath) -> Any:
    """Return the value at ``path``, or ``None`` when the path does not resolve."""
    value: Any = model
    try:
        for key in path:
            value = value.__dict__[key] if isinstance(value, BaseModel) else value[key]
    except (KeyError, IndexError, TypeError):
        return None
    return value


def _present(model: Tidy3dBaseModel, diagnostics: Iterable[ValidationDiagnostic]) -> None:
    """Log warnings in order, then raise the first error as the public model raised it.

    Only the first error's type is resolved from ``COMPONENT_ERROR_TYPES``. An error with a
    path raises there: a mapped type in Pydantic's ``Value error, `` form with the value at
    the path as its input, otherwise the bare message through the model's location helper.
    An error without a path raises its mapped type, ``ValueError`` by default.
    """
    for diagnostic in diagnostics:
        if diagnostic.severity == "warning":
            log.warning(diagnostic.message, custom_loc=list(diagnostic.path))
            continue
        type_path = COMPONENT_ERROR_TYPES.get(diagnostic.code)
        error_type = resolve(type_path) if type_path else None
        if not diagnostic.path:
            raise (error_type or ValueError)(diagnostic.message)
        if error_type is None:
            model._raise_validation_error_at_loc(diagnostic.message, *diagnostic.path)
        raise PydanticValidationError.from_exception_data(
            type(model).__name__,
            [
                InitErrorDetails(
                    type="value_error",
                    loc=diagnostic.path,
                    input=_value_at(model, diagnostic.path),
                    ctx={"error": error_type(diagnostic.message)},
                )
            ],
            hide_input=True,
        )


# Holds the field values an instance passed its construction checks with.
_VALIDATED_ON_CONSTRUCTION = "_validated_on_construction"


def validate_on_construction(model: Tidy3dBaseModel) -> Tidy3dBaseModel:
    """Run the full facade for registrations flagged ``on_construction``.

    Other models are left alone, but a missing family registration fails here. Pydantic runs
    this hook again when a parent takes the instance as a field. An instance whose field
    values are the ones it passed with is not checked again. A validated copy starts with an
    empty cache. A ``model_copy(update=...)`` shares the cache but holds a new value, so it
    is checked. An instance with a list or dict field other than ``attrs`` is always checked,
    since that field can change in place.
    """
    owner = _translating_class(type(model))
    if owner is None or not COMPONENT_SCOPES[_key(owner)].on_construction:
        return model
    values = tuple(model.__dict__.values())
    passed = model._cached_properties.get(_VALIDATED_ON_CONSTRUCTION)
    if (
        passed is not None
        and all(map(is_, passed, values))
        and not any(
            isinstance(value, (list, dict))
            for name, value in model.__dict__.items()
            if name != "attrs"
        )
    ):
        return model
    _present(model, validate(model))
    model._cached_properties[_VALIDATED_ON_CONSTRUCTION] = values
    return model


def run_model_rules(
    model: Tidy3dBaseModel,
    *rules: Callable[[Any], tuple[ValidationDiagnostic, ...]],
) -> Tidy3dBaseModel:
    """Translate once, run each scope's model-level rules in order, and present them.

    Call it from a public model's after validator with one rule callable per scope, each
    taking the translated schema. Rules run in the order given. The first scope that
    reports an error ends the run. Warnings are logged and the first error is raised as
    ``_present`` raises it.
    """
    schema = from_component(model)
    diagnostics: tuple[ValidationDiagnostic, ...] = ()
    for scope_rules in rules:
        reported = scope_rules(schema)
        diagnostics += reported
        if any(diagnostic.severity == "error" for diagnostic in reported):
            break
    _present(model, diagnostics)
    return model
