from __future__ import annotations

from collections.abc import Iterable
from typing import Any, get_args, get_origin

import tomlkit
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from tomlkit.items import Item, Table

from .registry import get_sections

Path = tuple[str, ...]


def collect_descriptions() -> dict[Path, str]:
    """Collect description strings for registered configuration fields."""

    descriptions: dict[Path, str] = {}
    for section_name, model in get_sections().items():
        base_path = tuple(segment for segment in section_name.split(".") if segment)
        section_doc = (model.__doc__ or "").strip()
        if section_doc and base_path:
            descriptions[base_path] = descriptions.get(
                base_path, section_doc.splitlines()[0].strip()
            )
        for field_name, field in model.model_fields.items():
            descriptions.update(_describe_field(field, prefix=(*base_path, field_name)))
    return descriptions


def _describe_field(field: FieldInfo, prefix: Path) -> dict[Path, str]:
    descriptions: dict[Path, str] = {}
    description = (field.description or "").strip()
    if description:
        descriptions[prefix] = description

    nested_models: Iterable[type[BaseModel]] = _iter_model_types(field.annotation)
    for model in nested_models:
        nested_doc = (model.__doc__ or "").strip()
        if nested_doc:
            descriptions[prefix] = descriptions.get(prefix, nested_doc.splitlines()[0].strip())
        for sub_name, sub_field in model.model_fields.items():
            descriptions.update(_describe_field(sub_field, prefix=(*prefix, sub_name)))
    return descriptions


def _iter_model_types(annotation: Any) -> Iterable[type[BaseModel]]:
    """Yield BaseModel subclasses referenced by a field annotation (if any)."""

    if annotation is None:
        return

    stack = [annotation]
    seen: set[type[BaseModel]] = set()

    while stack:
        current = stack.pop()
        if isinstance(current, type) and issubclass(current, BaseModel):
            if current not in seen:
                seen.add(current)
                yield current
            continue

        origin = get_origin(current)
        if origin is None:
            continue

        stack.extend(get_args(current))


def build_document(
    data: dict[str, Any],
    existing: tomlkit.TOMLDocument | None,
    descriptions: dict[Path, str] | None = None,
) -> tomlkit.TOMLDocument:
    """Return a TOML document populated with data and annotated comments."""

    descriptions = descriptions or collect_descriptions()
    document = existing if existing is not None else tomlkit.document()
    _prune_missing_keys(document, data.keys())
    for key, value in data.items():
        _apply_value(
            container=document,
            key=key,
            value=value,
            path=(key,),
            descriptions=descriptions,
            is_new=key not in document,
        )
    return document


def _prune_missing_keys(container: Table | tomlkit.TOMLDocument, keys: Iterable[str]) -> None:
    desired = set(keys)
    for existing_key in list(container.keys()):
        if existing_key not in desired:
            del container[existing_key]


def _apply_value(
    container: Table | tomlkit.TOMLDocument,
    key: str,
    value: Any,
    path: Path,
    descriptions: dict[Path, str],
    is_new: bool,
) -> None:
    description = descriptions.get(path)
    if isinstance(value, dict):
        existing = container.get(key)
        table = existing if isinstance(existing, Table) else tomlkit.table()
        _prune_missing_keys(table, value.keys())
        for sub_key, sub_value in value.items():
            _apply_value(
                container=table,
                key=sub_key,
                value=sub_value,
                path=(*path, sub_key),
                descriptions=descriptions,
                is_new=not isinstance(existing, Table) or sub_key not in table,
            )
        if key in container:
            container[key] = table
        else:
            if isinstance(container, tomlkit.TOMLDocument) and len(container) > 0:
                container.add(tomlkit.nl())
            container.add(key, table)
        return

    if value is None:
        return

    existing_item = container.get(key)
    new_item = tomlkit.item(value)
    if isinstance(existing_item, Item):
        new_item.trivia.comment = existing_item.trivia.comment
        new_item.trivia.comment_ws = existing_item.trivia.comment_ws
    elif description:
        new_item.comment(description)

    if key in container:
        container[key] = new_item
    else:
        container.add(key, new_item)
