"""Utilities for type & schema creation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Annotated, Any, get_args

from pydantic import TypeAdapter
from pydantic.json_schema import WithJsonSchema
from pydantic_core import core_schema

if TYPE_CHECKING:
    from typing import Literal

    from pydantic import GetCoreSchemaHandler


def _add_schema(arbitrary_type: type, title: str, field_type_str: str) -> None:
    """Adds a schema to the ``arbitrary_type`` class without subclassing."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls: type, _source_type: type, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def _serialize(value: Any, info: core_schema.SerializationInfo) -> Any:
            get_static = import_module("tidy3d.components.autograd.utils").get_static
            auto_serializer = import_module("tidy3d.components.types.base")._auto_serializer

            return auto_serializer(get_static(value), info)

        return core_schema.any_schema(
            metadata={"title": title, "type": field_type_str},
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize, info_arg=True
            ),
        )

    arbitrary_type.__get_pydantic_core_schema__ = __get_pydantic_core_schema__


def _tuple_item_schema(
    type_alias: Any,
    mode: Literal["validation", "serialization"] = "validation",
) -> dict[str, Any]:
    """Return explicit item-level JSON schema for a tuple alias."""
    args = get_args(type_alias)
    if not args:
        raise TypeError(f"Expected tuple type, got {type_alias!r}")

    adapter_config = {"arbitrary_types_allowed": True}
    if args[-1] is Ellipsis:
        item_schema = TypeAdapter(args[0], config=adapter_config).json_schema(mode=mode)
        return {"type": "array", "items": item_schema}

    items = [TypeAdapter(arg, config=adapter_config).json_schema(mode=mode) for arg in args]
    item_count = len(items)
    return {
        "type": "array",
        "minItems": item_count,
        "maxItems": item_count,
        "prefixItems": items,
    }


def _typed_tuple(type_alias: Any) -> Any:
    """Return a tuple alias with explicit item-level JSON schema."""
    validation_schema = _tuple_item_schema(type_alias)
    serialization_schema = _tuple_item_schema(type_alias, mode="serialization")
    if validation_schema == serialization_schema:
        return Annotated[type_alias, WithJsonSchema(validation_schema)]
    return Annotated[
        type_alias,
        WithJsonSchema(validation_schema, mode="validation"),
        WithJsonSchema(serialization_schema, mode="serialization"),
    ]


def complex_object_json_schema() -> dict[str, Any]:
    """Return JSON schema for serialized complex numbers."""
    return {
        "type": "object",
        "properties": {"real": {"type": "number"}, "imag": {"type": "number"}},
        "required": ["real", "imag"],
        "additionalProperties": False,
    }


def complex_json_schema(
    mode: Literal["validation", "serialization"] = "validation",
) -> dict[str, Any]:
    """Return JSON schema for complex values by schema mode."""
    if mode == "serialization":
        return complex_object_json_schema()
    return {
        "anyOf": [
            {"type": "number"},
            _tuple_item_schema(tuple[float, float]),
            complex_object_json_schema(),
        ]
    }
