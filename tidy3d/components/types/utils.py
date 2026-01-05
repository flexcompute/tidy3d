"""Utilities for type & schema creation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler


def _add_schema(arbitrary_type: type, title: str, field_type_str: str) -> None:
    """Adds a schema to the ``arbitrary_type`` class without subclassing."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls: type, _source_type: type, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def _serialize(value: Any, info: core_schema.SerializationInfo) -> Any:
            from tidy3d.components.autograd.utils import get_static
            from tidy3d.components.types.base import _auto_serializer

            return _auto_serializer(get_static(value), info)

        return core_schema.any_schema(
            metadata={"title": title, "type": field_type_str},
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize, info_arg=True
            ),
        )

    arbitrary_type.__get_pydantic_core_schema__ = __get_pydantic_core_schema__
