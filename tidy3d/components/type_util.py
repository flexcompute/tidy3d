"""Utilities for type & schema creation."""

from __future__ import annotations

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


def _add_schema(arbitrary_type: type, title: str, field_type_str: str) -> None:
    """Adds a schema to the ``arbitrary_type`` class without subclassing."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: type, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # `any_schema()` is enough because we only want to accept the value
        # the metadata will later show up in JSON-schema generation.
        return core_schema.any_schema(metadata={"title": title, "type": field_type_str})

    arbitrary_type.__get_pydantic_core_schema__ = __get_pydantic_core_schema__
