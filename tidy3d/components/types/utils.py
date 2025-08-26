"""Utilities for type & schema creation."""

from __future__ import annotations


def _add_schema(arbitrary_type: type, title: str, field_type_str: str) -> None:
    """Adds a schema to the ``arbitrary_type`` class without subclassing."""

    @classmethod
    def mod_schema_fn(cls, field_schema: dict) -> None:
        """Function that gets set to ``arbitrary_type.__modify_schema__``."""
        field_schema.update({"title": title, "type": field_type_str})

    arbitrary_type.__modify_schema__ = mod_schema_fn
