from __future__ import annotations

from tidy3d.exceptions import format_chained_exception_message


def test_format_chained_exception_message_appends_cause():
    exc = ValueError("boom")

    assert format_chained_exception_message("Wrapped error", exc) == (
        "Wrapped error (cause: ValueError: boom)"
    )


def test_format_chained_exception_message_omits_key_error_cause():
    exc = KeyError("Foo")

    assert format_chained_exception_message("Unknown type: Foo", exc) == "Unknown type: Foo"


def test_format_chained_exception_message_strips_trailing_colon():
    exc = RuntimeError("boom")

    assert format_chained_exception_message("Failed to parse:", exc) == (
        "Failed to parse (cause: RuntimeError: boom)"
    )
