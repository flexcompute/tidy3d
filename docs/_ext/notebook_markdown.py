"""Normalize notebook content that Pandoc cannot represent safely in RST."""

from __future__ import annotations

import json
import re
from typing import Any

_CODE_LINK_RE = re.compile(r"\[`(?P<label>[^`\]\n]+)`\]\((?P<target>[^)\n]+)\)")
_BACKTICK_RUN_RE = re.compile(r"`+")
_FENCE_RE = re.compile(
    r"^(?:(?: {0,3}>[ \t]?)|(?: {0,3}(?:[-+*]|\d{1,9}[.)])[ \t]))*"
    r" {0,3}(?P<fence>`{3,}|~{3,})"
)
_INDENTED_CODE_RE = re.compile(
    r"^(?:(?: {0,3}>[ \t]?)|(?: {0,3}(?:[-+*]|\d{1,9}[.)])[ \t]))*"
    r"(?: {4}|\t)"
)
_INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)(?:\\.|[^$\n])+?(?<!\\)\$")
_STRONG_RE = re.compile(r"\*\*(?P<body>(?:(?!\*\*).)+)\*\*")
_ORPHAN_CLOSING_HTML_RE = re.compile(r"(?:\s*</[A-Za-z][A-Za-z0-9_.:-]*\s*>)+\s*\Z")


def _bold_text(text: str) -> str:
    """Wrap the non-whitespace portion of *text* in strong emphasis."""
    if not text.strip():
        return text
    leading_length = len(text) - len(text.lstrip())
    trailing_length = len(text) - len(text.rstrip())
    leading = text[:leading_length]
    trailing = text[len(text) - trailing_length :] if trailing_length else ""
    end = len(text) - trailing_length if trailing_length else len(text)
    return f"{leading}**{text[leading_length:end]}**{trailing}"


def _separate_math_from_strong(match: re.Match[str]) -> str:
    """Keep inline math adjacent to, rather than nested in, strong emphasis."""
    body = match.group("body")
    math_spans = list(_INLINE_MATH_RE.finditer(body))
    if not math_spans:
        return match.group(0)

    normalized: list[str] = []
    cursor = 0
    for math_span in math_spans:
        normalized.append(_bold_text(body[cursor : math_span.start()]))
        normalized.append(math_span.group(0))
        cursor = math_span.end()
    normalized.append(_bold_text(body[cursor:]))
    return "".join(normalized)


def _normalize_markdown_line(line: str) -> str:
    line = _CODE_LINK_RE.sub(r"[\g<label>](\g<target>)", line)
    return _STRONG_RE.sub(_separate_math_from_strong, line)


def _block_code_ranges(markdown: str) -> list[tuple[int, int]]:
    """Return fenced and indented code ranges that must remain byte-for-byte intact."""
    ranges: list[tuple[int, int]] = []
    fence_character = ""
    fence_length = 0
    fence_start = 0
    offset = 0

    for line in markdown.splitlines(keepends=True):
        fence_match = _FENCE_RE.match(line)
        if fence_character:
            if fence_match:
                fence = fence_match.group("fence")
                suffix = line[fence_match.end() :].strip()
                if fence[0] == fence_character and len(fence) >= fence_length and not suffix:
                    ranges.append((fence_start, offset + len(line)))
                    fence_character = ""
                    fence_length = 0
        elif fence_match:
            fence = fence_match.group("fence")
            fence_character = fence[0]
            fence_length = len(fence)
            fence_start = offset
        elif _INDENTED_CODE_RE.match(line):
            ranges.append((offset, offset + len(line)))

        offset += len(line)

    if fence_character:
        ranges.append((fence_start, len(markdown)))

    return ranges


def _inline_code_ranges(markdown: str, start: int, end: int) -> list[tuple[int, int]]:
    """Return CommonMark-style backtick code spans within one prose range."""
    ranges: list[tuple[int, int]] = []
    cursor = start
    while opener := _BACKTICK_RUN_RE.search(markdown, cursor, end):
        opener_length = len(opener.group(0))
        candidate_cursor = opener.end()
        closer = None
        while candidate := _BACKTICK_RUN_RE.search(markdown, candidate_cursor, end):
            if len(candidate.group(0)) == opener_length:
                closer = candidate
                break
            candidate_cursor = candidate.end()

        if closer is None:
            cursor = opener.end()
            continue

        ranges.append((opener.start(), closer.end()))
        cursor = closer.end()

    return ranges


def _prose_ranges(length: int, protected: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return ranges outside sorted, non-overlapping protected Markdown regions."""
    prose: list[tuple[int, int]] = []
    cursor = 0
    for start, end in protected:
        if cursor < start:
            prose.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < length:
        prose.append((cursor, length))
    return prose


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and combine overlapping protected ranges."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def normalize_markdown_for_nbsphinx(markdown: str) -> str:
    """Remove renderer-incompatible nesting without changing Markdown code contexts."""
    block_ranges = _merge_ranges(_block_code_ranges(markdown))
    inline_ranges: list[tuple[int, int]] = []
    for start, end in _prose_ranges(len(markdown), block_ranges):
        inline_ranges.extend(_inline_code_ranges(markdown, start, end))

    linked_code_ranges = {
        (match.start("label") - 1, match.end("label") + 1)
        for match in _CODE_LINK_RE.finditer(markdown)
    }
    protected = _merge_ranges(
        [
            *block_ranges,
            *(code_range for code_range in inline_ranges if code_range not in linked_code_ranges),
        ]
    )

    normalized: list[str] = []
    cursor = 0
    for start, end in protected:
        normalized.append(_normalize_markdown_line(markdown[cursor:start]))
        normalized.append(markdown[start:end])
        cursor = end
    normalized.append(_normalize_markdown_line(markdown[cursor:]))

    return "".join(normalized)


def _source_text(source: Any) -> str | None:
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        return "".join(source)
    return None


def _restore_source_type(original: Any, normalized: str) -> str | list[str]:
    if isinstance(original, list):
        return normalized.splitlines(keepends=True)
    return normalized


def _sanitize_code_cell_outputs(cell: dict[str, Any]) -> bool:
    """Drop raw HTML outputs made entirely of orphan closing tags."""
    outputs = cell.get("outputs")
    if not isinstance(outputs, list):
        return False

    changed = False
    sanitized_outputs: list[Any] = []
    for output in outputs:
        if not isinstance(output, dict):
            sanitized_outputs.append(output)
            continue

        data = output.get("data")
        if not isinstance(data, dict):
            sanitized_outputs.append(output)
            continue

        raw_html = _source_text(data.get("text/html"))
        if raw_html is None or _ORPHAN_CLOSING_HTML_RE.fullmatch(raw_html) is None:
            sanitized_outputs.append(output)
            continue

        del data["text/html"]
        changed = True
        if data:
            sanitized_outputs.append(output)

    if changed:
        cell["outputs"] = sanitized_outputs
    return changed


def normalize_notebook_source(app, docname: str, source: list[str]) -> None:
    """Normalize an IPYNB before nbsphinx parses its Markdown and outputs."""
    del app, docname
    if not source:
        return

    try:
        notebook = json.loads(source[0])
    except (json.JSONDecodeError, TypeError):
        return

    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        return

    changed = False
    for cell in notebook["cells"]:
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") == "code":
            changed = _sanitize_code_cell_outputs(cell) or changed
            continue
        if cell.get("cell_type") != "markdown":
            continue
        original = cell.get("source", "")
        markdown = _source_text(original)
        if markdown is None:
            continue
        normalized = normalize_markdown_for_nbsphinx(markdown)
        if normalized != markdown:
            cell["source"] = _restore_source_type(original, normalized)
            changed = True

    if changed:
        source[0] = json.dumps(notebook, ensure_ascii=False)


def setup(app):
    app.connect("source-read", normalize_notebook_source)
    return {"parallel_read_safe": True}
