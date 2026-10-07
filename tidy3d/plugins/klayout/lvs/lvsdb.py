"""Syntax of KLayout LVS database (``.lvsdb``) files.

The reader turns a results file into records. It knows which keywords may appear in each part of
the file, in both of KLayout's keyword styles, but not what the records mean;
:mod:`tidy3d.plugins.klayout.lvs.results` builds the results from them.
"""

from __future__ import annotations

import gzip
import re
from collections import Counter
from typing import TYPE_CHECKING, Any, NamedTuple

from tidy3d.exceptions import FileError, format_chained_exception_message

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

GZIP_SIGNATURE = b"\x1f\x8b"


class Word(str):
    """A bare word, as opposed to a quoted string."""

    __slots__ = ()


class Group(tuple):
    """A parenthesized group that isn't a record, such as a point ``(dx dy)`` or ``()``."""

    __slots__ = ()


class _Star:
    """The ``*`` placeholder, which repeats a coordinate in the absolute point form."""

    def __repr__(self) -> str:
        return "*"


STAR = _Star()


class Record(NamedTuple):
    """A keyword with its parenthesized contents."""

    keyword: str
    """The keyword in its long form, whichever style the file uses."""
    items: tuple[Any, ...]
    """Atoms (words, strings, numbers, groups, ``STAR``) and child records, in file order."""
    offset: int
    """Byte offset of the keyword in the file, for error messages."""


# For each part of the file: the records it may contain, as (long keyword, short keyword, part
# of the file holding the record's contents). The same short keyword means different things in
# different parts, such as J for the layout section and J for a text label inside a net.
_RECORDS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "top": (
        ("version", "V", "value"),
        ("description", "B", "value"),
        ("layout", "J", "netlist"),
        ("reference", "H", "netlist"),
        ("xref", "Z", "xref"),
    ),
    "netlist": (
        ("version", "V", "value"),
        ("description", "B", "value"),
        ("top", "W", "value"),
        ("unit", "U", "value"),
        ("layer", "L", "value"),
        ("connect", "C", "value"),
        ("softconnect", "CS", "value"),
        ("global", "G", "value"),
        ("softglobal", "GS", "value"),
        ("message", "H", "message"),
        ("class", "K", "class"),
        ("device", "D", "abstract"),
        ("circuit", "X", "circuit"),
    ),
    "class": (("param", "E", "value"), ("terminal", "T", "value")),
    "abstract": (("terminal", "T", "terminal"),),
    "terminal": (("rect", "R", "shape"), ("polygon", "Q", "shape")),
    "circuit": (
        ("rect", "R", "value"),
        ("polygon", "Q", "value"),
        ("property", "F", "property"),
        ("net", "N", "net"),
        ("pin", "P", "pin"),
        ("device", "D", "device"),
        ("circuit", "X", "subcircuit"),
    ),
    "net": (
        ("name", "I", "value"),
        ("property", "F", "property"),
        ("rect", "R", "shape"),
        ("polygon", "Q", "shape"),
        ("text", "J", "shape"),
    ),
    "pin": (("name", "I", "value"), ("property", "F", "property")),
    "device": (
        ("name", "I", "value"),
        ("location", "Y", "value"),
        ("rotation", "O", "value"),
        ("scale", "S", "value"),
        ("param", "E", "value"),
        ("terminal", "T", "value"),
        ("device", "D", "device_part"),
        ("connect", "C", "value"),
        ("property", "F", "property"),
    ),
    "device_part": (
        ("location", "Y", "value"),
        ("rotation", "O", "value"),
        ("scale", "S", "value"),
    ),
    "subcircuit": (
        ("name", "I", "value"),
        ("location", "Y", "value"),
        ("rotation", "O", "value"),
        ("scale", "S", "value"),
        ("property", "F", "property"),
        ("pin", "P", "value"),
    ),
    "message": (
        ("description", "B", "value"),
        ("cell", "C", "value"),
        ("net", "N", "value"),
        ("cat", "X", "value"),
        ("polygon", "Q", "value"),
    ),
    "xref": (("log", "L", "log"), ("circuit", "X", "xref_circuit")),
    "xref_circuit": (
        ("description", "B", "value"),
        ("log", "L", "log"),
        ("xref", "Z", "pairs"),
    ),
    "pairs": (
        ("net", "N", "pair"),
        ("pin", "P", "pair"),
        ("device", "D", "pair"),
        ("circuit", "X", "pair"),
    ),
    "pair": (("description", "B", "value"),),
    "log": (("entry", "M", "message"),),
    "value": (),
    "shape": (),
    # Inside a skipped record, properties still need their own parser, because '#' starts a value.
    "skipped": (("property", "F", "property"),),
}

_TABLES: dict[str, dict[bytes, tuple[str, str]]] = {
    part: {key.encode(): (long, child) for long, short, child in entries for key in (long, short)}
    for part, entries in _RECORDS.items()
}

_TOKEN_RE = re.compile(
    rb"""[\s,]*(?:
        (?P<word>[A-Za-z0-9_.$+\-]+)
      | (?P<open>\()
      | (?P<close>\))
      | (?P<string>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
      | (?P<star>\*)
      | (?P<comment>\#[^\n]*)
      | (?P<end>\Z)
    )""",
    re.VERBOSE | re.DOTALL,
)

_VALUE_RE = re.compile(
    rb"""[\s,]*(?:
        (?P<double>\#\#[^\s,()\[\]{}]+)
      | (?P<integer>\#[lu]*[-+]?[0-9]+)
      | (?P<string>(?:'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")[bc]?)
      | (?P<list>\()
      | (?P<dict>\{)
      | (?P<object>\[)
      | (?P<arrow>=>)
      | (?P<close>[)}\]])
      | (?P<word>[A-Za-z0-9_.$+\-]+)
      | (?P<end>\Z)
    )""",
    re.VERBOSE | re.DOTALL,
)

_SEPARATORS_RE = re.compile(rb"[\s,]*")
_STRING_RE = re.compile(rb"'(?:[^'\\]|\\.)*'|" + rb'"(?:[^"\\]|\\.)*"', re.DOTALL)
_SKIP_RE = re.compile(rb"[()'\"#]")
_OBJECT_RE = re.compile(rb"""[\[\]'"]""")
_ESCAPE_RE = re.compile(rb"\\(?:([0-7]{1,3})|(.))", re.DOTALL)
_CONTROL_ESCAPES = {b"n": b"\n", b"r": b"\r", b"t": b"\t"}
_NUMBER_START = b"0123456789+-."
_WORD_VALUES = {b"nil": None, b"true": True, b"false": False}


def read_file(path: Path) -> bytes:
    """Return the contents of an LVS database file, decompressed if it is gzip-compressed."""
    try:
        data = path.read_bytes()
    except FileNotFoundError as error:
        raise FileError(
            format_chained_exception_message(f"LVS database file not found: '{path}'", error)
        ) from error
    except OSError as error:
        raise FileError(
            format_chained_exception_message(f"Could not read LVS database file '{path}'", error)
        ) from error
    if data.startswith(GZIP_SIGNATURE):
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError) as error:
            raise FileError(
                format_chained_exception_message(
                    f"Could not decompress LVS database file '{path}'", error
                )
            ) from error
    return data


def unquote(token: bytes) -> bytes:
    """Return the bytes a quoted string stands for, with its escapes resolved."""
    body = token[1:-1]
    if b"\\" in body:
        body = _ESCAPE_RE.sub(_replace_escape, body)
    return body


def _replace_escape(match: re.Match[bytes]) -> bytes:
    octal, char = match.groups()
    if octal is not None:
        return bytes((int(octal, 8) & 0xFF,))
    return _CONTROL_ESCAPES.get(char, char)


def _text(token: bytes) -> str:
    return unquote(token).decode("utf-8", errors="replace")


def _atom(token: bytes) -> int | float | Word:
    if token[:1] in _NUMBER_START:
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                pass
    return Word(token.decode("ascii"))


def hashable_key(value: Any) -> Any:
    """Return ``value`` in a form Python can use as a dictionary key.

    KLayout allows any property value as a key. Lists become tuples, and dictionaries become
    tuples of key-value pairs.
    """
    if isinstance(value, list):
        return tuple(hashable_key(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, hashable_key(item)) for key, item in value.items())
    return value


def decode_points(items: Sequence[Any], previous: list[int]) -> list[tuple[int, int]]:
    """Decode the points among ``items``, in database units.

    A point is either a group ``(dx dy)``, which is an offset from the previous point, or two
    numbers ``x y``, which are absolute and where ``STAR`` repeats the previous value.
    ``previous`` holds the previous point as ``[x, y]`` and is updated in place.

    Raises
    ------
    ValueError
        If an item isn't part of a point.
    """
    points = []
    pending = []
    for item in items:
        if isinstance(item, Group):
            if len(item) != 2 or not all(type(value) is int for value in item):
                raise ValueError(f"invalid point {item!r}")
            previous[0] += item[0]
            previous[1] += item[1]
            points.append((previous[0], previous[1]))
        elif item is STAR or type(item) is int:
            pending.append(item)
            if len(pending) == 2:
                if pending[0] is not STAR:
                    previous[0] = pending[0]
                if pending[1] is not STAR:
                    previous[1] = pending[1]
                points.append((previous[0], previous[1]))
                pending.clear()
        else:
            raise ValueError(f"unexpected {item!r} among points")
    if pending:
        raise ValueError("a point is missing its y coordinate")
    return points


def parse_polygon_text(text: str) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Parse KLayout's polygon notation ``(x,y;x,y;.../x,y;...)``: the hull, then any holes.

    Raises
    ------
    ValueError
        If the text isn't in that notation.
    """
    text = text.strip()
    if not (text.startswith("(") and text.endswith(")")):
        raise ValueError(f"invalid polygon {text!r}")
    contours = []
    for contour in text[1:-1].split("/"):
        points = []
        for point in contour.split(";"):
            x, y = point.split(",")
            points.append((float(x), float(y)))
        contours.append(tuple(points))
    return tuple(contours)


class Reader:
    """Reads the records of an LVS database.

    Parameters
    ----------
    data : bytes
        The file's contents, already decompressed.
    source : str
        The file's path, for error messages.
    skip_shapes : bool
        Whether to skip the shapes of nets and device-abstract terminals without reading them.
    """

    def __init__(self, data: bytes, source: str, skip_shapes: bool = False) -> None:
        self.data = data
        self.source = source
        self.skip_shapes = skip_shapes
        self.pos = 0
        self.skipped: Counter[str] = Counter()
        """Keywords of the records that were skipped as unknown, with their counts."""

    def error(self, reason: str, offset: int) -> FileError:
        """Return a ``FileError`` that locates ``offset`` in the file."""
        line = self.data.count(b"\n", 0, offset) + 1
        column = offset - self.data.rfind(b"\n", 0, offset)
        return FileError(
            f"Could not read LVS database '{self.source}': {reason} (line {line}, column {column})."
        )

    def sections(self) -> Iterator[Record]:
        """Yield the file's top-level records one at a time."""
        table = _TABLES["top"]
        while True:
            match = self._token(_TOKEN_RE)
            kind = match.lastgroup
            if kind == "end":
                return
            if kind == "comment":
                continue
            if kind == "word" and self.data.startswith(b"(", self.pos):
                record = self._record(match, table, "top")
                if record is not None:
                    yield record
                continue
            raise self.error("expected a record", match.start(kind))

    def _token(self, pattern: re.Pattern[bytes]) -> re.Match[bytes]:
        match = pattern.match(self.data, self.pos)
        if match is None:
            offset = _SEPARATORS_RE.match(self.data, self.pos).end()
            raise self.error(f"unexpected character {self.data[offset : offset + 1]!r}", offset)
        self.pos = match.end()
        return match

    def _record(
        self, match: re.Match[bytes], table: dict[bytes, tuple[str, str]], part: str
    ) -> Record | None:
        """Read the record whose keyword ``match`` holds; the cursor is at its '('."""
        word = match.group("word")
        offset = match.start("word")
        self.pos += 1
        entry = table.get(word)
        if entry is None:
            if part != "skipped":
                self.skipped[word.decode("ascii")] += 1
            self._items("skipped")
            return None
        keyword, child = entry
        if child == "property":
            return Record(keyword, self._values(offset), offset)
        if child == "shape" and self.skip_shapes:
            self._skip_group(offset)
            return None
        return Record(keyword, self._items(child), offset)

    def _items(self, part: str) -> tuple[Any, ...]:
        """Read atoms and records up to and including the closing ')'."""
        table = _TABLES[part]
        items = []
        while True:
            match = self._token(_TOKEN_RE)
            kind = match.lastgroup
            if kind == "word":
                if self.data.startswith(b"(", self.pos):
                    record = self._record(match, table, part)
                    if record is not None:
                        items.append(record)
                else:
                    items.append(_atom(match.group("word")))
            elif kind == "close":
                return tuple(items)
            elif kind == "string":
                items.append(_text(match.group("string")))
            elif kind == "open":
                items.append(Group(self._items("value")))
            elif kind == "star":
                items.append(STAR)
            elif kind == "end":
                raise self.error("the file ends inside a record", match.start(match.lastgroup))

    def _skip_group(self, offset: int) -> None:
        """Move past the ')' that closes the current group, without reading its contents."""
        depth = 1
        pos = self.pos
        while depth:
            match = _SKIP_RE.search(self.data, pos)
            if match is None:
                raise self.error("the file ends inside a record", offset)
            char = match.group()
            if char == b"(":
                depth += 1
                pos = match.end()
            elif char == b")":
                depth -= 1
                pos = match.end()
            elif char == b"#":
                # Shapes hold no properties, so '#' can only start a comment here.
                newline = self.data.find(b"\n", match.end())
                pos = len(self.data) if newline < 0 else newline
            else:
                string = _STRING_RE.match(self.data, match.start())
                if string is None:
                    raise self.error("unterminated string", match.start())
                pos = string.end()
        self.pos = pos

    def _values(self, offset: int) -> tuple[Any, ...]:
        """Read typed values up to and including the closing ')' of a property record."""
        values = []
        while True:
            match = self._token(_VALUE_RE)
            if match.lastgroup == "close":
                if match.group("close") != b")":
                    raise self.error(
                        "unbalanced brackets in a property", match.start(match.lastgroup)
                    )
                return tuple(values)
            values.append(self._value(match, offset))

    def _value(self, match: re.Match[bytes], offset: int) -> Any:
        kind = match.lastgroup
        text = match.group(kind)
        if kind == "integer":
            return int(text.lstrip(b"#lu"))
        if kind == "double":
            try:
                return float(text[2:])
            except ValueError as error:
                raise self.error(
                    format_chained_exception_message(f"invalid number {text!r}", error),
                    match.start(kind),
                ) from error
        if kind == "string":
            # A 'b' suffix marks a byte string, and a 'c' suffix a single character.
            suffix = text[-1:]
            if suffix in (b"b", b"c"):
                text = text[:-1]
            value = unquote(text)
            return value if suffix == b"b" else value.decode("utf-8", "replace")
        if kind == "list":
            return self._list(offset)
        if kind == "dict":
            return self._dict(offset)
        if kind == "object":
            return self._object(match.start(kind))
        if kind == "word":
            if text in _WORD_VALUES:
                return _WORD_VALUES[text]
            value = _atom(text)
            return str(value) if isinstance(value, Word) else value
        if kind == "end":
            raise self.error("the file ends inside a property", offset)
        raise self.error(f"unexpected {text!r} in a property", match.start(match.lastgroup))

    def _list(self, offset: int) -> list[Any]:
        values = []
        while True:
            match = self._token(_VALUE_RE)
            if match.lastgroup == "close":
                if match.group("close") != b")":
                    raise self.error(
                        "unbalanced brackets in a property", match.start(match.lastgroup)
                    )
                return values
            values.append(self._value(match, offset))

    def _dict(self, offset: int) -> dict[Any, Any]:
        values = {}
        while True:
            match = self._token(_VALUE_RE)
            if match.lastgroup == "close":
                if match.group("close") != b"}":
                    raise self.error(
                        "unbalanced brackets in a property", match.start(match.lastgroup)
                    )
                return values
            key = self._value(match, offset)
            arrow = self._token(_VALUE_RE)
            if arrow.lastgroup != "arrow":
                raise self.error("expected '=>' in a property", arrow.start(arrow.lastgroup))
            values[hashable_key(key)] = self._value(self._token(_VALUE_RE), offset)

    def _object(self, start: int) -> str:
        """Return an object reference such as ``[box:(0,0;1,1)]`` as its text."""
        depth = 1
        pos = start + 1
        while depth:
            match = _OBJECT_RE.search(self.data, pos)
            if match is None:
                raise self.error("the file ends inside a property", start)
            char = match.group()
            if char == b"[":
                depth += 1
                pos = match.end()
            elif char == b"]":
                depth -= 1
                pos = match.end()
            else:
                string = _STRING_RE.match(self.data, match.start())
                if string is None:
                    raise self.error("unterminated string", match.start())
                pos = string.end()
        self.pos = pos
        return self.data[start:pos].decode("utf-8", errors="replace")
