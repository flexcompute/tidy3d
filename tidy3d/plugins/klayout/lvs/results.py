"""Loading and summarizing KLayout LVS results."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from pydantic import Field

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.exceptions import format_chained_exception_message
from tidy3d.log import log
from tidy3d.plugins.klayout.lvs import lvsdb
from tidy3d.plugins.klayout.lvs.lvsdb import Group, Record, Word
from tidy3d.plugins.klayout.lvs.netlist import (
    LVSCircuit,
    LVSConnection,
    LVSDevice,
    LVSDeviceAbstract,
    LVSDeviceClass,
    LVSDevicePart,
    LVSLabel,
    LVSLayer,
    LVSNet,
    LVSNetlist,
    LVSParameterDefinition,
    LVSPin,
    LVSPlacement,
    LVSPolygon,
    LVSSubcircuit,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tidy3d.exceptions import FileError

LVSStatus = Literal["match", "warning", "mismatch", "nomatch", "skipped"]
LVSVerdict = Literal["match", "match_with_warnings", "mismatch"]
LVSSeverity = Literal["info", "warning", "error"]

_STATUSES: dict[str, LVSStatus] = {
    "match": "match",
    "1": "match",
    "nomatch": "nomatch",
    "X": "nomatch",
    "mismatch": "mismatch",
    "0": "mismatch",
    "warning": "warning",
    "W": "warning",
    "skipped": "skipped",
    "S": "skipped",
}
_FAILED_STATUSES = ("mismatch", "nomatch", "skipped")
_SEVERITIES: dict[str, LVSSeverity] = {
    "info": "info",
    "I": "info",
    "warning": "warning",
    "W": "warning",
    "error": "error",
    "E": "error",
}
_MIRROR_WORDS = ("mirror", "M")

# KLayout's built-in device classes, measured with KLayout 0.30.12: terminals, then parameters
# as (name, primary, default).
_RESISTOR_PARAMETERS = (
    ("R", True, 0.0),
    ("L", False, 0.0),
    ("W", False, 0.0),
    ("A", False, 0.0),
    ("P", False, 0.0),
)
_CAPACITOR_PARAMETERS = (("C", True, 0.0), ("A", False, 0.0), ("P", False, 0.0))
_MOS_PARAMETERS = (
    ("L", True, 0.0),
    ("W", True, 0.0),
    ("AS", False, 0.0),
    ("AD", False, 0.0),
    ("PS", False, 0.0),
    ("PD", False, 0.0),
)
_BJT_PARAMETERS = (
    ("AE", True, 0.0),
    ("PE", False, 0.0),
    ("AB", False, 0.0),
    ("PB", False, 0.0),
    ("AC", False, 0.0),
    ("PC", False, 0.0),
    ("NE", True, 1.0),
)
_DEVICE_TEMPLATES: dict[str, tuple[tuple[str, ...], tuple[tuple[str, bool, float], ...]]] = {
    "RES": (("A", "B"), _RESISTOR_PARAMETERS),
    "RES3": (("A", "B", "W"), _RESISTOR_PARAMETERS),
    "CAP": (("A", "B"), _CAPACITOR_PARAMETERS),
    "CAP3": (("A", "B", "W"), _CAPACITOR_PARAMETERS),
    "IND": (("A", "B"), (("L", True, 0.0),)),
    "DIODE": (("A", "C"), (("A", False, 0.0), ("P", False, 0.0))),
    "MOS3": (("S", "G", "D"), _MOS_PARAMETERS),
    "MOS4": (("S", "G", "D", "B"), _MOS_PARAMETERS),
    "BJT3": (("C", "B", "E"), _BJT_PARAMETERS),
    "BJT4": (("C", "B", "E", "S"), _BJT_PARAMETERS),
}

_SUPPORTED_VERSION = 1
_DEFAULT_UNIT = 0.001
_SUMMARY_LIMIT = 20


class LVSGeometry(Tidy3dBaseModel):
    """A marker polygon that a message points to."""

    hull: LVSPolygon = Field(title="Hull", description="The outer contour, in µm.")
    holes: tuple[LVSPolygon, ...] = Field(
        default=(), title="Holes", description="Contours cut from the hull, in µm."
    )


class LVSMessage(Tidy3dBaseModel):
    """A message that KLayout logged while extracting the netlist or comparing the netlists."""

    severity: LVSSeverity | None = Field(
        default=None, title="Severity", description="The message's severity, if it has one."
    )
    text: str = Field(default="", title="Text", description="The message text.")
    layout_circuit: str | None = Field(
        default=None,
        title="Layout circuit",
        description="Layout circuit of the circuit pair the message was logged under; ``None`` "
        "for extraction messages and messages that belong to no circuit pair.",
    )
    schematic_circuit: str | None = Field(
        default=None,
        title="Schematic circuit",
        description="Schematic circuit of the circuit pair the message was logged under.",
    )
    cell: str | None = Field(default=None, title="Cell", description="The cell the message names.")
    net: str | None = Field(default=None, title="Net", description="The net the message names.")
    category: str | None = Field(
        default=None, title="Category", description="The message's category."
    )
    category_description: str | None = Field(
        default=None, title="Category description", description="Description of the category."
    )
    geometry: LVSGeometry | None = Field(
        default=None,
        title="Geometry",
        description="A marker polygon, in µm, if the message has one.",
    )


class LVSPair(Tidy3dBaseModel):
    """Two objects that KLayout paired between the extracted netlist and the schematic.

    Either side is ``None`` when KLayout found no partner for the other.
    """

    circuit: str = Field(
        title="Circuit",
        description="The circuit pair this pair belongs to: the layout circuit's name, or the "
        "schematic circuit's name when there is no layout circuit.",
    )
    status: LVSStatus | None = Field(
        default=None,
        title="Status",
        description="KLayout's status for the pair, in the file's words; ``None`` if the file "
        "gives none.",
    )
    description: str | None = Field(
        default=None, title="Description", description="KLayout's text for the pair, if any."
    )


class LVSNetPair(LVSPair):
    """A net of the extracted netlist and the schematic net KLayout paired it with."""

    layout: LVSNet | None = Field(default=None, title="Layout", description="The layout net.")
    schematic: LVSNet | None = Field(
        default=None, title="Schematic", description="The schematic net."
    )


class LVSPinPair(LVSPair):
    """A pin of a layout circuit and the schematic pin KLayout paired it with."""

    layout: LVSPin | None = Field(default=None, title="Layout", description="The layout pin.")
    schematic: LVSPin | None = Field(
        default=None, title="Schematic", description="The schematic pin."
    )


class LVSDevicePair(LVSPair):
    """A layout device and the schematic device KLayout paired it with."""

    layout: LVSDevice | None = Field(default=None, title="Layout", description="The layout device.")
    schematic: LVSDevice | None = Field(
        default=None, title="Schematic", description="The schematic device."
    )


class LVSSubcircuitPair(LVSPair):
    """A layout subcircuit and the schematic subcircuit KLayout paired it with."""

    layout: LVSSubcircuit | None = Field(
        default=None, title="Layout", description="The layout subcircuit."
    )
    schematic: LVSSubcircuit | None = Field(
        default=None, title="Schematic", description="The schematic subcircuit."
    )


class LVSComparison(LVSPair):
    """The comparison of a layout circuit with a schematic circuit, and the pairs within it."""

    layout: LVSCircuit | None = Field(
        default=None, title="Layout", description="The layout circuit."
    )
    schematic: LVSCircuit | None = Field(
        default=None, title="Schematic", description="The schematic circuit."
    )
    messages: tuple[LVSMessage, ...] = Field(
        default=(), title="Messages", description="Messages KLayout logged for this comparison."
    )
    nets: tuple[LVSNetPair, ...] = Field(default=(), title="Nets", description="Net pairs.")
    pins: tuple[LVSPinPair, ...] = Field(default=(), title="Pins", description="Pin pairs.")
    devices: tuple[LVSDevicePair, ...] = Field(
        default=(), title="Devices", description="Device pairs."
    )
    subcircuits: tuple[LVSSubcircuitPair, ...] = Field(
        default=(), title="Subcircuits", description="Subcircuit pairs."
    )


class LVSResults(Tidy3dBaseModel):
    """The results of a KLayout LVS run, as recorded in its results file.

    Notes
    -----
    ``verdict`` summarizes the comparison:

    * ``'mismatch'``: a circuit pair's status isn't ``'match'``, or a comparison message is an
      error.
    * ``'match_with_warnings'``: otherwise, a net or device pair's status is ``'warning'``, or a
      comparison message is a warning.
    * ``'match'``: otherwise.

    KLayout's ``compare`` succeeds exactly when every circuit pair is ``'match'``. Checks that
    a runset calls after ``compare``, such as ``flag_missing_ports``, record failures as error
    messages. Extraction messages don't affect the verdict.

    Example
    -------
    >>> from tidy3d.plugins.klayout.lvs import LVSResults
    >>> results = LVSResults.load("lvs_results.lvsdb")  # doctest: +SKIP
    >>> print(results.verdict)  # doctest: +SKIP
    >>> print(results)  # doctest: +SKIP
    """

    verdict: LVSVerdict = Field(title="Verdict", description="The overall result of the run.")
    comparisons: tuple[LVSComparison, ...] = Field(
        default=(),
        title="Comparisons",
        description="One comparison per circuit pair, in file order.",
    )
    layout: LVSNetlist = Field(
        title="Layout", description="The netlist KLayout extracted from the layout."
    )
    schematic: LVSNetlist = Field(
        title="Schematic", description="The schematic netlist, as KLayout read it."
    )
    messages: tuple[LVSMessage, ...] = Field(
        default=(), title="Messages", description="Every message KLayout logged while comparing."
    )
    extraction_messages: tuple[LVSMessage, ...] = Field(
        default=(),
        title="Extraction messages",
        description="Messages KLayout logged while extracting the layout netlist.",
    )
    unit: float = Field(
        default=_DEFAULT_UNIT, title="Unit", description="The layout's database unit, in µm."
    )
    log: str | None = Field(
        default=None,
        title="Log",
        description="Everything KLayout printed during the run; ``None`` for loaded results.",
    )
    exit_code: int | None = Field(
        default=None,
        title="Exit code",
        description="KLayout's exit code; ``None`` for loaded results.",
    )

    @cached_property
    def top(self) -> LVSComparison | None:
        """The comparison that holds the layout's top circuit, if there is one."""
        if self.layout.top_circuit is None:
            return None
        for comparison in self.comparisons:
            if comparison.layout is not None and comparison.layout.name == self.layout.top_circuit:
                return comparison
        return None

    @cached_property
    def mismatches(self) -> tuple[LVSPair, ...]:
        """Pairs of every kind whose status is ``'mismatch'``, ``'nomatch'`` or ``'skipped'``."""
        return tuple(
            pair
            for comparison in self.comparisons
            for pair in _pairs(comparison)
            if pair.status in _FAILED_STATUSES
        )

    @cached_property
    def warnings(self) -> tuple[LVSPair, ...]:
        """Pairs whose status is ``'warning'``."""
        return tuple(
            pair
            for comparison in self.comparisons
            for pair in _pairs(comparison)
            if pair.status == "warning"
        )

    def comparison(self, name: str) -> LVSComparison:
        """Return the comparison of the circuit named ``name``, on either side.

        Raises
        ------
        KeyError
            If no comparison has a circuit named ``name``.
        """
        for comparison in self.comparisons:
            for circuit in (comparison.layout, comparison.schematic):
                if circuit is not None and circuit.name == name:
                    return comparison
        raise KeyError(f"No comparison has a circuit named '{name}'.")

    def __str__(self) -> str:
        """A summary of the verdict, the circuit pairs, mismatches, warnings and messages."""
        return _Summary(self).text()

    @classmethod
    def load(cls, resultsfile: str | Path, load_shapes: bool = True) -> LVSResults:
        """Load the results that a KLayout LVS run wrote to a ``.lvsdb`` file.

        Parameters
        ----------
        resultsfile : Union[str, Path]
            Path to the results file, which may be gzip-compressed.
        load_shapes : bool
            Whether to load the shapes and labels of nets and device abstracts. Circuit
            boundaries and message geometry are always loaded.

        Returns
        -------
        :class:`.LVSResults`
            The results in the file.

        Raises
        ------
        :class:`.FileError`
            If the file can't be read or parsed, or has no cross-reference section.

        Example
        -------
        >>> from tidy3d.plugins.klayout.lvs import LVSResults
        >>> results = LVSResults.load("lvs_results.lvsdb", load_shapes=False)  # doctest: +SKIP
        """
        return load_results(resultsfile, load_shapes=load_shapes)


def load_results(
    resultsfile: str | Path,
    load_shapes: bool = True,
    output: str | None = None,
    exit_code: int | None = None,
) -> LVSResults:
    """Load a results file, attaching the output and exit code of the run that wrote it."""
    path = Path(resultsfile)
    reader = lvsdb.Reader(lvsdb.read_file(path), str(path), skip_shapes=not load_shapes)
    builder = _Builder(reader)
    for section in reader.sections():
        builder.add(section)
    results = builder.results(output=output, exit_code=exit_code)
    if reader.skipped:
        records = ", ".join(f"'{name}' ({count})" for name, count in sorted(reader.skipped.items()))
        log.warning(
            f"Skipped unknown records in LVS database '{path}': {records}. The file may come "
            "from a newer KLayout version."
        )
    return results


def _pairs(comparison: LVSComparison) -> tuple[LVSPair, ...]:
    return (
        comparison,
        *comparison.nets,
        *comparison.pins,
        *comparison.devices,
        *comparison.subcircuits,
    )


def _verdict(comparisons: Sequence[LVSComparison], messages: Sequence[LVSMessage]) -> LVSVerdict:
    if any(comparison.status != "match" for comparison in comparisons) or any(
        message.severity == "error" for message in messages
    ):
        return "mismatch"
    if any(
        pair.status == "warning"
        for comparison in comparisons
        for pair in (*comparison.nets, *comparison.devices)
    ) or any(message.severity == "warning" for message in messages):
        return "match_with_warnings"
    return "match"


class _Units:
    """Converts database units to µm."""

    def __init__(self, unit: float) -> None:
        self.unit = unit
        # With a unit of 1/k µm, dividing by k gives the float nearest the decimal value,
        # where multiplying by the unit can be off in the last digit.
        inverse = round(1.0 / unit) if unit > 0 else 0
        self.divisor = inverse if inverse > 0 and abs(inverse * unit - 1.0) < 1e-9 else 0

    def value(self, value: float) -> float:
        return value / self.divisor if self.divisor else value * self.unit

    def polygon(self, points: Sequence[tuple[int, int]]) -> LVSPolygon:
        if self.divisor:
            divisor = self.divisor
            return tuple((x / divisor, y / divisor) for x, y in points)
        unit = self.unit
        return tuple((x * unit, y * unit) for x, y in points)


class _RawCircuit(NamedTuple):
    """A circuit whose nets still lack their connections."""

    record: Record
    name: str
    boundary: LVSPolygon | None
    properties: dict[Any, Any]
    nets: list[dict[str, Any]]
    pins: tuple[LVSPin, ...]
    devices: tuple[LVSDevice, ...]
    subcircuits: tuple[LVSSubcircuit, ...]


def _rect_polygon(corners: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """The box with these two corners, as a polygon in KLayout's clockwise order."""
    (x1, y1), (x2, y2) = corners
    left, right = min(x1, x2), max(x1, x2)
    bottom, top = min(y1, y2), max(y1, y2)
    return [(left, bottom), (left, top), (right, top), (right, bottom)]


class _Builder:
    """Builds :class:`.LVSResults` from the records of an LVS database."""

    def __init__(self, reader: lvsdb.Reader) -> None:
        self.reader = reader
        self.netlists: dict[str, LVSNetlist] = {}
        self.circuits: dict[str, dict[str, LVSCircuit]] = {}
        self.unit = _DEFAULT_UNIT
        self.extraction_messages: list[LVSMessage] = []
        self.comparisons: tuple[LVSComparison, ...] | None = None
        self.messages: list[LVSMessage] = []
        self.unknown_statuses: Counter[str] = Counter()
        self.unknown_templates: Counter[str] = Counter()

    def error(self, reason: str, record: Record) -> FileError:
        return self.reader.error(reason, record.offset)

    def add(self, section: Record) -> None:
        keyword = section.keyword
        if keyword == "version":
            self._check_version(section)
        elif keyword in ("layout", "reference"):
            side = "layout" if keyword == "layout" else "schematic"
            if side in self.netlists:
                raise self.error(f"a second '{keyword}' section", section)
            if self.comparisons is not None:
                raise self.error(f"the '{keyword}' section follows the 'xref' section", section)
            self.netlists[side] = self._netlist(section, side)
        elif keyword == "xref":
            if self.comparisons is not None:
                raise self.error("a second 'xref' section", section)
            self.comparisons = self._xref(section)

    def results(self, output: str | None, exit_code: int | None) -> LVSResults:
        if self.comparisons is None:
            raise self.reader.error("the file has no 'xref' (cross-reference) section", 0)
        for status, count in sorted(self.unknown_statuses.items()):
            log.warning(
                f"LVS database '{self.reader.source}' has {count} pair(s) with the unknown "
                f"status '{status}'; their status is read as None."
            )
        for template, count in sorted(self.unknown_templates.items()):
            log.warning(
                f"LVS database '{self.reader.source}' has {count} device class(es) based on the "
                f"unknown template '{template}'; only their own definitions are read."
            )
        messages = tuple(self.messages)
        return LVSResults(
            verdict=_verdict(self.comparisons, messages),
            comparisons=self.comparisons,
            layout=self.netlists.get("layout") or LVSNetlist(),
            schematic=self.netlists.get("schematic") or LVSNetlist(),
            messages=messages,
            extraction_messages=tuple(self.extraction_messages),
            unit=self.unit,
            log=output,
            exit_code=exit_code,
        )

    def _check_version(self, record: Record) -> None:
        version = self._int(record, 0)
        if version > _SUPPORTED_VERSION:
            raise self.error(
                f"format version {version} is newer than the versions this reader supports "
                f"(0 and {_SUPPORTED_VERSION})",
                record,
            )

    def _atom(self, record: Record, index: int) -> Any:
        if index >= len(record.items) or isinstance(record.items[index], Record):
            raise self.error(f"'{record.keyword}' is missing a value", record)
        return record.items[index]

    def _int(self, record: Record, index: int) -> int:
        value = self._atom(record, index)
        if type(value) is not int:
            raise self.error(f"expected an integer in '{record.keyword}', found {value!r}", record)
        return value

    def _float(self, record: Record, index: int) -> float:
        value = self._atom(record, index)
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise self.error(f"expected a number in '{record.keyword}', found {value!r}", record)
        try:
            return float(value)
        except ValueError as error:
            raise self.error(
                format_chained_exception_message(
                    f"expected a number in '{record.keyword}', found {value!r}", error
                ),
                record,
            ) from error

    def _text(self, record: Record, index: int = 0) -> str:
        value = self._atom(record, index)
        if isinstance(value, Group) or value is lvsdb.STAR:
            raise self.error(f"expected a name in '{record.keyword}', found {value!r}", record)
        return str(value)

    def _optional_name(self, record: Record, index: int) -> str | None:
        """A name, or ``None`` for ``()``, the placeholder for a missing object."""
        value = self._atom(record, index)
        if isinstance(value, Group):
            if value:
                raise self.error(f"expected a name or '()' in '{record.keyword}'", record)
            return None
        return str(value)

    def _optional_id(self, record: Record, index: int) -> int | None:
        value = self._atom(record, index)
        if isinstance(value, Group) and not value:
            return None
        if type(value) is not int:
            raise self.error(f"expected an ID or '()' in '{record.keyword}'", record)
        return value

    def _property(self, record: Record) -> tuple[Any, Any]:
        if len(record.items) != 2:
            raise self.error("a property must have a key and a value", record)
        key, value = record.items
        return lvsdb.hashable_key(key), value

    def _points(self, record: Record, items: Sequence[Any], previous: list[int]) -> list:
        try:
            return lvsdb.decode_points(items, previous)
        except ValueError as error:
            raise self.error(f"invalid points in '{record.keyword}': {error}", record) from error

    def _shape(
        self, record: Record, units: _Units, previous: list[int], has_layer: bool = True
    ) -> tuple[str | None, LVSPolygon]:
        """Decode a ``rect`` or ``polygon`` record into its layer and polygon."""
        layer = self._text(record, 0) if has_layer else None
        points = self._points(record, record.items[1 if has_layer else 0 :], previous)
        if record.keyword == "rect":
            if len(points) != 2:
                raise self.error("a 'rect' needs two corners", record)
            points = _rect_polygon(points)
        elif len(points) < 3:
            raise self.error("a 'polygon' needs at least three points", record)
        return layer, units.polygon(points)

    def _placement(self, record: Record, items: Sequence[Any], units: _Units) -> LVSPlacement:
        x = y = 0.0
        rotation = 0.0
        scale = 1.0
        mirror = False
        for item in items:
            if isinstance(item, Record):
                if item.keyword == "location":
                    x = units.value(self._int(item, 0))
                    y = units.value(self._int(item, 1))
                elif item.keyword == "rotation":
                    rotation = self._float(item, 0)
                elif item.keyword == "scale":
                    scale = self._float(item, 0)
            elif isinstance(item, Word) and item in _MIRROR_WORDS:
                mirror = True
        return LVSPlacement(x=x, y=y, rotation=rotation, mirror=mirror, scale=scale)

    @staticmethod
    def _has_placement(items: Sequence[Any]) -> bool:
        return any(
            (isinstance(item, Record) and item.keyword in ("location", "rotation", "scale"))
            or (isinstance(item, Word) and item in _MIRROR_WORDS)
            for item in items
        )

    def _message(
        self,
        record: Record,
        layout_circuit: str | None = None,
        schematic_circuit: str | None = None,
    ) -> LVSMessage:
        fields: dict[str, Any] = {
            "layout_circuit": layout_circuit,
            "schematic_circuit": schematic_circuit,
        }
        for item in record.items:
            if isinstance(item, Record):
                keyword = item.keyword
                if keyword == "description":
                    fields["text"] = self._text(item)
                elif keyword in ("cell", "net"):
                    fields[keyword] = self._text(item)
                elif keyword == "cat":
                    fields["category"] = self._text(item)
                    if len(item.items) > 1:
                        fields["category_description"] = self._text(item, 1)
                elif keyword == "polygon":
                    try:
                        hull, *holes = lvsdb.parse_polygon_text(self._text(item))
                    except ValueError as error:
                        raise self.error(f"invalid message geometry: {error}", item) from error
                    fields["geometry"] = LVSGeometry(hull=hull, holes=tuple(holes))
            elif isinstance(item, Word) and item in _SEVERITIES:
                fields["severity"] = _SEVERITIES[item]
        return LVSMessage(**fields)

    def _netlist(self, section: Record, side: str) -> LVSNetlist:
        records = [item for item in section.items if isinstance(item, Record)]
        unit = next(
            (self._float(record, 0) for record in records if record.keyword == "unit"),
            _DEFAULT_UNIT,
        )
        if unit <= 0:
            raise self.error(f"the database unit must be positive, found {unit}", section)
        if side == "layout":
            self.unit = unit
        units = _Units(unit)

        top_circuit = None
        layer_order: list[str] = []
        layer_sources: dict[str, str | None] = {}
        layer_links: dict[str, dict[str, list[str]]] = {}
        classes: dict[str, LVSDeviceClass] = {}
        abstracts: dict[str, LVSDeviceAbstract] = {}
        raw_circuits: dict[str, _RawCircuit] = {}

        def layer(name: str) -> dict[str, list[str]]:
            if name not in layer_links:
                layer_order.append(name)
                layer_links[name] = {
                    "connected_to": [],
                    "soft_connected_to": [],
                    "global_nets": [],
                    "soft_global_nets": [],
                }
            return layer_links[name]

        link_fields = {
            "connect": "connected_to",
            "softconnect": "soft_connected_to",
            "global": "global_nets",
            "softglobal": "soft_global_nets",
        }
        for record in records:
            keyword = record.keyword
            if keyword == "version":
                self._check_version(record)
            elif keyword == "top":
                top_circuit = self._text(record)
            elif keyword == "layer":
                name = self._text(record)
                layer(name)
                layer_sources[name] = self._text(record, 1) if len(record.items) > 1 else None
            elif keyword in link_fields:
                links = layer(self._text(record))[link_fields[keyword]]
                links.extend(self._text(record, index) for index in range(1, len(record.items)))
            elif keyword == "message":
                self.extraction_messages.append(self._message(record))
            elif keyword == "class":
                device_class = self._device_class(record)
                classes[device_class.name] = device_class
            elif keyword == "device":
                abstract = self._device_abstract(record, units)
                abstracts[abstract.name] = abstract
            elif keyword == "circuit":
                circuit = self._raw_circuit(record, units, classes, abstracts)
                raw_circuits[circuit.name] = circuit

        circuits = {name: self._circuit(raw, raw_circuits) for name, raw in raw_circuits.items()}
        self.circuits[side] = circuits
        layers = tuple(
            LVSLayer(
                name=name,
                source=layer_sources.get(name),
                **{field: tuple(values) for field, values in layer_links[name].items()},
            )
            for name in layer_order
        )
        return LVSNetlist(
            circuits=circuits,
            device_classes=classes,
            device_abstracts=abstracts,
            layers=layers,
            top_circuit=top_circuit,
        )

    def _device_class(self, record: Record) -> LVSDeviceClass:
        name = self._text(record, 0)
        atoms = [item for item in record.items[1:] if not isinstance(item, Record)]
        # An empty template, '', means the class starts from nothing.
        template = self._text(record, 1) if atoms and isinstance(atoms[0], str) else None
        template = template or None
        # A '1' after the template means the class's own definitions replace the template's.
        replaces = any(type(atom) is int and atom == 1 for atom in atoms[1:])
        terminals: list[str] = []
        parameters: dict[str, tuple[bool, float]] = {}
        if template is not None and template not in _DEVICE_TEMPLATES:
            self.unknown_templates[template] += 1
        elif template is not None and not replaces:
            template_terminals, template_parameters = _DEVICE_TEMPLATES[template]
            terminals.extend(template_terminals)
            parameters.update(
                (parameter, (primary, default))
                for parameter, primary, default in template_parameters
            )
        for item in record.items:
            if not isinstance(item, Record):
                continue
            if item.keyword == "param":
                parameters[self._text(item, 0)] = (
                    bool(self._int(item, 1)),
                    self._float(item, 2),
                )
            elif item.keyword == "terminal":
                terminal = self._text(item, 0)
                if terminal not in terminals:
                    terminals.append(terminal)
        return LVSDeviceClass(
            name=name,
            template=template,
            terminals=tuple(terminals),
            parameters=tuple(
                LVSParameterDefinition(name=parameter, primary=primary, default=default)
                for parameter, (primary, default) in parameters.items()
            ),
        )

    def _device_abstract(self, record: Record, units: _Units) -> LVSDeviceAbstract:
        name = self._text(record, 0)
        device_class = self._text(record, 1)
        terminal_shapes: dict[str, dict[str, tuple[LVSPolygon, ...]]] = {}
        for terminal in record.items[2:]:
            if not isinstance(terminal, Record):
                continue
            shapes: dict[str, list[LVSPolygon]] = {}
            previous = [0, 0]
            for shape in terminal.items[1:]:
                if isinstance(shape, Record):
                    layer, polygon = self._shape(shape, units, previous)
                    shapes.setdefault(layer, []).append(polygon)
            terminal_shapes[self._text(terminal, 0)] = {
                layer: tuple(polygons) for layer, polygons in shapes.items()
            }
        return LVSDeviceAbstract(
            name=name, device_class=device_class, terminal_shapes=terminal_shapes
        )

    def _raw_circuit(
        self,
        record: Record,
        units: _Units,
        classes: dict[str, LVSDeviceClass],
        abstracts: dict[str, LVSDeviceAbstract],
    ) -> _RawCircuit:
        name = self._text(record, 0)
        boundary = None
        properties: dict[Any, Any] = {}
        nets = []
        pins = []
        devices = []
        subcircuits = []
        for item in record.items[1:]:
            if not isinstance(item, Record):
                continue
            keyword = item.keyword
            if keyword in ("rect", "polygon"):
                _, boundary = self._shape(item, units, [0, 0], has_layer=False)
            elif keyword == "property":
                key, value = self._property(item)
                properties[key] = value
            elif keyword == "net":
                nets.append(self._raw_net(item, units))
            elif keyword == "pin":
                pins.append(self._pin(item, len(pins)))
            elif keyword == "device":
                devices.append(self._device(item, units, classes, abstracts))
            elif keyword == "circuit":
                subcircuits.append(self._subcircuit(item, units))
        return _RawCircuit(
            record,
            name,
            boundary,
            properties,
            nets,
            tuple(pins),
            tuple(devices),
            tuple(subcircuits),
        )

    def _raw_net(self, record: Record, units: _Units) -> dict[str, Any]:
        net: dict[str, Any] = {"id": self._int(record, 0)}
        properties = {}
        shapes: dict[str, list[LVSPolygon]] = {}
        labels = []
        # One net's shapes share the previous point, across shapes and layers.
        previous = [0, 0]
        for item in record.items[1:]:
            if not isinstance(item, Record):
                continue
            keyword = item.keyword
            if keyword == "name":
                net["name"] = self._text(item)
            elif keyword == "property":
                key, value = self._property(item)
                properties[key] = value
            elif keyword in ("rect", "polygon"):
                layer, polygon = self._shape(item, units, previous)
                shapes.setdefault(layer, []).append(polygon)
            elif keyword == "text":
                points = self._points(item, item.items[2:], previous)
                if len(points) != 1:
                    raise self.error("a 'text' needs one point", item)
                labels.append(
                    LVSLabel(
                        layer=self._text(item, 0),
                        text=self._text(item, 1),
                        position=units.polygon(points)[0],
                    )
                )
        net["properties"] = properties
        net["shapes"] = {layer: tuple(polygons) for layer, polygons in shapes.items()}
        net["labels"] = tuple(labels)
        return net

    def _pin(self, record: Record, index: int) -> LVSPin:
        fields: dict[str, Any] = {"index": index}
        properties = {}
        for item in record.items:
            if isinstance(item, Record):
                if item.keyword == "name":
                    fields["name"] = self._text(item)
                elif item.keyword == "property":
                    key, value = self._property(item)
                    properties[key] = value
            elif type(item) is int:
                fields["net"] = item
        return LVSPin(properties=properties, **fields)

    def _device(
        self,
        record: Record,
        units: _Units,
        classes: dict[str, LVSDeviceClass],
        abstracts: dict[str, LVSDeviceAbstract],
    ) -> LVSDevice:
        reference = self._text(record, 1)
        fields: dict[str, Any] = {"id": self._int(record, 0)}
        if reference in abstracts:
            fields["abstract"] = reference
            fields["device_class"] = abstracts[reference].device_class
        elif reference in classes:
            fields["device_class"] = reference
        else:
            raise self.error(
                f"device {fields['id']} refers to '{reference}', which is neither a device "
                "abstract nor a device class",
                record,
            )
        items = record.items[2:]
        if self._has_placement(items):
            fields["placement"] = self._placement(record, items, units)
        parameters = {}
        terminals = {}
        parts = []
        routes = []
        properties = {}
        for item in items:
            if not isinstance(item, Record):
                continue
            keyword = item.keyword
            if keyword == "name":
                fields["name"] = self._text(item)
            elif keyword == "param":
                parameters[self._text(item, 0)] = self._float(item, 1)
            elif keyword == "terminal":
                terminals[self._text(item, 0)] = self._int(item, 1) if len(item.items) > 1 else None
            elif keyword == "device":
                part = self._text(item, 0)
                if part not in abstracts:
                    raise self.error(f"device part '{part}' isn't a device abstract", item)
                parts.append(
                    LVSDevicePart(
                        abstract=part, placement=self._placement(item, item.items[1:], units)
                    )
                )
            elif keyword == "connect":
                routes.append((self._int(item, 0), self._text(item, 1), self._text(item, 2)))
            elif keyword == "property":
                key, value = self._property(item)
                properties[key] = value
        return LVSDevice(
            parameters=parameters,
            terminals=terminals,
            parts=tuple(parts),
            terminal_routes=tuple(routes),
            properties=properties,
            **fields,
        )

    def _subcircuit(self, record: Record, units: _Units) -> LVSSubcircuit:
        fields: dict[str, Any] = {
            "id": self._int(record, 0),
            "circuit": self._text(record, 1),
        }
        items = record.items[2:]
        if self._has_placement(items):
            fields["placement"] = self._placement(record, items, units)
        pins = {}
        properties = {}
        for item in items:
            if not isinstance(item, Record):
                continue
            if item.keyword == "name":
                fields["name"] = self._text(item)
            elif item.keyword == "pin":
                pins[self._int(item, 0)] = self._int(item, 1)
            elif item.keyword == "property":
                key, value = self._property(item)
                properties[key] = value
        return LVSSubcircuit(pins=pins, properties=properties, **fields)

    def _circuit(self, raw: _RawCircuit, raw_circuits: dict[str, _RawCircuit]) -> LVSCircuit:
        """Resolve a circuit's net connections now that every circuit's pins are known."""
        connections: dict[int, list[LVSConnection]] = {net["id"]: [] for net in raw.nets}

        def connect(net_id: int, connection: LVSConnection, what: str) -> None:
            if net_id not in connections:
                raise self.error(
                    f"{what} in circuit '{raw.name}' refers to net {net_id}, which the circuit "
                    "doesn't define",
                    raw.record,
                )
            connections[net_id].append(connection)

        for pin in raw.pins:
            if pin.net is not None:
                connect(
                    pin.net,
                    LVSConnection(kind="pin", pin=pin.index, name=pin.name),
                    f"pin {pin.index}",
                )
        for device in raw.devices:
            for terminal, net_id in device.terminals.items():
                if net_id is not None:
                    connect(
                        net_id,
                        LVSConnection(kind="device_terminal", device=device.id, name=terminal),
                        f"device {device.id}",
                    )
        for subcircuit in raw.subcircuits:
            placed = raw_circuits.get(subcircuit.circuit)
            if placed is None:
                raise self.error(
                    f"subcircuit {subcircuit.id} in circuit '{raw.name}' places circuit "
                    f"'{subcircuit.circuit}', which the netlist doesn't define",
                    raw.record,
                )
            for pin_index, net_id in subcircuit.pins.items():
                if not 0 <= pin_index < len(placed.pins):
                    raise self.error(
                        f"subcircuit {subcircuit.id} in circuit '{raw.name}' connects pin "
                        f"{pin_index} of '{subcircuit.circuit}', which has "
                        f"{len(placed.pins)} pins",
                        raw.record,
                    )
                connect(
                    net_id,
                    LVSConnection(
                        kind="subcircuit_pin",
                        subcircuit=subcircuit.id,
                        pin=pin_index,
                        name=placed.pins[pin_index].name,
                    ),
                    f"subcircuit {subcircuit.id}",
                )

        nets = tuple(
            LVSNet(
                id=net["id"],
                name=net.get("name"),
                connections=tuple(connections[net["id"]]),
                shapes=net["shapes"],
                labels=net["labels"],
                properties=net["properties"],
            )
            for net in raw.nets
        )
        return LVSCircuit(
            name=raw.name,
            boundary=raw.boundary,
            nets=nets,
            pins=raw.pins,
            devices=raw.devices,
            subcircuits=raw.subcircuits,
            properties=raw.properties,
        )

    def _xref(self, section: Record) -> tuple[LVSComparison, ...]:
        comparisons = []
        for item in section.items:
            if not isinstance(item, Record):
                continue
            if item.keyword == "log":
                self.messages.extend(self._log(item))
            elif item.keyword == "circuit":
                comparisons.append(self._comparison(item))
        return tuple(comparisons)

    def _log(
        self,
        record: Record,
        layout_circuit: str | None = None,
        schematic_circuit: str | None = None,
    ) -> list[LVSMessage]:
        return [
            self._message(entry, layout_circuit, schematic_circuit)
            for entry in record.items
            if isinstance(entry, Record)
        ]

    def _status(self, record: Record, index: int) -> tuple[LVSStatus | None, int]:
        """The status at ``index``, if one is there, and the index after it."""
        if index >= len(record.items) or isinstance(record.items[index], Record):
            return None, index
        word = str(record.items[index])
        status = _STATUSES.get(word)
        if status is None:
            self.unknown_statuses[word] += 1
        return status, index + 1

    def _description(self, record: Record) -> str | None:
        for item in record.items:
            if isinstance(item, Record) and item.keyword == "description":
                return self._text(item)
        return None

    def _side(self, side: str, name: str | None, record: Record) -> LVSCircuit | None:
        if name is None:
            return None
        circuit = self.circuits.get(side, {}).get(name)
        if circuit is None:
            raise self.error(
                f"the cross-reference names {side} circuit '{name}', which the file doesn't define",
                record,
            )
        return circuit

    def _comparison(self, record: Record) -> LVSComparison:
        layout_name = self._optional_name(record, 0)
        schematic_name = self._optional_name(record, 1)
        status, _ = self._status(record, 2)
        layout = self._side("layout", layout_name, record)
        schematic = self._side("schematic", schematic_name, record)
        label = layout_name if layout_name is not None else schematic_name or ""
        messages = []
        pairs: dict[str, list[LVSPair]] = {"net": [], "pin": [], "device": [], "circuit": []}
        for item in record.items:
            if not isinstance(item, Record):
                continue
            if item.keyword == "log":
                messages.extend(self._log(item, layout_name, schematic_name))
            elif item.keyword == "xref":
                for pair in item.items:
                    if isinstance(pair, Record):
                        pairs[pair.keyword].append(self._pair(pair, layout, schematic, label))
        self.messages.extend(messages)
        return LVSComparison(
            circuit=label,
            status=status,
            description=self._description(record),
            layout=layout,
            schematic=schematic,
            messages=tuple(messages),
            nets=tuple(pairs["net"]),
            pins=tuple(pairs["pin"]),
            devices=tuple(pairs["device"]),
            subcircuits=tuple(pairs["circuit"]),
        )

    def _pair(
        self,
        record: Record,
        layout: LVSCircuit | None,
        schematic: LVSCircuit | None,
        label: str,
    ) -> LVSPair:
        kind = record.keyword
        sides = []
        for index, (side, circuit) in enumerate((("layout", layout), ("schematic", schematic))):
            object_id = self._optional_id(record, index)
            if object_id is None:
                sides.append(None)
                continue
            if circuit is None:
                raise self.error(
                    f"a {kind} pair names {side} {kind} {object_id} in a comparison without a "
                    f"{side} circuit",
                    record,
                )
            try:
                if kind == "net":
                    sides.append(circuit.net(object_id))
                elif kind == "pin":
                    sides.append(circuit.pin(object_id))
                elif kind == "device":
                    sides.append(circuit.device(object_id))
                else:
                    sides.append(circuit.subcircuit(object_id))
            except (KeyError, IndexError) as error:
                raise self.error(
                    format_chained_exception_message(
                        f"the cross-reference names {kind} {object_id} of {side} circuit "
                        f"'{circuit.name}', which the circuit doesn't define",
                        error,
                    ),
                    record,
                ) from error
        status, _ = self._status(record, 2)
        fields = {
            "circuit": label,
            "status": status,
            "description": self._description(record),
            "layout": sides[0],
            "schematic": sides[1],
        }
        if kind == "net":
            return LVSNetPair(**fields)
        if kind == "pin":
            return LVSPinPair(**fields)
        if kind == "device":
            return LVSDevicePair(**fields)
        return LVSSubcircuitPair(**fields)


def _format_number(value: float) -> str:
    return f"{value:.10g}"


_PAIR_WORDS: dict[type[LVSPair], str] = {
    LVSComparison: "circuit",
    LVSNetPair: "net",
    LVSPinPair: "pin",
    LVSDevicePair: "device",
    LVSSubcircuitPair: "subcircuit",
}


class _Summary:
    """Builds the text that ``print(results)`` shows."""

    def __init__(self, results: LVSResults) -> None:
        self.results = results
        self.lines: list[str] = []

    def text(self) -> str:
        results = self.results
        self.lines.append(f"LVS verdict: {results.verdict}")
        self.lines.append(self._circuits_line())
        for title, statuses in (("Mismatches", _FAILED_STATUSES), ("Warnings", ("warning",))):
            for comparison in results.comparisons:
                entries = [
                    self._pair_lines(pair, comparison)
                    for pair in _pairs(comparison)[1:]
                    if pair.status in statuses
                ]
                self._section(f"{title} in {comparison.circuit}:", entries)
        self._section("Messages:", [[self._message_line(message)] for message in results.messages])
        self._section(
            "Extraction messages:",
            [[self._message_line(message)] for message in results.extraction_messages],
        )
        if results.exit_code is not None:
            self.lines.extend(("", f"KLayout exit code: {results.exit_code}"))
        return "\n".join(self.lines)

    def _circuits_line(self) -> str:
        comparisons = self.results.comparisons
        counts = Counter(comparison.status for comparison in comparisons)
        parts = []
        for status in ("match", "warning", *_FAILED_STATUSES, None):
            if not counts[status]:
                continue
            part = f"{counts[status]} {status or 'without status'}"
            if status != "match":
                names = [c.circuit for c in comparisons if c.status == status]
                shown = ", ".join(names[:_SUMMARY_LIMIT])
                if len(names) > _SUMMARY_LIMIT:
                    shown += f", and {len(names) - _SUMMARY_LIMIT} more"
                part += f" ({shown})"
            parts.append(part)
        return f"Circuits: {len(comparisons)} compared: {', '.join(parts) or 'none'}"

    def _section(self, title: str, entries: list[list[str]]) -> None:
        if not entries:
            return
        self.lines.extend(("", title))
        for entry in entries[:_SUMMARY_LIMIT]:
            self.lines.extend(entry)
        if len(entries) > _SUMMARY_LIMIT:
            self.lines.append(f"  ... and {len(entries) - _SUMMARY_LIMIT} more")

    def _pair_lines(self, pair: LVSPair, comparison: LVSComparison) -> list[str]:
        line = (
            f"  {_PAIR_WORDS[type(pair)]} {self._label(pair.layout)} <-> "
            f"{self._label(pair.schematic)}: {pair.status}"
        )
        details = []
        if isinstance(pair, LVSNetPair):
            for side, net, circuit in (
                ("layout", pair.layout, comparison.layout),
                ("schematic", pair.schematic, comparison.schematic),
            ):
                if net is not None and circuit is not None:
                    joins = ", ".join(
                        self._connection(connection, circuit) for connection in net.connections
                    )
                    details.append(f"    {side + ':':<10} {joins or 'nothing'}")
        elif isinstance(pair, LVSSubcircuitPair):
            side, subcircuit = (
                ("layout", pair.layout)
                if pair.layout is not None
                else ("schematic", pair.schematic)
            )
            if subcircuit is not None:
                line += f" ({side}: {self._placed(subcircuit)})"
        elif isinstance(pair, LVSDevicePair):
            for side, device, netlist in (
                ("layout", pair.layout, self.results.layout),
                ("schematic", pair.schematic, self.results.schematic),
            ):
                if device is not None:
                    details.append(f"    {side + ':':<10} {self._device(device, netlist)}")
        if pair.description:
            details.append(f"    {pair.description}")
        return [line, *details]

    @staticmethod
    def _label(item: Any) -> str:
        if item is None:
            return "(none)"
        if isinstance(item, LVSPin):
            return item.name or f"${item.index}"
        return item.name or f"${item.id}"

    @staticmethod
    def _connection(connection: LVSConnection, circuit: LVSCircuit) -> str:
        if connection.kind == "pin":
            return f"pin {connection.name or connection.pin}"
        if connection.kind == "device_terminal":
            device = circuit.device(connection.device)
            return f"{device.device_class} {device.name or f'${device.id}'} {connection.name}"
        subcircuit = circuit.subcircuit(connection.subcircuit)
        label = subcircuit.name or f"${subcircuit.id}"
        return f"{subcircuit.circuit} {label} {connection.name or connection.pin}"

    @staticmethod
    def _placed(subcircuit: LVSSubcircuit) -> str:
        text = subcircuit.circuit
        placement = subcircuit.placement
        if placement is not None:
            text += f" at ({_format_number(placement.x)}, {_format_number(placement.y)})"
            if placement.rotation:
                text += f", rotated {_format_number(placement.rotation)}°"
            if placement.mirror:
                text += ", mirrored"
            if placement.scale != 1:
                text += f", scaled by {_format_number(placement.scale)}"
        return text

    @staticmethod
    def _device(device: LVSDevice, netlist: LVSNetlist) -> str:
        device_class = netlist.device_classes.get(device.device_class)
        names = (
            [parameter.name for parameter in device_class.parameters if parameter.primary]
            if device_class is not None
            else list(device.parameters)
        )
        values = " ".join(
            f"{name}={_format_number(device.parameters[name])}"
            for name in names
            if name in device.parameters
        )
        return f"{device.device_class} {values}".rstrip()

    @staticmethod
    def _message_line(message: LVSMessage) -> str:
        where = message.layout_circuit or message.schematic_circuit or message.cell
        prefix = message.severity or "message"
        if where:
            prefix += f" in {where}"
        return f"  {prefix}: {message.text}"
