"""Netlists in KLayout LVS results: the netlist extracted from the layout and the schematic."""

from __future__ import annotations

import base64
import math
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, PlainSerializer

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.types import Coordinate2D
from tidy3d.exceptions import ValidationError

LVSPolygon = tuple[Coordinate2D, ...]
"""One polygon contour in µm. KLayout joins holes into the hull, so a contour is the whole polygon."""


def _encode_value(value: Any) -> Any:
    """Return a property key or value in a JSON form, tagging the types JSON lacks."""
    if isinstance(value, bytes):
        return {"bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": repr(value)}
    if isinstance(value, tuple):
        return {"tuple": [_encode_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, dict):
        return {"dict": _encode_properties(value)}
    return value


def _encode_properties(properties: dict[Any, Any]) -> list[list[Any]]:
    """Return properties as key/value pairs, because JSON keys can only be strings."""
    return [[_encode_value(key), _encode_value(value)] for key, value in properties.items()]


def _decode_value(value: Any) -> Any:
    """Return a property key or value from the JSON form that ``_encode_value`` writes."""
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if len(value) != 1:
        raise ValidationError(f"A tagged property value must have one key, not {len(value)}.")
    ((tag, content),) = value.items()
    if tag == "bytes":
        return base64.b64decode(content)
    if tag == "float":
        return float(content)
    if tag == "tuple":
        return tuple(_decode_value(item) for item in content)
    if tag == "dict":
        return _decode_properties(content)
    raise ValidationError(f"Unknown property value tag {tag!r}.")


def _decode_properties(value: Any) -> Any:
    """Return properties from their JSON form, a list of key/value pairs; leave others as they
    are."""
    if isinstance(value, list):
        return {_decode_value(key): _decode_value(item) for key, item in value}
    return value


LVSProperties = Annotated[
    dict[Any, Any],
    BeforeValidator(_decode_properties),
    PlainSerializer(_encode_properties, when_used="json"),
]
"""KLayout properties by key. Keys and values can be numbers, strings, byte strings, booleans,
``None``, lists and dictionaries; a list or dictionary used as a key becomes a tuple. JSON holds the
properties as key/value pairs, with bytes, tuples and non-finite numbers tagged, so they load back
unchanged."""


class LVSPlacement(Tidy3dBaseModel):
    """Placement of a subcircuit or device in the circuit that holds it.

    KLayout applies the magnification first, then the mirroring about the x-axis, then the
    counter-clockwise rotation, and then the displacement.
    """

    x: float = Field(title="x", description="Displacement along x, in µm.")
    y: float = Field(title="y", description="Displacement along y, in µm.")
    rotation: float = Field(
        default=0.0, title="Rotation", description="Counter-clockwise rotation, in degrees."
    )
    mirror: bool = Field(
        default=False,
        title="Mirror",
        description="Whether the placement mirrors about the x-axis before rotating.",
    )
    scale: float = Field(default=1.0, title="Scale", description="Magnification.")


class LVSLabel(Tidy3dBaseModel):
    """A text label attached to a net in the extracted netlist."""

    layer: str = Field(title="Layer", description="Name of the runset layer the label is on.")
    text: str = Field(title="Text", description="The label's text.")
    position: Coordinate2D = Field(title="Position", description="The label's anchor point, in µm.")


class LVSConnection(Tidy3dBaseModel):
    """Something a net connects to: a pin of its circuit, a pin of a subcircuit, or a device
    terminal."""

    kind: Literal["pin", "subcircuit_pin", "device_terminal"] = Field(
        title="Kind", description="What the net connects to."
    )
    pin: int | None = Field(
        default=None,
        title="Pin",
        description="Index of the circuit's own pin, or of the placed circuit's pin for "
        "``'subcircuit_pin'``.",
    )
    subcircuit: int | None = Field(
        default=None, title="Subcircuit", description="Subcircuit ID, for ``'subcircuit_pin'``."
    )
    device: int | None = Field(
        default=None, title="Device", description="Device ID, for ``'device_terminal'``."
    )
    name: str | None = Field(
        default=None, title="Name", description="Name of the pin or terminal, if it has one."
    )


class LVSNet(Tidy3dBaseModel):
    """A net: a set of connected shapes, and what they connect to."""

    id: int = Field(title="ID", description="The net's ID within its circuit, counting from 1.")
    name: str | None = Field(
        default=None,
        title="Name",
        description="The net's name as written, such as ``'o1'``, or ``'$3'`` for an unnamed "
        "layout net. ``None`` if the file gives no name.",
    )
    connections: tuple[LVSConnection, ...] = Field(
        default=(), title="Connections", description="The pins and device terminals the net joins."
    )
    shapes: dict[str, tuple[LVSPolygon, ...]] = Field(
        default_factory=dict,
        title="Shapes",
        description="The net's polygons by runset layer name, in µm, in the coordinates of the "
        "circuit that holds the net. Extracted netlist only.",
    )
    labels: tuple[LVSLabel, ...] = Field(
        default=(), title="Labels", description="Text labels on the net. Extracted netlist only."
    )
    properties: LVSProperties = Field(
        default_factory=dict, title="Properties", description="KLayout properties of the net."
    )


class LVSPin(Tidy3dBaseModel):
    """An external pin of a circuit."""

    index: int = Field(
        title="Index", description="The pin's position in its circuit's pin list, from 0."
    )
    name: str | None = Field(
        default=None, title="Name", description="The pin's name, if it has one."
    )
    net: int | None = Field(
        default=None,
        title="Net",
        description="ID of the net the pin connects to; ``None`` if none.",
    )
    properties: LVSProperties = Field(
        default_factory=dict, title="Properties", description="KLayout properties of the pin."
    )


class LVSSubcircuit(Tidy3dBaseModel):
    """A placed instance of another circuit."""

    id: int = Field(title="ID", description="The subcircuit's ID within its circuit.")
    circuit: str = Field(title="Circuit", description="Name of the placed circuit.")
    name: str | None = Field(
        default=None,
        title="Name",
        description="Instance name, such as ``'MMI_1'``. Schematic only, because layout "
        "instances have no names.",
    )
    placement: LVSPlacement | None = Field(
        default=None,
        title="Placement",
        description="The instance's placement. Extracted netlist only.",
    )
    pins: dict[int, int] = Field(
        default_factory=dict,
        title="Pins",
        description="Net ID by the placed circuit's pin index, for connected pins only.",
    )
    properties: LVSProperties = Field(
        default_factory=dict,
        title="Properties",
        description="KLayout properties of the subcircuit.",
    )


class LVSDevicePart(Tidy3dBaseModel):
    """One of the extra device abstracts that make up a combined device."""

    abstract: str = Field(title="Abstract", description="Name of the device abstract.")
    placement: LVSPlacement = Field(
        title="Placement", description="The part's placement relative to the device."
    )


class LVSDevice(Tidy3dBaseModel):
    """A device: a component that the runset recognizes from shapes, with measured parameters."""

    id: int = Field(title="ID", description="The device's ID within its circuit.")
    device_class: str = Field(title="Device class", description="Name of the device class.")
    name: str | None = Field(
        default=None, title="Name", description="The device's name. Schematic only."
    )
    abstract: str | None = Field(
        default=None,
        title="Abstract",
        description="Name of the device abstract that holds the device's geometry. Extracted "
        "netlist only.",
    )
    placement: LVSPlacement | None = Field(
        default=None,
        title="Placement",
        description="The device's placement. Extracted netlist only.",
    )
    parameters: dict[str, float] = Field(
        default_factory=dict,
        title="Parameters",
        description="Value by parameter name, as written, in the units the runset uses.",
    )
    terminals: dict[str, int | None] = Field(
        default_factory=dict,
        title="Terminals",
        description="ID of the connected net by terminal name; ``None`` if unconnected.",
    )
    parts: tuple[LVSDevicePart, ...] = Field(
        default=(),
        title="Parts",
        description="For a device that KLayout combined from several, the extra abstracts "
        "with their placements.",
    )
    terminal_routes: tuple[tuple[int, str, str], ...] = Field(
        default=(),
        title="Terminal routes",
        description="For a combined device: how outer terminals map to the terminals of its "
        "parts, as (part index, outer terminal, inner terminal).",
    )
    properties: LVSProperties = Field(
        default_factory=dict, title="Properties", description="KLayout properties of the device."
    )


class LVSParameterDefinition(Tidy3dBaseModel):
    """A parameter that a device class defines."""

    name: str = Field(title="Name", description="The parameter's name.")
    primary: bool = Field(
        title="Primary", description="Whether KLayout compares the parameter by default."
    )
    default: float = Field(title="Default", description="The parameter's default value.")


class LVSDeviceClass(Tidy3dBaseModel):
    """A device class, with its terminals and parameters."""

    name: str = Field(title="Name", description="The class name.")
    template: str | None = Field(
        default=None,
        title="Template",
        description="The built-in class it starts from, such as ``'RES'`` or ``'MOS4'``; "
        "``None`` for a class that starts from nothing.",
    )
    terminals: tuple[str, ...] = Field(
        default=(), title="Terminals", description="Terminal names, with the template applied."
    )
    parameters: tuple[LVSParameterDefinition, ...] = Field(
        default=(),
        title="Parameters",
        description="Parameter definitions, with the template applied.",
    )


class LVSDeviceAbstract(Tidy3dBaseModel):
    """The geometry of one kind of device in the extracted netlist."""

    name: str = Field(title="Name", description="The abstract's name.")
    device_class: str = Field(title="Device class", description="Name of the device class.")
    terminal_shapes: dict[str, dict[str, tuple[LVSPolygon, ...]]] = Field(
        default_factory=dict,
        title="Terminal shapes",
        description="Polygons by terminal name and then runset layer name, in µm, in the "
        "abstract's own coordinates.",
    )


class LVSLayer(Tidy3dBaseModel):
    """A layer that the runset uses for connectivity. Extracted netlist only."""

    name: str = Field(
        title="Name", description="The runset's name for the layer, such as ``'l2'``."
    )
    source: str | None = Field(
        default=None,
        title="Source",
        description="The layout layer it comes from, such as ``'101/0'``, if it is an input layer.",
    )
    connected_to: tuple[str, ...] = Field(
        default=(), title="Connected to", description="Layers this layer connects to."
    )
    soft_connected_to: tuple[str, ...] = Field(
        default=(), title="Soft connected to", description="Layers this layer soft-connects to."
    )
    global_nets: tuple[str, ...] = Field(
        default=(), title="Global nets", description="Global nets this layer connects to."
    )
    soft_global_nets: tuple[str, ...] = Field(
        default=(), title="Soft global nets", description="Global nets this layer soft-connects to."
    )


class LVSCircuit(Tidy3dBaseModel):
    """A circuit: one cell of the layout, or one subcircuit definition of the schematic."""

    name: str = Field(title="Name", description="The circuit name.")
    boundary: LVSPolygon | None = Field(
        default=None,
        title="Boundary",
        description="The cell outline, in µm. Extracted netlist only.",
    )
    nets: tuple[LVSNet, ...] = Field(default=(), title="Nets", description="The circuit's nets.")
    pins: tuple[LVSPin, ...] = Field(
        default=(), title="Pins", description="The circuit's pins, in pin order."
    )
    devices: tuple[LVSDevice, ...] = Field(default=(), title="Devices", description="The devices.")
    subcircuits: tuple[LVSSubcircuit, ...] = Field(
        default=(), title="Subcircuits", description="The placed circuits."
    )
    properties: LVSProperties = Field(
        default_factory=dict, title="Properties", description="KLayout properties of the circuit."
    )

    @cached_property
    def _nets_by_id(self) -> dict[int, LVSNet]:
        return {net.id: net for net in self.nets}

    @cached_property
    def _devices_by_id(self) -> dict[int, LVSDevice]:
        return {device.id: device for device in self.devices}

    @cached_property
    def _subcircuits_by_id(self) -> dict[int, LVSSubcircuit]:
        return {subcircuit.id: subcircuit for subcircuit in self.subcircuits}

    def net(self, net_id: int) -> LVSNet:
        """Return the net with ID ``net_id``.

        Raises
        ------
        KeyError
            If the circuit has no such net.
        """
        return self._nets_by_id[net_id]

    def device(self, device_id: int) -> LVSDevice:
        """Return the device with ID ``device_id``.

        Raises
        ------
        KeyError
            If the circuit has no such device.
        """
        return self._devices_by_id[device_id]

    def subcircuit(self, subcircuit_id: int) -> LVSSubcircuit:
        """Return the subcircuit with ID ``subcircuit_id``.

        Raises
        ------
        KeyError
            If the circuit has no such subcircuit.
        """
        return self._subcircuits_by_id[subcircuit_id]

    def pin(self, index: int) -> LVSPin:
        """Return the pin at position ``index``, counting from 0.

        Raises
        ------
        IndexError
            If the circuit has no such pin.
        """
        if index < 0:
            raise IndexError(f"Pin index {index} is negative.")
        return self.pins[index]


class LVSNetlist(Tidy3dBaseModel):
    """A netlist in LVS results: the one extracted from the layout, or the schematic."""

    circuits: dict[str, LVSCircuit] = Field(
        default_factory=dict,
        title="Circuits",
        description="Circuits by name, in file order, which lists children before parents.",
    )
    device_classes: dict[str, LVSDeviceClass] = Field(
        default_factory=dict, title="Device classes", description="Device classes by name."
    )
    device_abstracts: dict[str, LVSDeviceAbstract] = Field(
        default_factory=dict,
        title="Device abstracts",
        description="Device abstracts by name. Extracted netlist only.",
    )
    layers: tuple[LVSLayer, ...] = Field(
        default=(),
        title="Layers",
        description="The runset's connectivity layers. Extracted netlist only.",
    )
    top_circuit: str | None = Field(
        default=None,
        title="Top circuit",
        description="Name of the top circuit. Extracted netlist only.",
    )
