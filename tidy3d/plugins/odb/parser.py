"""ODB++ file parsing.

Low-level parsing functions for ODB++ text files including:
- Matrix file (layer/step definitions)
- Features file (L, P, A, S records)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Union


# --- Data Classes for Parsed Records ---


@dataclass
class LayerDef:
    """Layer definition from matrix file.

    Attributes
    ----------
    name : str
        Layer name (e.g., "TRACE", "GND").
    row : int
        Row number in the stackup (1-based).
    type : str
        Layer type: SIGNAL, POWER_GROUND, DIELECTRIC, DRILL, etc.
    polarity : str
        POSITIVE or NEGATIVE.
    context : str
        Layer context (typically "BOARD").
    start_name : str
        For drill layers, the starting layer name.
    end_name : str
        For drill layers, the ending layer name.
    """

    name: str
    row: int
    type: str
    polarity: str
    context: str = "BOARD"
    start_name: str = ""
    end_name: str = ""


@dataclass
class StepDef:
    """Step definition from matrix file.

    Attributes
    ----------
    name : str
        Step name (e.g., "pcb", "panel").
    col : int
        Column number (typically 1).
    """

    name: str
    col: int


@dataclass
class MatrixData:
    """Parsed matrix file data.

    Attributes
    ----------
    steps : list[StepDef]
        Step definitions.
    layers : list[LayerDef]
        Layer definitions sorted by row number.
    """

    steps: list[StepDef] = field(default_factory=list)
    layers: list[LayerDef] = field(default_factory=list)


@dataclass
class LineRecord:
    """Parsed L (line) record.

    Attributes
    ----------
    xs, ys : float
        Start point coordinates.
    xe, ye : float
        End point coordinates.
    symbol_num : int
        Symbol index from symbol table.
    polarity : str
        "P" (positive) or "N" (negative).
    dcode : int
        Gerber D-code number.
    attributes : dict
        Feature attributes.
    """

    xs: float
    ys: float
    xe: float
    ye: float
    symbol_num: int
    polarity: str
    dcode: int = 0
    attributes: dict = field(default_factory=dict)


@dataclass
class ArcRecord:
    """Parsed A (arc) record.

    Attributes
    ----------
    xs, ys : float
        Start point coordinates.
    xe, ye : float
        End point coordinates.
    xc, yc : float
        Center point coordinates.
    symbol_num : int
        Symbol index from symbol table.
    polarity : str
        "P" (positive) or "N" (negative).
    clockwise : bool
        True if arc goes clockwise.
    dcode : int
        Gerber D-code number.
    attributes : dict
        Feature attributes.
    """

    xs: float
    ys: float
    xe: float
    ye: float
    xc: float
    yc: float
    symbol_num: int
    polarity: str
    clockwise: bool
    dcode: int = 0
    attributes: dict = field(default_factory=dict)


@dataclass
class PadRecord:
    """Parsed P (pad) record.

    Attributes
    ----------
    x, y : float
        Pad center coordinates.
    symbol_num : int
        Symbol index from symbol table.
    polarity : str
        "P" (positive) or "N" (negative).
    dcode : int
        Gerber D-code number.
    orient_def : str
        Orientation definition string (rotation/mirror encoding).
    attributes : dict
        Feature attributes.
    """

    x: float
    y: float
    symbol_num: int
    polarity: str
    dcode: int
    orient_def: str
    attributes: dict = field(default_factory=dict)


@dataclass
class PolygonContour:
    """A single contour (outline or hole) within a surface.

    Attributes
    ----------
    is_hole : bool
        True if this is a hole (H), False if island (I).
    segments : list[dict]
        List of segment dictionaries with keys:
        - "type": "start", "line", or "arc"
        - For "start" and "line": "x", "y"
        - For "arc": "xe", "ye", "xc", "yc", "cw"
    """

    is_hole: bool
    segments: list[dict] = field(default_factory=list)


@dataclass
class SurfaceRecord:
    """Parsed S (surface) record.

    Attributes
    ----------
    polarity : str
        "P" (positive) or "N" (negative).
    dcode : int
        Gerber D-code number.
    contours : list[PolygonContour]
        List of polygon contours (islands and holes).
    attributes : dict
        Feature attributes.
    """

    polarity: str
    dcode: int
    contours: list[PolygonContour] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)


# Union type for all feature records
FeatureRecord = Union[LineRecord, ArcRecord, PadRecord, SurfaceRecord]


@dataclass
class FeaturesData:
    """Parsed features file data.

    Attributes
    ----------
    units : str
        "MM" or "INCH".
    symbols : dict[int, str]
        Symbol index to name mapping (e.g., {0: "r200", 1: "rect250x150"}).
    attr_names : dict[int, str]
        Attribute index to name mapping.
    attr_texts : dict[int, str]
        Attribute text index to text mapping.
    features : list[FeatureRecord]
        List of parsed feature records.
    """

    units: str = "MM"
    symbols: dict[int, str] = field(default_factory=dict)
    attr_names: dict[int, str] = field(default_factory=dict)
    attr_texts: dict[int, str] = field(default_factory=dict)
    features: list[FeatureRecord] = field(default_factory=list)


# --- Parsing Functions ---


def parse_matrix(content: str) -> MatrixData:
    """Parse matrix file content.

    Parameters
    ----------
    content : str
        Raw text content of matrix file.

    Returns
    -------
    MatrixData
        Parsed step and layer definitions.

    Example
    -------
    >>> content = open("mydesign.odb/matrix/matrix").read()
    >>> data = parse_matrix(content)
    >>> for layer in data.layers:
    ...     print(f"{layer.name}: {layer.type}")
    """
    data = MatrixData()

    # Parse STEP blocks
    for match in re.finditer(r"STEP\s*\{([^}]+)\}", content, re.DOTALL):
        block = match.group(1)
        step = _parse_block_to_dict(block)
        data.steps.append(
            StepDef(
                name=step.get("NAME", ""),
                col=int(step.get("COL", 1)),
            )
        )

    # Parse LAYER blocks
    for match in re.finditer(r"LAYER\s*\{([^}]+)\}", content, re.DOTALL):
        block = match.group(1)
        layer = _parse_block_to_dict(block)
        data.layers.append(
            LayerDef(
                name=layer.get("NAME", ""),
                row=int(layer.get("ROW", 0)),
                type=layer.get("TYPE", ""),
                polarity=layer.get("POLARITY", "POSITIVE"),
                context=layer.get("CONTEXT", "BOARD"),
                start_name=layer.get("START_NAME", ""),
                end_name=layer.get("END_NAME", ""),
            )
        )

    # Sort layers by row
    data.layers.sort(key=lambda x: x.row)

    return data


def parse_features(content: str) -> FeaturesData:
    """Parse features file content.

    Parameters
    ----------
    content : str
        Raw text content of features file.

    Returns
    -------
    FeaturesData
        Parsed symbols and feature records.

    Example
    -------
    >>> content = open("mydesign.odb/steps/pcb/layers/TRACE/features").read()
    >>> data = parse_features(content)
    >>> print(f"Units: {data.units}")
    >>> print(f"Symbols: {data.symbols}")
    >>> print(f"Features: {len(data.features)}")
    """
    data = FeaturesData()
    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip comments and empty lines
        if not line or line.startswith("#"):
            i += 1
            continue

        # UNITS directive
        if line.startswith("UNITS="):
            data.units = line.split("=")[1].strip()
            i += 1
            continue

        # Symbol definition: $<num> <name>
        if line.startswith("$"):
            match = re.match(r"\$(\d+)\s+(.+)", line)
            if match:
                data.symbols[int(match.group(1))] = match.group(2).strip()
            i += 1
            continue

        # Attribute name: @<num> <name>
        if line.startswith("@"):
            match = re.match(r"@(\d+)\s+(.+)", line)
            if match:
                data.attr_names[int(match.group(1))] = match.group(2).strip()
            i += 1
            continue

        # Attribute text: &<num> <text>
        if line.startswith("&"):
            match = re.match(r"&(\d+)\s+(.+)", line)
            if match:
                data.attr_texts[int(match.group(1))] = match.group(2).strip()
            i += 1
            continue

        # Skip F (feature count) lines
        if line.startswith("F "):
            i += 1
            continue

        # Skip ID lines
        if line.startswith("ID="):
            i += 1
            continue

        # Line record: L <xs> <ys> <xe> <ye> <sym_num> <pol> <dcode>
        if line.startswith("L "):
            record = _parse_line_record(line)
            if record:
                data.features.append(record)
            i += 1
            continue

        # Arc record: A <xs> <ys> <xe> <ye> <xc> <yc> <sym_num> <pol> <dcode> <cw>
        if line.startswith("A "):
            record = _parse_arc_record(line)
            if record:
                data.features.append(record)
            i += 1
            continue

        # Pad record: P <x> <y> <sym_num> <pol> <dcode> <orient>
        if line.startswith("P "):
            record = _parse_pad_record(line)
            if record:
                data.features.append(record)
            i += 1
            continue

        # Surface record: S <pol> <dcode>
        if line.startswith("S "):
            record, i = _parse_surface_record(lines, i)
            if record:
                data.features.append(record)
            continue

        # Skip unrecognized lines
        i += 1

    return data


def _parse_block_to_dict(block: str) -> dict[str, str]:
    """Parse KEY=VALUE lines from a block.

    Parameters
    ----------
    block : str
        Text block content.

    Returns
    -------
    dict[str, str]
        Parsed key-value pairs.
    """
    result: dict[str, str] = {}
    for line in block.strip().split("\n"):
        line = line.strip()
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def _parse_attributes(attr_str: str) -> dict:
    """Parse attribute string like '0=1,2=0,25=2'.

    Parameters
    ----------
    attr_str : str
        Attribute string.

    Returns
    -------
    dict
        Parsed attributes with integer keys.
    """
    if not attr_str:
        return {}
    attrs: dict = {}
    for part in attr_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                attrs[int(k)] = v
            except ValueError:
                pass
        else:
            # Boolean attribute (just the index)
            try:
                attrs[int(part)] = True
            except ValueError:
                pass
    return attrs


def _split_record_and_attrs(line: str) -> tuple[str, dict]:
    """Split record line and trailing attributes.

    Parameters
    ----------
    line : str
        Full record line.

    Returns
    -------
    tuple[str, dict]
        (main_part, attributes_dict)
    """
    # Attributes come after semicolon: L ... ;<attrs>
    if ";" in line:
        main, attr_part = line.split(";", 1)
        # May have multiple semicolon sections (attrs;ID=xxx)
        attrs = _parse_attributes(attr_part.split(";")[0])
        return main.strip(), attrs
    return line.strip(), {}


def _parse_line_record(line: str) -> Optional[LineRecord]:
    """Parse L record.

    Parameters
    ----------
    line : str
        L record line.

    Returns
    -------
    Optional[LineRecord]
        Parsed record or None if parsing failed.
    """
    main, attrs = _split_record_and_attrs(line)
    parts = main.split()
    if len(parts) < 7:
        return None

    try:
        return LineRecord(
            xs=float(parts[1]),
            ys=float(parts[2]),
            xe=float(parts[3]),
            ye=float(parts[4]),
            symbol_num=int(parts[5]),
            polarity=parts[6],
            dcode=int(parts[7]) if len(parts) > 7 else 0,
            attributes=attrs,
        )
    except (ValueError, IndexError):
        return None


def _parse_arc_record(line: str) -> Optional[ArcRecord]:
    """Parse A record.

    Parameters
    ----------
    line : str
        A record line.

    Returns
    -------
    Optional[ArcRecord]
        Parsed record or None if parsing failed.
    """
    main, attrs = _split_record_and_attrs(line)
    parts = main.split()
    if len(parts) < 10:
        return None

    try:
        return ArcRecord(
            xs=float(parts[1]),
            ys=float(parts[2]),
            xe=float(parts[3]),
            ye=float(parts[4]),
            xc=float(parts[5]),
            yc=float(parts[6]),
            symbol_num=int(parts[7]),
            polarity=parts[8],
            dcode=int(parts[9]) if len(parts) > 9 else 0,
            clockwise=parts[10].upper() == "Y" if len(parts) > 10 else False,
            attributes=attrs,
        )
    except (ValueError, IndexError):
        return None


def _parse_pad_record(line: str) -> Optional[PadRecord]:
    """Parse P record.

    Parameters
    ----------
    line : str
        P record line.

    Returns
    -------
    Optional[PadRecord]
        Parsed record or None if parsing failed.
    """
    main, attrs = _split_record_and_attrs(line)
    parts = main.split()
    if len(parts) < 5:
        return None

    try:
        # Join all parts from index 6 onwards (orient_type and optional angle)
        # For orient types 8/9, the angle is in the next field: "8 270.0"
        # This matches RF GUI: p.args.slice(5).join(' ')
        orient_def = " ".join(parts[6:]) if len(parts) > 6 else "0"

        return PadRecord(
            x=float(parts[1]),
            y=float(parts[2]),
            symbol_num=int(parts[3]),
            polarity=parts[4],
            dcode=int(parts[5]) if len(parts) > 5 else 0,
            orient_def=orient_def,
            attributes=attrs,
        )
    except (ValueError, IndexError):
        return None


def _parse_surface_record(lines: list[str], start_idx: int) -> tuple[Optional[SurfaceRecord], int]:
    """Parse S record spanning multiple lines until SE.

    Parameters
    ----------
    lines : list[str]
        All lines of the features file.
    start_idx : int
        Index of the S line.

    Returns
    -------
    tuple[Optional[SurfaceRecord], int]
        (parsed_record, next_line_index)
    """
    first_line = lines[start_idx].strip()
    main, attrs = _split_record_and_attrs(first_line)
    parts = main.split()

    surface = SurfaceRecord(
        polarity=parts[1] if len(parts) > 1 else "P",
        dcode=int(parts[2]) if len(parts) > 2 else 0,
        attributes=attrs,
    )

    i = start_idx + 1
    current_contour: Optional[PolygonContour] = None

    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("#"):
            i += 1
            continue

        # Surface end
        if line == "SE":
            i += 1
            break

        # Polygon begin: OB x y type
        if line.startswith("OB "):
            parts = line.split()
            if len(parts) >= 4:
                current_contour = PolygonContour(
                    is_hole=(parts[3].upper() == "H"),
                    segments=[{"type": "start", "x": float(parts[1]), "y": float(parts[2])}],
                )
                surface.contours.append(current_contour)
            i += 1
            continue

        # Polygon segment: OS x y
        if line.startswith("OS ") and current_contour:
            parts = line.split()
            if len(parts) >= 3:
                current_contour.segments.append(
                    {"type": "line", "x": float(parts[1]), "y": float(parts[2])}
                )
            i += 1
            continue

        # Polygon arc: OC xe ye xc yc cw
        if line.startswith("OC ") and current_contour:
            parts = line.split()
            if len(parts) >= 6:
                current_contour.segments.append(
                    {
                        "type": "arc",
                        "xe": float(parts[1]),
                        "ye": float(parts[2]),
                        "xc": float(parts[3]),
                        "yc": float(parts[4]),
                        "cw": parts[5].upper() == "Y" if len(parts) > 5 else False,
                    }
                )
            i += 1
            continue

        # Polygon end
        if line == "OE":
            current_contour = None
            i += 1
            continue

        i += 1

    return surface, i


# --- File I/O Helpers ---


def read_matrix(odb_path: Path) -> MatrixData:
    """Read and parse matrix file from ODB++ directory.

    Parameters
    ----------
    odb_path : Path
        Path to ODB++ root directory.

    Returns
    -------
    MatrixData
        Parsed matrix data.
    """
    matrix_path = odb_path / "matrix" / "matrix"
    content = matrix_path.read_text(encoding="utf-8", errors="ignore")
    return parse_matrix(content)


def read_features(odb_path: Path, step_name: str, layer_name: str) -> FeaturesData:
    """Read and parse features file for a specific layer.

    Parameters
    ----------
    odb_path : Path
        Path to ODB++ root directory.
    step_name : str
        Step name.
    layer_name : str
        Layer name.

    Returns
    -------
    FeaturesData
        Parsed features data, or empty FeaturesData if file doesn't exist.
    """
    features_path = odb_path / "steps" / step_name / "layers" / layer_name / "features"
    if not features_path.exists():
        return FeaturesData()
    content = features_path.read_text(encoding="utf-8", errors="ignore")
    return parse_features(content)


def iter_layers(odb_path: Path, step_name: str) -> Iterator[str]:
    """Iterate layer names for a step.

    Parameters
    ----------
    odb_path : Path
        Path to ODB++ root directory.
    step_name : str
        Step name.

    Yields
    ------
    str
        Layer names.
    """
    layers_dir = odb_path / "steps" / step_name / "layers"
    if layers_dir.exists():
        for layer_dir in layers_dir.iterdir():
            if layer_dir.is_dir():
                yield layer_dir.name


def read_profile(
    odb_path: Path, step_name: str, layer_name: Optional[str] = None
) -> Optional[SurfaceRecord]:
    """Read and parse profile file (board outline) for a step or layer.

    The profile file defines the board outline as a Surface record. This is
    used to auto-fill dielectric layers that have no explicit geometry.

    Parameters
    ----------
    odb_path : Path
        Path to ODB++ root directory.
    step_name : str
        Step name.
    layer_name : str, optional
        Layer name. If provided, looks for layer-specific profile first,
        then falls back to step profile.

    Returns
    -------
    Optional[SurfaceRecord]
        Parsed profile as a Surface record, or None if not found.

    Notes
    -----
    Profile lookup order:
    1. Layer-specific: steps/<step>/layers/<layer>/profile
    2. Step-level: steps/<step>/profile
    """
    # Try layer-specific profile first
    if layer_name:
        layer_profile_path = (
            odb_path / "steps" / step_name / "layers" / layer_name / "profile"
        )
        if layer_profile_path.exists():
            content = layer_profile_path.read_text(encoding="utf-8", errors="ignore")
            return _parse_profile_content(content)

    # Fall back to step-level profile
    step_profile_path = odb_path / "steps" / step_name / "profile"
    if step_profile_path.exists():
        content = step_profile_path.read_text(encoding="utf-8", errors="ignore")
        return _parse_profile_content(content)

    return None


def _parse_profile_content(content: str) -> Optional[SurfaceRecord]:
    """Parse profile file content into a SurfaceRecord.

    Parameters
    ----------
    content : str
        Raw profile file content.

    Returns
    -------
    Optional[SurfaceRecord]
        Parsed Surface record, or None if parsing failed.
    """
    # Profile files have the same format as features files but typically
    # contain just one Surface record representing the board outline
    data = parse_features(content)

    # Return the first Surface record found
    for feature in data.features:
        if isinstance(feature, SurfaceRecord):
            return feature

    return None


@dataclass
class ProfileData:
    """Parsed profile file data.

    Attributes
    ----------
    units : str
        "MM" or "INCH".
    surface : Optional[SurfaceRecord]
        Board outline as a Surface record.
    """

    units: str = "MM"
    surface: Optional[SurfaceRecord] = None


@dataclass
class EDAData:
    """Parsed EDA data file for net assignments.

    Attributes
    ----------
    units : str
        "MM" or "INCH".
    layer_names : list[str]
        Layer names in index order from LYR line.
    net_assignments : dict[tuple[str, int], str]
        Mapping from (layer_name, feature_index) to net name.
    """

    units: str = "MM"
    layer_names: list[str] = field(default_factory=list)
    net_assignments: dict[tuple[str, int], str] = field(default_factory=dict)


def read_profile_data(
    odb_path: Path, step_name: str, layer_name: Optional[str] = None
) -> ProfileData:
    """Read and parse profile file with units information.

    Parameters
    ----------
    odb_path : Path
        Path to ODB++ root directory.
    step_name : str
        Step name.
    layer_name : str, optional
        Layer name for layer-specific profile lookup.

    Returns
    -------
    ProfileData
        Parsed profile data including units.
    """
    result = ProfileData()

    # Try layer-specific profile first
    if layer_name:
        layer_profile_path = (
            odb_path / "steps" / step_name / "layers" / layer_name / "profile"
        )
        if layer_profile_path.exists():
            content = layer_profile_path.read_text(encoding="utf-8", errors="ignore")
            data = parse_features(content)
            result.units = data.units
            for feature in data.features:
                if isinstance(feature, SurfaceRecord):
                    result.surface = feature
                    return result

    # Fall back to step-level profile
    step_profile_path = odb_path / "steps" / step_name / "profile"
    if step_profile_path.exists():
        content = step_profile_path.read_text(encoding="utf-8", errors="ignore")
        data = parse_features(content)
        result.units = data.units
        for feature in data.features:
            if isinstance(feature, SurfaceRecord):
                result.surface = feature
                return result

    return result


def parse_eda_data(content: str) -> EDAData:
    """Parse EDA data file content for net assignments.

    Parameters
    ----------
    content : str
        Raw text content of eda/data file.

    Returns
    -------
    EDAData
        Parsed net assignment data.

    Notes
    -----
    EDA data format:
    - `LYR <layer1> <layer2> ...` - Layer names in index order
    - `NET <name>` - Start of a net block
    - `FID <type> <layer_idx> <feature_idx>` - Feature assignment
      - type: C (copper), L (line), H (hole), etc.

    Example
    -------
    >>> content = open("design.odb/steps/pcb/eda/data").read()
    >>> eda = parse_eda_data(content)
    >>> net = eda.net_assignments.get(("TRACE", 0))  # Get net for feature 0 on TRACE
    """
    data = EDAData()
    lines = content.split("\n")
    current_net: Optional[str] = None

    for line in lines:
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue

        # Units
        if line.startswith("UNITS="):
            data.units = line.split("=")[1].strip()
            continue

        # Layer index mapping: LYR <layer1> <layer2> ...
        if line.startswith("LYR "):
            # Parse layer names (space-separated after "LYR ")
            layer_part = line[4:].strip()
            data.layer_names = layer_part.split()
            continue

        # Net definition: NET <name> [;;ID=xxx]
        if line.startswith("NET "):
            # Extract net name (before any ";;" attributes)
            net_part = line[4:].split(";;")[0].strip()
            current_net = net_part
            continue

        # Feature ID: FID <type> <layer_idx> <feature_idx>
        if line.startswith("FID ") and current_net is not None:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # parts[1] = type (C, L, H, etc.)
                    layer_idx = int(parts[2])
                    feature_idx = int(parts[3])

                    # Map layer index to layer name
                    if 0 <= layer_idx < len(data.layer_names):
                        layer_name = data.layer_names[layer_idx]
                        data.net_assignments[(layer_name, feature_idx)] = current_net
                except (ValueError, IndexError):
                    pass
            continue

    return data


def read_eda_data(odb_path: Path, step_name: str) -> EDAData:
    """Read and parse EDA data file for net assignments.

    Parameters
    ----------
    odb_path : Path
        Path to ODB++ root directory.
    step_name : str
        Step name.

    Returns
    -------
    EDAData
        Parsed EDA data, or empty EDAData if file doesn't exist.
    """
    eda_path = odb_path / "steps" / step_name / "eda" / "data"
    if not eda_path.exists():
        return EDAData()

    content = eda_path.read_text(encoding="utf-8", errors="ignore")
    return parse_eda_data(content)

