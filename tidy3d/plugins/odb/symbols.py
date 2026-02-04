"""ODB++ standard symbol interpreter.

Parses ODB++ symbol names (e.g., "r200", "rect250x150", "oval100x50") into
structured parameter dictionaries for geometry construction.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

# Symbol type literals
SymbolType = Literal[
    "round", "square", "rect", "rounded_rect", "chamfered_rect",
    "oval", "diamond", "octagon",
    "donut_r", "donut_s", "donut_sr", "unknown"
]


@dataclass
class SymbolInfo:
    """Parsed ODB++ symbol information.

    Attributes
    ----------
    type : SymbolType
        Type of symbol: "round", "square", "rect", "rounded_rect",
        "chamfered_rect", "oval", "diamond", "octagon", "donut_r",
        "donut_s", "donut_sr", or "unknown".
    params : dict[str, float]
        Symbol parameters. Keys depend on type:
        - round: {"diameter": float}
        - square: {"side": float}
        - rect: {"width": float, "height": float}
        - rounded_rect: {"width", "height", "corner_radius", "corners"}
        - chamfered_rect: {"width", "height", "corner_radius", "corners"}
        - oval: {"width": float, "height": float}
        - diamond: {"width": float, "height": float}
        - octagon: {"width": float, "height": float, "corner": float}
        - donut_r: {"outer_diameter", "inner_diameter"}
        - donut_s: {"outer_side", "inner_side"}
        - donut_sr: {"outer_side", "inner_diameter"}
        - unknown: {"name": str}
    rotation : float
        Rotation in degrees (ODB++ v7+). Default 0.0.
    """

    type: SymbolType
    params: dict = field(default_factory=dict)
    rotation: float = 0.0

    @property
    def is_symmetric(self) -> bool:
        """True if symbol can be used for lines/arcs (round or square)."""
        return self.type in ("round", "square")

    @property
    def width(self) -> Optional[float]:
        """Width/diameter for line strokes, or None if not applicable."""
        if self.type == "round":
            return self.params.get("diameter")
        elif self.type == "square":
            return self.params.get("side")
        return None


# Symbol parsing patterns
# Format: prefix<num>, prefix<w>x<h>, etc.
# Optional rotation suffix: _<angle>
# Note: Order matters - more specific patterns must come before general ones
_PATTERNS: dict[str, re.Pattern] = {
    # r<d> or r<d>_<rotation>
    "round": re.compile(r"^r(\d+(?:\.\d+)?)(?:_(\d+(?:\.\d+)?))?$"),
    # s<s> or s<s>_<rotation>
    "square": re.compile(r"^s(\d+(?:\.\d+)?)(?:_(\d+(?:\.\d+)?))?$"),
    # rect<w>x<h>xr<rad> - rounded rectangle (must come before plain rect)
    # Optional corner selection: x<1234> where digits indicate which corners
    "rounded_rect": re.compile(
        r"^rect(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)xr(\d+(?:\.\d+)?)(?:x(\d+))?(?:_(\d+(?:\.\d+)?))?$"
    ),
    # rect<w>x<h>xc<rad> - chamfered rectangle (must come before plain rect)
    # Optional corner selection: x<1234> where digits indicate which corners
    "chamfered_rect": re.compile(
        r"^rect(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)xc(\d+(?:\.\d+)?)(?:x(\d+))?(?:_(\d+(?:\.\d+)?))?$"
    ),
    # rect<w>x<h> - plain rectangle
    "rect": re.compile(
        r"^rect(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(?:_(\d+(?:\.\d+)?))?$"
    ),
    # oval<w>x<h>
    "oval": re.compile(r"^oval(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(?:_(\d+(?:\.\d+)?))?$"),
    # di<w>x<h> (diamond)
    "diamond": re.compile(r"^di(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(?:_(\d+(?:\.\d+)?))?$"),
    # oct<w>x<h>x<r> (octagon)
    "octagon": re.compile(
        r"^oct(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(?:_(\d+(?:\.\d+)?))?$"
    ),
    # donut_r<outer>x<inner> (round donut/annular ring)
    "donut_r": re.compile(r"^donut_r(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(?:_(\d+(?:\.\d+)?))?$"),
    # donut_s<outer>x<inner> (square donut)
    "donut_s": re.compile(r"^donut_s(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(?:_(\d+(?:\.\d+)?))?$"),
    # donut_sr<outer>x<inner> (square with round hole)
    "donut_sr": re.compile(r"^donut_sr(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(?:_(\d+(?:\.\d+)?))?$"),
}

# Extensible registry for custom symbol types (dogbone, thermal, etc.)
_CUSTOM_PARSERS: dict[str, Callable[[str], Optional[SymbolInfo]]] = {}


def register_symbol_parser(prefix: str, parser: Callable[[str], Optional[SymbolInfo]]) -> None:
    """Register a custom symbol parser for future extensions.

    Parameters
    ----------
    prefix : str
        Symbol prefix to match (e.g., "dogbone", "thermal", "donut_r").
    parser : Callable[[str], Optional[SymbolInfo]]
        Function that takes a symbol name string and returns SymbolInfo
        if it matches, or None if it doesn't.

    Example
    -------
    >>> def parse_dogbone(name: str) -> Optional[SymbolInfo]:
    ...     # Parse dogbone<w>x<h>x<hs>x<vs>x<hc>xs pattern
    ...     match = re.match(r"^dogbone(\\d+)x...", name)
    ...     if match:
    ...         return SymbolInfo(type="unknown", params={"name": name})
    ...     return None
    >>>
    >>> register_symbol_parser("dogbone", parse_dogbone)
    """
    _CUSTOM_PARSERS[prefix] = parser


def parse_symbol(name: str) -> SymbolInfo:
    """Parse ODB++ symbol name into structured info.

    Parameters
    ----------
    name : str
        Symbol name, e.g., "r200", "rect250x150", "s29.62".

    Returns
    -------
    SymbolInfo
        Parsed symbol with type and dimension parameters.
        Returns type="unknown" for unrecognized symbols.

    Notes
    -----
    Dimensions are in symbol units (microns for MM, mils for INCH).
    Caller must apply unit conversion using `convert_to_microns()`.

    Supported standard symbols:
    - r<d>: round (circle) with diameter d
    - s<s>: square with side s
    - rect<w>x<h>: rectangle with width w, height h
    - rect<w>x<h>xr<rad>: rounded rectangle with corner radius rad
    - rect<w>x<h>xc<rad>: chamfered rectangle with corner chamfer rad
    - oval<w>x<h>: oval (stadium shape)
    - di<w>x<h>: diamond (rhombus)
    - oct<w>x<h>x<r>: octagon with corner size r
    - donut_r<od>x<id>: round donut (annular ring)
    - donut_s<od>x<id>: square donut
    - donut_sr<od>x<id>: square with round hole

    Example
    -------
    >>> info = parse_symbol("r200")
    >>> info.type
    'round'
    >>> info.params["diameter"]
    200.0
    >>>
    >>> info = parse_symbol("rect250x150")
    >>> info.type
    'rect'
    >>> info.params["width"], info.params["height"]
    (250.0, 150.0)
    """
    # Strip unit suffix if present (e.g., "oval210x170 M" or "rect3x5 I")
    parts = name.split()
    sym_name = parts[0]
    # Note: unit suffix is ignored here - caller uses file UNITS

    # Try standard patterns
    for sym_type, pattern in _PATTERNS.items():
        match = pattern.match(sym_name)
        if match:
            return _build_symbol_info(sym_type, match)

    # Try custom parsers
    for prefix, parser in _CUSTOM_PARSERS.items():
        if sym_name.startswith(prefix):
            result = parser(sym_name)
            if result is not None:
                return result

    # Unknown symbol
    warnings.warn(
        f"Unknown ODB++ symbol: '{name}'. Geometry will be skipped.",
        stacklevel=2,
    )
    return SymbolInfo(type="unknown", params={"name": name})


def _build_symbol_info(sym_type: str, match: re.Match) -> SymbolInfo:
    """Build SymbolInfo from regex match.

    Parameters
    ----------
    sym_type : str
        Symbol type string.
    match : re.Match
        Regex match object with captured groups.

    Returns
    -------
    SymbolInfo
        Constructed symbol info.
    """
    groups = match.groups()

    if sym_type == "round":
        return SymbolInfo(
            type="round",
            params={"diameter": float(groups[0])},
            rotation=float(groups[1]) if groups[1] else 0.0,
        )

    elif sym_type == "square":
        return SymbolInfo(
            type="square",
            params={"side": float(groups[0])},
            rotation=float(groups[1]) if groups[1] else 0.0,
        )

    elif sym_type == "rounded_rect":
        # rect<w>x<h>xr<rad> with optional corners x<1234>
        return SymbolInfo(
            type="rounded_rect",
            params={
                "width": float(groups[0]),
                "height": float(groups[1]),
                "corner_radius": float(groups[2]),
                "corners": groups[3] if groups[3] else "1234",  # All corners by default
            },
            rotation=float(groups[4]) if groups[4] else 0.0,
        )

    elif sym_type == "chamfered_rect":
        # rect<w>x<h>xc<rad> with optional corners x<1234>
        return SymbolInfo(
            type="chamfered_rect",
            params={
                "width": float(groups[0]),
                "height": float(groups[1]),
                "corner_radius": float(groups[2]),
                "corners": groups[3] if groups[3] else "1234",  # All corners by default
            },
            rotation=float(groups[4]) if groups[4] else 0.0,
        )

    elif sym_type == "rect":
        # Plain rect<w>x<h>
        return SymbolInfo(
            type="rect",
            params={
                "width": float(groups[0]),
                "height": float(groups[1]),
            },
            rotation=float(groups[2]) if groups[2] else 0.0,
        )

    elif sym_type == "oval":
        return SymbolInfo(
            type="oval",
            params={
                "width": float(groups[0]),
                "height": float(groups[1]),
            },
            rotation=float(groups[2]) if groups[2] else 0.0,
        )

    elif sym_type == "diamond":
        return SymbolInfo(
            type="diamond",
            params={
                "width": float(groups[0]),
                "height": float(groups[1]),
            },
            rotation=float(groups[2]) if groups[2] else 0.0,
        )

    elif sym_type == "octagon":
        return SymbolInfo(
            type="octagon",
            params={
                "width": float(groups[0]),
                "height": float(groups[1]),
                "corner": float(groups[2]),
            },
            rotation=float(groups[3]) if groups[3] else 0.0,
        )

    elif sym_type == "donut_r":
        return SymbolInfo(
            type="donut_r",
            params={
                "outer_diameter": float(groups[0]),
                "inner_diameter": float(groups[1]),
            },
            rotation=float(groups[2]) if groups[2] else 0.0,
        )

    elif sym_type == "donut_s":
        return SymbolInfo(
            type="donut_s",
            params={
                "outer_side": float(groups[0]),
                "inner_side": float(groups[1]),
            },
            rotation=float(groups[2]) if groups[2] else 0.0,
        )

    elif sym_type == "donut_sr":
        # Square with round hole
        return SymbolInfo(
            type="donut_sr",
            params={
                "outer_side": float(groups[0]),
                "inner_diameter": float(groups[1]),
            },
            rotation=float(groups[2]) if groups[2] else 0.0,
        )

    raise ValueError(f"Unhandled symbol type: {sym_type}")

