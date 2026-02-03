"""ODB++ utility functions for unit conversion and arc-to-bulge conversion."""

from __future__ import annotations

from math import atan2, pi, sqrt, tan
from typing import Literal

# Type aliases
Coordinate2D = tuple[float, float]
Units = Literal["MM", "INCH"]


def convert_to_microns(
    value: float,
    units: Units,
    is_symbol_dim: bool = False,
) -> float:
    """Convert ODB++ value to microns.

    Parameters
    ----------
    value : float
        Value in ODB++ units.
    units : Units
        "MM" or "INCH" from UNITS directive.
    is_symbol_dim : bool
        If True, value is a symbol dimension (already in microns/mils).
        If False, value is a coordinate (in mm/inches).

    Returns
    -------
    float
        Value in microns.

    Notes
    -----
    ODB++ symbol dimensions (e.g., r200 = round with diameter 200) are stored
    in sub-units: microns for MM mode, mils for INCH mode. Coordinates are
    stored in the main unit (mm or inches).

    For MM mode:
        - Coordinates: multiply by 1000 (mm → µm)
        - Symbol dims: already in microns, no conversion needed

    For INCH mode:
        - Coordinates: multiply by 25400 (inches → µm)
        - Symbol dims: multiply by 25.4 (mils → µm)
    """
    if units == "MM":
        if is_symbol_dim:
            return value  # Symbol dims in MM mode are already microns
        else:
            return value * 1000.0  # mm → µm
    else:  # INCH
        if is_symbol_dim:
            return value * 25.4  # mils → µm
        else:
            return value * 25400.0  # inches → µm


def arc_to_bulge(
    start: Coordinate2D,
    end: Coordinate2D,
    center: Coordinate2D,
    clockwise: bool,
) -> float:
    """Convert ODB++ arc (start, end, center, cw) to DXF bulge.

    Parameters
    ----------
    start : Coordinate2D
        Arc start point.
    end : Coordinate2D
        Arc end point.
    center : Coordinate2D
        Arc center point.
    clockwise : bool
        True if arc goes clockwise.

    Returns
    -------
    float
        DXF bulge value: tan(included_angle/4), positive for CCW.

    Notes
    -----
    Full circles (start ≈ end) return bulge = 1.0 (semicircle approximation).
    Caller should detect this case and split into two arcs if needed.

    The bulge convention:
    - Positive bulge: CCW arc (bulges to the LEFT of edge direction)
    - Negative bulge: CW arc (bulges to the RIGHT of edge direction)
    - |bulge| = 1.0: semicircle
    """
    # Check for full circle (start ≈ end)
    dist_start_end = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    radius = sqrt((start[0] - center[0]) ** 2 + (start[1] - center[1]) ** 2)

    if radius < 1e-12:
        return 0.0  # Degenerate arc

    if dist_start_end < radius * 1e-9:
        # Full circle - return semicircle bulge (will need to split externally)
        return -1.0 if clockwise else 1.0

    # Calculate angles from center
    angle_start = atan2(start[1] - center[1], start[0] - center[0])
    angle_end = atan2(end[1] - center[1], end[0] - center[0])

    # Calculate arc angle (CCW positive convention)
    theta = angle_end - angle_start

    if clockwise:
        if theta > 0:
            theta -= 2 * pi
    else:
        if theta < 0:
            theta += 2 * pi

    # Handle near-zero angles
    if abs(theta) < 1e-12:
        return 0.0

    # Bulge = tan(|θ|/4), sign matches CCW=positive
    bulge = tan(abs(theta) / 4)
    return -bulge if clockwise else bulge


def is_full_circle(
    start: Coordinate2D,
    end: Coordinate2D,
    center: Coordinate2D,
    tolerance: float = 1e-9,
) -> bool:
    """Check if arc represents a full circle (start ≈ end).

    Parameters
    ----------
    start : Coordinate2D
        Arc start point.
    end : Coordinate2D
        Arc end point.
    center : Coordinate2D
        Arc center point (used to calculate relative tolerance).
    tolerance : float
        Relative tolerance for comparing start and end positions.

    Returns
    -------
    bool
        True if the arc is a full circle.
    """
    dist_start_end = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    radius = sqrt((start[0] - center[0]) ** 2 + (start[1] - center[1]) ** 2)

    if radius < 1e-12:
        return False

    return dist_start_end < radius * tolerance


def split_full_circle(
    start: Coordinate2D,
    center: Coordinate2D,
    clockwise: bool,
) -> tuple[Coordinate2D, float, float]:
    """Split a full circle into two semicircles.

    Parameters
    ----------
    start : Coordinate2D
        Start point of the full circle (also the end point).
    center : Coordinate2D
        Center of the circle.
    clockwise : bool
        True if arc goes clockwise.

    Returns
    -------
    tuple[Coordinate2D, float, float]
        (midpoint, bulge1, bulge2) where midpoint is the point opposite
        to start, and bulge1/bulge2 are the bulges for the two semicircles.

    Notes
    -----
    The first semicircle goes from start to midpoint, the second from
    midpoint back to start. Both semicircles have |bulge| = 1.0.
    """
    # Midpoint is diametrically opposite to start
    dx = start[0] - center[0]
    dy = start[1] - center[1]
    midpoint = (center[0] - dx, center[1] - dy)

    # Both semicircles have bulge magnitude of 1.0
    bulge = -1.0 if clockwise else 1.0

    return midpoint, bulge, bulge

