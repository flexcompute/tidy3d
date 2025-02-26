"""Exact and approximate formulas in microwave engineering.

References
----------
[1]     F. E. Terman, Radio engineers' handbook, McGraw-Hill Book Company, Inc., 1943.

[2]     E. B. Rosa, The self and mutal inductances of linear conductors, US Department of
        Commerce and Labor, Bureau of Standards, 1908.

[3]     Y. Y. Iossel, E. S. Kochanov, and M. G. Strunskii, “The Calculation of Electrical Capacitance,”
        Foreign Technology Division Air Force Systems Command U.S. Air Force, 1971.
"""

import numpy as np

from ....constants import EPSILON_0
from ...geometry.base import Geometry
from ...types import Axis


def inductance_straight_rectangular_wire(
    size: tuple[float, float, float], current_axis: Axis
) -> float:
    """Computes the self-inductance of a finite length of wire with a rectangular cross section.
    Equation 26 [1] and Equation 21 [2].

    Parameters
    ----------
    size :
        Tuple representing the size of the rectangular wire segment in microns.
    current_axis :
        The axis along which the current is flowing.

    Returns
    -------
    float
        The self-inductance in Henrys."""

    length, transverse_sizes = Geometry.pop_axis(size, current_axis)
    b_plus_c = transverse_sizes[0] + transverse_sizes[1]
    log_term = np.log(2 * length / b_plus_c) + 0.5 + 0.2235 * (b_plus_c / length)
    L0 = 2 * (length) * log_term * 1e-13
    return L0


def mutual_inductance_colinear_wire_segments(l1: float, l2: float, d: float) -> float:
    """Computes the mutual inductance between two wire segments arranged in the same line,
    Equation 73 [1].

    Parameters
    ----------
    l1 :
        Length of the first segment in microns.
    l2 :
        Length of the second segment in microns.
    d :
        Separation distance in microns between segment end points.

    Returns
    -------
    float
        The mutual inductance in Henrys."""

    sum_all = l1 + l2 + d
    sum_l1 = l1 + d
    sum_l2 = l2 + d
    M = (
        1e-7
        * (
            sum_all * np.log(sum_all)
            + d * np.log(d)
            - sum_l1 * np.log(sum_l1)
            - sum_l2 * np.log(sum_l2)
        )
        * 1e-6
    )
    return M


def total_inductance_colinear_rectangular_wire_segments(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    d: float,
    current_axis: Axis,
) -> float:
    """Computes the total inductance of a pair of finite length of colinear wire segments with a
    rectangular cross section, Equation 69 [1].

    Parameters
    ----------
    first :
        Size of the first rectangular wire segment.
    second :
        Size of the the second rectangular wire segment.
    d :
        Separation distance in microns between wire end points.
    current_axis :
        The axis along which the current is flowing.

    Returns
    -------
    float
        The total inductance in Henrys."""

    Lfirst = inductance_straight_rectangular_wire(first, current_axis)
    Lsecond = inductance_straight_rectangular_wire(second, current_axis)
    M = mutual_inductance_colinear_wire_segments(first[current_axis], second[current_axis], d)
    return Lfirst + Lsecond + 2 * M


def capacitance_rectangular_sheets(
    width: float, length: float, d: float, eps_r: float = 1
) -> float:
    """Computes the capacitance between a pair of rectangular sheets, Equation 4-28 [3].

    Parameters
    ----------
    width :
        Width of sheets in microns.
    length :
        Length of each sheet in microns.
    d :
        Separation distance in microns between wire end points.

    Returns
    -------
    float
        The capacitance in Farads."""
    # Approximation valid when sheets are thin, width >> length, and length >> d
    C = 2 / np.pi * EPSILON_0 * eps_r * width * np.log(4 * (1 + 2 * length / d))
    return C


def capacitance_colinear_cylindrical_wire_segments(
    radius: float, length: float, d: float, eps_r: float = 1
) -> float:
    """Computes the capacitance between a pair of colinear cylindrical wires of finite length,
    Equation 3-43 [3].

    Parameters
    ----------
    radius :
        Radius of the wire segment in microns.
    length :
        Length of each wire segment in microns.
    d :
        Separation distance in microns between wire end points.

    Returns
    -------
    float
        The capacitance in Farads."""
    # Approximation valid when wires are thin and long (radius << length)

    m = d / 2
    if m < length:
        ratio = m / length
        D2 = (
            0.434
            + ratio * np.log10(4 * ratio)
            + (1 + ratio) * np.log10(1 + ratio)
            - (1 + 2 * ratio) * np.log10(1 + 2 * ratio)
        )
    else:
        ml = m / length
        lm = length / m
        D2 = (
            0.133
            + ml * (1 + lm) * np.log10(1 + lm)
            - (2 * ml) * (1 + lm / 2) * np.log10(1 + lm / 2)
        )
    C = np.pi * EPSILON_0 * eps_r * length / (np.log(length / radius) - 2.303 * D2)
    return C
