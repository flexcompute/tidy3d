"""Tests microwave tools."""

from math import isclose

import numpy as np
from tidy3d.components.microwave.formulas.circuit_parameters import (
    capacitance_colinear_cylindrical_wire_segments,
    capacitance_rectangular_sheets,
    inductance_straight_rectangular_wire,
    mutual_inductance_colinear_wire_segments,
    total_inductance_colinear_rectangular_wire_segments,
)
from tidy3d.constants import EPSILON_0


def test_inductance_formulas():
    """Run the formulas for inductance and compare to precomputed results."""
    bar_size = (1000e4, 1e4, 1e4)  # case from reference
    L1 = inductance_straight_rectangular_wire(bar_size, 0)
    assert isclose(L1, 14.816e-6, rel_tol=1e-4)
    length = 1e3
    L2 = mutual_inductance_colinear_wire_segments(length, length, length / 10)
    assert isclose(L2, 0.11181e-9, rel_tol=1e-4)
    side = length / 10
    L3 = total_inductance_colinear_rectangular_wire_segments(
        (side, length, side), (side, length, side), length / 10, 1
    )
    assert isclose(L3, 1.3625e-9, rel_tol=1e-4)


def test_capacitance_formulas():
    """Run the formulas for capacitance and compare to precomputed results."""
    width = 3e3
    length = 1e3
    d = length / 4.5  # case from reference
    C1 = capacitance_rectangular_sheets(width, length, d)
    result = 2.347 * EPSILON_0 * width  # from reference
    assert isclose(C1, result, rel_tol=1e-3)

    # case from reference
    radius = 0.1e-3
    C2 = capacitance_colinear_cylindrical_wire_segments(radius, length, length / 5)
    D2 = 0.345
    C_ref = np.pi * EPSILON_0 * length / (np.log(length / radius) - 2.303 * D2)
    assert isclose(C2, C_ref, rel_tol=1e-3)

    # case from reference
    C3 = capacitance_colinear_cylindrical_wire_segments(radius, length, length * 5)
    D2 = 0.144
    C_ref = np.pi * EPSILON_0 * length / (np.log(length / radius) - 2.303 * D2)
    assert isclose(C3, C_ref, rel_tol=1e-2)
