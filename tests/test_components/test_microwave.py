"""Tests microwave tools."""

from __future__ import annotations

from math import isclose

import numpy as np
import pytest
import xarray as xr

from tidy3d.components.data.monitor_data import FreqDataArray
from tidy3d.components.microwave.data.monitor_data import AntennaMetricsData
from tidy3d.components.microwave.formulas.circuit_parameters import (
    capacitance_colinear_cylindrical_wire_segments,
    capacitance_rectangular_sheets,
    inductance_straight_rectangular_wire,
    mutual_inductance_colinear_wire_segments,
    total_inductance_colinear_rectangular_wire_segments,
)
from tidy3d.constants import EPSILON_0

from ..test_data.test_monitor_data import make_directivity_data


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


def test_antenna_parameters():
    """Test basic antenna parameters computation and validation."""

    # Create from random directivity data
    directivity_data = make_directivity_data()
    f = directivity_data.coords["f"]
    power_inc = FreqDataArray(0.8 * np.ones(len(f)), coords={"f": f})
    power_refl = 0.25 * power_inc
    antenna_params = AntennaMetricsData.from_directivity_data(
        directivity_data, power_inc, power_refl
    )

    # Test that all essential parameters exist and are correct type
    assert isinstance(antenna_params.radiation_efficiency, FreqDataArray)
    assert isinstance(antenna_params.reflection_efficiency, FreqDataArray)
    assert np.allclose(antenna_params.reflection_efficiency, 0.75)
    assert isinstance(antenna_params.gain, xr.DataArray)
    assert isinstance(antenna_params.realized_gain, xr.DataArray)

    # Test partial gain computations in linear basis
    partial_gain_linear = antenna_params.partial_gain(pol_basis="linear")
    assert isinstance(partial_gain_linear, xr.Dataset)
    assert "Gtheta" in partial_gain_linear
    assert "Gphi" in partial_gain_linear

    # Test partial gain computations in linear basis with tilt angle = 0 matches partial gain in the original basis
    partial_gain_linear_tilted = antenna_params.partial_gain(pol_basis="linear", tilt_angle=0)
    assert isinstance(partial_gain_linear_tilted, xr.Dataset)
    assert "Gco" in partial_gain_linear_tilted
    assert "Gcross" in partial_gain_linear_tilted
    assert np.allclose(partial_gain_linear_tilted.Gco, partial_gain_linear.Gtheta)
    assert np.allclose(partial_gain_linear_tilted.Gcross, partial_gain_linear.Gphi)

    # Test validation of tilt angle that only works with linear basis
    with pytest.raises(ValueError):
        antenna_params.partial_gain(pol_basis="circular", tilt_angle=1)

    # Test partial gain computations in circular basis
    partial_gain_circular = antenna_params.partial_gain(pol_basis="circular")
    assert isinstance(partial_gain_circular, xr.Dataset)
    assert "Gright" in partial_gain_circular
    assert "Gleft" in partial_gain_circular

    # Test partial realized gain computations in both bases
    assert isinstance(antenna_params.partial_realized_gain("linear"), xr.Dataset)
    assert isinstance(antenna_params.partial_realized_gain("circular"), xr.Dataset)

    # Test validation of pol_basis parameter
    with pytest.raises(ValueError):
        antenna_params.partial_gain("invalid")
    with pytest.raises(ValueError):
        antenna_params.partial_realized_gain("invalid")
