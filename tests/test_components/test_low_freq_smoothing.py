"""Test low frequency smoothing specification."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import tidy3d as td


def test_low_freq_smoothing_spec_initialization_default_values():
    """Test that LowFrequencySmoothingSpec initializes with correct default values."""
    spec = td.LowFrequencySmoothingSpec(monitors=["monitor1"])
    assert spec.min_sampling_time == 1
    assert spec.max_sampling_time == 5
    assert spec.order == 1
    assert spec.max_deviation == 0.5


def test_empty_monitors():
    """Test that LowFrequencySmoothingSpec raises an error if monitors are not provided."""
    with pytest.raises(ValidationError):
        td.LowFrequencySmoothingSpec(monitors=[])


def test_monitors_exist():
    """Test that LowFrequencySmoothingSpec raises an error if monitors do not exist."""
    sim = td.Simulation(
        size=(1, 1, 1),
        monitors=[
            td.ModeMonitor(
                name="monitor1",
                mode_spec=td.ModeSpec(num_modes=1),
                freqs=[td.C_0],
                size=(0.5, 0.5, 0),
            )
        ],
        run_time=1e-12,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=20, wavelength=1),
        low_freq_smoothing=td.LowFrequencySmoothingSpec(monitors=["monitor1"]),
    )

    with pytest.raises(ValidationError):
        sim = td.Simulation(
            size=(1, 1, 1),
            monitors=[],
            run_time=1e-12,
            grid_spec=td.GridSpec.auto(min_steps_per_wvl=20, wavelength=1),
            low_freq_smoothing=td.LowFrequencySmoothingSpec(monitors=["monitor1"]),
        )
