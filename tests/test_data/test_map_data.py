from __future__ import annotations

import collections.abc

import pytest
from pydantic import ValidationError

from tidy3d import SimulationDataMap

from ..utils import SAMPLE_SIMULATIONS, AssertLogStr, run_emulated

# Reusable test constants
# Base simulation data is needed to generate the SimulationData objects
SIM_MAP_DATA = {
    "sim_A": SAMPLE_SIMULATIONS["full_fdtd"],
    "sim_B": SAMPLE_SIMULATIONS["full_fdtd"].updated_copy(run_time=2e-12),
}

# Generate SimulationData once to be reused
SIM_DATA_MAP_DATA = {
    "data_A": run_emulated(SIM_MAP_DATA["sim_A"]),
    "data_B": run_emulated(SIM_MAP_DATA["sim_B"]),
}


def make_simulation_data_map() -> SimulationDataMap:
    """Factory function to create a standard SimulationDataMap instance."""
    return SimulationDataMap(
        keys=tuple(SIM_DATA_MAP_DATA.keys()), values=tuple(SIM_DATA_MAP_DATA.values())
    )


def test_simulation_data_map_creation():
    """Tests successful creation and basic properties of a SimulationDataMap."""
    sd_map = make_simulation_data_map()
    assert isinstance(sd_map, collections.abc.Mapping)
    assert len(sd_map) == len(SIM_DATA_MAP_DATA)
    assert sd_map["data_A"] == SIM_DATA_MAP_DATA["data_A"]
    assert list(sd_map.keys()) == list(SIM_DATA_MAP_DATA.keys())


def test_simulation_data_map_invalid_type_raises_error():
    """Tests that a ValidationError is raised for incorrect value types."""
    invalid_data = {"data_A": "not simulation data"}
    with pytest.raises(ValidationError):
        SimulationDataMap(keys=tuple(invalid_data.keys()), values=tuple(invalid_data.values()))


def test_simulation_data_map_mismatched_lengths_raises_error():
    """Tests that a ValidationError is raised for mismatched keys and values lengths."""
    with pytest.raises(ValidationError, match="Length of 'keys' and 'values' must be the same"):
        SimulationDataMap(keys=("data_A",), values=tuple(SIM_DATA_MAP_DATA.values()))


def test_simulation_data_map_getitem_success():
    """Tests successful retrieval of simulation data by its string key."""
    sd_map = make_simulation_data_map()
    assert sd_map["data_A"] == SIM_DATA_MAP_DATA["data_A"]
    assert sd_map["data_B"] == SIM_DATA_MAP_DATA["data_B"]


def test_simulation_data_map_getitem_key_error():
    """Tests that a KeyError is raised when retrieving a non-existent key."""
    sd_map = make_simulation_data_map()
    with pytest.raises(KeyError):
        _ = sd_map["not_a_real_key"]


def test_simulation_data_map_roundtrip_no_spurious_errors():
    """Regression test for FXC-5469: deserializing SimulationDataMap should not
    produce spurious ERROR logs from pydantic trying ModeSimulation validators."""
    sd_map = make_simulation_data_map()

    with AssertLogStr("ERROR", excludes_str=""):
        SimulationDataMap.model_validate(sd_map.model_dump())
