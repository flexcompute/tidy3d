from __future__ import annotations

import collections.abc

import pytest
from pydantic import ValidationError

from tidy3d import SimulationMap

from ..utils import SAMPLE_SIMULATIONS, AssertLogStr

# Reusable test constants
SIM_MAP_DATA = {
    "sim_A": SAMPLE_SIMULATIONS["full_fdtd"],
    "sim_B": SAMPLE_SIMULATIONS["full_fdtd"].updated_copy(run_time=2e-12),
}


def make_simulation_map() -> SimulationMap:
    """Factory function to create a standard SimulationMap instance."""
    return SimulationMap(keys=tuple(SIM_MAP_DATA.keys()), values=tuple(SIM_MAP_DATA.values()))


def test_simulation_map_creation():
    """Tests successful creation and basic properties of a SimulationMap."""
    s_map = make_simulation_map()
    assert isinstance(s_map, collections.abc.Mapping)
    assert len(s_map) == len(SIM_MAP_DATA)
    assert s_map["sim_A"] == SIM_MAP_DATA["sim_A"]
    assert list(s_map.keys()) == list(SIM_MAP_DATA.keys())


def test_simulation_map_invalid_type_raises_error():
    """Tests that a ValidationError is raised for incorrect value types."""
    invalid_data = {"sim_A": "not a simulation"}
    with pytest.raises(ValidationError):
        SimulationMap(keys=tuple(invalid_data.keys()), values=tuple(invalid_data.values()))


def test_simulation_map_mismatched_lengths_raises_error():
    """Tests that a ValidationError is raised for mismatched keys and values lengths."""
    with pytest.raises(ValidationError, match="Length of 'keys' and 'values' must be the same"):
        SimulationMap(keys=("sim_A",), values=tuple(SIM_MAP_DATA.values()))


def test_simulation_map_getitem_success():
    """Tests successful retrieval of a simulation by its string key."""
    s_map = make_simulation_map()
    assert s_map["sim_A"] == SIM_MAP_DATA["sim_A"]
    assert s_map["sim_B"] == SIM_MAP_DATA["sim_B"]


def test_simulation_map_getitem_key_error():
    """Tests that a KeyError is raised when retrieving a non-existent key."""
    s_map = make_simulation_map()
    with pytest.raises(KeyError):
        _ = s_map["not_a_real_key"]


def test_simulation_map_roundtrip_no_spurious_errors():
    """Regression test for FXC-5469: deserializing SimulationMap should not
    produce spurious ERROR logs from pydantic trying ModeSimulation validators."""
    s_map = make_simulation_map()

    with AssertLogStr("ERROR", excludes_str=""):
        SimulationMap.model_validate(s_map.model_dump())
