from __future__ import annotations

import pydantic.v1 as pydantic
import pytest

from tidy3d import IndexSimulation, IndexSimulationData, Simulation, SimulationData

from ..utils import SAMPLE_SIMULATIONS, run_emulated

# Reusable test constants
SIM_INDEX = ("sim_A", "sim_B")
DATA_INDEX = ("data_A", "data_B")


def make_simulations() -> tuple[Simulation, ...]:
    """Creates a tuple of simple, distinct Simulation objects for testing."""
    sim1 = SAMPLE_SIMULATIONS["full_fdtd"]
    sim2 = SAMPLE_SIMULATIONS["full_fdtd"].updated_copy(run_time=2e-12)
    return (sim1, sim2)


def make_simulation_data() -> tuple[SimulationData, ...]:
    """Creates a tuple of simple SimulationData objects for testing."""
    sims = make_simulations()
    data1 = run_emulated(sims[0])
    data2 = run_emulated(sims[1])
    return (data1, data2)


def make_index_simulation(**kwargs) -> IndexSimulation:
    """Factory function to create a standard IndexSimulation instance."""
    return IndexSimulation(index=SIM_INDEX, simulation=make_simulations(), **kwargs)


def make_index_simulation_data(**kwargs) -> IndexSimulationData:
    """Factory function to create a standard IndexSimulationData instance."""
    return IndexSimulationData(index=DATA_INDEX, data=make_simulation_data(), **kwargs)


def test_index_simulation_data_creation():
    """Tests successful creation of an IndexSimulationData instance."""
    sim_data = make_simulation_data()
    container = make_index_simulation_data()
    assert container.index == DATA_INDEX
    assert container.data == sim_data
    assert len(container.index) == len(container.data)


def test_index_simulation_data_mismatched_length_raises_error():
    """Tests that a ValueError is raised for mismatched index and data lengths."""
    sim_data = make_simulation_data()
    with pytest.raises(pydantic.ValidationError, match="Length of 'index' and 'data'"):
        IndexSimulationData(index=("only_one_index",), data=sim_data)


def test_index_simulation_data_getitem_success():
    """Tests successful retrieval of simulation data by its string index."""
    container = make_index_simulation_data()
    sim_data = make_simulation_data()
    assert container[DATA_INDEX[0]] == sim_data[0]
    assert container[DATA_INDEX[1]] == sim_data[1]


def test_index_simulation_data_getitem_key_error():
    """Tests that a KeyError is raised when retrieving a non-existent index."""
    container = make_index_simulation_data()
    with pytest.raises(KeyError, match="Index 'not_a_real_key' not found."):
        _ = container["not_a_real_key"]
