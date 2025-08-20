from __future__ import annotations

import pydantic.v1 as pydantic
import pytest

from tidy3d import IndexSimulation, Simulation

from ..utils import SAMPLE_SIMULATIONS

# Reusable test constants
SIM_INDEX = ("sim_A", "sim_B")
DATA_INDEX = ("data_A", "data_B")


def make_simulations() -> tuple[Simulation, ...]:
    """Creates a tuple of simple, distinct Simulation objects for testing."""
    sim1 = SAMPLE_SIMULATIONS["full_fdtd"]
    sim2 = SAMPLE_SIMULATIONS["full_fdtd"]
    return (sim1, sim2)


def make_index_simulation(**kwargs) -> IndexSimulation:
    """Factory function to create a standard IndexSimulation instance."""
    return IndexSimulation(index=SIM_INDEX, simulation=make_simulations(), **kwargs)


def test_index_simulation_creation():
    """Tests successful creation of an IndexSimulation instance."""
    sims = make_simulations()
    container = make_index_simulation()
    assert container.index == SIM_INDEX
    assert container.simulation == sims
    assert len(container.index) == len(container.simulation)


def test_index_simulation_mismatched_length_raises_error():
    """Tests that a ValueError is raised for mismatched index and simulation lengths."""
    sims = make_simulations()
    with pytest.raises(pydantic.ValidationError, match="Length of 'index' and 'simulation'"):
        IndexSimulation(index=("only_one_index",), simulation=sims)


def test_index_simulation_getitem_success():
    """Tests successful retrieval of a simulation by its string index."""
    container = make_index_simulation()
    sims = make_simulations()
    assert container[SIM_INDEX[0]] == sims[0]
    assert container[SIM_INDEX[1]] == sims[1]


def test_index_simulation_getitem_key_error():
    """Tests that a KeyError is raised when retrieving a non-existent index."""
    container = make_index_simulation()
    with pytest.raises(KeyError, match="Index 'not_a_real_key' not found."):
        _ = container["not_a_real_key"]
