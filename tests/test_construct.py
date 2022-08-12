"""Test loading simulation and other objects without validation."""
import json
import numpy as np
import pytest
import pydantic
import tidy3d as td
from tidy3d.log import SetupError, ValidationError
from .utils import SIM_FULL, clear_tmp
import json
import dill as pickle


def load_sim_dict_json_string() -> dict:
    """Load a sim_dict from file."""
    return json.loads(SIM_FULL.copy()._json_string())


def load_sim_dict_json() -> dict:
    """Load a sim_dict from .json()."""
    return json.loads(SIM_FULL.json())


def load_sim_dict() -> dict:
    """Load a sim_dict from file."""
    return SIM_FULL.dict()


@clear_tmp
def load_sim_dict_json_file():
    path = "tests/tmp/simulation.json"
    SIM_FULL.to_file(path)

    # now load a sim with the original monitors
    with open(path, "r") as fjson:
        sim_dict = json.load(fjson)
    return sim_dict


def test_construct_simple():
    """Use with box"""
    _ = td.Box.load_without_validation(size=(1, 1, 1))


# test all of these sim_dicts in the tests below
sim_dicts = [
    load_sim_dict(),
    load_sim_dict_json(),
    load_sim_dict_json_string(),
    load_sim_dict_json_file(),
]


@pytest.mark.parametrize("sim_dict", sim_dicts)
def test_construct_skips_validation(sim_dict):
    """Does the validation occur?"""

    # add an extra monitor to the sim_dict with identical name
    sim_dict_test = sim_dict.copy()
    new_monitors = tuple(list(sim_dict_test["monitors"]) + [sim_dict_test["monitors"][0]])
    sim_dict_test["monitors"] = new_monitors

    # should error if loading without construct
    with pytest.raises(SetupError):
        sim = td.Simulation.parse_obj(sim_dict_test)

    # should not error if loading with construct
    sim_fail = td.Simulation.load_without_validation(**sim_dict_test)

    # make sure sims were actually added, even though it failed
    assert len(sim_fail.monitors) > 0, "monitor list not added"
    assert len(sim_fail.structures) > 0, "structure list not added"
    assert len(sim_fail.sources) > 0, "sources list not added"


@pytest.mark.parametrize("sim_dict", sim_dicts)
def test_constructed_is_identical(sim_dict):
    """Does the loaded sim match original?"""

    # now load a sim with the original monitors
    sim_constructed = td.Simulation.load_without_validation(**sim_dict)

    # make sure sims were actually added
    assert len(sim_constructed.monitors) > 0, "monitor list not added"
    assert len(sim_constructed.structures) > 0, "structure list not added"
    assert len(sim_constructed.sources) > 0, "sources list not added"

    # do something with the loaded sim
    for monitor in sim_constructed.monitors:
        monitor.storage_size(10, np.arange(10))

    assert SIM_FULL == sim_constructed, "sim loaded without validation was not equal to original."


@pytest.mark.parametrize("sim_dict", sim_dicts)
def test_pickle(sim_dict):
    """Test if dill can pickle the constructed simulation (needed for parallelism)"""

    sim_constructed = td.Simulation.load_without_validation(**sim_dict)
    pickle.dumps(sim_constructed)
