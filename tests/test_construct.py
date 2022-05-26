"""Test loading simulation and other objects without validation."""
import json

import pytest
import pydantic
import tidy3d as td
from tidy3d.log import SetupError, ValidationError
from .utils import SIM_FULL, clear_tmp


def load_sim_dict() -> dict:
    """Load a sim_dict from file."""
    return SIM_FULL.copy().dict()


def test_construct():
    """Does the validation occur?"""

    sim_dict = load_sim_dict()

    # add an extra monitor to the sim_dict with identical name
    sim_dict["monitors"].append(sim_dict["monitors"][0])

    # should error if loading without construct
    with pytest.raises(SetupError):
        sim = td.Simulation.parse_obj(sim_dict)

    # should not error if loading with construct
    sim2 = td.Simulation.construct(sim_dict)

    assert len(sim2.monitors) > 0, "monitor list not added"

    # do something with the loaded sim
    for monitor in sim2.monitors:
        monitor.storage_size(10, np.arange(10))

    assert SIM_FULL == sim2


def test_construct_recursive():
    """Does the validation occur on subclasses?"""

    sim_dict = load_sim_dict()

    # remove frequencies from the FreqMonitors, which should trigger validation error
    for structure in sim_dict["structures"]:
        geometry = structure["geometry"]
        if geometry["type"] == "Box":
            geometry["center"] = ("Infinity", "Infinity", "Infinity")

    # should error if loading without construct
    with pytest.raises(ValidationError):
        sim = td.Simulation.parse_obj(sim_dict)

    # should not error if loading with construct
    sim2 = td.Simulation.construct(sim_dict)

    assert len(sim2.structures) > 0, "structure list not added"

    # do something with the loaded sim
    for structure in sim2.structures:
        structure.geometry.bounds

    assert SIM_FULL == sim2
