"""Test running tidy3d as command line application."""

import tidy3d as td
import pytest
from tidy3d.__main__ import main

DEFAULT_PATH = "tests/tmp/sim.json"


def save_sim_to_path(path: str) -> None:
    sim = td.Simulation(size=(1, 1, 1), grid_spec=td.GridSpec.auto(wavelength=1.0), run_time=1e-12)
    sim.to_file(path)


@pytest.mark.parametrize("extension", (".json", ".yaml"))
def test_main(extension):
    path = f"tests/tmp/sim{extension}"
    save_sim_to_path(path)
    main([path, "--test_only"])
