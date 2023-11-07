"""Test running tidy3d as command line application."""

import tidy3d as td
import pytest
from tidy3d.__main__ import main


def save_sim_to_path(path: str) -> None:
    sim = td.Simulation(size=(1, 1, 1), grid_spec=td.GridSpec.auto(wavelength=1.0), run_time=1e-12)
    sim.to_file(path)


@pytest.mark.parametrize("extension", (".json", ".yaml"))
def test_main(extension, tmp_path):
    path = str((tmp_path / "sim").with_suffix(extension))
    save_sim_to_path(path)
    main([path, "--test_only"])
