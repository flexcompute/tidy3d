"""Tests generation of pythons script from simulation file."""
import os

import tidy3d as td
from make_script import main
from ..utils import clear_tmp


@clear_tmp
def test_make_script():

    # make a sim
    simulation = td.Simulation(
        size=(1, 1, 1),
        sources=(
            td.PointDipole(
                polarization="Ex", source_time=td.GaussianPulse(freq0=2e14, fwidth=1e14)
            ),
        ),
        monitors=(td.FluxMonitor(size=(0, 1, 1), freqs=[1e14], name="flux"),),
        run_time=1e-12,
    )

    sim_path = "tests/tmp/sim.json"
    out_path = "tests/tmp/sim.py"

    # save it to file, assuring it does not exist already
    simulation.to_file(sim_path)
    assert not os.path.exists(out_path), f"out file {out_path} already exists."

    # generate out script from the simulation file
    main([sim_path, out_path])

    # make sure that file was created and is not empty
    assert os.path.exists(out_path), f"out file {out_path} wasn't created."
    assert os.stat(out_path).st_size > 0, f"out file {out_path} is empty."
