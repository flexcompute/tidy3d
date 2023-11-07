"""Tests generation of pythons script from simulation file."""
import tidy3d as td
from make_script import main


def test_make_script(tmp_path):

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

    sim_path = tmp_path / "sim.json"
    out_path = tmp_path / "sim.py"

    # save it to file, assuring it does not exist already
    simulation.to_file(str(sim_path))
    assert not out_path.exists(), f"out file {out_path} already exists."

    # generate out script from the simulation file
    main([str(sim_path), str(out_path)])

    # make sure that file was created and is not empty
    assert out_path.is_file(), f"out file {out_path} wasn't created."
    assert len(out_path.read_text()) > 0, f"out file {out_path} is empty."
