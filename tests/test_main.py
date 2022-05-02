import tidy3d as td


def test_main():
    s = td.Simulation(size=(1, 1, 1), grid_spec=td.GridSpec.auto(wavelength=1.0), run_time=1e-12)
    path = "tests/tmp/sim.json"
    s.to_file(path)

    from tidy3d.__main__ import main

    main([path, "--test_only"])
