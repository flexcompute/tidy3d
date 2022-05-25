"""Test the grid caching."""
import tidy3d as td


def make_sim():
    """Generate a starting simulation."""
    return td.Simulation(
        size=(10, 10, 10),
        run_time=1e-12,
        structures=[
            td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium(permittivity=4))
        ],
        grid_spec=td.GridSpec(wavelength=1),
    )


def test_no_cache():
    """test without anything"""

    sim = make_sim()
    sim.grid


def test_cached():

    sim = make_sim()
    for i in range(5):
        sim.grid


def test_frozen():

    sim = make_sim()
    td.config.freeze_cache = True
    sim.grid
    assert len(sim._frozen_property_values) > 0
    td.config.freeze_cache = False
    assert len(sim._frozen_property_values) == 0
    sim.grid
