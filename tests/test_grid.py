""" test the grid operations """

import numpy as np

import tidy3d as td
from tidy3d.components.grid import Coords, FieldGrid, YeeGrid, Grid


def test_coords():
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    z = np.linspace(-1, 1, 100)
    c = Coords(x=x, y=y, z=z)


def test_field_grid():
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    z = np.linspace(-1, 1, 100)
    c = Coords(x=x, y=y, z=z)
    f = FieldGrid(x=c, y=c, z=c)


def test_grid():

    boundaries_x = np.arange(-1, 2, 1)
    boundaries_y = np.arange(-2, 3, 1)
    boundaries_z = np.arange(-3, 4, 1)
    boundaries = Coords(x=boundaries_x, y=boundaries_y, z=boundaries_z)
    g = Grid(boundaries=boundaries)

    assert np.all(g.centers.x == np.array([-0.5, 0.5]))
    assert np.all(g.centers.y == np.array([-1.5, -0.5, 0.5, 1.5]))
    assert np.all(g.centers.z == np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]))

    for s in g.cell_sizes.dict().values():
        assert np.all(s == 1.0)

    assert np.all(g.yee.E.x.x == np.array([-0.5, 0.5]))
    assert np.all(g.yee.E.x.y == np.array([-2, -1, 0, 1]))
    assert np.all(g.yee.E.x.z == np.array([-3, -2, -1, 0, 1, 2]))


def test_sim_grid():

    sim = td.Simulation(
        size=(4, 4, 4),
        grid_size=(1, 1, 1),
    )

    for c in sim.grid.centers.dict().values():
        assert np.all(c == np.array([-1.5, -0.5, 0.5, 1.5]))
    for b in sim.grid.boundaries.dict().values():
        assert np.all(b == np.array([-2, -1, 0, 1, 2]))


def test_sim_discretize_vol():

    sim = td.Simulation(
        size=(4, 4, 4),
        grid_size=(1, 1, 1),
    )

    vol = td.Box(size=(1.9, 1.9, 1.9))

    subgrid = sim.discretize(vol)

    for b in subgrid.boundaries.dict().values():
        assert np.all(b == np.array([-1, 0, 1]))

    for c in subgrid.centers.dict().values():
        assert np.all(c == np.array([-0.5, 0.5]))

    plane = td.Box(size=(6, 6, 0))


def test_sim_discretize_plane():

    sim = td.Simulation(
        size=(4, 4, 4),
        grid_size=(1, 1, 1),
    )

    plane = td.Box(size=(6, 6, 0))

    subgrid = sim.discretize(plane)

    assert np.all(subgrid.boundaries.x == np.array([-2, -1, 0, 1, 2]))
    assert np.all(subgrid.boundaries.y == np.array([-2, -1, 0, 1, 2]))
    assert np.all(subgrid.boundaries.z == np.array([0, 1]))

    assert np.all(subgrid.centers.x == np.array([-1.5, -0.5, 0.5, 1.5]))
    assert np.all(subgrid.centers.y == np.array([-1.5, -0.5, 0.5, 1.5]))
    assert np.all(subgrid.centers.z == np.array([0.5]))
