""" test the grid operations """

import numpy as np

import tidy3d as td
from tidy3d.components.grid import Coords, FieldGrid, YeeGrid, Grid
from tidy3d.components.base import TYPE_TAG_STR


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

    for s in g.sizes.dict(exclude={TYPE_TAG_STR}).values():
        assert np.all(s == 1.0)

    assert np.all(g.yee.E.x.x == np.array([-0.5, 0.5]))
    assert np.all(g.yee.E.x.y == np.array([-2, -1, 0, 1]))
    assert np.all(g.yee.E.x.z == np.array([-3, -2, -1, 0, 1, 2]))


def test_sim_nonuniform_small():
    # tests when the nonuniform grid doesnt cover the simulation size

    size_x = 18
    num_layers_pml_x = 2
    grid_size_x = [2, 1, 3]
    sim = td.Simulation(
        center=(1, 0, 0),
        size=(size_x, 4, 4),
        grid_size=(grid_size_x, 1, 1),
        pml_layers=[td.PML(num_layers=num_layers_pml_x), None, None],
    )

    bound_coords = sim.grid.boundaries.x
    dls = np.diff(bound_coords)

    dl_min = grid_size_x[0]
    dl_max = grid_size_x[-1]

    # checks the bounds were adjusted correctly
    # (smaller than sim size as is, but larger than sim size with one dl added on each edge)
    assert np.sum(dls) <= size_x + num_layers_pml_x * dl_min + num_layers_pml_x * dl_max
    assert (
        np.sum(dls) + dl_min + dl_max
        >= size_x + num_layers_pml_x * dl_min + num_layers_pml_x * dl_max
    )

    # tests that PMLs were added correctly
    for i in range(num_layers_pml_x):
        assert np.diff(bound_coords[i : i + 2]) == dl_min
        assert np.diff(bound_coords[-2 - i : len(bound_coords) - i]) == dl_max

    # tests that all the grid sizes are in there
    for size in grid_size_x:
        assert size in dls

    # tests that nothing but the grid sizes are in there
    for dl in dls:
        assert dl in grid_size_x

    # tests that it gives exactly what we expect
    assert np.all(bound_coords == np.array([-12, -10, -8, -6, -4, -2, 0, 1, 4, 7, 10, 13, 16]))


def test_sim_nonuniform_large():
    # tests when the nonuniform grid extends beyond the simulation size

    size_x = 18
    num_layers_pml_x = 2
    grid_size_x = [2, 3, 4, 1, 2, 1, 3, 1, 2, 3, 4]
    sim = td.Simulation(
        center=(1, 0, 0),
        size=(size_x, 4, 4),
        grid_size=(grid_size_x, 1, 1),
        pml_layers=[td.PML(num_layers=num_layers_pml_x), None, None],
    )

    bound_coords = sim.grid.boundaries.x
    dls = np.diff(bound_coords)

    dl_min = grid_size_x[0]
    dl_max = grid_size_x[-1]

    # checks the bounds were adjusted correctly
    # (smaller than sim size as is, but larger than sim size with one dl added on each edge)
    assert np.sum(dls) <= size_x + num_layers_pml_x * dl_min + num_layers_pml_x * dl_max
    assert (
        np.sum(dls) + dl_min + dl_max
        >= size_x + num_layers_pml_x * dl_min + num_layers_pml_x * dl_max
    )

    # tests that PMLs were added correctly
    for i in range(num_layers_pml_x):
        assert np.diff(bound_coords[i : i + 2]) == grid_size_x[0]
        assert np.diff(bound_coords[-2 - i : len(bound_coords) - i]) == grid_size_x[-1]

    # tests that all the grid sizes are in there
    for size in grid_size_x:
        assert size in dls

    # tests that nothing but the grid sizes are in there
    for dl in dls:
        assert dl in grid_size_x

    # tests that it gives exactly what we expect
    # assert np.all(bound_coords == np.array([-12,-10,-8,-6,-4,-2,0,1,4,7,10,13,16]))


def test_sim_grid():

    sim = td.Simulation(
        size=(4, 4, 4),
        grid_size=(1, 1, 1),
    )

    for c in sim.grid.centers.dict(exclude={TYPE_TAG_STR}).values():
        assert np.all(c == np.array([-1.5, -0.5, 0.5, 1.5]))
    for b in sim.grid.boundaries.dict(exclude={TYPE_TAG_STR}).values():
        assert np.all(b == np.array([-2, -1, 0, 1, 2]))


def test_sim_pml_grid():

    sim = td.Simulation(
        size=(4, 4, 4),
        grid_size=(1, 1, 1),
        pml_layers=(td.PML(num_layers=2), td.Absorber(num_layers=2), td.StablePML(num_layers=2)),
    )

    for c in sim.grid.centers.dict(exclude={TYPE_TAG_STR}).values():
        assert np.all(c == np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]))
    for b in sim.grid.boundaries.dict(exclude={TYPE_TAG_STR}).values():
        assert np.all(b == np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]))


def test_sim_discretize_vol():

    sim = td.Simulation(
        size=(4, 4, 4),
        grid_size=(1, 1, 1),
    )

    vol = td.Box(size=(1.9, 1.9, 1.9))

    subgrid = sim.discretize(vol)

    for b in subgrid.boundaries.dict(exclude={TYPE_TAG_STR}).values():
        assert np.all(b == np.array([-1, 0, 1]))

    for c in subgrid.centers.dict(exclude={TYPE_TAG_STR}).values():
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
