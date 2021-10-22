""" test the grid operations """

import numpy as np

import tidy3d as td
from tidy3d.components.grid import Coords, FieldGrid, YeeGrid, Grid
from tidy3d.components.grid import GridSpec, GridSpec1DSize, GridSpec1DCoords, GridSpec1DResolution
from tidy3d.components.grid import GridSpec1DAuto


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
    cell_boundaries = Coords(x=boundaries_x, y=boundaries_y, z=boundaries_z)
    g = Grid(cell_boundaries=cell_boundaries)

    assert np.all(g.cell_centers.x == np.array([-0.5, 0.5]))
    assert np.all(g.cell_centers.y == np.array([-1.5, -0.5, 0.5, 1.5]))
    assert np.all(g.cell_centers.z == np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]))

    assert np.all(g.cell_sizes.x == 1.0)
    assert np.all(g.cell_sizes.y == 1.0)
    assert np.all(g.cell_sizes.z == 1.0)

    yee_Exz = g.yee.E.x.z
