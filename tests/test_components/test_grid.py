"""Tests grid operations."""
import pytest
import numpy as np

import tidy3d as td
from tidy3d.components.grid.grid import Coords, FieldGrid, YeeGrid, Grid
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.exceptions import SetupError


def make_grid():
    boundaries_x = np.arange(-1, 2, 1)
    boundaries_y = np.arange(-2, 3, 1)
    boundaries_z = np.arange(-3, 4, 1)
    boundaries = Coords(x=boundaries_x, y=boundaries_y, z=boundaries_z)
    return Grid(boundaries=boundaries)


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
    g = make_grid()

    assert np.all(g.centers.x == np.array([-0.5, 0.5]))
    assert np.all(g.centers.y == np.array([-1.5, -0.5, 0.5, 1.5]))
    assert np.all(g.centers.z == np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]))

    for s in g.sizes.dict(exclude={TYPE_TAG_STR}).values():
        assert np.all(np.array(s) == 1.0)

    assert np.all(g.yee.E.x.x == np.array([-0.5, 0.5]))
    assert np.all(g.yee.E.x.y == np.array([-2, -1, 0, 1]))
    assert np.all(g.yee.E.x.z == np.array([-3, -2, -1, 0, 1, 2]))


def test_grid_dict():
    g = make_grid()
    yee = g.yee
    gd = yee.grid_dict


def test_primal_steps():
    g = make_grid()
    ps = g._primal_steps


def test_dual_steps():
    g = make_grid()
    ps = g._dual_steps


def test_num_cells():
    g = make_grid()
    nc = g.num_cells


def test_getitem():
    g = make_grid()
    _ = g["Ex"]
    with pytest.raises(SetupError):
        _ = g["NOT_A_GRID_KEY"]


def test_extend_grid():
    """This test should clarify the expected behavior of various extensions."""
    g = make_grid()
    center_y = g.centers.to_list[1][g.num_cells[1] // 2]
    # 2d box on the left of a grid center
    box_left = td.Box(center=(0, center_y - 1e-5, 0), size=(2, 0, 6))
    # 2d box on the right of a grid center
    box_right = td.Box(center=(0, center_y + 1e-5, 0), size=(2, 0, 6))
    inds_l_0_0 = g.discretize_inds(box=box_left, extend=False, extend_2d_normal=False)[1]
    inds_r_0_0 = g.discretize_inds(box=box_right, extend=False, extend_2d_normal=False)[1]
    inds_l_1_0 = g.discretize_inds(box=box_left, extend=True, extend_2d_normal=False)[1]
    inds_r_1_0 = g.discretize_inds(box=box_right, extend=True, extend_2d_normal=False)[1]
    inds_l_0_1 = g.discretize_inds(box=box_left, extend=False, extend_2d_normal=True)[1]
    inds_r_0_1 = g.discretize_inds(box=box_right, extend=False, extend_2d_normal=True)[1]
    inds_l_1_1 = g.discretize_inds(box=box_left, extend=True, extend_2d_normal=True)[1]
    inds_r_1_1 = g.discretize_inds(box=box_right, extend=True, extend_2d_normal=True)[1]

    assert np.diff(inds_l_0_0) == np.diff(inds_r_0_0)
    assert np.diff(inds_l_0_0) == np.diff(inds_l_1_0) - 2
    assert np.diff(inds_r_0_0) == np.diff(inds_r_1_0) - 1
    assert np.diff(inds_l_0_0) == np.diff(inds_l_0_1) - 1
    assert np.diff(inds_r_0_0) == np.diff(inds_r_0_1) - 1
    assert np.diff(inds_l_0_0) == np.diff(inds_l_1_1) - 2
    assert np.diff(inds_r_0_0) == np.diff(inds_r_1_1) - 2


def test_extended_subspace():
    g = make_grid()
    coords = g.extended_subspace(axis=0, ind_beg=-4, ind_end=6, periodic=False)
    assert np.allclose(coords, np.arange(-5, 5))
    coords = g.extended_subspace(axis=0, ind_beg=-4, ind_end=6, periodic=True)
    assert np.allclose(coords, np.arange(-5, 5))


def test_sim_nonuniform_small():
    # tests when the nonuniform grid doesnt cover the simulation size

    size_x = 18
    num_layers_pml_x = 2
    grid_size_x = [2, 1, 3]
    sim = td.Simulation(
        center=(1, 0, 0),
        size=(size_x, 4, 4),
        grid_spec=td.GridSpec(
            grid_x=td.CustomGrid(dl=grid_size_x),
            grid_y=td.UniformGrid(dl=1.0),
            grid_z=td.UniformGrid(dl=1.0),
        ),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=num_layers_pml_x),
            y=td.Boundary.periodic(),
            z=td.Boundary.periodic(),
        ),
        run_time=1e-12,
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
        grid_spec=td.GridSpec(
            grid_x=td.CustomGrid(dl=grid_size_x),
            grid_y=td.UniformGrid(dl=1.0),
            grid_z=td.UniformGrid(dl=1.0),
        ),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=num_layers_pml_x),
            y=td.Boundary.periodic(),
            z=td.Boundary.periodic(),
        ),
        run_time=1e-12,
    )

    bound_coords = sim.grid.boundaries.x
    dls = np.diff(bound_coords)

    dl_min = dls[0]
    dl_max = dls[-1]

    # checks the bounds were adjusted correctly
    # (smaller than sim size as is, but larger than sim size with one dl added on each edge)
    assert np.sum(dls) <= size_x + num_layers_pml_x * dl_min + num_layers_pml_x * dl_max
    assert (
        np.sum(dls) + dl_min + dl_max
        >= size_x + num_layers_pml_x * dl_min + num_layers_pml_x * dl_max
    )

    # tests that PMLs were added correctly
    for i in range(num_layers_pml_x):
        assert np.diff(bound_coords[i : i + 2]) == dls[0]
        assert np.diff(bound_coords[-2 - i : len(bound_coords) - i]) == dls[-1]

    # tests that nothing but the grid sizes are in there
    for dl in dls:
        assert dl in grid_size_x


def test_sim_grid():

    sim = td.Simulation(
        size=(4, 4, 4),
        grid_spec=td.GridSpec.uniform(1.0),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    for c in sim.grid.centers.dict(exclude={TYPE_TAG_STR}).values():
        assert np.all(c == np.array([-1.5, -0.5, 0.5, 1.5]))
    for b in sim.grid.boundaries.dict(exclude={TYPE_TAG_STR}).values():
        assert np.all(b == np.array([-2, -1, 0, 1, 2]))


def test_sim_symmetry_grid():
    """tests that a grid symmetric w.r.t. the simulation center is created in presence of
    symmetries."""

    grid_1d = td.CustomGrid(dl=[2, 1, 3, 2])
    sim = td.Simulation(
        center=(1, 1, 1),
        size=(11, 11, 11),
        grid_spec=td.GridSpec(grid_x=grid_1d, grid_y=grid_1d, grid_z=grid_1d),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=2),
            y=td.Boundary.pml(num_layers=2),
            z=td.Boundary.pml(num_layers=2),
        ),
        symmetry=(0, 1, -1),
        run_time=1e-12,
    )

    coords_x, coords_y, coords_z = sim.grid.boundaries.to_list

    # Assert coords size is odd
    assert len(coords_x) % 2 != 0
    assert len(coords_y) % 2 != 0
    assert len(coords_z) % 2 != 0

    # Assert the dls along the symmetric axes are symmetric
    dls_y = np.diff(coords_y)
    dls_z = np.diff(coords_z)
    assert np.all(dls_y[len(dls_y) // 2 - 1 :: -1] == dls_y[len(dls_y) // 2 :])
    assert np.all(dls_z[len(dls_z) // 2 - 1 :: -1] == dls_z[len(dls_z) // 2 :])


def test_sim_pml_grid():

    sim = td.Simulation(
        size=(4, 4, 4),
        grid_spec=td.GridSpec.uniform(1.0),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=2),
            y=td.Boundary.absorber(num_layers=2),
            z=td.Boundary.stable_pml(num_layers=2),
        ),
        run_time=1e-12,
    )

    for c in sim.grid.centers.dict(exclude={TYPE_TAG_STR}).values():
        assert np.all(c == np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]))
    for b in sim.grid.boundaries.dict(exclude={TYPE_TAG_STR}).values():
        assert np.all(b == np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]))


def test_sim_discretize_vol():

    sim = td.Simulation(
        size=(4, 4, 4),
        grid_spec=td.GridSpec.uniform(1.0),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
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
        grid_spec=td.GridSpec.uniform(1.0),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    plane = td.Box(size=(6, 6, 0))

    subgrid = sim.discretize(plane)

    assert np.all(subgrid.boundaries.x == np.array([-2, -1, 0, 1, 2]))
    assert np.all(subgrid.boundaries.y == np.array([-2, -1, 0, 1, 2]))
    assert np.all(subgrid.boundaries.z == np.array([0, 1]))

    assert np.all(subgrid.centers.x == np.array([-1.5, -0.5, 0.5, 1.5]))
    assert np.all(subgrid.centers.y == np.array([-1.5, -0.5, 0.5, 1.5]))
    assert np.all(subgrid.centers.z == np.array([0.5]))


def test_grid_auto_uniform():
    """Compare GridSpec.auto and GridSpec.uniform in a simulation without structures."""

    sim_uniform = td.Simulation(
        size=(4, 4, 4),
        grid_spec=td.GridSpec.uniform(0.1),
        run_time=1e-12,
        medium=td.Medium(permittivity=4),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    sim_auto = td.Simulation(
        size=(4, 4, 4),
        grid_spec=td.GridSpec.auto(wavelength=2.4, min_steps_per_wvl=12),
        run_time=1e-12,
        medium=td.Medium(permittivity=4),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    bounds_uniform = sim_uniform.grid.boundaries.to_list
    bounds_auto = sim_auto.grid.boundaries.to_list

    for b_uniform, b_auto in zip(bounds_uniform, bounds_auto):
        assert np.allclose(b_uniform, b_auto)
