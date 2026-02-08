"""Tests grid operations."""

from __future__ import annotations

import numpy as np
import pytest

import tidy3d as td
from tidy3d.components.grid.grid import Coords, FieldGrid, Grid
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
    _ = Coords(x=x, y=y, z=z)


def test_coords_arrays_are_immutable():
    """Test that arrays in Coords objects are immutable.

    This ensures that numpy arrays in Pydantic models cannot be modified,
    enforcing true immutability for these data structures.
    """

    # Create original arrays
    x_orig = np.array([1.0, 2.0, 3.0])
    y_orig = np.array([4.0, 5.0, 6.0])
    z_orig = np.array([7.0, 8.0, 9.0])

    # Create Coords object
    coords = Coords(x=x_orig, y=y_orig, z=z_orig)

    # Get dictionary
    coord_dict = coords.to_dict

    # Verify we got the right values
    assert np.array_equal(coord_dict["x"], x_orig)
    assert np.array_equal(coord_dict["y"], y_orig)
    assert np.array_equal(coord_dict["z"], z_orig)

    # Verify arrays are not writeable
    assert not coord_dict["x"].flags.writeable
    assert not coord_dict["y"].flags.writeable
    assert not coord_dict["z"].flags.writeable

    # Attempting to modify the arrays should raise an error
    with pytest.raises(ValueError, match="output array is read-only"):
        coord_dict["x"] -= 10

    with pytest.raises(ValueError, match="output array is read-only"):
        coord_dict["y"] *= 2

    with pytest.raises(ValueError, match="output array is read-only"):
        coord_dict["z"] += 100

    # Arrays should still have original values
    assert np.array_equal(coord_dict["x"], x_orig)
    assert np.array_equal(coord_dict["y"], y_orig)
    assert np.array_equal(coord_dict["z"], z_orig)


def test_grid_boundaries_modification_pattern():
    """Test the pattern of modifying grid boundaries after retrieval.

    This demonstrates that arrays are immutable and shows the correct
    pattern for creating modified versions.
    """

    # Create a grid for testing boundary modification
    boundaries_x = np.array([-1.0, 0.0, 1.0])
    boundaries_y = np.array([-1.0, 0.0, 1.0])
    boundaries_z = np.array([-1.0, 0.0, 1.0])
    coords = Coords(x=boundaries_x, y=boundaries_y, z=boundaries_z)
    grid = Grid(boundaries=coords)

    # Store original boundary values
    original_x = grid.boundaries.x.copy()
    original_y = grid.boundaries.y.copy()
    original_z = grid.boundaries.z.copy()

    # Get boundaries dictionary
    boundaries = grid.boundaries.to_dict
    center = [0.5, 0.5, 0.5]  # Simulate an offset value

    # Verify that direct modification fails due to immutability
    with pytest.raises(ValueError, match="output array is read-only"):
        boundaries["x"] -= center[0]

    # Show the correct pattern: make copies when modification is needed
    boundaries_copy = {k: v.copy() for k, v in boundaries.items()}

    # Now we can modify the copies
    for dim, dim_name in enumerate(boundaries_copy.keys()):
        boundaries_copy[dim_name] -= center[dim]

    # Create a new grid with modified boundaries
    offset_coords = Coords(**boundaries_copy)
    offset_grid = Grid(boundaries=offset_coords)

    # Verify original grid is unchanged
    assert np.array_equal(grid.boundaries.x, original_x)
    assert np.array_equal(grid.boundaries.y, original_y)
    assert np.array_equal(grid.boundaries.z, original_z)

    # Verify offset grid has the expected modified values
    assert np.array_equal(offset_grid.boundaries.x, original_x - 0.5)
    assert np.array_equal(offset_grid.boundaries.y, original_y - 0.5)
    assert np.array_equal(offset_grid.boundaries.z, original_z - 0.5)


def test_field_grid():
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    z = np.linspace(-1, 1, 100)
    c = Coords(x=x, y=y, z=z)
    _ = FieldGrid(x=c, y=c, z=c)


def test_grid():
    boundaries_x = np.arange(-1, 2, 1)
    boundaries_y = np.arange(-2, 3, 1)
    boundaries_z = np.arange(-3, 4, 1)
    _ = Coords(x=boundaries_x, y=boundaries_y, z=boundaries_z)
    g = make_grid()

    assert np.all(g.centers.x == np.array([-0.5, 0.5]))
    assert np.all(g.centers.y == np.array([-1.5, -0.5, 0.5, 1.5]))
    assert np.all(g.centers.z == np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]))

    for dim in "xyz":
        s = g.sizes.model_dump()[dim]
        assert np.all(np.array(s) == 1.0)

    assert np.all(g.yee.E.x.x == np.array([-0.5, 0.5]))
    assert np.all(g.yee.E.x.y == np.array([-2, -1, 0, 1]))
    assert np.all(g.yee.E.x.z == np.array([-3, -2, -1, 0, 1, 2]))


def test_grid_dict():
    g = make_grid()
    yee = g.yee
    _ = yee.grid_dict


def test_primal_steps():
    g = make_grid()
    _ = g._primal_steps


def test_dual_steps():
    g = make_grid()
    _ = g._dual_steps


def test_num_cells():
    g = make_grid()
    _ = g.num_cells


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
    inds_l_0_0 = g.discretize_inds(box=box_left, extend=False)[1]
    inds_r_0_0 = g.discretize_inds(box=box_right, extend=False)[1]
    inds_l_1_0 = g.discretize_inds(box=box_left, extend=True)[1]
    inds_r_1_0 = g.discretize_inds(box=box_right, extend=True)[1]

    assert np.diff(inds_l_0_0) == np.diff(inds_r_0_0)
    assert np.diff(inds_l_0_0) == np.diff(inds_l_1_0) - 2
    assert np.diff(inds_r_0_0) == np.diff(inds_r_1_0) - 1


def test_extended_subspace():
    g = make_grid()
    coords = g.extended_subspace(axis=0, ind_beg=-4, ind_end=6, periodic=False)
    assert np.allclose(coords, np.arange(-5, 5))
    coords = g.extended_subspace(axis=0, ind_beg=-4, ind_end=6, periodic=True)
    assert np.allclose(coords, np.arange(-5, 5))


def test_sim_nonuniform_small():
    # tests when the nonuniform grid does not cover the simulation size

    size_x = 18
    num_layers_pml_x = 6
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


def test_sim_nonuniform_large():
    # tests when the nonuniform grid extends beyond the simulation size

    size_x = 18
    num_layers_pml_x = 6
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

    for dim in "xyz":
        c = sim.grid.centers.model_dump()[dim]
        assert np.all(c == np.array([-1.5, -0.5, 0.5, 1.5]))

    for dim in "xyz":
        b = sim.grid.boundaries.model_dump()[dim]
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
            x=td.Boundary.pml(num_layers=6),
            y=td.Boundary.pml(num_layers=6),
            z=td.Boundary.pml(num_layers=6),
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
            x=td.Boundary.pml(num_layers=6),
            y=td.Boundary.absorber(num_layers=6),
            z=td.Boundary.stable_pml(num_layers=6),
        ),
        run_time=1e-12,
    )

    for dim in "xyz":
        c = sim.grid.centers.model_dump()[dim]
        assert np.all(c == np.arange(-7.5, 8, 1))

    for dim in "xyz":
        b = sim.grid.boundaries.model_dump()[dim]
        assert np.all(b == np.arange(-8, 8.5, 1))


def test_sim_discretize_vol():
    sim = td.Simulation(
        size=(4, 4, 4),
        grid_spec=td.GridSpec.uniform(1.0),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    vol = td.Box(size=(1.9, 1.9, 1.9))

    subgrid = sim.discretize(vol)

    for dim in "xyz":
        b = subgrid.boundaries.model_dump()[dim]
        assert np.all(b == np.array([-1, 0, 1]))

    for dim in "xyz":
        c = subgrid.centers.model_dump()[dim]
        assert np.all(c == np.array([-0.5, 0.5]))

    _ = td.Box(size=(6, 6, 0))


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


def test_get_geo_inds_no_span():
    g = make_grid()
    # Full-domain box: spans all cells along each axis
    geo_full = td.Box(center=(0, 0, 0), size=(2, 4, 6))

    # Expected: expand discretize_inds by 2 and clip to Yee E-grid sizes
    raw_inds = g.discretize_inds(geo_full.bounding_box, extend=False)
    lengths = [len(arr) for arr in g.yee.E.x.to_list]  # [Nx, Ny, Nz] on E.x grid
    expand = 2
    expected = np.array(
        [
            [
                max(raw_inds[i][0] - expand, 0),
                min(raw_inds[i][1] + expand, lengths[i]),
            ]
            for i in range(3)
        ]
    )

    inds = g._get_geo_inds(geo_full)
    assert np.array_equal(inds, expected)


def test_get_geo_inds_with_span():
    g = make_grid()
    geo_full = td.Box(center=(0, 0, 0), size=(2, 4, 6))

    # span_inds are in the same index space as discretize_inds (boundaries index space)
    # Choose a restricted span to intersect with the full-geometry indices
    span_inds = np.array(
        [
            [1, 2],  # x boundaries indices (restrict to right cell)
            [1, 3],  # y boundaries indices
            [2, 5],  # z boundaries indices
        ]
    )

    # Expected: intersect raw inds with span, then expand and clip
    raw_inds = g.discretize_inds(geo_full.bounding_box, extend=False)
    intersect = np.array(
        [
            [max(raw_inds[i][0], span_inds[i][0]), min(raw_inds[i][1], span_inds[i][1])]
            for i in range(3)
        ]
    )
    lengths = [len(arr) for arr in g.yee.E.x.to_list]
    expand = 2
    expected = np.array(
        [
            [
                max(intersect[i][0] - expand, 0),
                min(intersect[i][1] + expand, lengths[i]),
            ]
            for i in range(3)
        ]
    )

    inds = g._get_geo_inds(geo_full, span_inds=span_inds)
    assert np.array_equal(inds, expected)


def test_discretize_inds_relax_precision():
    """Test that relax_precision handles floating-point precision issues at cell boundaries."""
    g = make_grid()

    # Create a box where boundaries are exactly on cell boundaries
    box_exact = td.Box(center=(0, 0, 0), size=(2, 2, 2))
    inds_exact = g.discretize_inds(box=box_exact, extend=False, relax_precision=False)

    # Create a box where min boundary is slightly below the cell boundary (precision issue)
    # The grid has boundaries at x = -1, 0, 1; y = -2, -1, 0, 1, 2; z = -3, -2, -1, 0, 1, 2, 3
    # A box from -1 to 1 should span indices [0, 2] in x
    eps = 1e-14  # Small epsilon to simulate floating-point precision issues
    box_min_below = td.Box(center=(0, 0, 0), size=(2 + eps, 2 + eps, 2 + eps))

    # Without relax_precision, the slightly larger box may include extra cells
    inds_no_relax = g.discretize_inds(box=box_min_below, extend=False, relax_precision=False)

    # With relax_precision, the indices should match the exact case since boundaries are "close enough"
    inds_with_relax = g.discretize_inds(box=box_min_below, extend=False, relax_precision=True)

    # The relaxed precision should produce the same result as the exact box
    assert inds_exact == inds_with_relax
    assert inds_no_relax != inds_with_relax

    # Test case where box max is slightly below a cell boundary
    box_max_below = td.Box(center=(0, 0, 0), size=(2 - eps, 2 - eps, 2 - eps))
    inds_max_no_relax = g.discretize_inds(box=box_max_below, extend=False, relax_precision=False)
    inds_max_with_relax = g.discretize_inds(box=box_max_below, extend=False, relax_precision=True)

    # With relaxed precision, the boundaries close to cell boundaries should be treated as equal
    assert inds_exact == inds_max_with_relax
    assert inds_max_no_relax != inds_max_with_relax


def test_fine_mesh_info_uniform_grid():
    """Test that fine_mesh_info returns empty dict for uniform grids."""
    # Create a uniform grid
    boundaries_x = np.linspace(-1, 1, 11)  # uniform spacing of 0.2
    boundaries_y = np.linspace(-2, 2, 21)  # uniform spacing of 0.2
    boundaries_z = np.linspace(-3, 3, 31)  # uniform spacing of 0.2
    boundaries = Coords(x=boundaries_x, y=boundaries_y, z=boundaries_z)
    g = Grid(boundaries=boundaries)

    # Uniform grids should return empty dict
    info = g.fine_mesh_info
    assert info == {}


def test_fine_mesh_info_single_dimension_varying():
    """Test fine_mesh_info with varying cell sizes in one dimension."""
    # Create grid with varying x, uniform y and z
    x = np.array([0.0, 0.01, 0.02, 0.1, 0.2, 0.5])  # varying cell sizes
    y = np.linspace(-1, 1, 11)  # uniform
    z = np.linspace(-1, 1, 12)  # uniform
    boundaries = Coords(x=x, y=y, z=z)
    g = Grid(boundaries=boundaries)

    info = g.fine_mesh_info

    # Should have entries only for x dimension
    assert len(info) > 0
    for key in info.keys():
        dim, _ = key
        assert dim == "x"

    # The minimum cell size is 0.01 (between 0.0 and 0.01, and between 0.01 and 0.02)
    # Centers at these locations are 0.005 and 0.015
    assert ("x", 0.005) in info
    assert ("x", 0.015) in info
    assert np.isclose(info[("x", 0.005)], 0.01, rtol=1e-6)
    assert np.isclose(info[("x", 0.015)], 0.01, rtol=1e-6)


def test_fine_mesh_info_multiple_dimensions():
    """Test fine_mesh_info with varying cell sizes in multiple dimensions."""
    # Create grid with varying sizes in x and y
    x = np.array([0.0, 0.01, 0.02, 0.15])  # min size 0.01
    y = np.array([-1.0, -0.99, -0.98, -0.5, 0.0])  # min size 0.01
    z = np.linspace(-1, 1, 11)  # uniform
    boundaries = Coords(x=x, y=y, z=z)
    g = Grid(boundaries=boundaries)

    info = g.fine_mesh_info

    # Should have entries for both x and y dimensions
    x_entries = [key for key in info.keys() if key[0] == "x"]
    y_entries = [key for key in info.keys() if key[0] == "y"]
    assert len(x_entries) > 0
    assert len(y_entries) > 0

    # Check that all cell sizes are near minimum
    min_size = g.min_size
    for size in info.values():
        assert size <= min_size * 1.05  # within 5% tolerance


def test_fine_mesh_info_tolerance_threshold():
    """Test that fine_mesh_info includes cells within tolerance of minimum."""
    # Create grid with minimum size 0.1 and slightly larger cells
    x = np.array(
        [
            0.0,
            0.1,  # size 0.1 (min)
            0.2,  # size 0.1 (min)
            0.304,  # size 0.104 (within 5% tolerance)
            0.5,  # size 0.196 (too large)
        ]
    )
    y = np.linspace(-1, 1, 11)
    z = np.linspace(-1, 1, 11)
    boundaries = Coords(x=x, y=y, z=z)
    g = Grid(boundaries=boundaries)

    info = g.fine_mesh_info

    # Should include the first three cells (sizes 0.1, 0.1, 0.104)
    # but not the last one (size 0.196)
    assert len(info) == 3

    # Check that the expected centers are in info (with tolerance for floating point)
    x_dims = [key for key in info.keys() if key[0] == "x"]
    x_coords = [coord for dim, coord in x_dims]

    assert any(np.isclose(coord, 0.05, atol=1e-9) for coord in x_coords)
    assert any(np.isclose(coord, 0.15, atol=1e-9) for coord in x_coords)
    assert any(np.isclose(coord, 0.252, atol=1e-9) for coord in x_coords)
