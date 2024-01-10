import pytest
import pydantic.v1 as pd
import numpy as np
from matplotlib import pyplot as plt

import tidy3d as td

from ..utils import STL_GEO, assert_log_level, log_capture


def make_eme_sim():
    # general simulation parameters
    lambda0 = 1
    freq0 = td.C_0 / lambda0
    freqs = [freq0]
    sim_size = 3 * lambda0, 3 * lambda0, 3 * lambda0
    waveguide_size = (lambda0 / 2, lambda0, td.inf)
    min_steps_per_wvl = 20

    # EME parameters
    monitor_size = (2 * lambda0, 2 * lambda0, 0.1 * lambda0)
    eme_num_cells = 5  # EME grid num cells
    eme_axis = 2

    # Structures and FDTD grid
    waveguide_geometry = td.Box(size=waveguide_size)
    waveguide_medium = td.Medium(permittivity=2, conductivity=1e-6)
    waveguide = td.Structure(geometry=waveguide_geometry, medium=waveguide_medium)
    grid_spec = td.GridSpec.auto(wavelength=lambda0, min_steps_per_wvl=min_steps_per_wvl)

    # EME setup
    mode_spec = td.ModeSpec(num_modes=5, num_pml=(10, 10))
    eme_grid_spec = td.EMEUniformGrid(num_cells=eme_num_cells, mode_spec=mode_spec)

    # field monitor stores field on FDTD grid
    field_monitor = td.EMEFieldMonitor(size=(0, td.inf, td.inf), name="field", colocate=True)

    coeff_monitor = td.EMECoefficientMonitor(
        size=monitor_size,
        name="coeffs",
    )

    mode_monitor = td.EMEModeSolverMonitor(
        size=(td.inf, td.inf, td.inf),
        name="modes",
    )

    monitors = [mode_monitor, coeff_monitor, field_monitor]
    structures = [waveguide]

    sim = td.EMESimulation(
        size=sim_size,
        monitors=monitors,
        structures=structures,
        grid_spec=grid_spec,
        axis=eme_axis,
        eme_grid_spec=eme_grid_spec,
        freqs=freqs,
    )
    return sim


def test_eme_grid():
    sim_geom = td.Box(size=(4, 4, 4), center=(0, 0, 0))
    axis = 2

    # make a uniform grid
    mode_spec = td.ModeSpec(num_modes=4)
    uniform_grid_spec = td.EMEUniformGrid(num_cells=4, mode_spec=mode_spec)
    uniform_grid = uniform_grid_spec.make_grid(sim_geom.center, sim_geom.size, axis)

    # make a nonuniform grid
    mode_spec1 = td.ModeSpec(num_modes=3)
    mode_spec2 = td.ModeSpec(num_modes=1)
    uniform_grid1 = td.EMEUniformGrid(num_cells=2, mode_spec=mode_spec1)
    uniform_grid2 = td.EMEUniformGrid(num_cells=4, mode_spec=mode_spec2)
    composite_grid_spec = td.EMECompositeGrid(
        subgrids=[uniform_grid1, uniform_grid2], subgrid_boundaries=[0]
    )
    composite_grid = composite_grid_spec.make_grid(sim_geom.center, sim_geom.size, axis)

    # test grid generation
    assert uniform_grid.axis == 2
    assert composite_grid.axis == 2

    assert uniform_grid.mode_specs == [mode_spec] * 4
    assert composite_grid.mode_specs == [mode_spec1] * 2 + [mode_spec2] * 4

    assert np.array_equal(uniform_grid.boundaries, [-2, -1, 0, 1, 2])
    assert np.array_equal(composite_grid.boundaries, [-2, -1, 0, 0.5, 1, 1.5, 2])

    assert np.array_equal(uniform_grid.centers, [-1.5, -0.5, 0.5, 1.5])
    assert np.array_equal(composite_grid.centers, [-1.5, -0.5, 0.25, 0.75, 1.25, 1.75])

    assert np.array_equal(uniform_grid.lengths, [1, 1, 1, 1])
    assert np.array_equal(composite_grid.lengths, [1, 1, 0.5, 0.5, 0.5, 0.5])

    assert uniform_grid.num_cells == 4
    assert composite_grid.num_cells == 6

    # test that mode planes span sim and lie at cell centers
    for grid in [uniform_grid, composite_grid]:
        for center, mode_plane in zip(grid.centers, grid.mode_planes):
            for dim in [0, 1, 2]:
                if dim == axis:
                    assert mode_plane.center[dim] == center
                    assert mode_plane.size[dim] == 0
                else:
                    assert mode_plane.center[dim] == sim_geom.center[dim]
                    assert mode_plane.size[dim] == sim_geom.size[dim]

    # test that boundary planes span sim and lie at cell boundaries
    for grid in [uniform_grid, composite_grid]:
        for boundary, boundary_plane in zip(grid.boundaries, grid.boundary_planes):
            for dim in [0, 1, 2]:
                if dim == axis:
                    assert boundary_plane.center[dim] == boundary
                    assert boundary_plane.size[dim] == 0
                else:
                    assert boundary_plane.center[dim] == sim_geom.center[dim]
                    assert boundary_plane.size[dim] == sim_geom.size[dim]

    # test that cells have correct centers and sizes
    for grid in [uniform_grid, composite_grid]:
        for center, length, cell in zip(grid.centers, grid.lengths, grid.cells):
            for dim in [0, 1, 2]:
                if dim == axis:
                    assert cell.center[dim] == center
                    assert cell.size[dim] == length
                else:
                    assert boundary_plane.center[dim] == sim_geom.center[dim]
                    assert boundary_plane.size[dim] == sim_geom.size[dim]

    # test cell_indices_in_box
    box = td.Box(center=(0, 0, 0.75), size=(td.inf, td.inf, 0.6))
    assert uniform_grid.cell_indices_in_box(box) == [2, 3]
    assert composite_grid.cell_indices_in_box(box) == [2, 3, 4]

    # test composite grid subgrid boundaries validator
    with pytest.raises(pd.ValidationError):
        # need right number
        _ = composite_grid_spec.updated_copy(subgrid_boundaries=[0, 2])
    with pytest.raises(pd.ValidationError):
        # need increasing
        _ = composite_grid_spec.updated_copy(
            subgris=[uniform_grid1, uniform_grid2, uniform_grid1], subgrid_boundaries=[0, -2]
        )

    # test grid boundaries validator
    # fine to not span entire simulation
    _ = uniform_grid.updated_copy(boundaries=[-1.5, -1, 0, 1, 1.5])
    with pytest.raises(pd.ValidationError):
        # need inside sim domain
        _ = uniform_grid.updated_copy(boundaries=[-2, -1, 0, 1, 3])
    with pytest.raises(pd.ValidationError):
        # need inside sim domain
        _ = uniform_grid.updated_copy(boundaries=[-3, -1, 0, 1, 2])
    with pytest.raises(pd.ValidationError):
        # need increasing
        _ = uniform_grid.updated_copy(boundaries=[-2, -1, 0, 1, 0.5])
    with pytest.raises(pd.ValidationError):
        # need one more boundary than mode_Spec
        _ = uniform_grid.updated_copy(boundaries=[-2, -1, 0, 1])


def test_eme_monitor():
    _ = td.EMEModeSolverMonitor(
        center=(1, 2, 3), size=(2, 2, 2), freqs=[300e12], mode_indices=[0, 1], name="eme_modes"
    )
    _ = td.EMEFieldMonitor(
        center=(1, 2, 3),
        size=(2, 2, 0),
        freqs=[300e12],
        mode_indices=[0, 1],
        colocate=False,
        name="eme_field",
    )
    _ = td.EMECoefficientMonitor(
        center=(1, 2, 3), size=(2, 2, 2), freqs=[300e12], mode_indices=[0, 1], name="eme_coeffs"
    )


def test_eme_simulation(log_capture):
    sim = make_eme_sim()

    # need at least one freq
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(freqs=[])
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(freqs=None)

    # test warning for not providing wavelength in autogrid
    log_capture
    assert_log_level(log_capture, None)
    grid_spec = td.GridSpec.auto(min_steps_per_wvl=20)
    sim = sim.updated_copy(grid_spec=grid_spec)
    assert_log_level(log_capture, "WARNING")
    # multiple freqs are ok, but not for autogrid
    _ = sim.updated_copy(grid_spec=td.GridSpec.uniform(dl=1), freqs=[1e10, 2e10])
    # TODO: validator on grid_spec doesn't run when freqs is updated
    # with pytest.raises(pd.ValidationError):
    #    _ = sim.updated_copy(freqs=[1, 2])


def test_eme_dataset():
    pass


def test_eme_monitor_data():
    pass


def test_eme_sim_data():
    pass
