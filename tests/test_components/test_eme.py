import numpy as np
import pydantic.v1 as pd
import pytest
import tidy3d as td
from matplotlib import pyplot as plt
from tidy3d.exceptions import SetupError, ValidationError

from ..utils import AssertLogLevel

np.random.seed(4)

f, AX = plt.subplots()


def make_eme_sim():
    # general simulation parameters
    lambda0 = 1
    freq0 = td.C_0 / lambda0
    freqs = [freq0]
    sim_size = 3 * lambda0, 3 * lambda0, 3 * lambda0
    waveguide_size = (lambda0 / 2, lambda0, td.inf)
    min_steps_per_wvl = 10

    # EME parameters
    monitor_size = (2 * lambda0, 2 * lambda0, 0.1 * lambda0)
    eme_num_cells = 5  # EME grid num cells
    eme_axis = 2

    # Structures and FDTD grid
    waveguide_geometry = td.Box(size=waveguide_size)
    waveguide_medium = td.Medium(permittivity=2, conductivity=1e-6)
    waveguide = td.Structure(geometry=waveguide_geometry, medium=waveguide_medium)
    override = td.Structure(geometry=waveguide_geometry, medium=td.Medium(permittivity=2))
    grid_spec = td.GridSpec.auto(
        wavelength=lambda0, min_steps_per_wvl=min_steps_per_wvl, override_structures=[override]
    )

    # EME setup
    mode_spec = td.EMEModeSpec(num_modes=10, num_pml=(10, 10))
    eme_uniform_grid = td.EMEUniformGrid(num_cells=eme_num_cells, mode_spec=mode_spec)
    eme_port_grid = td.EMEUniformGrid(num_cells=1, mode_spec=mode_spec.updated_copy(num_modes=5))
    eme_grid_spec = td.EMECompositeGrid(
        subgrids=[eme_port_grid, eme_uniform_grid, eme_port_grid], subgrid_boundaries=[-1, 1]
    )

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

    modes_in = td.ModeSolverMonitor(
        size=(td.inf, td.inf, 0),
        center=(0, 0, -lambda0),
        freqs=[freq0],
        mode_spec=td.ModeSpec(),
        name="modes_in",
    )
    modes_out = td.ModeSolverMonitor(
        size=(td.inf, td.inf, 0),
        center=(0, 0, lambda0),
        freqs=[freq0],
        mode_spec=td.ModeSpec(),
        name="modes_out",
    )

    monitors = [mode_monitor, coeff_monitor, field_monitor, modes_in, modes_out]
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


def test_sim_version_update():
    sim = make_eme_sim()
    sim_dict = sim.dict()
    sim_dict["version"] = "ancient_version"

    with AssertLogLevel("WARNING"):
        sim_new = td.EMESimulation.parse_obj(sim_dict)

    assert sim_new.version == td.__version__


def test_eme_grid():
    sim_geom = td.Box(size=(4, 4, 4), center=(0, 0, 0))
    axis = 2

    # make a uniform grid
    mode_spec = td.EMEModeSpec(num_modes=4)
    uniform_grid_spec = td.EMEUniformGrid(num_cells=4, mode_spec=mode_spec)
    uniform_grid = uniform_grid_spec.make_grid(
        center=sim_geom.center, size=sim_geom.size, axis=axis
    )

    # make a nonuniform grid
    mode_spec1 = td.EMEModeSpec(num_modes=3)
    mode_spec2 = td.EMEModeSpec(num_modes=1)
    uniform_grid1 = td.EMEUniformGrid(num_cells=2, mode_spec=mode_spec1)
    uniform_grid2 = td.EMEUniformGrid(num_cells=4, mode_spec=mode_spec2)
    composite_grid_spec = td.EMECompositeGrid(
        subgrids=[uniform_grid1, uniform_grid2], subgrid_boundaries=[0]
    )
    composite_grid = composite_grid_spec.make_grid(
        center=sim_geom.center, size=sim_geom.size, axis=axis
    )
    explicit_grid_spec = td.EMEExplicitGrid(boundaries=[0], mode_specs=[mode_spec1, mode_spec2])
    explicit_grid = explicit_grid_spec.make_grid(
        center=sim_geom.center, size=sim_geom.size, axis=axis
    )

    nested_composite_grid_spec = td.EMECompositeGrid(
        subgrids=[composite_grid_spec, uniform_grid_spec], subgrid_boundaries=[1]
    )
    nested_composite_grid = nested_composite_grid_spec.make_grid(
        center=sim_geom.center, size=sim_geom.size, axis=axis
    )

    # test grid generation
    assert uniform_grid.axis == 2
    assert composite_grid.axis == 2
    assert explicit_grid.axis == 2

    assert uniform_grid.mode_specs == [mode_spec] * 4
    assert composite_grid.mode_specs == [mode_spec1] * 2 + [mode_spec2] * 4
    assert explicit_grid.mode_specs == [mode_spec1, mode_spec2]

    assert np.array_equal(uniform_grid.boundaries, [-2, -1, 0, 1, 2])
    assert np.array_equal(composite_grid.boundaries, [-2, -1, 0, 0.5, 1, 1.5, 2])
    assert np.array_equal(explicit_grid.boundaries, [-2, 0, 2])

    assert np.array_equal(uniform_grid.centers, [-1.5, -0.5, 0.5, 1.5])
    assert np.array_equal(composite_grid.centers, [-1.5, -0.5, 0.25, 0.75, 1.25, 1.75])
    assert np.array_equal(explicit_grid.centers, [-1, 1])

    assert np.array_equal(uniform_grid.lengths, [1, 1, 1, 1])
    assert np.array_equal(composite_grid.lengths, [1, 1, 0.5, 0.5, 0.5, 0.5])
    assert np.array_equal(explicit_grid.lengths, [2, 2])

    assert uniform_grid.num_cells == 4
    assert composite_grid.num_cells == 6
    assert explicit_grid.num_cells == 2

    grids = [uniform_grid, composite_grid, explicit_grid, nested_composite_grid]
    # test that mode planes span sim and lie at cell centers
    for grid in grids:
        for center, mode_plane in zip(grid.centers, grid.mode_planes):
            for dim in [0, 1, 2]:
                if dim == axis:
                    assert mode_plane.center[dim] == center
                    assert mode_plane.size[dim] == 0
                else:
                    assert mode_plane.center[dim] == sim_geom.center[dim]
                    assert mode_plane.size[dim] == sim_geom.size[dim]

    # test that boundary planes span sim and lie at cell boundaries
    for grid in grids:
        for boundary, boundary_plane in zip(grid.boundaries, grid.boundary_planes):
            for dim in [0, 1, 2]:
                if dim == axis:
                    assert boundary_plane.center[dim] == boundary
                    assert boundary_plane.size[dim] == 0
                else:
                    assert boundary_plane.center[dim] == sim_geom.center[dim]
                    assert boundary_plane.size[dim] == sim_geom.size[dim]

    # test that cells have correct centers and sizes
    for grid in grids:
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
            subgrids=[uniform_grid1, uniform_grid2, uniform_grid1, uniform_grid2],
            subgrid_boundaries=[0, 2, 1],
        )
    # need inside sim domain
    composite_grid_spec_outside = composite_grid_spec.updated_copy(subgrid_boundaries=[-5])
    with pytest.raises(ValidationError):
        _ = composite_grid_spec_outside.make_grid(
            center=sim_geom.center, size=sim_geom.size, axis=axis
        )
    composite_grid_spec_outside = composite_grid_spec.updated_copy(subgrid_boundaries=[5])
    with pytest.raises(ValidationError):
        _ = composite_grid_spec_outside.make_grid(
            center=sim_geom.center, size=sim_geom.size, axis=axis
        )

    # test explicit grid boundaries validator
    with pytest.raises(pd.ValidationError):
        # need right number
        _ = explicit_grid_spec.updated_copy(boundaries=[0, 1])
    with pytest.raises(pd.ValidationError):
        # need increasing
        _ = explicit_grid_spec.updated_copy(
            boundaries=[0, 1, 0.5], mode_specs=[mode_spec1, mode_spec1, mode_spec1, mode_spec1]
        )
    # need inside sim domain
    explicit_grid_spec_outside = explicit_grid_spec.updated_copy(boundaries=[-5])
    with pytest.raises(ValidationError):
        _ = explicit_grid_spec_outside.make_grid(
            center=sim_geom.center, size=sim_geom.size, axis=axis
        )
    explicit_grid_spec_outside = explicit_grid_spec.updated_copy(boundaries=[5])
    with pytest.raises(ValidationError):
        _ = explicit_grid_spec_outside.make_grid(
            center=sim_geom.center, size=sim_geom.size, axis=axis
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

    # test max num cells
    large_grid = td.EMEUniformGrid(num_cells=1000, mode_spec=td.EMEModeSpec())
    with pytest.raises(pd.ValidationError):
        _ = large_grid.make_grid(center=sim_geom.center, size=sim_geom.size, axis=axis)
    too_many_modes = td.EMEUniformGrid(num_cells=1, mode_spec=td.EMEModeSpec(num_modes=1000))
    with pytest.raises(pd.ValidationError):
        _ = too_many_modes.make_grid(center=sim_geom.center, size=sim_geom.size, axis=axis)


def test_eme_monitor():
    _ = td.EMEModeSolverMonitor(
        center=(1, 2, 3), size=(2, 2, 2), freqs=[300e12], num_modes=2, name="eme_modes"
    )
    _ = td.EMEFieldMonitor(
        center=(1, 2, 3),
        size=(2, 2, 0),
        freqs=[300e12],
        num_modes=2,
        colocate=False,
        name="eme_field",
    )
    _ = td.EMECoefficientMonitor(
        center=(1, 2, 3), size=(2, 2, 2), freqs=[300e12], num_modes=2, name="eme_coeffs"
    )


def test_eme_simulation():
    sim = make_eme_sim()
    _ = sim.plot(x=0, ax=AX)
    _ = sim.plot(y=0, ax=AX)
    _ = sim.plot(z=0, ax=AX)
    _ = sim.plot_grid(x=0, ax=AX)
    _ = sim.plot_grid(y=0, ax=AX)
    _ = sim.plot_grid(z=0, ax=AX)
    _ = sim.plot_eps(x=0, ax=AX)
    _ = sim.plot_eps(y=0, ax=AX)
    _ = sim.plot_eps(z=0, ax=AX)
    sim2 = sim.updated_copy(axis=1)
    _ = sim2.plot(x=0, ax=AX)
    _ = sim2.plot(y=0, ax=AX)
    _ = sim2.plot(z=0, ax=AX)

    # must be 3D
    with pytest.raises(pd.ValidationError):
        _ = td.EMESimulation(
            size=(0, 2, 2),
            freqs=[td.C_0],
            axis=2,
            eme_grid_spec=td.EMEUniformGrid(num_cells=2, mode_spec=td.EMEModeSpec()),
        )
    with pytest.raises(pd.ValidationError):
        _ = td.EMESimulation(
            size=(2, 2, 0),
            freqs=[td.C_0],
            axis=2,
            eme_grid_spec=td.EMEUniformGrid(num_cells=2, mode_spec=td.EMEModeSpec()),
        )

    # need at least one freq
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(freqs=[])
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(freqs=None)

    # no symmetry in propagation direction
    with pytest.raises(SetupError):
        _ = sim.updated_copy(symmetry=(0, 0, 1))

    # test warning for not providing wavelength in autogrid
    grid_spec = td.GridSpec.auto(min_steps_per_wvl=20)
    with AssertLogLevel("INFO"):
        sim = sim.updated_copy(grid_spec=grid_spec)
    # multiple freqs are ok, but not for autogrid
    _ = sim.updated_copy(grid_spec=td.GridSpec.uniform(dl=0.2), freqs=[1e10] + list(sim.freqs))
    with pytest.raises(SetupError):
        _ = td.EMESimulation(
            size=sim.size,
            freqs=list(sim.freqs) + [1e10],
            monitors=sim.monitors,
            structures=sim.structures,
            grid_spec=grid_spec,
            axis=sim.axis,
            eme_grid_spec=sim.eme_grid_spec,
        )

    # test port offsets
    with pytest.raises(ValidationError):
        _ = sim.updated_copy(port_offsets=[sim.size[sim.axis] * 2 / 3, sim.size[sim.axis] * 2 / 3])

    # test duplicate freqs
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(freqs=list(sim.freqs) + list(sim.freqs))
    # test empty freqs
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(freqs=[])

    # test unsupported media
    # fully anisotropic
    perm_diag = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    cond_diag = [[4, 0, 0], [0, 5, 0], [0, 0, 6]]
    rot = td.RotationAroundAxis(axis=(1, 2, 3), angle=1.23)
    perm = rot.rotate_tensor(perm_diag)
    cond = rot.rotate_tensor(cond_diag)
    med = td.FullyAnisotropicMedium(permittivity=perm, conductivity=cond)
    struct = sim.structures[0].updated_copy(medium=med)
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(structures=[struct])
    # warn for time modulated
    FREQ_MODULATE = 1e12
    AMP_TIME = 1.1
    PHASE_TIME = 0
    CW = td.ContinuousWaveTimeModulation(freq0=FREQ_MODULATE, amplitude=AMP_TIME, phase=PHASE_TIME)
    ST = td.SpaceTimeModulation(
        time_modulation=CW,
    )
    MODULATION_SPEC = td.ModulationSpec()
    modulation_spec = MODULATION_SPEC.updated_copy(permittivity=ST)
    modulated = td.Medium(permittivity=2, modulation_spec=modulation_spec)
    struct = sim.structures[0].updated_copy(medium=modulated)
    with AssertLogLevel("WARNING"):
        _ = td.EMESimulation(
            size=sim.size,
            monitors=sim.monitors,
            structures=[struct],
            grid_spec=grid_spec,
            axis=sim.axis,
            eme_grid_spec=sim.eme_grid_spec,
            freqs=sim.freqs,
        )
    # warn for nonlinear
    nonlinear = td.Medium(
        permittivity=2, nonlinear_spec=td.NonlinearSpec(models=[td.KerrNonlinearity(n2=1)])
    )
    struct = sim.structures[0].updated_copy(medium=nonlinear)
    with AssertLogLevel("WARNING"):
        _ = td.EMESimulation(
            size=sim.size,
            monitors=sim.monitors,
            structures=[struct],
            grid_spec=grid_spec,
            axis=sim.axis,
            eme_grid_spec=sim.eme_grid_spec,
            freqs=sim.freqs,
        )

    # test from_scene
    _ = td.EMESimulation.from_scene(
        scene=sim.scene,
        eme_grid_spec=sim.eme_grid_spec,
        freqs=sim.freqs,
        axis=sim.axis,
        size=sim.size,
    )

    # test monitor setup
    monitor = sim.monitors[0].updated_copy(freqs=[sim.freqs[0], sim.freqs[0]])
    with pytest.raises(SetupError):
        _ = sim.updated_copy(monitors=[monitor])
    monitor = sim.monitors[0].updated_copy(freqs=[5e10])
    with pytest.raises(SetupError):
        _ = sim.updated_copy(monitors=[monitor])
    monitor = sim.monitors[0].updated_copy(num_modes=1000)
    with pytest.raises(SetupError):
        _ = sim.updated_copy(monitors=[monitor])
    monitor = sim.monitors[2].updated_copy(num_modes=6)
    with pytest.raises(SetupError):
        _ = sim.updated_copy(monitors=[monitor])

    # test monitor at simulation bounds
    monitor = sim.monitors[-1].updated_copy(center=[0, 0, -sim.size[2] / 2])
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(monitors=[monitor])

    # test boundary and source validation
    with pytest.raises(SetupError):
        _ = sim.updated_copy(boundary_spec=td.BoundarySpec.all_sides(td.Periodic()))

    # test max sim size and freqs
    sim_bad = sim.updated_copy(size=(1000, 1000, 1000))
    with pytest.raises(SetupError):
        sim_bad.validate_pre_upload()
    sim_bad = sim.updated_copy(size=(1000, 500, 3), monitors=[], store_port_modes=True)
    with pytest.raises(SetupError):
        sim_bad.validate_pre_upload()
    sim_bad = sim.updated_copy(size=(1000, 500, 3), monitors=[], store_port_modes=False)
    with pytest.raises(SetupError):
        sim_bad.validate_pre_upload()
    sim_bad = sim.updated_copy(size=(500, 500, 3), monitors=[])
    with AssertLogLevel("WARNING", "slow-down"):
        sim_bad.validate_pre_upload()

    sim_bad = sim.updated_copy(
        freqs=list(sim.freqs) + list(1e14 * np.linspace(1, 2, 1000)),
        grid_spec=sim.grid_spec.updated_copy(wavelength=1),
    )
    with pytest.raises(SetupError):
        sim_bad.validate_pre_upload()
    large_monitor = sim.monitors[2].updated_copy(size=(td.inf, td.inf, td.inf))
    _ = sim.updated_copy(
        size=(10, 10, 10),
        monitors=[large_monitor],
        freqs=list(1e14 * np.linspace(1, 2, 1)),
        grid_spec=sim.grid_spec.updated_copy(wavelength=1),
    )
    sim_bad = sim.updated_copy(
        size=(10, 10, 10),
        monitors=[large_monitor],
        freqs=list(1e14 * np.linspace(1, 2, 5)),
        grid_spec=sim.grid_spec.updated_copy(wavelength=1),
    )
    with AssertLogLevel("WARNING", contains_str="estimated storage"):
        sim_bad.validate_pre_upload()
    sim_bad = sim.updated_copy(
        size=(10, 10, 10),
        monitors=[large_monitor],
        freqs=list(1e14 * np.linspace(1, 2, 20)),
        grid_spec=sim.grid_spec.updated_copy(wavelength=1),
    )
    with pytest.raises(SetupError):
        sim_bad.validate_pre_upload()
    sim_bad = sim.updated_copy(
        size=(10, 10, 10),
        monitors=[large_monitor, large_monitor.updated_copy(name="lmon2")],
        freqs=list(1e14 * np.linspace(1, 2, 5)),
        grid_spec=sim.grid_spec.updated_copy(wavelength=1),
    )
    with pytest.raises(SetupError):
        sim_bad.validate_pre_upload()

    # test monitor that does not intersect any EME cells
    mode_monitor = td.EMEModeSolverMonitor(
        size=(0.1, 0.1, 0.1),
        center=(0, 0, -1.5),
        name="modes",
    )
    with pytest.raises(SetupError):
        _ = sim.updated_copy(monitors=[mode_monitor], port_offsets=(0.5, 0.5))
    # test eme cell interval space
    mode_monitor = mode_monitor.updated_copy(
        size=(td.inf, td.inf, td.inf), eme_cell_interval_space=8
    )
    sim2 = sim.updated_copy(monitors=[mode_monitor])
    assert sim2._monitor_num_eme_cells(monitor=mode_monitor) == 2

    # test monitor num modes
    sim_tmp = sim.updated_copy(monitors=[sim.monitors[0].updated_copy(num_modes=1)])
    assert sim_tmp._monitor_num_modes_cell(monitor=sim_tmp.monitors[0], cell_index=0) == 1

    # test monitor num freqs
    sim_tmp = sim.updated_copy(monitors=[sim.monitors[0].updated_copy(freqs=[sim.freqs[0]])])
    assert sim_tmp._monitor_num_freqs(monitor=sim_tmp.monitors[0]) == 1

    # test sweep
    sweep_sim = sim.updated_copy(
        sweep_spec=td.EMELengthSweep(scale_factors=list(np.linspace(1, 2, 10)))
    )
    assert sweep_sim._sweep_cells
    assert not sweep_sim._sweep_interfaces
    assert sweep_sim._num_sweep_cells == 10
    assert sweep_sim._num_sweep_interfaces == 1
    assert sweep_sim._num_sweep_modes == 1
    _ = sim.updated_copy(
        sweep_spec=td.EMELengthSweep(
            scale_factors=np.stack((np.linspace(1, 2, 7), np.linspace(1, 2, 7)))
        )
    )
    with pytest.raises(SetupError):
        _ = sim.updated_copy(sweep_spec=td.EMELengthSweep(scale_factors=[]))
    with pytest.raises(SetupError):
        _ = sim.updated_copy(
            sweep_spec=td.EMELengthSweep(
                scale_factors=np.stack(
                    (
                        np.stack((np.linspace(1, 2, 7), np.linspace(1, 2, 7))),
                        np.stack((np.linspace(1, 2, 7), np.linspace(1, 2, 7))),
                    )
                )
            )
        )
    # second shape of length sweep must equal number of cells
    with pytest.raises(SetupError):
        _ = sim.updated_copy(sweep_spec=td.EMELengthSweep(scale_factors=np.array([[1, 2], [3, 4]])))
    _ = sim.updated_copy(sweep_spec=td.EMEModeSweep(num_modes=list(np.arange(1, 5))))
    # test sweep size limit
    with pytest.raises(SetupError):
        _ = sim.updated_copy(sweep_spec=td.EMELengthSweep(scale_factors=[]))
    sim_bad = sim.updated_copy(
        sweep_spec=td.EMELengthSweep(scale_factors=list(np.linspace(1, 2, 200)))
    )
    with pytest.raises(SetupError):
        sim_bad.validate_pre_upload()
    # can't exceed max num modes
    with pytest.raises(SetupError):
        _ = sim.updated_copy(sweep_spec=td.EMEModeSweep(num_modes=list(np.arange(150, 200))))

    # don't warn in these two cases
    with AssertLogLevel(None):
        sim_good = sim.updated_copy(
            constraint="passive",
            eme_grid_spec=td.EMEUniformGrid(num_cells=1, mode_spec=td.EMEModeSpec(num_modes=40)),
            grid_spec=sim.grid_spec.updated_copy(wavelength=1),
        )
        sim_good.validate_pre_upload()
        sim_good = sim.updated_copy(
            constraint=None,
            eme_grid_spec=td.EMEUniformGrid(num_cells=1, mode_spec=td.EMEModeSpec(num_modes=60)),
            grid_spec=sim.grid_spec.updated_copy(wavelength=1),
        )
        sim_good.validate_pre_upload()
    # warn about num modes with constraint
    sim_bad = sim.updated_copy(
        constraint="passive",
        eme_grid_spec=td.EMEUniformGrid(num_cells=1, mode_spec=td.EMEModeSpec(num_modes=60)),
    )
    with AssertLogLevel("WARNING", contains_str="constraint"):
        sim_bad.validate_pre_upload()

    _ = sim.port_modes_monitor

    # test freq sweep
    sim = sim.updated_copy(sweep_spec=None)
    assert sim._num_sweep == 1
    assert not sim._sweep_modes
    sim = sim.updated_copy(sweep_spec=td.EMELengthSweep(scale_factors=[1, 2]))
    assert not sim._sweep_modes
    assert sim._num_sweep == 2
    sim = sim.updated_copy(sweep_spec=td.EMEFreqSweep(freq_scale_factors=[1, 2]))
    assert sim._sweep_modes
    assert sim._num_sweep == 2
    assert sim._monitor_num_sweep(sim.monitors[0]) == 1
    sim = sim.updated_copy(monitors=[sim.monitors[0].updated_copy(num_sweep=None)])
    assert sim._monitor_num_sweep(sim.monitors[0]) == 2
    with pytest.raises(SetupError):
        _ = sim.updated_copy(monitors=[sim.monitors[0].updated_copy(num_sweep=4)])
    with pytest.raises(SetupError):
        _ = sim.updated_copy(sweep_spec=td.EMEFreqSweep(freq_scale_factors=[1e-10, 2]))

    with pytest.raises(SetupError):
        _ = sim.updated_copy(
            eme_grid_spec=td.EMEExplicitGrid(
                boundaries=[-sim.size[2] / 2 + 0.001],
                mode_specs=[td.EMEModeSpec(), td.EMEModeSpec()],
            )
        )
    with pytest.raises(SetupError):
        _ = sim.updated_copy(
            eme_grid_spec=td.EMEExplicitGrid(
                boundaries=[sim.size[2] / 2 - 0.001],
                mode_specs=[td.EMEModeSpec(), td.EMEModeSpec()],
            )
        )
    with pytest.raises(SetupError):
        _ = sim.updated_copy(
            monitors=[
                td.ModeSolverMonitor(
                    center=[0, 0, sim.size[2] / 2 - 0.001],
                    size=[td.inf, td.inf, 0],
                    name="modes",
                    freqs=sim.freqs,
                    mode_spec=td.ModeSpec(),
                )
            ]
        )


def _get_eme_scalar_mode_field_data_array(num_sweep=0):
    x = np.linspace(-1, 1, 35)
    y = np.linspace(-1, 1, 38)
    z = [3]
    f = [td.C_0, 3e14]
    mode_index = np.arange(10)
    eme_cell_index = np.arange(7)
    if num_sweep != 0:
        sweep_index = np.arange(num_sweep)
    else:
        sweep_index = [0]
    coords = dict(
        x=x,
        y=y,
        z=z,
        f=f,
        sweep_index=sweep_index,
        eme_cell_index=eme_cell_index,
        mode_index=mode_index,
    )
    data = td.EMEScalarModeFieldDataArray(
        (1 + 1j)
        * np.random.random(
            (len(x), len(y), 1, 2, len(sweep_index), len(eme_cell_index), len(mode_index))
        ),
        coords=coords,
    )
    data[:, :, :, :, 0, :, 1] = np.nan
    if num_sweep == 0:
        data = data.drop_vars("sweep_index")
    return data


def test_eme_scalar_mode_field_data_array():
    _ = _get_eme_scalar_mode_field_data_array()


def _get_eme_scalar_field_data_array(num_sweep=0):
    x = [0]
    y = np.linspace(-1.5, 1.5, 38)
    z = np.linspace(-1.5, 1.5, 35)
    f = [td.C_0, 3e14]
    mode_index = np.arange(5)
    eme_port_index = [0, 1]
    if num_sweep != 0:
        sweep_index = np.arange(num_sweep)
    else:
        sweep_index = [0]
    coords = dict(
        x=x,
        y=y,
        z=z,
        f=f,
        sweep_index=sweep_index,
        eme_port_index=eme_port_index,
        mode_index=mode_index,
    )
    data = td.EMEScalarFieldDataArray(
        (1 + 1j) * np.random.random((len(x), len(y), len(z), 2, len(sweep_index), 2, 5)),
        coords=coords,
    )
    data[:, :, :, :, 0, 0, 0] = np.nan
    if num_sweep == 0:
        data = data.drop_vars("sweep_index")
    return data


def test_eme_scalar_field_data_array():
    _ = _get_eme_scalar_field_data_array()


def _get_eme_smatrix_data_array(num_modes_in=2, num_modes_out=3, num_freqs=2, num_sweep=0):
    if num_modes_in != 0:
        mode_index_in = np.arange(num_modes_in)
    else:
        mode_index_in = [0]
    if num_modes_out != 0:
        mode_index_out = np.arange(num_modes_out)
    else:
        mode_index_out = [0]
    if num_sweep != 0:
        sweep_index = np.arange(num_sweep)
    else:
        sweep_index = [0]

    f = td.C_0 * np.linspace(1, 2, num_freqs)

    data = (1 + 1j) * np.random.random(
        (len(f), len(mode_index_out), len(mode_index_in), len(sweep_index))
    )
    coords = dict(
        f=f, mode_index_out=mode_index_out, mode_index_in=mode_index_in, sweep_index=sweep_index
    )
    smatrix_entry = td.EMESMatrixDataArray(data, coords=coords)

    if num_modes_in == 0:
        smatrix_entry = smatrix_entry.drop_vars("mode_index_in")
    if num_modes_out == 0:
        smatrix_entry = smatrix_entry.drop_vars("mode_index_out")
    if num_sweep == 0:
        smatrix_entry = smatrix_entry.drop_vars("sweep_index")

    return smatrix_entry


def _get_eme_smatrix_dataset(num_modes_1=3, num_modes_2=4, num_sweep=0):
    S11 = _get_eme_smatrix_data_array(
        num_modes_in=num_modes_1, num_modes_out=num_modes_1, num_sweep=num_sweep
    )
    S12 = _get_eme_smatrix_data_array(
        num_modes_in=num_modes_2, num_modes_out=num_modes_1, num_sweep=num_sweep
    )
    S21 = _get_eme_smatrix_data_array(
        num_modes_in=num_modes_1, num_modes_out=num_modes_2, num_sweep=num_sweep
    )
    S22 = _get_eme_smatrix_data_array(
        num_modes_in=num_modes_2, num_modes_out=num_modes_2, num_sweep=num_sweep
    )
    return td.EMESMatrixDataset(S11=S11, S12=S12, S21=S21, S22=S22)


def _get_eme_coeff_data_array(num_sweep=0):
    f = [2e14]
    mode_index_out = [0, 1]
    mode_index_in = [0, 1, 2]
    eme_cell_index = np.arange(6)
    eme_port_index = [0, 1]
    if num_sweep != 0:
        sweep_index = np.arange(num_sweep)
    else:
        sweep_index = [0]
    coords = dict(
        f=f,
        sweep_index=sweep_index,
        eme_port_index=eme_port_index,
        eme_cell_index=eme_cell_index,
        mode_index_out=mode_index_out,
        mode_index_in=mode_index_in,
    )
    data = td.EMECoefficientDataArray(
        (1 + 1j)
        * np.random.random(
            (
                len(f),
                len(sweep_index),
                len(eme_port_index),
                len(eme_cell_index),
                len(mode_index_out),
                len(mode_index_in),
            ),
        ),
        coords=coords,
    )
    if num_sweep == 0:
        data = data.drop_vars("sweep_index")
    return data


def _get_eme_coeff_dataset(num_sweep=0):
    A = _get_eme_coeff_data_array(num_sweep=num_sweep)
    B = _get_eme_coeff_data_array(num_sweep=num_sweep)
    return td.EMECoefficientDataset(A=A, B=B)


def test_eme_coeff_data_array():
    _ = _get_eme_coeff_data_array()
    _ = _get_eme_coeff_data_array(num_sweep=3)


def _get_eme_mode_index_data_array(num_sweep=0):
    f = [td.C_0, 3e14]
    mode_index = np.arange(10)
    eme_cell_index = np.arange(7)
    if num_sweep != 0:
        sweep_index = np.arange(num_sweep)
    else:
        sweep_index = [0]
    coords = dict(
        f=f, sweep_index=sweep_index, eme_cell_index=eme_cell_index, mode_index=mode_index
    )
    data = td.EMEModeIndexDataArray(
        (1 + 1j)
        * np.random.random((len(f), len(sweep_index), len(eme_cell_index), len(mode_index))),
        coords=coords,
    )
    if num_sweep == 0:
        data = data.drop_vars("sweep_index")
    return data


def test_eme_mode_index_data_array():
    _ = _get_eme_mode_index_data_array()


def test_eme_smatrix_data_array():
    _ = _get_eme_smatrix_data_array()


def _get_eme_mode_solver_dataset(num_sweep=0):
    n_complex = _get_eme_mode_index_data_array(num_sweep=num_sweep)
    field = _get_eme_scalar_mode_field_data_array(num_sweep=num_sweep)
    fields = {key: field for key in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]}

    return td.EMEModeSolverDataset(n_complex=n_complex, **fields)


def _get_eme_field_dataset(num_sweep=0):
    field = _get_eme_scalar_field_data_array(num_sweep=num_sweep)
    fields = {key: field for key in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]}
    return td.EMEFieldDataset(**fields)


def test_eme_dataset():
    # test s matrix
    _ = _get_eme_smatrix_dataset()
    _ = _get_eme_smatrix_dataset(num_modes_1=0)
    _ = _get_eme_smatrix_dataset(num_modes_2=0)
    _ = _get_eme_smatrix_dataset(num_modes_1=0, num_modes_2=0)
    _ = _get_eme_smatrix_dataset(num_sweep=5)

    # test coefficient
    _ = _get_eme_coeff_dataset()

    # test field
    _ = _get_eme_field_dataset()

    # test mode solver
    _ = _get_eme_mode_solver_dataset()


def _get_eme_mode_solver_data(num_sweep=0):
    dataset = _get_eme_mode_solver_dataset(num_sweep=num_sweep)
    kwargs = dataset.field_components
    monitor = td.EMEModeSolverMonitor(
        size=(td.inf, td.inf, td.inf),
        name="modes",
    )
    n_complex = _get_eme_mode_index_data_array(num_sweep=num_sweep)
    kwargs.update({"n_complex": n_complex})
    if num_sweep != 0:
        sweep_index = np.arange(num_sweep)
    else:
        sweep_index = [0]
    grid_primal_correction_data = np.ones(
        (
            len(n_complex.f),
            len(sweep_index),
            len(n_complex.eme_cell_index),
            len(n_complex.mode_index),
        )
    )
    grid_dual_correction_data = grid_primal_correction_data
    grid_correction_coords = dict(
        f=n_complex.f,
        sweep_index=sweep_index,
        eme_cell_index=n_complex.eme_cell_index,
        mode_index=n_complex.mode_index,
    )
    grid_primal_correction = td.components.data.data_array.EMEFreqModeDataArray(
        grid_primal_correction_data, coords=grid_correction_coords
    )
    grid_dual_correction = td.components.data.data_array.EMEFreqModeDataArray(
        grid_dual_correction_data, coords=grid_correction_coords
    )
    if num_sweep == 0:
        grid_primal_correction = grid_primal_correction.drop_vars("sweep_index")
        grid_dual_correction = grid_dual_correction.drop_vars("sweep_index")
    return td.EMEModeSolverData(
        monitor=monitor,
        grid_primal_correction=grid_primal_correction,
        grid_dual_correction=grid_dual_correction,
        **kwargs,
    )


def _get_eme_field_data(num_sweep=0):
    dataset = _get_eme_field_dataset(num_sweep=num_sweep)
    kwargs = dataset.field_components
    monitor = td.EMEFieldMonitor(size=(0, td.inf, td.inf), name="field", colocate=True)
    return td.EMEFieldData(monitor=monitor, **kwargs)


def _get_eme_coeff_data(num_sweep=0):
    dataset = _get_eme_coeff_dataset(num_sweep=num_sweep)
    monitor = td.EMECoefficientMonitor(
        size=(td.inf, td.inf, td.inf),
        name="coeffs",
    )
    return td.EMECoefficientData(monitor=monitor, A=dataset.A, B=dataset.B)


def _get_mode_solver_data(modes_out=False, num_modes=3):
    offset = 1 if modes_out else -1
    name = "modes_out" if modes_out else "modes_in"
    monitor = td.ModeSolverMonitor(
        size=(td.inf, td.inf, 0),
        center=(0, 0, offset),
        freqs=[td.C_0],
        mode_spec=td.ModeSpec(num_modes=num_modes),
        name=name,
    )
    eme_mode_data = _get_eme_mode_solver_data()
    kwargs = dict(eme_mode_data._grid_correction_dict, **eme_mode_data.field_components)
    mode_index = np.arange(num_modes)
    kwargs = {key: field.isel(eme_cell_index=0, drop=True) for key, field in kwargs.items()}
    kwargs = {key: field.isel(mode_index=mode_index) for key, field in kwargs.items()}
    kwargs = {key: field.isel(sweep_index=0) for key, field in kwargs.items()}
    n_complex = eme_mode_data.n_complex.isel(eme_cell_index=0, drop=True)
    n_complex = n_complex.isel(mode_index=mode_index)
    n_complex = n_complex.isel(sweep_index=0)
    kwargs.update({"n_complex": n_complex})
    sim = make_eme_sim()
    grid_expanded = sim.discretize_monitor(monitor)
    return td.ModeSolverData(monitor=monitor, grid_expanded=grid_expanded, **kwargs)


def test_eme_monitor_data():
    _ = _get_eme_mode_solver_data()
    _ = _get_eme_field_data()
    _ = _get_eme_coeff_data()
    _ = _get_mode_solver_data()
    _ = _get_eme_mode_solver_data(num_sweep=3)
    _ = _get_eme_field_data(num_sweep=3)
    _ = _get_eme_coeff_data(num_sweep=3)


def _get_eme_port_modes(num_sweep=0):
    mode_data = _get_eme_mode_solver_data(num_sweep=num_sweep)
    n_complex = mode_data.n_complex
    kwargs = dict(mode_data._grid_correction_dict, **mode_data.field_components)
    kwargs = {
        key: field.isel(
            eme_cell_index=[0, len(n_complex.eme_cell_index) - 1], mode_index=np.arange(5)
        )
        for key, field in kwargs.items()
    }
    n_complex = n_complex.isel(eme_cell_index=[0, len(n_complex.eme_cell_index) - 1])
    return mode_data.updated_copy(n_complex=n_complex, **kwargs)


def test_eme_sim_data():
    sim = make_eme_sim()
    mode_monitor_data = _get_eme_mode_solver_data()
    coeff_monitor_data = _get_eme_coeff_data()
    field_monitor_data = _get_eme_field_data()
    modes_in_data = _get_mode_solver_data(modes_out=False, num_modes=3)
    modes_out_data = _get_mode_solver_data(modes_out=True, num_modes=2)
    data = [
        mode_monitor_data,
        coeff_monitor_data,
        field_monitor_data,
        modes_in_data,
        modes_out_data,
    ]
    port_modes = _get_eme_port_modes()
    smatrix = _get_eme_smatrix_dataset(num_modes_1=5, num_modes_2=5)

    sim_data = td.EMESimulationData(simulation=sim, data=data, smatrix=smatrix, port_modes=None)
    with pytest.raises(SetupError):
        _ = sim_data.port_modes_tuple
    with pytest.raises(SetupError):
        _ = sim_data.port_modes_list_sweep

    sim_data = td.EMESimulationData(
        simulation=sim, data=data, smatrix=smatrix, port_modes=port_modes
    )
    _ = sim_data.port_modes_tuple
    _ = sim_data.port_modes_list_sweep

    # test smatrix_in_basis
    smatrix_in_basis = sim_data.smatrix_in_basis(modes1=modes_in_data, modes2=modes_out_data)
    assert len(smatrix_in_basis.S11.f) == 1
    assert len(smatrix_in_basis.S21.mode_index_in) == 3
    assert len(smatrix_in_basis.S21.mode_index_out) == 2
    assert len(smatrix_in_basis.S12.mode_index_in) == 2
    assert len(smatrix_in_basis.S12.mode_index_out) == 3
    assert len(smatrix_in_basis.S11.mode_index_in) == 3
    assert len(smatrix_in_basis.S11.mode_index_out) == 3
    assert len(smatrix_in_basis.S22.mode_index_in) == 2
    assert len(smatrix_in_basis.S22.mode_index_out) == 2
    monitor_in = td.FieldMonitor(
        size=(td.inf, td.inf, 0),
        center=(0, 0, -1),
        freqs=[td.C_0],
        name="in",
    )
    monitor_out = monitor_in.updated_copy(center=(0, 0, 1))
    kwargs = {
        key: field.isel(mode_index=0, drop=True)
        for key, field in modes_in_data.field_components.items()
    }
    modes_in0 = td.components.data.monitor_data.ElectromagneticFieldData(
        **kwargs, monitor=monitor_in, grid_expanded=modes_in_data.grid_expanded
    )
    kwargs = {
        key: field.isel(mode_index=0, drop=True)
        for key, field in modes_out_data.field_components.items()
    }
    modes_out0 = td.components.data.monitor_data.ElectromagneticFieldData(
        **kwargs, monitor=monitor_out, grid_expanded=modes_out_data.grid_expanded
    )
    smatrix_in_basis = sim_data.smatrix_in_basis(modes1=modes_in0, modes2=modes_out_data)
    assert len(smatrix_in_basis.S11.coords) == 1
    assert len(smatrix_in_basis.S12.coords) == 2
    assert len(smatrix_in_basis.S21.coords) == 2
    assert len(smatrix_in_basis.S22.coords) == 3
    smatrix_in_basis = sim_data.smatrix_in_basis(modes1=modes_in_data, modes2=modes_out0)
    assert len(smatrix_in_basis.S11.coords) == 3
    assert len(smatrix_in_basis.S12.coords) == 2
    assert len(smatrix_in_basis.S21.coords) == 2
    assert len(smatrix_in_basis.S22.coords) == 1
    smatrix_in_basis = sim_data.smatrix_in_basis(modes1=modes_in0, modes2=modes_out0)
    assert len(smatrix_in_basis.S11.coords) == 1
    assert len(smatrix_in_basis.S12.coords) == 1
    assert len(smatrix_in_basis.S21.coords) == 1
    assert len(smatrix_in_basis.S22.coords) == 1

    with pytest.raises(SetupError):
        _ = sim_data.updated_copy(port_modes=None).smatrix_in_basis(
            modes1=modes_in_data, modes2=modes_out_data
        )
    with pytest.raises(SetupError):
        _ = sim_data.updated_copy(port_modes=None).field_in_basis(
            field=sim_data["field"], modes=modes_in_data, port_index=0
        )

    # test field in basis
    field_in_basis = sim_data.field_in_basis(field=sim_data["field"], port_index=0)
    assert "mode_index" in field_in_basis.Ex.coords
    field_in_basis = sim_data.field_in_basis(field=sim_data["field"], modes=modes_in0, port_index=0)
    assert "mode_index" not in field_in_basis.Ex.coords
    field_in_basis = sim_data.field_in_basis(field=sim_data["field"], modes=modes_in0, port_index=1)
    assert "mode_index" not in field_in_basis.Ex.coords

    # test plotting
    _ = sim_data.plot_field(
        "field", "Ex", eme_port_index=0, val="real", f=td.C_0, mode_index=0, ax=AX
    )
    _ = sim_data.plot_field(
        "field", "Ex", eme_port_index=0, val="imag", f=td.C_0, mode_index=0, ax=AX
    )
    _ = sim_data.plot_field(
        "field", "Ex", eme_port_index=0, val="abs", f=td.C_0, mode_index=0, ax=AX
    )
    _ = sim_data.plot_field(
        "field", "Ex", eme_port_index=0, val="abs", f=td.C_0, mode_index=0, scale="dB", ax=AX
    )
    _ = sim_data.plot_field(
        "field", "S", eme_port_index=0, val="abs", f=td.C_0, mode_index=0, ax=AX
    )
    _ = sim_data.plot_field(
        "field", "Sx", eme_port_index=0, val="abs", f=td.C_0, mode_index=0, ax=AX
    )
    _ = sim_data.plot_field(
        "field", "Sx", eme_port_index=0, val="real", f=td.C_0, mode_index=0, ax=AX
    )
    _ = sim_data.plot_field(
        "field", "Sx", eme_port_index=0, val="imag", f=td.C_0, mode_index=0, ax=AX
    )
    _ = sim_data.plot_field(
        "field", "Sx", eme_port_index=0, val="abs^2", f=td.C_0, mode_index=0, ax=AX
    )
    _ = sim_data.plot_field(
        "field", "Sx", eme_port_index=0, val="phase", f=td.C_0, mode_index=0, ax=AX
    )
    _ = sim_data.plot_field(
        "field", "S", eme_port_index=0, val="real", f=td.C_0, mode_index=0, ax=AX
    )
    _ = sim_data.plot_field(
        "field", "S", eme_port_index=0, val="imag", f=td.C_0, mode_index=0, ax=AX
    )
    _ = sim_data.plot_field(
        "field", "S", eme_port_index=0, val="abs^2", f=td.C_0, mode_index=0, ax=AX
    )
    _ = sim_data.plot_field(
        "field", "E", eme_port_index=0, val="abs^2", f=td.C_0, mode_index=0, ax=AX
    )

    # test smatrix in basis with sweep
    smatrix = _get_eme_smatrix_dataset(num_modes_1=5, num_modes_2=5, num_sweep=10)
    sim = sim.updated_copy(sweep_spec=td.EMELengthSweep(scale_factors=np.linspace(1, 2, 10)))
    sim_data = td.EMESimulationData(
        simulation=sim, data=data, smatrix=smatrix, port_modes=port_modes
    )

    # test smatrix_in_basis
    smatrix_in_basis = sim_data.smatrix_in_basis(modes1=modes_in_data, modes2=modes_out_data)
    assert len(smatrix_in_basis.S11.f) == 1
    assert len(smatrix_in_basis.S21.mode_index_in) == 3
    assert len(smatrix_in_basis.S21.mode_index_out) == 2
    assert len(smatrix_in_basis.S12.mode_index_in) == 2
    assert len(smatrix_in_basis.S12.mode_index_out) == 3
    assert len(smatrix_in_basis.S11.mode_index_in) == 3
    assert len(smatrix_in_basis.S11.mode_index_out) == 3
    assert len(smatrix_in_basis.S22.mode_index_in) == 2
    assert len(smatrix_in_basis.S22.mode_index_out) == 2
    monitor_in = td.FieldMonitor(
        size=(td.inf, td.inf, 0),
        center=(0, 0, -1),
        freqs=[td.C_0],
        name="in",
    )
    monitor_out = monitor_in.updated_copy(center=(0, 0, 1))
    kwargs = {
        key: field.isel(mode_index=0, drop=True)
        for key, field in modes_in_data.field_components.items()
    }
    modes_in0 = td.components.data.monitor_data.ElectromagneticFieldData(
        **kwargs, monitor=monitor_in, grid_expanded=modes_in_data.grid_expanded
    )
    kwargs = {
        key: field.isel(mode_index=0, drop=True)
        for key, field in modes_out_data.field_components.items()
    }
    modes_out0 = td.components.data.monitor_data.ElectromagneticFieldData(
        **kwargs, monitor=monitor_out, grid_expanded=modes_out_data.grid_expanded
    )
    smatrix_in_basis = sim_data.smatrix_in_basis(modes1=modes_in0, modes2=modes_out_data)
    assert len(smatrix_in_basis.S11.coords) == 2
    assert len(smatrix_in_basis.S12.coords) == 3
    assert len(smatrix_in_basis.S21.coords) == 3
    assert len(smatrix_in_basis.S22.coords) == 4
    smatrix_in_basis = sim_data.smatrix_in_basis(modes1=modes_in_data, modes2=modes_out0)
    assert len(smatrix_in_basis.S11.coords) == 4
    assert len(smatrix_in_basis.S12.coords) == 3
    assert len(smatrix_in_basis.S21.coords) == 3
    assert len(smatrix_in_basis.S22.coords) == 2
    smatrix_in_basis = sim_data.smatrix_in_basis(modes1=modes_in0, modes2=modes_out0)
    assert len(smatrix_in_basis.S11.coords) == 2
    assert len(smatrix_in_basis.S12.coords) == 2
    assert len(smatrix_in_basis.S21.coords) == 2
    assert len(smatrix_in_basis.S22.coords) == 2
    smatrix_in_basis = sim_data.smatrix_in_basis(modes1=modes_in0)
    assert len(smatrix_in_basis.S11.coords) == 2
    assert len(smatrix_in_basis.S12.coords) == 3
    assert len(smatrix_in_basis.S21.coords) == 3
    assert len(smatrix_in_basis.S22.coords) == 4
    smatrix_in_basis = sim_data.smatrix_in_basis(modes2=modes_out0)
    assert len(smatrix_in_basis.S11.coords) == 4
    assert len(smatrix_in_basis.S12.coords) == 3
    assert len(smatrix_in_basis.S21.coords) == 3
    assert len(smatrix_in_basis.S22.coords) == 2
    smatrix_in_basis = sim_data.smatrix_in_basis()
    assert len(smatrix_in_basis.S11.coords) == 4
    assert len(smatrix_in_basis.S12.coords) == 4
    assert len(smatrix_in_basis.S21.coords) == 4
    assert len(smatrix_in_basis.S22.coords) == 4
    _ = sim_data.port_modes_tuple
    assert len(sim_data.port_modes_list_sweep) == 1

    # test freq sweep smatrix_in_basis
    sim = sim.updated_copy(sweep_spec=td.EMEFreqSweep(freq_scale_factors=np.linspace(1, 2, 10)))
    port_modes = _get_eme_port_modes(num_sweep=10)
    sim_data = td.EMESimulationData(
        simulation=sim, data=data, smatrix=smatrix, port_modes=port_modes
    )
    with pytest.raises(SetupError):
        _ = sim_data.port_modes_tuple
    assert len(sim_data.port_modes_list_sweep) == 10
    smatrix_in_basis = sim_data.smatrix_in_basis(modes1=modes_in0, modes2=modes_out_data)
    assert len(smatrix_in_basis.S11.sweep_index) == 10
    assert len(smatrix_in_basis.S11.coords) == 2
    assert len(smatrix_in_basis.S12.coords) == 3
    assert len(smatrix_in_basis.S21.coords) == 3
    assert len(smatrix_in_basis.S22.coords) == 4
    smatrix_in_basis = sim_data.smatrix_in_basis(modes1=modes_in_data, modes2=modes_out0)
    assert len(smatrix_in_basis.S11.coords) == 4
    assert len(smatrix_in_basis.S12.coords) == 3
    assert len(smatrix_in_basis.S21.coords) == 3
    assert len(smatrix_in_basis.S22.coords) == 2
    smatrix_in_basis = sim_data.smatrix_in_basis(modes1=modes_in0, modes2=modes_out0)
    assert len(smatrix_in_basis.S11.coords) == 2
    assert len(smatrix_in_basis.S12.coords) == 2
    assert len(smatrix_in_basis.S21.coords) == 2
    assert len(smatrix_in_basis.S22.coords) == 2
    smatrix_in_basis = sim_data.smatrix_in_basis(modes1=modes_in0)
    assert len(smatrix_in_basis.S11.coords) == 2
    assert len(smatrix_in_basis.S12.coords) == 3
    assert len(smatrix_in_basis.S21.coords) == 3
    assert len(smatrix_in_basis.S22.coords) == 4
    smatrix_in_basis = sim_data.smatrix_in_basis(modes2=modes_out0)
    assert len(smatrix_in_basis.S11.coords) == 4
    assert len(smatrix_in_basis.S12.coords) == 3
    assert len(smatrix_in_basis.S21.coords) == 3
    assert len(smatrix_in_basis.S22.coords) == 2
    smatrix_in_basis = sim_data.smatrix_in_basis()
    assert len(smatrix_in_basis.S11.coords) == 4
    assert len(smatrix_in_basis.S12.coords) == 4
    assert len(smatrix_in_basis.S21.coords) == 4
    assert len(smatrix_in_basis.S22.coords) == 4

    # test field in basis with freq sweep
    field_monitor_data = _get_eme_field_data(num_sweep=10)
    data[2] = field_monitor_data
    sim_data = sim_data.updated_copy(data=data)
    field_in_basis = sim_data.field_in_basis(field=sim_data["field"], port_index=0)
    assert len(field_in_basis.Ex.sweep_index) == 10
    assert "mode_index" in field_in_basis.Ex.coords
    field_in_basis = sim_data.field_in_basis(field=sim_data["field"], modes=modes_in0, port_index=0)
    assert "mode_index" not in field_in_basis.Ex.coords
    field_in_basis = sim_data.field_in_basis(field=sim_data["field"], modes=modes_in0, port_index=1)
    assert "mode_index" not in field_in_basis.Ex.coords


def test_eme_sim_subsection():
    eme_sim = td.EMESimulation(
        axis=2,
        size=(2, 2, 2),
        freqs=[td.C_0],
        grid_spec=td.GridSpec.auto(),
        eme_grid_spec=td.EMEUniformGrid(num_cells=2, mode_spec=td.EMEModeSpec()),
    )
    # check 3d subsection
    region = td.Box(size=(2, 2, 1))
    subsection = eme_sim.subsection(region=region)
    assert subsection.size[2] == 1

    # check 3d subsection with identical eme grid
    region = td.Box(size=(2, 2, 1))
    subsection = eme_sim.subsection(region=region, eme_grid_spec="identical")
    assert subsection.size[2] == 2
    region = td.Box(size=(2, 2, 0.5), center=(0, 0, 0.5))
    subsection = eme_sim.subsection(region=region, eme_grid_spec="identical")
    assert subsection.size[2] == 1

    # 2d subsection errors
    region = td.Box(size=(2, 2, 0))
    with pytest.raises(pd.ValidationError):
        subsection = eme_sim.subsection(region=region)
