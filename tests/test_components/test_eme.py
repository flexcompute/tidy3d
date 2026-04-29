from __future__ import annotations

import numpy as np
import pydantic as pd
import pytest
from matplotlib import pyplot as plt

import tidy3d as td
from tidy3d.exceptions import SetupError, Tidy3dImportError, ValidationError

from ..utils import AssertLogLevel, assert_single_value_error_loc

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


@pytest.fixture(name="eme_base_sim")
def fixture_eme_base_sim():
    return make_eme_sim()


def _matched_lorentz_media_yy_zz(freq0: float) -> tuple[td.Lorentz, td.Lorentz]:
    """Two Lorentz media that agree at ``freq0`` and diverge away from it."""
    eps_inf = 2.0
    delta_eps_yy = 1.0
    resonance_yy = 1.5 * freq0
    resonance_zz = 2.0 * freq0
    delta_eps_zz = delta_eps_yy * resonance_yy**2 / (resonance_yy**2 - freq0**2)
    delta_eps_zz *= (resonance_zz**2 - freq0**2) / resonance_zz**2
    return (
        td.Lorentz(eps_inf=eps_inf, coeffs=[(delta_eps_yy, resonance_yy, 0.0)]),
        td.Lorentz(eps_inf=eps_inf, coeffs=[(delta_eps_zz, resonance_zz, 0.0)]),
    )


def test_sim_version_update():
    sim = make_eme_sim()
    sim_dict = sim.model_dump()
    sim_dict["version"] = "ancient_version"

    with AssertLogLevel("WARNING"):
        sim_new = td.EMESimulation.model_validate(sim_dict)

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
                    assert mode_plane.size[dim] == td.inf

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
    max_grid = td.EMEUniformGrid(num_cells=500, mode_spec=td.EMEModeSpec())
    _ = max_grid.make_grid(center=sim_geom.center, size=sim_geom.size, axis=axis)
    too_large_grid = td.EMEUniformGrid(num_cells=501, mode_spec=td.EMEModeSpec())
    with pytest.raises(pd.ValidationError):
        _ = too_large_grid.make_grid(center=sim_geom.center, size=sim_geom.size, axis=axis)
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
    # test default fields
    monitor_default = td.EMECoefficientMonitor(
        center=(1, 2, 3), size=(2, 2, 2), freqs=[300e12], num_modes=2, name="eme_coeffs"
    )
    assert monitor_default.fields == (
        "A",
        "B",
    )

    # test custom fields subset
    monitor_subset = td.EMECoefficientMonitor(
        center=(1, 2, 3),
        size=(2, 2, 2),
        freqs=[300e12],
        num_modes=2,
        fields=["A", "B", "overlaps"],
        name="eme_coeffs_subset",
    )
    assert monitor_subset.fields == ("A", "B", "overlaps")

    # test storage_size varies with fields
    num_cells, num_transverse, num_eme_cells, num_virtual_eme_cells = 100, 50, 5, 5
    num_freqs, num_modes = 2, 3
    size_default = monitor_default.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=None,
    )
    size_subset = monitor_subset.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=None,
    )
    assert size_subset > size_default
    assert size_subset > 0

    # test empty fields gives zero storage
    monitor_empty = td.EMECoefficientMonitor(
        center=(1, 2, 3),
        size=(2, 2, 2),
        freqs=[300e12],
        num_modes=2,
        fields=[],
        name="eme_coeffs_empty",
    )
    size_empty = monitor_empty.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=None,
    )
    assert size_empty == 0


def test_eme_monitor_storage_size_with_sweep_spec():
    """Test that storage_size correctly handles different sweep_spec types."""
    import numpy as np

    num_cells, num_transverse, num_eme_cells, num_virtual_eme_cells = 100, 50, 5, 5
    num_freqs, num_modes = 2, 3
    num_sweep = 10

    # Create sweep specs
    length_sweep = td.EMELengthSweep(scale_factors=np.linspace(0.5, 1.5, num_sweep))
    mode_sweep = td.EMEModeSweep(num_modes=np.arange(1, num_sweep + 1))
    freq_sweep = td.EMEFreqSweep(freq_scale_factors=np.linspace(0.9, 1.1, num_sweep))

    # Verify sweep_spec properties
    assert length_sweep.sweep_modes is False
    assert length_sweep.sweep_interfaces is False
    assert length_sweep.sweep_cells is True

    assert mode_sweep.sweep_modes is False
    assert mode_sweep.sweep_interfaces is True
    assert mode_sweep.sweep_cells is True

    assert freq_sweep.sweep_modes is True
    assert freq_sweep.sweep_interfaces is True
    assert freq_sweep.sweep_cells is True

    # Monitor with only A and B fields (uses full sweep)
    monitor_ab = td.EMECoefficientMonitor(
        center=(1, 2, 3),
        size=(2, 2, 2),
        freqs=[300e12],
        num_modes=2,
        num_sweep=None,
        fields=["A", "B"],
        name="eme_coeffs_ab",
    )

    # Monitor with only n_complex and flux (uses sweep_modes)
    monitor_nf = td.EMECoefficientMonitor(
        center=(1, 2, 3),
        size=(2, 2, 2),
        freqs=[300e12],
        num_modes=2,
        num_sweep=None,
        fields=["n_complex", "flux"],
        name="eme_coeffs_nf",
    )

    # Monitor with only interface_smatrices (uses sweep_interfaces)
    monitor_is = td.EMECoefficientMonitor(
        center=(1, 2, 3),
        size=(2, 2, 2),
        freqs=[300e12],
        num_modes=2,
        num_sweep=None,
        fields=["interface_smatrices"],
        name="eme_coeffs_is",
    )

    # Monitor with overlaps (uses sweep_modes)
    monitor_ov = td.EMECoefficientMonitor(
        center=(1, 2, 3),
        size=(2, 2, 2),
        freqs=[300e12],
        num_modes=2,
        num_sweep=None,
        fields=["overlaps"],
        name="eme_coeffs_ov",
    )
    monitor_ab_limited = td.EMECoefficientMonitor(
        center=(1, 2, 3),
        size=(2, 2, 2),
        freqs=[300e12],
        num_modes=2,
        num_sweep=3,
        fields=["A", "B"],
        name="eme_coeffs_ab_limited",
    )

    # Get storage sizes with no sweep (baseline)
    size_ab_none = monitor_ab.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=None,
    )
    size_nf_none = monitor_nf.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=None,
    )
    size_is_none = monitor_is.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=None,
    )
    size_ov_none = monitor_ov.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=None,
    )

    # Test EMELengthSweep: only A, B should scale with sweep
    size_ab_length = monitor_ab.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=length_sweep,
    )
    size_nf_length = monitor_nf.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=length_sweep,
    )
    size_is_length = monitor_is.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=length_sweep,
    )
    size_ov_length = monitor_ov.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=length_sweep,
    )

    # A, B should scale by num_sweep
    assert size_ab_length == size_ab_none * num_sweep
    # n_complex, flux should NOT scale (sweep_modes=False)
    assert size_nf_length == size_nf_none
    # interface_smatrices should NOT scale (sweep_interfaces=False)
    assert size_is_length == size_is_none
    # overlaps should NOT scale (sweep_modes=False)
    assert size_ov_length == size_ov_none

    # Test EMEModeSweep: A, B, interface_smatrices should scale
    size_ab_mode = monitor_ab.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=mode_sweep,
    )
    size_nf_mode = monitor_nf.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=mode_sweep,
    )
    size_is_mode = monitor_is.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=mode_sweep,
    )
    size_ov_mode = monitor_ov.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=mode_sweep,
    )

    # A, B should scale by num_sweep
    assert size_ab_mode == size_ab_none * num_sweep
    # n_complex, flux should NOT scale (sweep_modes=False)
    assert size_nf_mode == size_nf_none
    # interface_smatrices SHOULD scale (sweep_interfaces=True)
    assert size_is_mode == size_is_none * num_sweep
    # overlaps should NOT scale (sweep_modes=False)
    assert size_ov_mode == size_ov_none
    # monitor.num_sweep limits scaling
    size_ab_mode_limited = monitor_ab_limited.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=mode_sweep,
    )
    assert size_ab_mode_limited == size_ab_none * 3

    # Test EMEFreqSweep: everything should scale
    size_ab_freq = monitor_ab.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=freq_sweep,
    )
    size_nf_freq = monitor_nf.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=freq_sweep,
    )
    size_is_freq = monitor_is.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=freq_sweep,
    )
    size_ov_freq = monitor_ov.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=freq_sweep,
    )

    # All fields should scale by num_sweep
    assert size_ab_freq == size_ab_none * num_sweep
    assert size_nf_freq == size_nf_none * num_sweep
    assert size_is_freq == size_is_none * num_sweep
    assert size_ov_freq == size_ov_none * num_sweep

    # Test EMEModeSolverMonitor with sweep_spec
    mode_solver_monitor = td.EMEModeSolverMonitor(
        center=(1, 2, 3),
        size=(2, 2, 2),
        freqs=[300e12],
        num_modes=2,
        num_sweep=None,
        name="eme_mode_solver",
    )
    mode_solver_monitor_limited = td.EMEModeSolverMonitor(
        center=(1, 2, 3),
        size=(2, 2, 2),
        freqs=[300e12],
        num_modes=2,
        num_sweep=4,
        name="eme_mode_solver_limited",
    )

    size_ms_none = mode_solver_monitor.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=None,
    )
    size_ms_length = mode_solver_monitor.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=length_sweep,
    )
    size_ms_mode = mode_solver_monitor.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=mode_sweep,
    )
    size_ms_freq = mode_solver_monitor.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=freq_sweep,
    )

    # EMEModeSolverMonitor only scales with sweep_modes (EMEFreqSweep)
    assert size_ms_length == size_ms_none
    assert size_ms_mode == size_ms_none
    assert size_ms_freq == size_ms_none * num_sweep
    size_ms_freq_limited = mode_solver_monitor_limited.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=freq_sweep,
    )
    assert size_ms_freq_limited == size_ms_none * 4

    # Test EMEFieldMonitor with sweep_spec (uses full sweep)
    field_monitor = td.EMEFieldMonitor(
        center=(1, 2, 3),
        size=(2, 2, 0),
        freqs=[300e12],
        num_modes=2,
        num_sweep=None,
        name="eme_field",
    )
    field_monitor_limited = td.EMEFieldMonitor(
        center=(1, 2, 3),
        size=(2, 2, 0),
        freqs=[300e12],
        num_modes=2,
        num_sweep=2,
        name="eme_field_limited",
    )

    size_fm_none = field_monitor.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=None,
    )
    size_fm_length = field_monitor.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=length_sweep,
    )
    size_fm_mode = field_monitor.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=mode_sweep,
    )
    size_fm_freq = field_monitor.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=freq_sweep,
    )

    # EMEFieldMonitor uses full sweep for all sweep types
    assert size_fm_length == size_fm_none * num_sweep
    assert size_fm_mode == size_fm_none * num_sweep
    assert size_fm_freq == size_fm_none * num_sweep
    size_fm_freq_limited = field_monitor_limited.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=freq_sweep,
    )
    assert size_fm_freq_limited == size_fm_none * 2

    # Test EMEPeriodicitySweep: all sweep properties are False, so only A, B scale
    periodicity_sweep = td.EMEPeriodicitySweep(
        num_reps=[{"unit_cell": i} for i in range(1, num_sweep + 1)]
    )

    assert periodicity_sweep.sweep_modes is False
    assert periodicity_sweep.sweep_interfaces is False
    assert periodicity_sweep.sweep_cells is False

    size_ab_period = monitor_ab.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=periodicity_sweep,
    )
    size_nf_period = monitor_nf.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=periodicity_sweep,
    )
    size_is_period = monitor_is.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=periodicity_sweep,
    )
    size_ov_period = monitor_ov.storage_size(
        num_cells,
        num_transverse,
        num_eme_cells,
        num_virtual_eme_cells,
        num_freqs,
        num_modes,
        sweep_spec=periodicity_sweep,
    )

    # A, B should scale by num_sweep (full sweep)
    assert size_ab_period == size_ab_none * num_sweep
    # n_complex, flux should NOT scale
    assert size_nf_period == size_nf_none
    # interface_smatrices should NOT scale
    assert size_is_period == size_is_none
    # overlaps should NOT scale
    assert size_ov_period == size_ov_none


def test_eme_simulation(eme_base_sim):
    sim = eme_base_sim
    # no log except deprecated coeffs monitor
    with AssertLogLevel(None):
        _ = sim.updated_copy(monitors=[sim.monitors[0], *list(sim.monitors[2:])])
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

    # need at least one freq
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(freqs=[])
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(freqs=None)

    # no symmetry in propagation direction
    with pytest.raises(pd.ValidationError) as excinfo:
        _ = sim.updated_copy(symmetry=(0, 0, 1))
    assert_single_value_error_loc(
        excinfo,
        ("symmetry", 2),
        "Symmetry in the propagation direction is not currently supported.",
    )

    # test warning for not providing wavelength in autogrid
    grid_spec = td.GridSpec.auto(min_steps_per_wvl=20)
    sim = sim.updated_copy(grid_spec=grid_spec)
    with AssertLogLevel("INFO", contains_str="wavelength"):
        _ = sim.updated_copy(monitors=[])
    # multiple freqs are ok, but not for autogrid
    _ = sim.updated_copy(
        grid_spec=td.GridSpec.uniform(dl=0.2), freqs=[10000000000.0, *list(sim.freqs)]
    )
    with AssertLogLevel("INFO", contains_str="wavelength"):
        _ = sim.updated_copy(
            freqs=[*list(sim.freqs), 10000000000.0], grid_spec=grid_spec, monitors=[]
        )

    # test port offsets
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(port_offsets=(sim.size[sim.axis] * 2 / 3, sim.size[sim.axis] * 2 / 3))

    # test duplicate freqs
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(freqs=list(sim.freqs) + list(sim.freqs))

    # test anisotropic media support (reciprocal fully anisotropic only)
    perm_diag = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    cond_diag = [[4, 0, 0], [0, 5, 0], [0, 0, 6]]
    rot = td.RotationAroundAxis(axis=(1, 2, 3), angle=1.23)
    perm = rot.rotate_tensor(perm_diag)
    cond = rot.rotate_tensor(cond_diag)
    med = td.FullyAnisotropicMedium(permittivity=perm, conductivity=cond)
    struct = sim.structures[0].updated_copy(medium=med)
    _ = sim.updated_copy(structures=(struct,))
    _ = sim.updated_copy(medium=med)

    diag_aniso_med = td.AnisotropicMedium(
        xx=td.Medium(permittivity=2),
        yy=td.Medium(permittivity=3),
        zz=td.Medium(permittivity=4),
    )
    diag_struct = sim.structures[0].updated_copy(medium=diag_aniso_med)
    _ = sim.updated_copy(structures=(diag_struct,))
    _ = sim.updated_copy(medium=diag_aniso_med)

    bend_mode_spec = td.EMEModeSpec(num_modes=1, bend_radius=10.0, bend_axis=1)
    repeated_grid = td.EMEUniformGrid(num_cells=1, mode_spec=bend_mode_spec, num_reps=2)
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(
            structures=(struct,),
            monitors=(sim.monitors[0],),
            eme_grid_spec=repeated_grid,
        )
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(
            structures=(diag_struct,),
            monitors=(sim.monitors[0],),
            eme_grid_spec=repeated_grid,
        )
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(
            medium=diag_aniso_med,
            monitors=(sim.monitors[0],),
            eme_grid_spec=repeated_grid,
        )

    invariant_aniso_med = td.AnisotropicMedium(
        xx=td.Medium(permittivity=2),
        yy=td.Medium(permittivity=3),
        zz=td.Medium(permittivity=3),
    )
    invariant_struct = sim.structures[0].updated_copy(medium=invariant_aniso_med)
    invariant_repeated_grid = td.EMEUniformGrid(
        num_cells=1,
        mode_spec=td.EMEModeSpec(num_modes=1, bend_radius=10.0, bend_axis=0),
        num_reps=2,
    )
    _ = sim.updated_copy(
        structures=(invariant_struct,),
        monitors=(sim.monitors[0],),
        eme_grid_spec=invariant_repeated_grid,
    )

    lossy_invariant_aniso_med = td.AnisotropicMedium(
        xx=td.Medium(permittivity=2),
        yy=td.Medium(permittivity=3, conductivity=1e8),
        zz=td.Medium(permittivity=3, conductivity=1e8),
    )
    lossy_invariant_struct = sim.structures[0].updated_copy(medium=lossy_invariant_aniso_med)
    _ = sim.updated_copy(
        structures=(lossy_invariant_struct,),
        monitors=(sim.monitors[0],),
        eme_grid_spec=invariant_repeated_grid,
    )

    invariant_periodic_grid = td.EMEUniformGrid(
        num_cells=1,
        mode_spec=td.EMEModeSpec(num_modes=1, bend_radius=10.0, bend_axis=0),
        name="periodic_unit",
    )
    invariant_periodic_sweep = td.EMEPeriodicitySweep(num_reps=[{"periodic_unit": 2}])
    _ = sim.updated_copy(
        structures=(lossy_invariant_struct,),
        monitors=(sim.monitors[0],),
        eme_grid_spec=invariant_periodic_grid,
        sweep_spec=invariant_periodic_sweep,
    )

    invariant_named_aniso_med = td.AnisotropicMedium(
        xx=td.Medium(permittivity=2),
        yy=td.Medium(permittivity=3, name="yy_component"),
        zz=td.Medium(permittivity=3, name="zz_component"),
    )
    invariant_named_struct = sim.structures[0].updated_copy(medium=invariant_named_aniso_med)
    _ = sim.updated_copy(
        structures=(invariant_named_struct,),
        monitors=(sim.monitors[0],),
        eme_grid_spec=invariant_repeated_grid,
    )

    freq0 = float(sim.freqs[0])
    dispersive_invariant_aniso_med = td.AnisotropicMedium(
        xx=td.Medium(permittivity=2),
        yy=td.Sellmeier.from_dispersion(n=np.sqrt(3), freq=freq0, dn_dwvl=-0.1),
        zz=td.Sellmeier.from_dispersion(n=np.sqrt(3), freq=freq0, dn_dwvl=-1.0),
    )
    dispersive_invariant_struct = sim.structures[0].updated_copy(
        medium=dispersive_invariant_aniso_med
    )
    _ = sim.updated_copy(
        structures=(dispersive_invariant_struct,),
        monitors=(sim.monitors[0],),
        eme_grid_spec=invariant_repeated_grid,
    )
    with AssertLogLevel("ERROR", contains_str="'bend_medium_frame=\"co_rotating\"'"):
        with pytest.raises(pd.ValidationError):
            _ = sim.updated_copy(
                structures=(dispersive_invariant_struct,),
                monitors=(sim.monitors[0],),
                eme_grid_spec=invariant_repeated_grid,
                sweep_spec=td.EMEFreqSweep(freq_scale_factors=[1.0, 1.1]),
            )

    medium_yy, medium_zz = _matched_lorentz_media_yy_zz(freq0)
    assert np.isclose(medium_yy.eps_model(freq0), medium_zz.eps_model(freq0))
    assert not np.isclose(medium_yy.eps_model(freq0 * 1.1), medium_zz.eps_model(freq0 * 1.1))
    group_index_invariant_struct = sim.structures[0].updated_copy(
        medium=td.AnisotropicMedium(xx=td.Medium(permittivity=2), yy=medium_yy, zz=medium_zz)
    )
    group_index_repeated_grid = td.EMEUniformGrid(
        num_cells=1,
        mode_spec=td.EMEModeSpec(num_modes=1, bend_radius=10.0, bend_axis=0, group_index_step=0.1),
        num_reps=2,
    )
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(
            structures=(group_index_invariant_struct,),
            monitors=(sim.monitors[0],),
            eme_grid_spec=group_index_repeated_grid,
        )

    periodic_grid = td.EMEUniformGrid(num_cells=1, mode_spec=bend_mode_spec, name="periodic_unit")
    periodic_sweep = td.EMEPeriodicitySweep(num_reps=[{"periodic_unit": 2}])
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(
            structures=(struct,),
            monitors=(sim.monitors[0],),
            eme_grid_spec=periodic_grid,
            sweep_spec=periodic_sweep,
        )

    length_sweep = td.EMELengthSweep(scale_factors=[1.0, 1.1])
    with AssertLogLevel("ERROR", contains_str="'bend_medium_frame=\"co_rotating\"'"):
        with pytest.raises(pd.ValidationError):
            _ = sim.updated_copy(
                structures=(struct,),
                monitors=(sim.monitors[0],),
                eme_grid_spec=periodic_grid,
                sweep_spec=length_sweep,
            )

    second_cell_only_struct = td.Structure(
        geometry=td.Box(center=(0, 0, 0.75), size=(0.5, 1.0, 1.4)),
        medium=diag_aniso_med,
    )
    preceding_bent_grid = td.EMEExplicitGrid(
        boundaries=[0],
        mode_specs=[
            td.EMEModeSpec(num_modes=1, bend_radius=10.0, bend_axis=1),
            td.EMEModeSpec(num_modes=1, bend_radius=10.0, bend_axis=1),
        ],
    )
    with AssertLogLevel("ERROR", contains_str="separate simulations"):
        with pytest.raises(pd.ValidationError):
            _ = sim.updated_copy(
                structures=(second_cell_only_struct,),
                monitors=(sim.monitors[0],),
                eme_grid_spec=preceding_bent_grid,
                sweep_spec=td.EMELengthSweep(scale_factors=[[1.1, 1.0]]),
            )

    nested_repeated_grid = td.EMECompositeGrid(
        subgrids=[
            td.EMEUniformGrid(num_cells=1, mode_spec=bend_mode_spec, num_reps=2),
            td.EMEUniformGrid(num_cells=1, mode_spec=td.EMEModeSpec(num_modes=1)),
        ],
        subgrid_boundaries=[0],
        num_reps=2,
    )
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(
            structures=(second_cell_only_struct,),
            monitors=(sim.monitors[0],),
            eme_grid_spec=nested_repeated_grid,
        )

    straight_repeated_grid = td.EMECompositeGrid(
        subgrids=[
            td.EMEUniformGrid(num_cells=1, mode_spec=bend_mode_spec),
            td.EMEUniformGrid(num_cells=1, mode_spec=td.EMEModeSpec(num_modes=1), num_reps=2),
        ],
        subgrid_boundaries=[0],
    )
    _ = sim.updated_copy(
        structures=(struct,),
        monitors=(sim.monitors[0],),
        eme_grid_spec=straight_repeated_grid,
    )
    _ = sim.updated_copy(
        structures=(struct,),
        monitors=(sim.monitors[0],),
        eme_grid_spec=straight_repeated_grid,
        sweep_spec=td.EMELengthSweep(scale_factors=[[1.0, 1.2], [1.0, 0.8]]),
    )
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(
            structures=(struct,),
            monitors=(sim.monitors[0],),
            eme_grid_spec=straight_repeated_grid,
            sweep_spec=td.EMELengthSweep(scale_factors=[[1.1, 1.0], [0.9, 1.0]]),
        )

    straight_periodic_grid = td.EMECompositeGrid(
        subgrids=[
            td.EMEUniformGrid(num_cells=1, mode_spec=bend_mode_spec),
            td.EMEUniformGrid(
                num_cells=1, mode_spec=td.EMEModeSpec(num_modes=1), name="periodic_unit"
            ),
        ],
        subgrid_boundaries=[0],
    )
    _ = sim.updated_copy(
        structures=(struct,),
        monitors=(sim.monitors[0],),
        eme_grid_spec=straight_periodic_grid,
        sweep_spec=periodic_sweep,
    )

    # non-reciprocal permittivity tensor is still unsupported in EME
    perm_nonreciprocal = [[1, 1, 0], [-1, 2, 0], [0, 0, 3]]
    med = td.FullyAnisotropicMedium.model_construct(
        permittivity=perm_nonreciprocal, conductivity=cond_diag
    )
    struct = sim.structures[0].copy(validate=False, update={"medium": med})
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(structures=(struct,))
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(medium=med)

    # non-reciprocal conductivity tensor is still unsupported in EME
    cond_nonreciprocal = [[4, 1, 0], [-1, 5, 0], [0, 0, 6]]
    med = td.FullyAnisotropicMedium(permittivity=perm_diag, conductivity=cond_nonreciprocal)
    struct = sim.structures[0].updated_copy(medium=med)
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(structures=(struct,))
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(medium=med)
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
            structures=(struct,),
            grid_spec=grid_spec,
            axis=sim.axis,
            eme_grid_spec=sim.eme_grid_spec,
            freqs=sim.freqs,
        )
    # warn for nonlinear
    nonlinear = td.Medium(
        permittivity=2,
        nonlinear_spec=td.NonlinearSpec(models=(td.NonlinearSusceptibility(chi3=1),)),
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
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(monitors=(monitor,))
    monitor = sim.monitors[0].updated_copy(freqs=[5e10])
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(monitors=(monitor,))
    monitor = sim.monitors[0].updated_copy(num_modes=1000)
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(monitors=(monitor,))
    monitor = sim.monitors[2].updated_copy(num_modes=6)
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(monitors=(monitor,))

    # test monitor at simulation bounds
    monitor = sim.monitors[-1].updated_copy(center=[0, 0, -sim.size[2] / 2])
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(monitors=(monitor,))

    # test max sim size and freqs
    sim_bad = sim.updated_copy(size=(150, 150, 3))
    with pytest.raises(SetupError):
        sim_bad.validate_pre_upload()
    sim_bad = sim.updated_copy(size=(50, 50, 3), monitors=())
    with AssertLogLevel("WARNING", "slow-down"):
        sim_bad.validate_pre_upload()

    sim_ok = sim.updated_copy(
        freqs=list(sim.freqs) + list(1e14 * np.linspace(1, 2, 1000)),
        grid_spec=sim.grid_spec.updated_copy(wavelength=1),
    )
    sim_ok.validate_pre_upload()
    eme_grid_spec_no_interp = td.EMECompositeGrid(
        subgrids=[
            s.updated_copy(interp_spec=None, path="mode_spec")
            for s in sim_ok.eme_grid_spec.subgrids
        ],
        subgrid_boundaries=[-1, 1],
    )
    sim_bad = sim_ok.updated_copy(eme_grid_spec=eme_grid_spec_no_interp)
    with pytest.raises(SetupError):
        sim_bad.validate_pre_upload()
    sim_bad = sim_ok.updated_copy(
        freqs=list(sim.freqs) + list(1e14 * np.linspace(1, 2, 5000)),
    )
    with pytest.raises(SetupError):
        sim_bad.validate_pre_upload()
    sim_bad = sim_ok.updated_copy(
        freqs=list(sim.freqs) + list(1e14 * np.linspace(1, 2, 100)),
        eme_grid_spec=eme_grid_spec_no_interp,
    )
    with AssertLogLevel("WARNING", contains_str="expensive"):
        sim_bad.validate_pre_upload()
    large_monitor = sim.monitors[2].updated_copy(size=(td.inf, td.inf, td.inf))
    _ = sim.updated_copy(
        size=(10, 10, 10),
        monitors=(large_monitor,),
        freqs=list(1e14 * np.linspace(1, 2, 1)),
        grid_spec=sim.grid_spec.updated_copy(wavelength=1),
    )
    sim_bad = sim.updated_copy(
        size=(10, 10, 10),
        monitors=(large_monitor,),
        freqs=list(1e14 * np.linspace(1, 2, 5)),
        grid_spec=sim.grid_spec.updated_copy(wavelength=1),
    )
    with AssertLogLevel("WARNING", contains_str="estimated storage"):
        sim_bad.validate_pre_upload()
    # coeffs warning
    sim_bad = sim.updated_copy(
        size=(10, 10, 10),
        monitors=[],
        store_port_modes=False,
        freqs=list(1e14 * np.linspace(1, 2, 100)),
        eme_grid_spec=td.EMEUniformGrid(mode_spec=td.EMEModeSpec(num_modes=100), num_cells=100),
        grid_spec=sim.grid_spec.updated_copy(wavelength=1),
    )
    with AssertLogLevel("WARNING", contains_str="store_coeffs"):
        sim_bad.updated_copy(store_coeffs=True).validate_pre_upload()
    # port_modes warning
    sim_bad = sim.updated_copy(
        size=(10, 10, 10),
        monitors=[],
        store_coeffs=False,
        freqs=list(1e14 * np.linspace(1, 2, 100)),
        eme_grid_spec=td.EMEUniformGrid(mode_spec=td.EMEModeSpec(num_modes=100), num_cells=100),
        grid_spec=sim.grid_spec.updated_copy(wavelength=1),
    )
    with AssertLogLevel("WARNING", contains_str="store_port_modes"):
        sim_bad.updated_copy(store_port_modes=True).validate_pre_upload()
    sim_bad = sim.updated_copy(
        size=(10, 10, 10),
        monitors=(large_monitor,),
        freqs=list(1e14 * np.linspace(1, 2, 20)),
        grid_spec=sim.grid_spec.updated_copy(wavelength=1),
    )
    with pytest.raises(SetupError):
        sim_bad.validate_pre_upload()
    sim_bad = sim.updated_copy(
        size=(10, 10, 10),
        monitors=(large_monitor, large_monitor.updated_copy(name="lmon2")),
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
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(monitors=(mode_monitor,), port_offsets=(0.5, 0.5))
    # test eme cell interval space
    mode_monitor = mode_monitor.updated_copy(
        size=(td.inf, td.inf, td.inf), eme_cell_interval_space=8
    )
    sim2 = sim.updated_copy(monitors=(mode_monitor,))
    assert sim2._monitor_num_eme_cells(monitor=mode_monitor) == 2

    # test monitor num modes
    sim_tmp = sim.updated_copy(monitors=(sim.monitors[0].updated_copy(num_modes=1),))
    assert sim_tmp._monitor_num_modes_cell(monitor=sim_tmp.monitors[0], cell_index=0) == 1

    # test monitor num freqs
    sim_tmp = sim.updated_copy(monitors=(sim.monitors[0].updated_copy(freqs=[sim.freqs[0]]),))
    assert sim_tmp._monitor_num_freqs(monitor=sim_tmp.monitors[0]) == 1

    # test sweep
    with pytest.raises(pd.ValidationError) as excinfo:
        _ = sim.updated_copy(sweep_spec=td.EMELengthSweep(scale_factors=[1.0, 1.1]))
    assert_single_value_error_loc(
        excinfo,
        ("monitors", 2),
        "Monitor 'field' at 'monitors[2]' is an 'EMEFieldMonitor'",
    )
    sim_no_field = sim.updated_copy(
        monitors=[mnt for mnt in sim.monitors if not isinstance(mnt, td.EMEFieldMonitor)]
    )
    sweep_sim = sim_no_field.updated_copy(
        sweep_spec=td.EMELengthSweep(scale_factors=list(np.linspace(1, 2, 10)))
    )
    assert sweep_sim._sweep_cells
    assert not sweep_sim._sweep_interfaces
    assert sweep_sim._num_sweep_cells == 10
    assert sweep_sim._num_sweep_interfaces == 1
    assert sweep_sim._num_sweep_modes == 1
    _ = sim_no_field.updated_copy(
        sweep_spec=td.EMELengthSweep(
            scale_factors=np.stack((np.linspace(1, 2, 7), np.linspace(1, 2, 7)))
        ),
    )
    with pytest.raises(pd.ValidationError):
        _ = sim_no_field.updated_copy(sweep_spec=td.EMELengthSweep(scale_factors=[]))
    with pytest.raises(pd.ValidationError):
        _ = sim_no_field.updated_copy(
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
    with pytest.raises(pd.ValidationError):
        _ = sim_no_field.updated_copy(
            sweep_spec=td.EMELengthSweep(scale_factors=np.array([[1, 2], [3, 4]]))
        )
    _ = sim.updated_copy(sweep_spec=td.EMEModeSweep(num_modes=list(np.arange(1, 5))))
    # test sweep size limit
    with pytest.raises(pd.ValidationError):
        _ = sim_no_field.updated_copy(sweep_spec=td.EMELengthSweep(scale_factors=[]))
    sim_bad = sim_no_field.updated_copy(
        sweep_spec=td.EMELengthSweep(scale_factors=list(np.linspace(1, 2, 200)))
    )
    with pytest.raises(SetupError):
        sim_bad.validate_pre_upload()
    # can't exceed max num modes
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(sweep_spec=td.EMEModeSweep(num_modes=list(np.arange(150, 200))))

    # don't warn in these two cases
    with AssertLogLevel(None):
        sim_good = sim.updated_copy(
            constraint="passive",
            eme_grid_spec=td.EMEUniformGrid(num_cells=1, mode_spec=td.EMEModeSpec(num_modes=40)),
            grid_spec=sim.grid_spec.updated_copy(wavelength=1),
            monitors=[],
        )
        sim_good.validate_pre_upload()
        sim_good = sim.updated_copy(
            constraint=None,
            eme_grid_spec=td.EMEUniformGrid(num_cells=1, mode_spec=td.EMEModeSpec(num_modes=60)),
            grid_spec=sim.grid_spec.updated_copy(wavelength=1),
            monitors=[],
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

    # test coeffs_full_monitor
    coeffs_monitor = sim.coeffs_full_monitor
    assert isinstance(coeffs_monitor, td.EMECoefficientMonitor)
    assert coeffs_monitor.name == "_eme_coeffs_full_monitor"
    # coeffs monitor is included in _monitors_full when store_coeffs=True
    sim_with_coeffs = sim.updated_copy(store_coeffs=True, monitors=[])
    assert any(m.name == "_eme_coeffs_full_monitor" for m in sim_with_coeffs._monitors_full)
    # coeffs monitor is not included when store_coeffs=False
    sim_no_coeffs = sim.updated_copy(store_coeffs=False, monitors=[])
    assert not any(m.name == "_eme_coeffs_full_monitor" for m in sim_no_coeffs._monitors_full)

    # test freq sweep
    sim = sim.updated_copy(sweep_spec=None)
    assert sim._num_sweep == 1
    assert not sim._sweep_modes
    sim = sim_no_field.updated_copy(sweep_spec=td.EMELengthSweep(scale_factors=[1, 2]))
    assert not sim._sweep_modes
    assert sim._num_sweep == 2
    with AssertLogLevel("WARNING", contains_str="'EMEFreqSweep' is deprecated"):
        sim = sim.updated_copy(sweep_spec=td.EMEFreqSweep(freq_scale_factors=[1, 2]))
    assert sim._sweep_modes
    assert sim._num_sweep == 2
    assert sim._monitor_num_sweep(sim.monitors[0]) == 1
    sim = sim.updated_copy(monitors=(sim.monitors[0].updated_copy(num_sweep=None),))
    assert sim._monitor_num_sweep(sim.monitors[0]) == 2
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(monitors=(sim.monitors[0].updated_copy(num_sweep=4),))
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(sweep_spec=td.EMEFreqSweep(freq_scale_factors=[1e-10, 2]))

    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(
            eme_grid_spec=td.EMEExplicitGrid(
                boundaries=(-sim.size[2] / 2 + 0.001,),
                mode_specs=(td.EMEModeSpec(), td.EMEModeSpec()),
            )
        )
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(
            eme_grid_spec=td.EMEExplicitGrid(
                boundaries=(sim.size[2] / 2 - 0.001,),
                mode_specs=(td.EMEModeSpec(), td.EMEModeSpec()),
            )
        )
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(
            monitors=(
                td.ModeSolverMonitor(
                    center=(0, 0, sim.size[2] / 2 - 0.001),
                    size=(td.inf, td.inf, 0),
                    name="modes",
                    freqs=sim.freqs,
                    mode_spec=td.ModeSpec(),
                ),
            )
        )


def test_eme_bend_medium_frames():
    sim = make_eme_sim()
    monitor = (sim.monitors[0],)

    diag_aniso_med = td.AnisotropicMedium(
        xx=td.Medium(permittivity=2),
        yy=td.Medium(permittivity=3),
        zz=td.Medium(permittivity=4),
    )
    diag_struct = sim.structures[0].updated_copy(medium=diag_aniso_med)

    global_bent_mode_spec = td.EMEModeSpec(
        num_modes=1,
        bend_radius=10.0,
        bend_axis=1,
        bend_medium_frame="global",
    )
    co_rotating_bent_mode_spec = global_bent_mode_spec.updated_copy(bend_medium_frame="co_rotating")

    global_repeated_grid = td.EMEUniformGrid(
        num_cells=1,
        mode_spec=global_bent_mode_spec,
        num_reps=2,
    )
    co_rotating_repeated_grid = td.EMEUniformGrid(
        num_cells=1,
        mode_spec=co_rotating_bent_mode_spec,
        num_reps=2,
    )

    with AssertLogLevel("ERROR", contains_str="check convergence"):
        with pytest.raises(pd.ValidationError):
            _ = sim.updated_copy(
                structures=(diag_struct,),
                monitors=monitor,
                eme_grid_spec=global_repeated_grid,
            )

    _ = sim.updated_copy(
        structures=(diag_struct,),
        monitors=monitor,
        eme_grid_spec=co_rotating_repeated_grid,
    )

    coords = {
        "x": np.linspace(-0.25, 0.25, 2),
        "y": np.linspace(-0.5, 0.5, 2),
        "z": np.linspace(-1.5, 1.5, 3),
    }
    permittivity = td.SpatialDataArray(np.full((2, 2, 3), 2.5), coords=coords)
    custom_medium = td.CustomMedium(permittivity=permittivity)
    custom_struct = sim.structures[0].updated_copy(medium=custom_medium)

    global_bent_grid = td.EMEUniformGrid(num_cells=1, mode_spec=global_bent_mode_spec)
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(
            structures=(custom_struct,),
            monitors=monitor,
            eme_grid_spec=global_bent_grid,
        )

    co_rotating_bent_grid = td.EMEUniformGrid(num_cells=1, mode_spec=co_rotating_bent_mode_spec)
    _ = sim.updated_copy(
        structures=(custom_struct,),
        monitors=monitor,
        eme_grid_spec=co_rotating_bent_grid,
    )


def _with_eme_custom_medium_global_bend(sim):
    coords = {
        "x": np.linspace(-0.25, 0.25, 2),
        "y": np.linspace(-0.5, 0.5, 2),
        "z": np.linspace(-1.5, 1.5, 3),
    }
    custom_medium = td.CustomMedium(
        permittivity=td.SpatialDataArray(np.full((2, 2, 3), 2.5), coords=coords)
    )
    custom_struct = sim.structures[0].updated_copy(medium=custom_medium)
    global_bent_grid = td.EMEUniformGrid(
        num_cells=1,
        mode_spec=td.EMEModeSpec(
            num_modes=1,
            bend_radius=10.0,
            bend_axis=1,
            bend_medium_frame="global",
        ),
    )

    return sim.updated_copy(
        structures=(custom_struct,),
        monitors=(sim.monitors[0],),
        eme_grid_spec=global_bent_grid,
    )


def _with_eme_anisotropic_global_repeated_bend(sim):
    diag_aniso_med = td.AnisotropicMedium(
        xx=td.Medium(permittivity=2),
        yy=td.Medium(permittivity=3),
        zz=td.Medium(permittivity=4),
    )
    diag_struct = sim.structures[0].updated_copy(medium=diag_aniso_med)
    global_repeated_grid = td.EMEUniformGrid(
        num_cells=1,
        mode_spec=td.EMEModeSpec(
            num_modes=1,
            bend_radius=10.0,
            bend_axis=1,
            bend_medium_frame="global",
        ),
        num_reps=2,
    )

    return sim.updated_copy(
        structures=(diag_struct,),
        monitors=(sim.monitors[0],),
        eme_grid_spec=global_repeated_grid,
    )


@pytest.mark.parametrize(
    "sim_updater,expected_loc,message_contains",
    [
        (
            _with_eme_custom_medium_global_bend,
            ("eme_grid_spec",),
            "Custom media are not currently supported",
        ),
        (
            _with_eme_anisotropic_global_repeated_bend,
            ("eme_grid_spec",),
            "nontrivial relative bend rotation",
        ),
    ],
    ids=["eme_custom_medium_global_bend", "eme_anisotropic_repeated_bend"],
)
def test_eme_bend_validation_error_locs(eme_base_sim, sim_updater, expected_loc, message_contains):
    with pytest.raises(pd.ValidationError) as excinfo:
        _ = sim_updater(eme_base_sim)
    assert_single_value_error_loc(excinfo, expected_loc, message_contains)


def test_eme_anisotropic_bend_validation_uses_cell_specific_freqs():
    sim = make_eme_sim()
    freq0 = float(sim.freqs[0])
    medium_yy, medium_zz = _matched_lorentz_media_yy_zz(freq0)

    first_cell_only_struct = td.Structure(
        geometry=td.Box(center=(0, 0, -0.75), size=(0.5, 1.0, 1.4)),
        medium=td.AnisotropicMedium(
            xx=td.Medium(permittivity=2),
            yy=medium_yy,
            zz=medium_zz,
        ),
    )
    bent_repeated_grid = td.EMEUniformGrid(
        num_cells=1,
        mode_spec=td.EMEModeSpec(num_modes=1, bend_radius=10.0, bend_axis=0),
        num_reps=2,
    )
    extra_sampling_grid = td.EMEUniformGrid(
        num_cells=1,
        mode_spec=td.EMEModeSpec(num_modes=1, group_index_step=0.1),
    )
    mixed_grid = td.EMECompositeGrid(
        subgrids=[bent_repeated_grid, extra_sampling_grid],
        subgrid_boundaries=[0],
    )

    assert np.isclose(medium_yy.eps_model(freq0), medium_zz.eps_model(freq0))
    assert not np.isclose(medium_yy.eps_model(freq0 * 1.1), medium_zz.eps_model(freq0 * 1.1))

    _ = sim.updated_copy(
        structures=(first_cell_only_struct,),
        monitors=(sim.monitors[0],),
        eme_grid_spec=mixed_grid,
    )


def test_eme_periodicity_sweep_reuses_grid_rotation_validation_data(monkeypatch):
    sim = make_eme_sim()
    invariant_aniso_struct = sim.structures[0].updated_copy(
        medium=td.AnisotropicMedium(
            xx=td.Medium(permittivity=2),
            yy=td.Medium(permittivity=2),
            zz=td.Medium(permittivity=2),
        )
    )
    periodic_sweep = td.EMEPeriodicitySweep(num_reps=[{"periodic_unit": 2}, {"periodic_unit": 3}])
    sim = sim.updated_copy(
        structures=(invariant_aniso_struct,),
        monitors=(sim.monitors[0],),
        eme_grid_spec=td.EMEUniformGrid(
            num_cells=1,
            mode_spec=td.EMEModeSpec(
                num_modes=1,
                bend_radius=10.0,
                bend_axis=1,
                bend_medium_frame="global",
            ),
            name="periodic_unit",
        ),
        sweep_spec=periodic_sweep,
    )

    original_method = td.EMESimulation._grid_rotation_validation_data
    num_calls = 0

    def wrapped_grid_rotation_validation_data(self, *args, **kwargs):
        nonlocal num_calls
        num_calls += 1
        return original_method(self, *args, **kwargs)

    monkeypatch.setattr(
        td.EMESimulation,
        "_grid_rotation_validation_data",
        wrapped_grid_rotation_validation_data,
    )

    sim._validate_anisotropic_bend_repetitions()

    assert num_calls == 1 + len(periodic_sweep.num_reps)


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
    coords = {
        "x": x,
        "y": y,
        "z": z,
        "f": f,
        "sweep_index": sweep_index,
        "eme_cell_index": eme_cell_index,
        "mode_index": mode_index,
    }
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
    coords = {
        "x": x,
        "y": y,
        "z": z,
        "f": f,
        "sweep_index": sweep_index,
        "eme_port_index": eme_port_index,
        "mode_index": mode_index,
    }
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
    coords = {
        "f": f,
        "mode_index_out": mode_index_out,
        "mode_index_in": mode_index_in,
        "sweep_index": sweep_index,
    }
    smatrix_entry = td.EMESMatrixDataArray(data, coords=coords)

    if num_modes_in == 0:
        smatrix_entry = smatrix_entry.drop_vars("mode_index_in")
    if num_modes_out == 0:
        smatrix_entry = smatrix_entry.drop_vars("mode_index_out")
    if num_sweep == 0:
        smatrix_entry = smatrix_entry.drop_vars("sweep_index")

    return smatrix_entry


def _get_eme_interface_smatrix_data_array(
    num_modes_in=2, num_modes_out=3, num_freqs=2, num_sweep=0
):
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
    eme_cell_index = np.arange(3)

    f = td.C_0 * np.linspace(1, 2, num_freqs)

    data = (1 + 1j) * np.random.random(
        (len(f), len(sweep_index), len(eme_cell_index), len(mode_index_out), len(mode_index_in))
    )
    coords = {
        "f": f,
        "sweep_index": sweep_index,
        "eme_cell_index": eme_cell_index,
        "mode_index_out": mode_index_out,
        "mode_index_in": mode_index_in,
    }
    smatrix_entry = td.EMEInterfaceSMatrixDataArray(data, coords=coords)

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


def _get_eme_interface_smatrix_dataset(num_modes_1=3, num_modes_2=4, num_sweep=0):
    S11 = _get_eme_interface_smatrix_data_array(
        num_modes_in=num_modes_1, num_modes_out=num_modes_1, num_sweep=num_sweep
    )
    S12 = _get_eme_interface_smatrix_data_array(
        num_modes_in=num_modes_2, num_modes_out=num_modes_1, num_sweep=num_sweep
    )
    S21 = _get_eme_interface_smatrix_data_array(
        num_modes_in=num_modes_1, num_modes_out=num_modes_2, num_sweep=num_sweep
    )
    S22 = _get_eme_interface_smatrix_data_array(
        num_modes_in=num_modes_2, num_modes_out=num_modes_2, num_sweep=num_sweep
    )
    return td.EMEInterfaceSMatrixDataset(S11=S11, S12=S12, S21=S21, S22=S22)


def _get_eme_overlaps_dataset(num_modes_1=3, num_modes_2=4, num_sweep=0):
    O11 = _get_eme_interface_smatrix_data_array(
        num_modes_in=num_modes_1, num_modes_out=num_modes_1, num_sweep=num_sweep
    )
    O12 = _get_eme_interface_smatrix_data_array(
        num_modes_in=num_modes_2, num_modes_out=num_modes_1, num_sweep=num_sweep
    )
    O21 = _get_eme_interface_smatrix_data_array(
        num_modes_in=num_modes_1, num_modes_out=num_modes_2, num_sweep=num_sweep
    )
    return td.EMEOverlapDataset(O11=O11, O12=O12, O21=O21)


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
    coords = {
        "f": f,
        "sweep_index": sweep_index,
        "eme_port_index": eme_port_index,
        "eme_cell_index": eme_cell_index,
        "mode_index_out": mode_index_out,
        "mode_index_in": mode_index_in,
    }
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
    flux = _get_eme_flux_data_array(num_sweep=num_sweep)
    n_complex = _get_eme_mode_index_data_array(num_sweep=num_sweep)
    interface_smatrices = _get_eme_interface_smatrix_dataset(num_sweep=num_sweep)
    overlaps = _get_eme_overlaps_dataset(num_sweep=num_sweep)
    return td.EMECoefficientDataset(
        A=A,
        B=B,
        flux=flux,
        n_complex=n_complex,
        interface_smatrices=interface_smatrices,
        overlaps=overlaps,
    )


def test_eme_normalize_coeff_dataset():
    coeffs = _get_eme_coeff_dataset()
    coeffs_normalized = coeffs.normalized_copy
    assert coeffs_normalized.flux is None
    with pytest.raises(ValidationError):
        _ = coeffs_normalized.normalized_copy


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
    coords = {
        "f": f,
        "sweep_index": sweep_index,
        "eme_cell_index": eme_cell_index,
        "mode_index": mode_index,
    }
    data = td.EMEModeIndexDataArray(
        (1 + 1j)
        * np.random.random((len(f), len(sweep_index), len(eme_cell_index), len(mode_index))),
        coords=coords,
    )
    if num_sweep == 0:
        data = data.drop_vars("sweep_index")
    return data


def _get_eme_flux_data_array(num_sweep=0):
    f = [td.C_0, 3e14]
    mode_index = np.arange(10)
    eme_cell_index = np.arange(7)
    if num_sweep != 0:
        sweep_index = np.arange(num_sweep)
    else:
        sweep_index = [0]
    coords = {
        "f": f,
        "sweep_index": sweep_index,
        "eme_cell_index": eme_cell_index,
        "mode_index": mode_index,
    }
    data = td.EMEFluxDataArray(
        np.random.random((len(f), len(sweep_index), len(eme_cell_index), len(mode_index))),
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
    fields = dict.fromkeys(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"], field)

    return td.EMEModeSolverDataset(n_complex=n_complex, **fields)


def _get_eme_field_dataset(num_sweep=0):
    field = _get_eme_scalar_field_data_array(num_sweep=num_sweep)
    fields = dict.fromkeys(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"], field)
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
    grid_correction_coords = {
        "f": n_complex.f,
        "sweep_index": sweep_index,
        "eme_cell_index": n_complex.eme_cell_index,
        "mode_index": n_complex.mode_index,
    }
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
        propagation_axis=2,
        grid_primal_correction=grid_primal_correction,
        grid_dual_correction=grid_dual_correction,
        **kwargs,
    )


@pytest.mark.slow
def _get_eme_field_data(num_sweep=0):
    dataset = _get_eme_field_dataset(num_sweep=num_sweep)
    kwargs = dataset.field_components
    monitor = td.EMEFieldMonitor(size=(0, td.inf, td.inf), name="field", colocate=True)
    return td.EMEFieldData(monitor=monitor, propagation_axis=0, **kwargs)


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
        mode_spec=td.ModeSpec(
            num_modes=num_modes, interp_spec=td.ModeInterpSpec.cheb(num_points=3, reduce_data=True)
        ),
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
    return td.ModeSolverData(
        monitor=monitor,
        grid_expanded=grid_expanded,
        **kwargs,
    )


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


@pytest.mark.slow
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

    sim_data = td.EMESimulationData(simulation=sim, data=data, smatrix=smatrix, port_modes_raw=None)
    with pytest.raises(SetupError):
        _ = sim_data.port_modes_tuple
    with pytest.raises(SetupError):
        _ = sim_data.port_modes_list_sweep

    sim_data = td.EMESimulationData(
        simulation=sim, data=data, smatrix=smatrix, port_modes_raw=port_modes
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
        _ = sim_data.updated_copy(port_modes_raw=None).smatrix_in_basis(
            modes1=modes_in_data, modes2=modes_out_data
        )
    with pytest.raises(SetupError):
        _ = sim_data.updated_copy(port_modes_raw=None).field_in_basis(
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
    _ = sim_data.plot_field(
        "field", "Ex", eme_port_index=0, val="real", f=td.C_0, mode_index=0, cmap="plasma", ax=AX
    )
    _ = sim_data.plot_field(
        "field",
        "Ex",
        eme_port_index=0,
        val="real",
        f=td.C_0,
        mode_index=0,
        cmap=plt.get_cmap("cividis"),
        ax=AX,
    )

    # test smatrix in basis with sweep
    smatrix = _get_eme_smatrix_dataset(num_modes_1=5, num_modes_2=5, num_sweep=10)
    sim_sweep = sim.updated_copy(
        sweep_spec=td.EMELengthSweep(scale_factors=np.linspace(1, 2, 10)), monitors=[]
    )
    sim_data = td.EMESimulationData(
        simulation=sim_sweep, data=[], smatrix=smatrix, port_modes_raw=port_modes
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

    with AssertLogLevel("WARNING", contains_str="flux"):
        _ = sim_data._extract_mode_solver_data(
            data=sim_data.port_modes.updated_copy(
                monitor=sim.port_modes_monitor.updated_copy(size=(0, td.inf, td.inf))
            ),
            eme_cell_index=0,
        )

    # test _validate_interp_specs error for inconsistent interp_specs
    mode_spec1 = td.EMEModeSpec(num_modes=10, interp_spec=td.ModeInterpSpec.cheb(num_points=3))
    mode_spec2 = td.EMEModeSpec(num_modes=10, interp_spec=td.ModeInterpSpec.cheb(num_points=5))
    eme_grid_spec_inconsistent = td.EMECompositeGrid(
        subgrids=[
            td.EMEUniformGrid(num_cells=2, mode_spec=mode_spec1),
            td.EMEUniformGrid(num_cells=2, mode_spec=mode_spec2),
        ],
        subgrid_boundaries=[0],
    )
    with pytest.raises(pd.ValidationError):
        sim_interp_test = sim.updated_copy(eme_grid_spec=eme_grid_spec_inconsistent)

    # test _validate_interp_specs no error for consistent interp_specs
    mode_spec_consistent = td.EMEModeSpec(
        num_modes=10, interp_spec=td.ModeInterpSpec.cheb(num_points=4)
    )
    eme_grid_spec_consistent = td.EMECompositeGrid(
        subgrids=[
            td.EMEUniformGrid(num_cells=2, mode_spec=mode_spec_consistent),
            td.EMEUniformGrid(num_cells=2, mode_spec=mode_spec_consistent),
        ],
        subgrid_boundaries=[0],
    )
    with AssertLogLevel(None):
        sim_interp_test = sim.updated_copy(eme_grid_spec=eme_grid_spec_consistent)

    # test freq sweep smatrix_in_basis
    sim = sim.updated_copy(sweep_spec=td.EMEFreqSweep(freq_scale_factors=np.linspace(1, 2, 10)))
    port_modes = _get_eme_port_modes(num_sweep=10)
    sim_data = td.EMESimulationData(
        simulation=sim, data=data, smatrix=smatrix, port_modes_raw=port_modes
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
    sim_data = sim_data.updated_copy(data=tuple(data))
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

    # 2d subsection
    region = td.Box(size=(2, 2, 0))
    subsection = eme_sim.subsection(region=region)
    assert subsection.size[2] == 0


def test_eme_periodicity():
    # give the middle subgrid a name
    sim = make_eme_sim()
    sim = sim.updated_copy(name="a", path="eme_grid_spec/subgrids/1")

    # directly give it num_reps
    # can't have field monitor
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(num_reps=2, path="eme_grid_spec/subgrids/1")

    # EMEPeriodicitySweep validation
    with pytest.raises(pd.ValidationError):
        _ = td.EMEPeriodicitySweep(num_reps=[{"a": n} for n in range(150000, 150003)])
    sweep_spec = td.EMEPeriodicitySweep(num_reps=[{"a": n} for n in range(1, 4)])
    # still can't have field monitor
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(sweep_spec=sweep_spec)

    # remove the field monitor, now it passes
    desired_cell_index_pairs = set([(i, i + 1) for i in range(6)] + [(5, 1)])
    sim = sim.updated_copy(
        monitors=tuple(m for m in sim.monitors if not isinstance(m, td.EMEFieldMonitor))
    )
    sim2 = sim.updated_copy(num_reps=2, path="eme_grid_spec/subgrids/1")
    assert set(sim2.cell_index_pairs) == desired_cell_index_pairs
    # sweep can't have coeff monitor
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(sweep_spec=sweep_spec)
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(sweep_spec=sweep_spec, store_coeffs=True, monitors=[])
    # remove coeff monitor too, now it passes
    with AssertLogLevel(None):
        sim = sim.updated_copy(
            monitors=tuple(m for m in sim.monitors if not isinstance(m, td.EMECoefficientMonitor))
        )
        sim2 = sim.updated_copy(sweep_spec=sweep_spec)
        assert set(sim2.cell_index_pairs) == desired_cell_index_pairs


def test_eme_grid_from_structures():
    sim = make_eme_sim()
    eme_grid_spec = td.EMEExplicitGrid.from_structures(
        structures=sim.structures, axis=2, mode_spec=td.EMEModeSpec(num_modes=1)
    )
    sim = sim.updated_copy(eme_grid_spec=eme_grid_spec)
    eme_grid_spec = td.EMECompositeGrid.from_structure_groups(
        structure_groups=[[], [td.Box(center=(0, 0, 0), size=(1, 1, 1))], []],
        axis=2,
        mode_specs=[td.EMEModeSpec(num_modes=1)] * 3,
        names=[None, "wg", None],
        num_reps=[1, 2, 1],
    )
    sim = sim.updated_copy(eme_grid_spec=eme_grid_spec, monitors=())
    with pytest.raises(ValidationError):
        _ = td.EMECompositeGrid.from_structure_groups(
            structure_groups=(),
            axis=2,
            mode_specs=[],
            names=[None, "wg", None],
            num_reps=[1, 2, 1],
        )
    with pytest.raises(ValidationError):
        _ = td.EMECompositeGrid.from_structure_groups(
            structure_groups=[([], [td.Box(center=(0, 0, 0), size=(1, 1, 1))], [])],
            axis=2,
            mode_specs=[td.EMEModeSpec(num_modes=1)] * 2,
            names=[None, "wg", None],
            num_reps=[1, 2, 1],
        )
    with pytest.raises(ValidationError):
        _ = td.EMECompositeGrid.from_structure_groups(
            structure_groups=[[], [td.Box(center=(0, 0, 0), size=(1, 1, 1))], []],
            axis=2,
            mode_specs=[td.EMEModeSpec(num_modes=1)] * 3,
            names=[None, "wg"],
            num_reps=[1, 2, 1],
        )
    with pytest.raises(ValidationError):
        _ = td.EMECompositeGrid.from_structure_groups(
            structure_groups=[[], [td.Box(center=(0, 0, 0), size=(1, 1, 1))], []],
            axis=2,
            mode_specs=[td.EMEModeSpec(num_modes=1)] * 3,
            names=[None, "wg", None],
            num_reps=[1, 2],
        )
    with pytest.raises(ValidationError):
        _ = td.EMECompositeGrid.from_structure_groups(
            structure_groups=[
                [],
                [td.Box(center=(0, 0, 0), size=(1, 1, 1))],
                [td.Box(center=(0, 0, 3), size=(1, 1, 1))],
            ],
            axis=2,
            mode_specs=[td.EMEModeSpec(num_modes=1)] * 3,
            names=[None, "wg", None],
            num_reps=[1, 2, 1],
        )
    _ = td.EMECompositeGrid.from_structure_groups(
        structure_groups=[
            [],
            [td.Box(center=(0, 0, 0), size=(1, 1, 1))],
            [td.Box(center=(0, 0, 1), size=(1, 1, 1))],
        ],
        axis=2,
        mode_specs=[td.EMEModeSpec(num_modes=1)] * 3,
        names=[None, "wg", None],
        num_reps=[1, 2, 1],
    )
    with pytest.raises(ValidationError):
        _ = td.EMECompositeGrid.from_structure_groups(
            structure_groups=[[], [], [td.Box(center=(0, 0, 3), size=(1, 1, 1))]],
            axis=2,
            mode_specs=[td.EMEModeSpec(num_modes=1)] * 3,
            names=[None, "wg", None],
            num_reps=[1, 2, 1],
        )


def test_eme_sim_2d():
    freq0 = td.C_0 / 1.55
    sim_size = (3, 0, 3)
    eme_grid_spec = td.EMEUniformGrid(num_cells=5, mode_spec=td.EMEModeSpec())
    monitor = td.EMEFieldMonitor(size=(td.inf, td.inf, td.inf), name="field")
    eme_sim = td.EMESimulation(
        size=sim_size,
        axis=2,
        freqs=[freq0],
        eme_grid_spec=eme_grid_spec,
        monitors=(monitor,),
        port_offsets=(0.5, 0),
    )


# --- Local staged propagation tests ---


def _mda(data, freqs, nm0, nm1):
    """Helper to wrap numpy array into EMESMatrixDataArray with singleton sweep_index."""
    from tidy3d.components.data.data_array import EMESMatrixDataArray

    data = np.asarray(data)
    if data.ndim == 3:
        data = data[:, None, :, :]
    return EMESMatrixDataArray(
        data,
        coords={
            "f": freqs,
            "sweep_index": [0],
            "mode_index_out": np.arange(nm0),
            "mode_index_in": np.arange(nm1),
        },
    )


def _local_eme_basis_modes(sim, cell_index, name, num_modes=4):
    """Build real ModeSolverData on an EME cell plane for smatrix_in_basis tests."""
    plane = sim.eme_grid.mode_planes[cell_index]
    mode_index = np.arange(num_modes)
    freqs = list(sim.freqs)
    normal_dim = "xyz"[sim.axis]
    plane_coord = plane.center[sim.axis]
    monitor = td.ModeSolverMonitor(
        size=plane.size,
        center=plane.center,
        freqs=freqs,
        mode_spec=td.ModeSpec(num_modes=num_modes),
        name=name,
        colocate=True,
        use_colocated_integration=True,
    )
    grid_boundaries = {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [-0.5, 0.5]}
    grid_boundaries[normal_dim] = [plane_coord]
    field_coords = {"x": [0.0], "y": [0.0], "z": [0.0]}
    field_coords[normal_dim] = [plane_coord]
    grid = td.Grid(
        boundaries=td.Coords(
            **grid_boundaries,
        )
    )
    coords = {
        **field_coords,
        "f": freqs,
        "mode_index": mode_index,
    }

    def field(values):
        values = np.asarray(values, dtype=np.complex128)
        data = np.broadcast_to(
            values.reshape(1, 1, 1, 1, num_modes),
            (1, 1, 1, len(freqs), num_modes),
        ).copy()
        return td.ScalarModeFieldDataArray(data, coords=coords)

    values = np.arange(1, num_modes + 1, dtype=np.complex128)
    zeros = np.zeros(num_modes, dtype=np.complex128)
    n_complex = td.ModeIndexDataArray(
        np.ones((len(freqs), num_modes), dtype=np.complex128),
        coords={"f": freqs, "mode_index": mode_index},
    )
    return td.ModeSolverData(
        monitor=monitor,
        Ex=field(values),
        Ey=field(values + 1),
        Ez=field(zeros),
        Hx=field(values + 2),
        Hy=field(values + 3),
        Hz=field(zeros),
        n_complex=n_complex,
        grid_expanded=grid,
    )


def test_smatrix_in_basis_allows_truncated_mode_sweep_port_axes():
    """Mode-sweep S-matrix axes may be a valid prefix of solved port modes."""
    from tidy3d.packaging import check_tidy3d_extras_licensed_feature

    try:
        check_tidy3d_extras_licensed_feature("local_eme", quiet=True)
    except Tidy3dImportError as exc:
        pytest.skip(f"tidy3d-extras local EME is unavailable: {exc}")

    sim = make_local_eme_sim(
        num_cells=2,
        num_modes=4,
        sweep_spec=td.EMEModeSweep(num_modes=[2, 3]),
    )
    port_modes1 = _local_eme_basis_modes(sim, 0, "modes_in")
    port_modes2 = _local_eme_basis_modes(sim, 1, "modes_out")
    freqs = list(sim.freqs)
    smatrix = td.EMESMatrixDataset(
        S11=_mda(np.zeros((len(freqs), 3, 3), dtype=complex), sim.freqs, 3, 3),
        S12=_mda(np.zeros((len(freqs), 3, 3), dtype=complex), sim.freqs, 3, 3),
        S21=_mda(np.zeros((len(freqs), 3, 3), dtype=complex), sim.freqs, 3, 3),
        S22=_mda(np.zeros((len(freqs), 3, 3), dtype=complex), sim.freqs, 3, 3),
    )

    result = sim.smatrix_in_basis(
        smatrix,
        (port_modes1, port_modes2),
        modes1=port_modes1,
        modes2=port_modes2,
    )

    assert result.S11.shape == (len(freqs), 1, 4, 4)
    assert result.S12.shape == (len(freqs), 1, 4, 4)
    np.testing.assert_array_equal(result.S11.mode_index_in.values, [0, 1, 2, 3])
    np.testing.assert_array_equal(result.S22.mode_index_in.values, [0, 1, 2, 3])


def make_local_eme_sim(num_cells=3, num_modes=4, sweep_spec=None, constraint="passive"):
    """Create a small EMESimulation for local propagation testing."""
    lambda0 = 1.55
    freq0 = td.C_0 / lambda0
    return td.EMESimulation(
        size=(2 * lambda0, 2 * lambda0, 3 * lambda0),
        structures=[
            td.Structure(
                geometry=td.Box(size=(lambda0 / 2, lambda0 / 2, td.inf)),
                medium=td.Medium(permittivity=2.25),
            )
        ],
        grid_spec=td.GridSpec.auto(wavelength=lambda0, min_steps_per_wvl=6),
        axis=2,
        eme_grid_spec=td.EMEUniformGrid(
            num_cells=num_cells,
            mode_spec=td.EMEModeSpec(num_modes=num_modes, num_pml=(6, 6)),
        ),
        freqs=[freq0],
        sweep_spec=sweep_spec,
        constraint=constraint,
    )


def _mixed_mode_basis(modes, mixing, mode_index_start=0):
    """Return a ModeSolverData copy whose fields are linear combinations of modes."""
    mixing = np.asarray(mixing, dtype=complex)
    new_mode_index = np.arange(mode_index_start, mode_index_start + mixing.shape[0])

    mixed_fields = {}
    for field_name, field_data in modes.field_components.items():
        data = np.einsum("...p,ap->...a", field_data.to_numpy(), mixing)
        coords = {dim: field_data.coords[dim].to_numpy() for dim in field_data.dims}
        coords["mode_index"] = new_mode_index
        mixed_fields[field_name] = type(field_data)(data, coords=coords)

    n_complex = modes.n_complex
    n_data = np.tile(n_complex.to_numpy()[:, :1], (1, mixing.shape[0]))
    n_coords = {"f": n_complex.f.to_numpy(), "mode_index": new_mode_index}
    mixed_n_complex = type(n_complex)(n_data, coords=n_coords)

    return modes.updated_copy(
        **mixed_fields,
        n_complex=mixed_n_complex,
        grid_primal_correction=1,
        grid_dual_correction=1,
        deep=False,
        validate=False,
    )


def _single_field_basis(modes, mode_index=0, name="field_basis"):
    """Return a single-vector field basis without a mode_index coordinate."""
    fields = {
        key: field.isel(mode_index=mode_index, drop=True)
        for key, field in modes.field_components.items()
    }
    monitor = td.FieldMonitor(
        center=modes.monitor.center,
        size=(td.inf, td.inf, 0),
        freqs=list(modes.monitor.freqs),
        name=name,
    )
    return td.components.data.monitor_data.ElectromagneticFieldData(
        **fields,
        monitor=monitor,
        grid_expanded=modes.grid_expanded,
    )


def _smatrix_block_with_sweep(block):
    """Return an S-matrix block as (f, sweep_index, mode_index_out, mode_index_in)."""
    if "sweep_index" in block.dims:
        return block.transpose("f", "sweep_index", "mode_index_out", "mode_index_in").to_numpy()
    return block.transpose("f", "mode_index_out", "mode_index_in").to_numpy()[:, None, :, :]


def _overlap_block(modes, port_modes, freqs, port_mode_index):
    overlaps = modes.outer_dot(port_modes, conjugate=False, bidirectional=True).sel(
        f=freqs,
        mode_index_1=port_mode_index,
    )
    if "mode_index_0" not in overlaps.dims:
        overlaps = overlaps.expand_dims(dim={"mode_index_0": [0]}, axis=1)
    return overlaps.transpose("f", "mode_index_0", "mode_index_1").to_numpy()


def _assert_smatrix_in_basis_matches_outer_dot_oracle(
    sim,
    smatrix,
    actual,
    port_modes,
    modes1,
    modes2,
):
    """Check local smatrix_in_basis against the public overlap contraction formula."""
    port_cell_1 = sim.eme_grid_spec.virtual_cell_indices[0]
    port_cell_2 = sim.eme_grid_spec.virtual_cell_indices[-1]
    port_modes1 = sim.stage_cell_modes(port_modes[0], cell_index=port_cell_1).modes
    port_modes2 = sim.stage_cell_modes(port_modes[1], cell_index=port_cell_2).modes

    freqs = smatrix.S11.f.to_numpy()
    mi1 = smatrix.S11.mode_index_in.to_numpy()
    mi2 = smatrix.S22.mode_index_in.to_numpy()
    O1 = _overlap_block(modes1, port_modes1, freqs, mi1)
    O2 = _overlap_block(modes2, port_modes2, freqs, mi2)

    S11 = _smatrix_block_with_sweep(smatrix.S11)
    S12 = _smatrix_block_with_sweep(smatrix.S12)
    S21 = _smatrix_block_with_sweep(smatrix.S21)
    S22 = _smatrix_block_with_sweep(smatrix.S22)

    expected11 = np.einsum("fap,fspq,fbq->fsab", O1, S11, O1)
    expected12 = np.einsum("fap,fspq,fbq->fsab", O1, S12, O2)
    expected21 = np.einsum("fap,fspq,fbq->fsab", O2, S21, O1)
    expected22 = np.einsum("fap,fspq,fbq->fsab", O2, S22, O2)

    np.testing.assert_allclose(
        _smatrix_block_with_sweep(actual.S11), expected11, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        _smatrix_block_with_sweep(actual.S12), expected12, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        _smatrix_block_with_sweep(actual.S21), expected21, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        _smatrix_block_with_sweep(actual.S22), expected22, rtol=1e-10, atol=1e-12
    )


def test_eme_stage_models():
    """Stage model creation and identity stamping."""
    S = _mda(np.eye(2, dtype=complex).reshape(1, 2, 2), [2e14], 2, 2)
    nc = td.ModeIndexDataArray(
        (1.5 + 0j) * np.ones((1, 2)), coords={"f": [2e14], "mode_index": [0, 1]}
    )
    fl = td.FreqModeDataArray(
        (0.5 + 0j) * np.ones((1, 2)), coords={"f": [2e14], "mode_index": [0, 1]}
    )

    # Cell overlap with identity
    co = td.EMEStageCellOverlap(cell_index=2, n_complex=nc, complex_flux=fl, self_overlap=S)
    assert co.cell_index == 2

    # Interface overlap with identity
    io = td.EMEStageInterfaceOverlap(cell_index=0, right_cell_index=1, O12=S, O21=S)
    assert io.cell_index == 0
    assert io.right_cell_index == 1

    # Cell S-matrix with identity
    csm = td.EMEStageCellSMatrix(cell_index=1, sweep_index=3, S11=S, S12=S, S21=S, S22=S)
    assert csm.cell_index == 1
    assert csm.sweep_index == 3
    assert isinstance(csm, td.EMESMatrixDataset)

    # Interface S-matrix with identity
    ism = td.EMEStageInterfaceSMatrix(
        cell_index=0, right_cell_index=1, sweep_index=0, S11=S, S12=S, S21=S, S22=S
    )
    assert ism.right_cell_index == 1
    assert isinstance(ism, td.EMESMatrixDataset)


def test_eme_stage_serialization():
    """HDF5 round-trip for stage models."""
    import os
    import tempfile

    S = _mda(np.eye(2, dtype=complex).reshape(1, 2, 2) * (0.7 + 0.3j), [2e14], 2, 2)
    nc = td.ModeIndexDataArray(
        (1.5 + 0j) * np.ones((1, 2)), coords={"f": [2e14], "mode_index": [0, 1]}
    )
    fl = td.FreqModeDataArray(
        (0.5 + 0j) * np.ones((1, 2)), coords={"f": [2e14], "mode_index": [0, 1]}
    )

    def _roundtrip(obj, cls):
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
            path = f.name
        try:
            obj.to_hdf5(path)
            return cls.from_hdf5(path)
        finally:
            os.unlink(path)

    # Cell overlap round-trip
    co = td.EMEStageCellOverlap(cell_index=0, n_complex=nc, complex_flux=fl, self_overlap=S)
    co2 = _roundtrip(co, td.EMEStageCellOverlap)
    assert co2.cell_index == 0
    np.testing.assert_allclose(co2.self_overlap.values, S.values)

    # Cell S-matrix round-trip
    csm = td.EMEStageCellSMatrix(cell_index=1, sweep_index=0, S11=S, S12=S, S21=S, S22=S)
    csm2 = _roundtrip(csm, td.EMEStageCellSMatrix)
    assert csm2.cell_index == 1
    np.testing.assert_allclose(csm2.S21.values, S.values)

    # Interface S-matrix round-trip
    ism = td.EMEStageInterfaceSMatrix(
        cell_index=0, right_cell_index=1, sweep_index=0, S11=S, S12=S, S21=S, S22=S
    )
    ism2 = _roundtrip(ism, td.EMEStageInterfaceSMatrix)
    assert ism2.cell_index == 0
    assert ism2.right_cell_index == 1
    np.testing.assert_allclose(ism2.S12.values, S.values)


def test_eme_mode_simulations():
    """mode_simulations property returns correct ModeSimulation objects."""
    from tidy3d.components.mode.simulation import ModeSimulation

    sim = make_local_eme_sim(num_cells=3)
    mode_sims = sim.mode_simulations
    assert len(mode_sims) == 3
    for ms in mode_sims:
        assert isinstance(ms, ModeSimulation)
        np.testing.assert_array_equal(ms.freqs, sim.freqs)
        assert ms.plane.size.count(0.0) == 1

    # Property works even with a sweep_spec — always returns full modes
    sweep = td.EMEModeSweep(num_modes=[2, 4])
    sim_sweep = make_local_eme_sim(num_modes=4, sweep_spec=sweep)
    mode_sims = sim_sweep.mode_simulations
    for ms in mode_sims:
        assert ms.mode_spec.num_modes == 4

    # Bent anisotropic media in the global frame is rejected by the local
    # path: subpixel runs before the bend rotation and does not yet
    # support fully anisotropic tensors.
    diag_aniso_med = td.AnisotropicMedium(
        xx=td.Medium(permittivity=2),
        yy=td.Medium(permittivity=3),
        zz=td.Medium(permittivity=4),
    )
    base_sim = make_eme_sim()
    bent_sim = base_sim.updated_copy(
        structures=(base_sim.structures[0].updated_copy(medium=diag_aniso_med),),
        eme_grid_spec=td.EMEUniformGrid(
            num_cells=3,
            mode_spec=td.EMEModeSpec(
                num_modes=2,
                bend_radius=10.0,
                bend_axis=1,
                bend_medium_frame="global",
            ),
        ),
    )
    with pytest.raises(SetupError, match="bend_medium_frame"):
        _ = bent_sim.mode_simulations

    # The co-rotating frame is fine because each cell's local frame
    # already encodes the bend orientation, so no explicit rotation is
    # required at the solver boundary.
    co_rotating_sim = bent_sim.updated_copy(
        eme_grid_spec=td.EMEUniformGrid(
            num_cells=3,
            mode_spec=td.EMEModeSpec(
                num_modes=2,
                bend_radius=10.0,
                bend_axis=1,
                bend_medium_frame="co_rotating",
            ),
        ),
    )
    assert len(co_rotating_sim.mode_simulations) == 3


def test_eme_cell_lengths():
    """_get_cell_lengths resolves from grid and sweep."""
    sim = make_local_eme_sim(num_cells=3)
    lengths = sim._get_cell_lengths(None)
    assert len(lengths) == 3 and all(L > 0 for L in lengths)
    sweep = td.EMELengthSweep(scale_factors=[2.0, 0.5])
    sim2 = make_local_eme_sim(num_cells=3, sweep_spec=sweep)
    base = sim2._get_cell_lengths(None)
    scaled = sim2._get_cell_lengths(0)
    for b, s in zip(base, scaled):
        np.testing.assert_allclose(s, b * 2.0)


def test_eme_local_monitor_warning_dedup_key():
    """Helper covers three requirements: fire when monitors are present, dedupe
    via log_once for identical monitor sets, and use a key that distinguishes
    sets differing in type or placement (so different sims in the same process
    don't silently collide)."""
    sim = make_local_eme_sim(num_cells=3)

    mnt_field_a = td.EMEFieldMonitor(size=(0, td.inf, td.inf), name="field", colocate=True)
    mnt_field_b = td.EMEFieldMonitor(size=(td.inf, 0, td.inf), name="field", colocate=True)
    mnt_mode = td.EMEModeSolverMonitor(size=(0, td.inf, td.inf), name="field")

    sim_a = sim.updated_copy(monitors=[mnt_field_a])

    # First call for a given monitor set warns.
    with AssertLogLevel("WARNING", contains_str="field"):
        sim_a._warn_if_local_ignores_monitors()

    # Identical monitor set → deduped by log_once.
    with AssertLogLevel(None):
        sim.updated_copy(monitors=[mnt_field_a])._warn_if_local_ignores_monitors()

    # Different placement (same name + type) → distinct key, warns again.
    with AssertLogLevel("WARNING", contains_str="field"):
        sim.updated_copy(monitors=[mnt_field_b])._warn_if_local_ignores_monitors()

    # Different type (same name + placement) → distinct key, warns again.
    with AssertLogLevel("WARNING", contains_str="field"):
        sim.updated_copy(monitors=[mnt_mode])._warn_if_local_ignores_monitors()


@pytest.mark.numerical
def test_eme_local_warns_when_monitors_dropped():
    """Explicit per-element staged propagation wires the monitor-drop warning
    — not just the convenience helpers. Covers the regression where
    compute_cell_smatrix / compute_interface_smatrix / compute_smatrix bypassed
    the three originally-hooked entry points."""
    sim = make_local_eme_sim(num_cells=2, num_modes=3)
    mnt = td.EMEFieldMonitor(size=(0, td.inf, td.inf), name="staged_monitor", colocate=True)
    sim_with_mnt = sim.updated_copy(monitors=[mnt])

    mode_data = [ms.run_local() for ms in sim_with_mnt.mode_simulations]

    # mode_simulations fired the log-once warning above; clear the cache so the
    # staged-flow call sites below get a fair check on their own.
    td.log._static_cache.clear()

    with AssertLogLevel("WARNING", contains_str="staged_monitor") as ctx:
        cell_modes = [
            sim_with_mnt.stage_cell_modes(md, cell_index=i) for i, md in enumerate(mode_data)
        ]
        cell_overlaps = [sim_with_mnt.compute_cell_overlap(cm) for cm in cell_modes]
        iface_overlaps = [
            sim_with_mnt.compute_interface_overlap(cell_modes[li], cell_modes[ri])
            for li, ri in sim_with_mnt.cell_index_pairs
        ]
        cell_sms = [sim_with_mnt.compute_cell_smatrix(co) for co in cell_overlaps]
        iface_sms = [
            sim_with_mnt.compute_interface_smatrix(cell_overlaps[li], cell_overlaps[ri], io)
            for (li, ri), io in zip(sim_with_mnt.cell_index_pairs, iface_overlaps)
        ]
        sim_with_mnt.compute_smatrix(cell_overlaps, cell_sms, iface_sms)

    monitor_warns = [msg for _, msg in ctx.records if "staged_monitor" in msg]
    assert len(monitor_warns) == 1, (
        f"Expected the monitor-drop warning exactly once across the explicit "
        f"staged pipeline; got {len(monitor_warns)}."
    )


@pytest.mark.numerical
def test_eme_local_tunneling():
    """Tunneling with passive constraint: unitarity and reciprocity."""
    lambda0 = 1
    freq0 = td.C_0 / lambda0
    n1, n2 = 2, 1
    L = lambda0 / 4
    sim = td.EMESimulation(
        size=(lambda0 / 3, lambda0 / 15, L + lambda0),
        structures=[
            td.Structure(
                geometry=td.Box(center=(0, 0, 0), size=(td.inf, td.inf, L)),
                medium=td.Medium(permittivity=n2**2),
            )
        ],
        medium=td.Medium(permittivity=n1**2),
        freqs=[freq0],
        axis=2,
        grid_spec=td.GridSpec.auto(wavelength=lambda0, min_steps_per_wvl=30),
        eme_grid_spec=td.EMEExplicitGrid(
            boundaries=[-L / 2, L / 2],
            mode_specs=[td.EMEModeSpec(num_modes=1)] * 3,
        ),
    )
    mode_data = [ms.run_local() for ms in sim.mode_simulations]
    smatrix = sim.propagate(mode_data)
    S11 = smatrix.S11.values.squeeze()
    S12 = smatrix.S12.values.squeeze()
    S21 = smatrix.S21.values.squeeze()
    S22 = smatrix.S22.values.squeeze()
    assert abs(abs(S11) ** 2 + abs(S21) ** 2 - 1.0) < 0.02
    assert abs(abs(S22) ** 2 + abs(S12) ** 2 - 1.0) < 0.02
    assert abs(S21 - S12) < 0.02


@pytest.mark.numerical
def test_eme_local_tir():
    """Total internal reflection: multi-mode interface with passive constraint."""
    lambda0 = 1
    freq0 = td.C_0 / lambda0
    sim = td.EMESimulation(
        size=(lambda0 / 3, lambda0 / 15, 3 * lambda0),
        structures=[
            td.Structure(
                geometry=td.Box.from_bounds(rmin=(-100, -100, -100), rmax=(100, 100, 0)),
                medium=td.Medium(permittivity=4),
            ),
            td.Structure(
                geometry=td.Box.from_bounds(rmin=(-100, -100, 0), rmax=(100, 100, 100)),
                medium=td.Medium(permittivity=1),
            ),
        ],
        grid_spec=td.GridSpec.auto(wavelength=lambda0, min_steps_per_wvl=50),
        axis=2,
        eme_grid_spec=td.EMEUniformGrid(num_cells=2, mode_spec=td.EMEModeSpec(num_modes=10)),
        freqs=[freq0],
        normalize=False,
        constraint="passive",
    )
    mode_data = [ms.run_local() for ms in sim.mode_simulations]
    smatrix = sim.propagate(mode_data)
    R = abs(smatrix.S11.values.squeeze()[0, 0])
    assert R > 0.99


@pytest.mark.numerical
def test_eme_local_staged_vs_oneshot():
    """Explicit per-element staged pipeline matches propagate."""
    sim = make_local_eme_sim(num_cells=2, num_modes=3)
    mode_data = [ms.run_local() for ms in sim.mode_simulations]

    # One-shot
    sm_oneshot = sim.propagate(mode_data)

    # Explicit per-element pipeline
    cell_modes = [sim.stage_cell_modes(md, cell_index=i) for i, md in enumerate(mode_data)]
    cell_overlaps = [sim.compute_cell_overlap(cm) for cm in cell_modes]
    iface_overlaps = [
        sim.compute_interface_overlap(cell_modes[li], cell_modes[ri])
        for li, ri in sim.cell_index_pairs
    ]
    cell_sms = [sim.compute_cell_smatrix(co) for co in cell_overlaps]
    iface_sms = [
        sim.compute_interface_smatrix(cell_overlaps[li], cell_overlaps[ri], io)
        for (li, ri), io in zip(sim.cell_index_pairs, iface_overlaps)
    ]
    sm_staged = sim.compute_smatrix(cell_overlaps, cell_sms, iface_sms)

    np.testing.assert_allclose(sm_oneshot.S21.values, sm_staged.S21.values, rtol=1e-12)
    np.testing.assert_allclose(sm_oneshot.S11.values, sm_staged.S11.values, atol=1e-14)

    port_modes = (mode_data[0], mode_data[-1])
    mixed1 = _mixed_mode_basis(
        mode_data[0].modes_raw,
        [[1.0, 0.25, -0.1], [0.2j, 0.8, 0.3]],
        mode_index_start=10,
    )
    mixed2 = _mixed_mode_basis(
        mode_data[-1].modes_raw,
        [[0.7, -0.2j, 0.15], [-0.1, 1.1, 0.25j]],
        mode_index_start=20,
    )
    rebased_mixed = sim.smatrix_in_basis(sm_oneshot, port_modes, modes1=mixed1, modes2=mixed2)
    _assert_smatrix_in_basis_matches_outer_dot_oracle(
        sim, sm_oneshot, rebased_mixed, port_modes, mixed1, mixed2
    )
    np.testing.assert_array_equal(rebased_mixed.S11.mode_index_in.values, [10, 11])
    np.testing.assert_array_equal(rebased_mixed.S22.mode_index_in.values, [20, 21])

    field1 = _single_field_basis(mode_data[0].modes_raw, mode_index=0, name="field_basis_1")
    rebased_field = sim.smatrix_in_basis(sm_oneshot, port_modes, modes1=field1, modes2=mixed2)
    _assert_smatrix_in_basis_matches_outer_dot_oracle(
        sim, sm_oneshot, rebased_field, port_modes, field1, mixed2
    )
    assert "mode_index_in" not in rebased_field.S11.coords
    assert "mode_index_out" not in rebased_field.S11.coords
    assert "mode_index_in" not in rebased_field.S21.coords

    raw0 = mode_data[0].modes_raw
    nc_vals = raw0.n_complex.values.copy()
    nc_vals[..., 1] = complex(np.nan, np.nan)
    new_nc = type(raw0.n_complex)(nc_vals, coords=dict(raw0.n_complex.coords))
    mode_data_nan = [
        mode_data[0].updated_copy(modes_raw=raw0.updated_copy(n_complex=new_nc)),
        *mode_data[1:],
    ]
    smatrix_nan = sim.propagate(mode_data_nan)
    assert smatrix_nan.S11.sizes["mode_index_in"] == 2
    filtered0 = sim.stage_cell_modes(mode_data_nan[0], cell_index=0).modes
    rebased_nan = sim.smatrix_in_basis(
        smatrix_nan, (mode_data_nan[0], mode_data_nan[-1]), modes1=filtered0
    )
    assert rebased_nan.S11.sizes["mode_index_in"] == 2
    assert np.isfinite(rebased_nan.S21.values).any()

    # Periodicity sweep: virtual cell indices repeat (e.g. [0, 1, 0, 1]),
    # exercising dict-based lookup by cell_index rather than list position.
    lambda0 = 1.55
    freq0 = td.C_0 / lambda0
    periodic_sim = td.EMESimulation(
        size=(2 * lambda0, 2 * lambda0, 3 * lambda0),
        structures=[
            td.Structure(
                geometry=td.Box(size=(lambda0 / 2, lambda0 / 2, td.inf)),
                medium=td.Medium(permittivity=2.25),
            )
        ],
        grid_spec=td.GridSpec.auto(wavelength=lambda0, min_steps_per_wvl=6),
        axis=2,
        eme_grid_spec=td.EMEExplicitGrid(
            boundaries=[0.0],
            mode_specs=[td.EMEModeSpec(num_modes=3, num_pml=(6, 6))] * 2,
            name="unit",
        ),
        freqs=[freq0],
        sweep_spec=td.EMEPeriodicitySweep(num_reps=[{"unit": 1}, {"unit": 3}]),
        constraint="passive",
    )
    pmode_data = [ms.run_local() for ms in periodic_sim.mode_simulations]
    sm_periodic = periodic_sim.propagate(pmode_data)
    assert sm_periodic.S21.shape[1] == 2  # two sweep points
    for si in range(2):
        T = abs(sm_periodic.S21.isel(sweep_index=si).values.squeeze()) ** 2
        assert T.sum() > 0


@pytest.mark.numerical
def test_eme_local_length_sweep():
    """Length sweep via propagate."""
    lambda0 = 1
    freq0 = td.C_0 / lambda0
    sim = td.EMESimulation(
        size=(lambda0 / 3, lambda0 / 15, lambda0 / 4 + lambda0),
        structures=[
            td.Structure(
                geometry=td.Box(center=(0, 0, 0), size=(td.inf, td.inf, lambda0 / 4)),
                medium=td.Medium(permittivity=1),
            )
        ],
        medium=td.Medium(permittivity=4),
        freqs=[freq0],
        axis=2,
        grid_spec=td.GridSpec.auto(wavelength=lambda0, min_steps_per_wvl=30),
        eme_grid_spec=td.EMEExplicitGrid(
            boundaries=[-lambda0 / 8, lambda0 / 8],
            mode_specs=[td.EMEModeSpec(num_modes=1)] * 3,
        ),
        sweep_spec=td.EMELengthSweep(scale_factors=[0.5, 1.0, 2.0]),
        constraint=None,
    )
    mode_data = [ms.run_local() for ms in sim.mode_simulations]
    smatrix = sim.propagate(mode_data)
    # Shorter barrier -> higher transmission
    T = [float(abs(smatrix.S21.isel(sweep_index=si).values.squeeze()) ** 2) for si in range(3)]
    assert T[0] > T[2]


def test_eme_propagate_rejects_freq_sweep():
    """The local staged path rejects EMEFreqSweep at every entry point."""
    from tidy3d.exceptions import SetupError

    sim = make_local_eme_sim(num_cells=2, sweep_spec=td.EMEFreqSweep(freq_scale_factors=[1.0, 1.1]))

    # mode_simulations fails before the caller ever pays for a mode solve.
    with pytest.raises(SetupError, match="EMEFreqSweep"):
        _ = sim.mode_simulations

    # Drop the EMEFreqSweep to get mode data we can hand to other entry points,
    # then reinstate it on the sim and confirm each of them also rejects.
    sim_ms = sim.updated_copy(sweep_spec=None)
    mode_data = [ms.run_local() for ms in sim_ms.mode_simulations]

    with pytest.raises(SetupError, match="EMEFreqSweep"):
        sim.propagate(mode_data)

    with pytest.raises(SetupError, match="EMEFreqSweep"):
        sim.compute_overlaps(mode_data)


def test_eme_stack_sweep_points_nan_pads_ragged_modes():
    """Stacked mode-sweep blocks preserve NaN padding for missing mode entries."""
    from tidy3d.components.eme.simulation import _stack_sweep_points

    freqs = [2e14]
    per_point_blocks = []
    for si, n in enumerate([1, 2, 4]):
        block = np.full((1, 1, n, n), fill_value=complex(si + 1, 0))
        per_point_blocks.append(
            td.EMESMatrixDataArray(
                block,
                coords={
                    "f": freqs,
                    "sweep_index": [si],
                    "mode_index_out": np.arange(n),
                    "mode_index_in": np.arange(n),
                },
            )
        )

    stacked = _stack_sweep_points(per_point_blocks)

    assert stacked.shape == (1, 3, 4, 4)

    sweep0 = stacked.isel(sweep_index=0, f=0).values
    assert sweep0[0, 0] == complex(1, 0)
    nan0 = np.isnan(sweep0)
    assert nan0.sum() == 15 and not nan0[0, 0]

    sweep1 = stacked.isel(sweep_index=1, f=0).values
    assert np.all(sweep1[:2, :2] == complex(2, 0))
    assert np.all(np.isnan(sweep1[2:, :]))
    assert np.all(np.isnan(sweep1[:, 2:]))

    sweep2 = stacked.isel(sweep_index=2, f=0).values
    assert np.all(sweep2 == complex(3, 0))
    assert not np.isnan(sweep2).any()


def test_eme_sim_data_smatrix_in_basis_preserves_pass_through_ragged_axis():
    """Partial rebasing under EMEModeSweep keeps NaN-padded pass-through port axes."""
    sim = make_eme_sim().updated_copy(
        sweep_spec=td.EMEModeSweep(num_modes=[2, 5]),
        monitors=[],
    )
    smatrix = _get_eme_smatrix_dataset(num_modes_1=5, num_modes_2=5, num_sweep=2)
    nan = complex(np.nan, np.nan)

    def _updated_smatrix_array(data_array, values):
        return data_array.copy(data=values)

    S12_values = smatrix.S12.values.copy()
    S21_values = smatrix.S21.values.copy()
    S22_values = smatrix.S22.values.copy()
    S12_values[:, 0, :, 2:] = nan
    S21_values[:, 0, 2:, :] = nan
    S22_values[:, 0, 2:, :] = nan
    S22_values[:, 0, :, 2:] = nan
    S12_values[:, 1, :, 4:] = nan
    S21_values[:, 1, 4:, :] = nan
    S22_values[:, 1, 4:, :] = nan
    S22_values[:, 1, :, 4:] = nan
    smatrix = td.EMESMatrixDataset(
        S11=smatrix.S11,
        S12=_updated_smatrix_array(smatrix.S12, S12_values),
        S21=_updated_smatrix_array(smatrix.S21, S21_values),
        S22=_updated_smatrix_array(smatrix.S22, S22_values),
    )
    sim_data = td.EMESimulationData(
        simulation=sim,
        data=[],
        smatrix=smatrix,
        port_modes_raw=_get_eme_port_modes(num_sweep=2),
    )

    rebased = sim_data.smatrix_in_basis(modes1=_get_mode_solver_data(num_modes=1))

    assert rebased.S12.shape == (1, 2, 1, 5)
    assert rebased.S21.shape == (1, 2, 5, 1)
    assert rebased.S22.shape == (1, 2, 5, 5)
    np.testing.assert_array_equal(rebased.S12.mode_index_in.values, np.arange(5))
    assert np.isnan(rebased.S22.isel(f=0, sweep_index=0, mode_index_in=4, mode_index_out=4).item())
    assert np.isfinite(
        rebased.S22.isel(f=0, sweep_index=1, mode_index_in=0, mode_index_out=0).item()
    )


def test_eme_sim_data_smatrix_in_basis_partial_ragged_matches_oracle():
    """Partial rebasing under EMEModeSweep matches an explicit per-sweep oracle."""
    sim = make_eme_sim().updated_copy(
        sweep_spec=td.EMEModeSweep(num_modes=[2, 5]),
        monitors=[],
    )
    smatrix_template = _get_eme_smatrix_dataset(num_modes_1=5, num_modes_2=5, num_sweep=2)

    def _deterministic_block(block, offset):
        values = offset + np.arange(block.size, dtype=float).reshape(block.shape)
        return block.copy(data=values + 1j * (values + 0.25))

    smatrix = td.EMESMatrixDataset(
        S11=_deterministic_block(smatrix_template.S11, 100),
        S12=_deterministic_block(smatrix_template.S12, 200),
        S21=_deterministic_block(smatrix_template.S21, 300),
        S22=_deterministic_block(smatrix_template.S22, 400),
    )

    nan = complex(np.nan, np.nan)
    S11_values = smatrix.S11.values.copy()
    S12_values = smatrix.S12.values.copy()
    S21_values = smatrix.S21.values.copy()
    S22_values = smatrix.S22.values.copy()
    # Sweep point 0 keeps only modes 0 and 1 on both ports. Sweep point 1
    # keeps all five modes, giving a ragged mode sweep after stacking.
    S11_values[:, 0, 2:, :] = nan
    S11_values[:, 0, :, 2:] = nan
    S12_values[:, 0, 2:, :] = nan
    S12_values[:, 0, :, 2:] = nan
    S21_values[:, 0, 2:, :] = nan
    S21_values[:, 0, :, 2:] = nan
    S22_values[:, 0, 2:, :] = nan
    S22_values[:, 0, :, 2:] = nan
    smatrix = td.EMESMatrixDataset(
        S11=smatrix.S11.copy(data=S11_values),
        S12=smatrix.S12.copy(data=S12_values),
        S21=smatrix.S21.copy(data=S21_values),
        S22=smatrix.S22.copy(data=S22_values),
    )
    sim_data = td.EMESimulationData(
        simulation=sim,
        data=[],
        smatrix=smatrix,
        port_modes_raw=_get_eme_port_modes(num_sweep=2),
    )
    modes1 = _get_mode_solver_data(num_modes=1)

    rebased = sim_data.smatrix_in_basis(modes1=modes1)
    mode_spec1 = modes1.monitor.mode_spec
    interp_spec1 = mode_spec1.interp_spec if mode_spec1 is not None else None
    freqs = rebased.S11.f.values
    port_modes1 = sim_data.port_modes_list_sweep[0][0]

    for sweep_index in rebased.S11.sweep_index.values:
        S11 = smatrix.S11.sel(f=freqs, sweep_index=sweep_index)
        S12 = smatrix.S12.sel(f=freqs, sweep_index=sweep_index)
        S21 = smatrix.S21.sel(f=freqs, sweep_index=sweep_index)
        S22 = smatrix.S22.sel(f=freqs, sweep_index=sweep_index)

        diag1_nan = np.isnan(np.diagonal(S11.to_numpy(), axis1=-2, axis2=-1)).any(axis=0)
        keep_inds1 = np.where(~diag1_nan)[0]
        keep_mode_inds1 = [S11.mode_index_in[i] for i in keep_inds1]

        S11 = S11.sel(mode_index_in=keep_mode_inds1, mode_index_out=keep_mode_inds1)
        S12 = S12.sel(mode_index_out=keep_mode_inds1)
        S21 = S21.sel(mode_index_in=keep_mode_inds1)

        O1 = modes1.outer_dot(port_modes1, conjugate=False)
        if interp_spec1 is not None:
            O1 = modes1._interp_dataarray_in_freq(O1, freqs=freqs, method=interp_spec1.method)
        O1 = O1.sel(f=freqs, mode_index_1=keep_mode_inds1)

        O1out = O1.rename(mode_index_0="mode_index_out", mode_index_1="mode_index_out_old")
        O1in = O1.rename(mode_index_0="mode_index_in", mode_index_1="mode_index_in_old")
        expected11 = (
            O1out.dot(
                S11.rename(
                    mode_index_in="mode_index_in_old",
                    mode_index_out="mode_index_out_old",
                ),
                dim="mode_index_out_old",
            )
            .dot(O1in, dim="mode_index_in_old")
            .transpose("f", "mode_index_out", "mode_index_in")
        )
        expected12 = O1out.dot(
            S12.rename(mode_index_out="mode_index_out_old"),
            dim="mode_index_out_old",
        ).transpose("f", "mode_index_out", "mode_index_in")
        expected21 = (
            S21.rename(mode_index_in="mode_index_in_old")
            .dot(O1in, dim="mode_index_in_old")
            .transpose("f", "mode_index_out", "mode_index_in")
        )
        expected22 = S22.transpose("f", "mode_index_out", "mode_index_in")

        np.testing.assert_allclose(
            rebased.S11.sel(sweep_index=sweep_index)
            .transpose("f", "mode_index_out", "mode_index_in")
            .values,
            expected11.values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            rebased.S12.sel(sweep_index=sweep_index)
            .transpose("f", "mode_index_out", "mode_index_in")
            .values,
            expected12.values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            rebased.S21.sel(sweep_index=sweep_index)
            .transpose("f", "mode_index_out", "mode_index_in")
            .values,
            expected21.values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            rebased.S22.sel(sweep_index=sweep_index)
            .transpose("f", "mode_index_out", "mode_index_in")
            .values,
            expected22.values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )


@pytest.mark.numerical
def test_eme_local_mode_sweep():
    """Mode sweep via propagate."""
    sim = make_local_eme_sim(
        num_cells=2,
        num_modes=4,
        sweep_spec=td.EMEModeSweep(num_modes=[1, 2, 4]),
    )
    mode_data = [ms.run_local() for ms in sim.mode_simulations]
    smatrix = sim.propagate(mode_data)
    assert smatrix.S21.shape[1] == 3
    for si in range(3):
        T = abs(smatrix.S21.isel(sweep_index=si).values.squeeze()) ** 2
        # Truncated-away modes are NaN-padded
        # (see test_eme_stack_sweep_points_nan_pads_ragged_modes).
        assert np.nansum(T) > 0
