from __future__ import annotations

import io

import numpy as np
import pydantic as pd
import pytest
from matplotlib import pyplot as plt
from rich.console import Console

import tidy3d as td
from tidy3d.components.data.data_array import EMETraceMetricDataArray
from tidy3d.components.eme import simulation as eme_simulation
from tidy3d.exceptions import SetupError, Tidy3dImportError, ValidationError
from tidy3d.log import LogHandler, log

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


def test_skip_size_checks_bypasses_eme_upload_size_limits(monkeypatch):
    sim = make_eme_sim().updated_copy(monitors=())
    max_num_freqs = eme_simulation.MAX_NUM_FREQS
    max_num_sweep = eme_simulation.MAX_NUM_SWEEP

    monkeypatch.setattr(eme_simulation, "MAX_NUM_FREQS", 0)
    with pytest.raises(SetupError, match="frequencies"):
        sim.validate_pre_upload()

    sweep_sim = sim.updated_copy(sweep_spec=td.EMELengthSweep(scale_factors=[1.0]))
    monkeypatch.setattr(eme_simulation, "MAX_NUM_SWEEP", 0)
    with pytest.raises(SetupError, match="sweep_spec"):
        sweep_sim.validate_pre_upload()

    monkeypatch.setattr(eme_simulation, "MAX_NUM_FREQS", max_num_freqs)
    monkeypatch.setattr(eme_simulation, "MAX_NUM_SWEEP", max_num_sweep)
    monkeypatch.setattr(eme_simulation, "MAX_MODE_NUM_CELLS", 0)
    with pytest.raises(SetupError, match="transverse directions"):
        sim.validate_pre_upload()

    monkeypatch.setattr(eme_simulation, "MAX_NUM_FREQS", 0)
    monkeypatch.setattr(eme_simulation, "MAX_NUM_SWEEP", 0)
    with td.config as scoped_config:
        scoped_config.simulation.skip_size_checks = True
        sim.validate_pre_upload()
        sweep_sim.validate_pre_upload()


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


def test_eme_mode_spec_increasing_mode_tolerance_default():
    mode_spec = td.EMEModeSpec()

    assert mode_spec.increasing_mode_tolerance == pytest.approx(1e-12)
    assert td.EMEModeSpec(increasing_mode_tolerance=0).increasing_mode_tolerance == 0


@pytest.mark.parametrize("num_pml", [(0, 0), (1, 1)])
def test_eme_internal_mode_solves_use_corrected_solver_grid(eme_base_sim, num_pml):
    """EME mode solves use the corrected mode-solver grid and branch convention."""
    sim = eme_base_sim.updated_copy(
        eme_grid_spec=td.EMEUniformGrid(
            num_cells=1,
            mode_spec=td.EMEModeSpec(num_modes=1, num_pml=num_pml),
        )
    )
    mode_solver = sim.mode_simulations[0]._mode_solver

    assert mode_solver.conjugated_dot_product is True
    # EME integrates overlaps/flux on the native Yee grid (discrete biorthogonality),
    # so the internal mode solves declare Yee integration, not colocated.
    assert mode_solver.use_colocated_integration is False
    expected_bounds = mode_solver._compute_solver_field_bounds(
        grid=mode_solver.simulation.grid,
        plane=mode_solver.plane,
        normal_axis=mode_solver.normal_axis,
        symmetry=mode_solver.simulation.symmetry,
        symmetry_center=mode_solver.simulation.center,
    )
    assert mode_solver._solver_field_bounds == expected_bounds
    assert mode_solver.to_mode_solver_monitor(name="test").attrs == {}
    assert sim.mode_solver_monitors[0].conjugated_dot_product is True
    assert sim.mode_solver_monitors[0].use_colocated_integration is False
    assert sim.mode_solver_monitors[0].attrs == {}


def test_smatrix_in_basis_convention_resolution():
    """The basis change uses one integration convention = the port (S-matrix)
    convention, defaulting to Yee, and falls back to colocated only when a target
    basis is stored colocated (Yee integration then impossible). A new basis's own
    ``use_colocated_integration`` is not honored -- the rebasing inner product is
    fixed by the S-matrix, not the target basis's preference. On a one-sided rebase
    only the rebased port contributes; the untouched port's convention is ignored
    (it is unused in the basis change), matching the extras kernel."""
    from tidy3d.components.eme.data.sim_data import _rebasing_colocated

    class _Mon:
        def __init__(self, colocate, uci):
            self.colocate = colocate
            self.use_colocated_integration = uci

    class _M:
        def __init__(self, colocate, uci):
            self.monitor = _Mon(colocate, uci)

    port_yee = _M(colocate=False, uci=False)  # EME port modes
    # EME default: Yee port + Yee-stored new basis -> Yee.
    assert not _rebasing_colocated(port_yee, port_yee, _M(False, False), _M(False, False))
    # A colocated-stored target basis forces colocated (unavoidable).
    assert _rebasing_colocated(port_yee, port_yee, _M(True, True), None)
    # Colocated port (S-matrix) convention -> colocated.
    assert _rebasing_colocated(_M(False, True), port_yee, _M(False, False), None)
    # A Yee-stored new basis with uci=True stays Yee -- its uci is not honored.
    assert not _rebasing_colocated(port_yee, port_yee, _M(False, True), None)
    # One-sided rebase: a colocated *untouched* port must NOT force colocated.
    # Only port 2 (Yee) is rebased here, so the result stays Yee despite port 1
    # being colocated; symmetric for the other side.
    assert not _rebasing_colocated(_M(False, True), port_yee, None, _M(False, False))
    assert not _rebasing_colocated(port_yee, _M(True, True), _M(False, False), None)
    # The rebased side's colocated port is still honored on a one-sided rebase.
    assert _rebasing_colocated(_M(False, True), port_yee, _M(False, False), None)


def test_field_in_basis_port_expansion_coeffs_gram_correction():
    """`field_in_basis` Gram-corrects the port->new overlap into expansion
    coefficients ``d = O @ G_port^{-1}`` (a raw overlap ``O`` assumes orthonormal port
    modes). Verifies the helper against a direct per-frequency inverse, that it reduces
    to ``O`` when the port Gram is identity, and that a dropped (NaN) port column is
    excluded from the inverse and left NaN."""
    import xarray as xr

    from tidy3d.components.eme.data.sim_data import _port_expansion_coeffs

    rng = np.random.default_rng(0)
    nf, n_new, n_port = 2, 3, 4

    def _da(arr, n1):
        return xr.DataArray(
            arr,
            dims=["f", "mode_index_0", "mode_index_1"],
            coords={
                "f": [1e14, 2e14],
                "mode_index_0": np.arange(arr.shape[1]),
                "mode_index_1": np.arange(n1),
            },
        )

    ovl = rng.standard_normal((nf, n_new, n_port)) + 1j * rng.standard_normal((nf, n_new, n_port))
    A = rng.standard_normal((nf, n_port, n_port)) + 1j * rng.standard_normal((nf, n_port, n_port))
    g_port = A + A.transpose(0, 2, 1) + 4.0 * np.eye(n_port)[None]  # symmetric, invertible per freq

    coeffs = _port_expansion_coeffs(_da(ovl, n_port), _da(g_port, n_port)).to_numpy()
    np.testing.assert_allclose(
        coeffs, np.einsum("fac,fcb->fab", ovl, np.linalg.inv(g_port)), atol=1e-10
    )

    # Identity port Gram -> reduces to the raw overlap.
    eye = np.broadcast_to(np.eye(n_port), (nf, n_port, n_port)).astype(complex)
    coeffs_id = _port_expansion_coeffs(_da(ovl, n_port), _da(eye, n_port)).to_numpy()
    np.testing.assert_allclose(coeffs_id, ovl, atol=1e-12)

    # Dropped (NaN) port column -> excluded from the inverse, left NaN; kept block exact.
    ovl_drop = ovl.copy()
    ovl_drop[:, :, 1] = np.nan
    keep = [0, 2, 3]
    coeffs_drop = _port_expansion_coeffs(_da(ovl_drop, n_port), _da(g_port, n_port)).to_numpy()
    assert np.all(np.isnan(coeffs_drop[:, :, 1]))
    np.testing.assert_allclose(
        coeffs_drop[:, :, keep],
        np.einsum("fac,fcb->fab", ovl_drop[:, :, keep], np.linalg.inv(g_port[:, keep][:, :, keep])),
        atol=1e-10,
    )


def test_trial_basis_mode_inds_drops_nan_diagonal():
    """``field_in_basis`` rebases through ``smatrix_in_basis``'s trial basis: a port
    mode with a NaN S-matrix diagonal (sweep-truncated, or increasing-/ModeSortSpec-
    filtered) is dropped and the rest kept -- per sweep, sliced to the rebased
    frequencies, and robustly when ``sweep_index`` is an unlabeled axis (where a naive
    ``np.where`` on the 2D diagonal would return the sweep axis as all-zero indices)."""
    import xarray as xr

    from tidy3d.components.eme.data.sim_data import _trial_basis_mode_inds

    n = 5
    ff = [1e14, 2e14]
    eye = np.broadcast_to(np.eye(n), (2, n, n)).astype(complex)
    mode_coords = {"mode_index_out": np.arange(n), "mode_index_in": np.arange(n)}
    dims = ["f", "mode_index_out", "mode_index_in"]
    coords = {"f": ff, **mode_coords}

    def _block(arr, dims, coords):
        return xr.DataArray(arr, dims=dims, coords=coords)

    # all diagonals finite -> every mode kept
    assert _trial_basis_mode_inds(_block(eye, dims, coords), 0, ff) == [0, 1, 2, 3, 4]

    # NaN the diagonal of mode 2 -> dropped, the rest kept
    dropped = eye.copy()
    dropped[:, 2, 2] = np.nan
    assert _trial_basis_mode_inds(_block(dropped, dims, coords), 0, ff) == [0, 1, 3, 4]

    # frequency subset: a mode dropped only at a frequency outside the rebased subset
    # stays kept (the block is sliced to f first, like smatrix_in_basis)
    freq_dep = eye.copy()
    freq_dep[1, 3, 3] = np.nan  # mode 3 NaN only at f=2e14
    assert _trial_basis_mode_inds(_block(freq_dep, dims, coords), 0, ff) == [0, 1, 2, 4]
    assert _trial_basis_mode_inds(_block(freq_dep, dims, coords), 0, [1e14]) == [0, 1, 2, 3, 4]

    # unlabeled sweep_index axis (a dim without a coordinate): the mode positions must
    # still come through rather than the sweep axis
    sweep_dims = ["f", "sweep_index", "mode_index_out", "mode_index_in"]
    got = _trial_basis_mode_inds(_block(dropped[:, None], sweep_dims, coords), 0, ff)
    assert got == [0, 1, 3, 4]

    # labeled multi-sweep: the requested sweep's kept set is selected
    multi = np.broadcast_to(np.eye(n), (2, 2, n, n)).astype(complex).copy()
    multi[:, 0, 3, 3] = np.nan  # sweep 0 drops mode 3
    multi[:, 1, 4, 4] = np.nan  # sweep 1 drops mode 4
    sweep_coords = {"f": ff, "sweep_index": [0, 1], **mode_coords}
    assert _trial_basis_mode_inds(_block(multi, sweep_dims, sweep_coords), 0, ff) == [0, 1, 2, 4]
    assert _trial_basis_mode_inds(_block(multi, sweep_dims, sweep_coords), 1, ff) == [0, 1, 2, 3]

    # unlabeled MULTI-sweep (sweep dim, no coordinate): each sweep is still selected by
    # position rather than collapsed over all sweeps (regression for "Unlabeled sweep
    # drops wrong modes" -- collapsing would drop both modes 3 and 4 for every sweep).
    multi_unlabeled = _block(multi, sweep_dims, coords)
    assert _trial_basis_mode_inds(multi_unlabeled, 0, ff) == [0, 1, 2, 4]
    assert _trial_basis_mode_inds(multi_unlabeled, 1, ff) == [0, 1, 2, 3]


def test_eme_mode_data_flux_no_use_colocated_integration():
    """``EMEModeSolverData`` carries an ``EMEModeSolverMonitor``, which has no
    ``use_colocated_integration`` field, so the flux helpers must fall back to the
    monitor's ``colocate`` convention rather than raising ``AttributeError`` on that
    field. (The multi-cell fixture then fails the normal 2D tangential-field check,
    which is unrelated to the fallback being exercised.)"""
    data = _get_eme_mode_solver_data()
    assert not hasattr(data.monitor, "use_colocated_integration")
    try:
        _ = data.complex_flux
    except AttributeError as e:
        assert "use_colocated_integration" not in str(e)
    except Exception:
        pass  # downstream errors (e.g. multi-cell data is not 2D) are unrelated


def test_force_integration_convention_raises_when_unforceable():
    """A monitor without ``use_colocated_integration`` (e.g. ``EMEModeSolverMonitor``)
    can't be forced to colocated when stored Yee (``colocate=False``); the basis change
    raises a clear ``SetupError`` instead of silently mixing conventions (the
    colocate=False EME-monitor-on-a-different-grid case), rather than no-op'ing on Yee."""
    from tidy3d.components.eme.data.sim_data import _force_integration_convention

    data = _get_eme_mode_solver_data()
    assert not hasattr(data.monitor, "use_colocated_integration")
    yee = data.updated_copy(monitor=data.monitor.updated_copy(colocate=False), validate=False)
    # required convention matches the `colocate` storage -> no-op
    assert _force_integration_convention(yee, colocated=False) is yee
    # required colocated, only Yee storage and no flag to force it -> raise
    with pytest.raises(SetupError):
        _force_integration_convention(yee, colocated=True)


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
    too_many_modes = td.EMEUniformGrid(num_cells=1, mode_spec=td.EMEModeSpec(num_modes=1001))
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
        yy=td.Medium(permittivity=3, conductivity=1e2),
        zz=td.Medium(permittivity=3, conductivity=1e2),
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


def _with_finite_fields(mode_data):
    """Replace NaN-padded field values in mode data with finite values.

    The shared synthetic field fixtures NaN-pad some modes' fields (e.g. sweep
    0, mode 1) that the S-matrix nonetheless treats as kept. A kept mode with a
    non-finite field makes the basis-change self-Gram non-finite, which
    ``smatrix_in_basis`` (correctly) rejects with a ``SetupError``. Tests that
    exercise the basis change itself -- rather than that non-finite-Gram guard
    -- use finite mode data so the kept modes are well-defined.
    """
    fields_no_nan = {}
    for name, field in mode_data.field_components.items():
        arr = field.to_numpy().copy()
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            arr[nan_mask] = 1.0 + 1.0j
        fields_no_nan[name] = field.copy(data=arr)
    return mode_data.updated_copy(**fields_no_nan)


def _get_eme_port_modes_finite(num_sweep=0):
    """``_get_eme_port_modes`` with NaN-padded fields replaced by finite values."""
    return _with_finite_fields(_get_eme_port_modes(num_sweep=num_sweep))


@pytest.mark.slow
def test_eme_sim_data():
    sim = make_eme_sim()
    mode_monitor_data = _get_eme_mode_solver_data()
    coeff_monitor_data = _get_eme_coeff_data()
    field_monitor_data = _get_eme_field_data()
    # Finite new-basis modes: the synthetic fixture NaN-pads a kept mode's
    # field, which would trip the non-finite-Gram guard in 'smatrix_in_basis'.
    modes_in_data = _with_finite_fields(_get_mode_solver_data(modes_out=False, num_modes=3))
    modes_out_data = _with_finite_fields(_get_mode_solver_data(modes_out=True, num_modes=2))
    data = [
        mode_monitor_data,
        coeff_monitor_data,
        field_monitor_data,
        modes_in_data,
        modes_out_data,
    ]
    port_modes = _get_eme_port_modes_finite()
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
    field = sim_data["field"]
    # The shared field fixture NaN-pads a kept mode (sweep 0, mode 1); use finite fields
    # so the basis change runs (the all-NaN case is exercised as a raise below).
    field_finite = _with_finite_fields(field)
    field_in_basis = sim_data.field_in_basis(field=field, port_index=0)
    assert "mode_index" in field_in_basis.Ex.coords
    field_in_basis = sim_data.field_in_basis(field=field_finite, modes=modes_in0, port_index=0)
    assert "mode_index" not in field_in_basis.Ex.coords
    field_in_basis = sim_data.field_in_basis(field=field_finite, modes=modes_in0, port_index=1)
    assert "mode_index" not in field_in_basis.Ex.coords
    # An active trial-basis mode whose field slice is all NaN is incomplete data and
    # raises (the fixture's kept-but-NaN-field mode), not a silent drop.
    with pytest.raises(SetupError):
        sim_data.field_in_basis(field=field, modes=modes_in0, port_index=0)
    # Field data missing trial-basis modes (fewer recorded than the port basis) raises a
    # clear SetupError, not a raw IndexError when indexing the field array by keep_inds.
    field_one_mode = field_finite.updated_copy(
        validate=False,
        **{key: fc.isel(mode_index=[0]) for key, fc in field_finite.field_components.items()},
    )
    with pytest.raises(SetupError):
        _ = sim_data.field_in_basis(field=field_one_mode, modes=modes_in0, port_index=0)
    # skip_gram_normalization opt-out is wired through the field_in_basis path.
    field_in_basis_skip = sim_data.field_in_basis(
        field=field_finite, modes=modes_in0, port_index=0, skip_gram_normalization=True
    )
    assert "mode_index" not in field_in_basis_skip.Ex.coords
    # Incomplete target data (non-finite fields) raises rather than silently dropping a
    # present mode's contribution -- matching smatrix_in_basis and the changelog.
    modes_in0_incomplete = modes_in0.updated_copy(Ex=modes_in0.Ex * np.nan)
    with pytest.raises(SetupError):
        sim_data.field_in_basis(field=field_finite, modes=modes_in0_incomplete, port_index=0)

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
    port_modes = _get_eme_port_modes_finite(num_sweep=10)
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
    field_finite = _with_finite_fields(sim_data["field"])
    field_in_basis = sim_data.field_in_basis(field=sim_data["field"], port_index=0)
    assert len(field_in_basis.Ex.sweep_index) == 10
    assert "mode_index" in field_in_basis.Ex.coords
    field_in_basis = sim_data.field_in_basis(field=field_finite, modes=modes_in0, port_index=0)
    assert "mode_index" not in field_in_basis.Ex.coords
    field_in_basis = sim_data.field_in_basis(field=field_finite, modes=modes_in0, port_index=1)
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


def _trace_metric(freqs, nm_left, nm_right):
    """Build a small trace metric matrix for stage model tests."""
    ntrace = 2 * (nm_left + nm_right)
    data = np.eye(ntrace, dtype=complex).reshape(1, 1, ntrace, ntrace)
    return EMETraceMetricDataArray(
        data,
        coords={
            "f": freqs,
            "sweep_index": [0],
            "trace_index_out": np.arange(ntrace),
            "trace_index_in": np.arange(ntrace),
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


def test_eme_simulation_smatrix_in_basis_honors_skip_gram_normalization():
    """`EMESimulation.smatrix_in_basis` (the local entry point) honors
    `skip_gram_normalization`, mirroring the data-path regression: the opt-out
    skips the Gram normalization and changes the rebased S-matrix (for a basis
    that is not orthonormal in the overlap convention). Guards that the flag is
    actually wired through the local path and both paths produce finite output."""
    from tidy3d.packaging import check_tidy3d_extras_licensed_feature

    try:
        check_tidy3d_extras_licensed_feature("local_eme", quiet=True)
    except Tidy3dImportError as exc:
        pytest.skip(f"tidy3d-extras local EME is unavailable: {exc}")

    sim = make_local_eme_sim(num_cells=2, num_modes=3)
    port_modes1 = _local_eme_basis_modes(sim, 0, "modes_in")
    port_modes2 = _local_eme_basis_modes(sim, 1, "modes_out")
    freqs = list(sim.freqs)
    rng = np.random.default_rng(0)

    def _blk():
        data = rng.standard_normal((len(freqs), 3, 3)) + 1j * rng.standard_normal(
            (len(freqs), 3, 3)
        )
        return _mda(data, sim.freqs, 3, 3)

    smatrix = td.EMESMatrixDataset(S11=_blk(), S12=_blk(), S21=_blk(), S22=_blk())
    ports = (port_modes1, port_modes2)

    default = sim.smatrix_in_basis(smatrix, ports, modes1=port_modes1, modes2=port_modes2)
    skipped = sim.smatrix_in_basis(
        smatrix, ports, modes1=port_modes1, modes2=port_modes2, skip_gram_normalization=True
    )
    blocks = ("S11", "S12", "S21", "S22")
    for blk in blocks:
        d = getattr(default, blk).values
        s = getattr(skipped, blk).values
        assert d.shape == s.shape
        assert np.all(np.isfinite(d)) and np.all(np.isfinite(s))
    # The opt-out actually takes effect on the local path (Gram normalization changed).
    max_diff = max(
        float(np.max(np.abs(getattr(default, blk).values - getattr(skipped, blk).values)))
        for blk in blocks
    )
    assert max_diff > 1e-6, "skip_gram_normalization should change the local-path result"


def test_eme_simulation_smatrix_in_basis_one_sided_modesimulationdata_port():
    """A one-sided local rebase (only ``modes1`` given) leaves the port-2 entry
    untouched -- and the common call pattern passes it as a ``ModeSimulationData``,
    which has no ``monitor``. The convention forcing must skip that unused side
    rather than reach for ``.monitor`` and raise. Guards the one-sided local path:
    no crash, finite output, the untouched S22 block passes through unchanged, and
    the rebased result does not depend on the untouched port's type."""
    from tidy3d.components.mode.data.sim_data import ModeSimulationData
    from tidy3d.packaging import check_tidy3d_extras_licensed_feature

    try:
        check_tidy3d_extras_licensed_feature("local_eme", quiet=True)
    except Tidy3dImportError as exc:
        pytest.skip(f"tidy3d-extras local EME is unavailable: {exc}")

    sim = make_local_eme_sim(num_cells=2, num_modes=3)
    port_modes1 = _local_eme_basis_modes(sim, 0, "modes_in", num_modes=3)
    port_modes2 = _local_eme_basis_modes(sim, 1, "modes_out", num_modes=3)
    # Wrap the port-2 modes as a ModeSimulationData (the realistic per-cell type,
    # which carries no `.monitor`) for the untouched side.
    port_modes2_sim_data = ModeSimulationData(
        simulation=td.ModeSimulation.from_simulation(
            simulation=sim,
            plane=sim.eme_grid.mode_planes[1],
            mode_spec=td.ModeSpec(num_modes=3),
            freqs=list(sim.freqs),
        ),
        modes_raw=port_modes2,
    )
    assert not hasattr(port_modes2_sim_data, "monitor")

    freqs = list(sim.freqs)
    rng = np.random.default_rng(0)

    def _blk():
        data = rng.standard_normal((len(freqs), 3, 3)) + 1j * rng.standard_normal(
            (len(freqs), 3, 3)
        )
        return _mda(data, sim.freqs, 3, 3)

    smatrix = td.EMESMatrixDataset(S11=_blk(), S12=_blk(), S21=_blk(), S22=_blk())

    # One-sided rebase: rebase port 1 only; port 2 (ModeSimulationData) is untouched.
    result = sim.smatrix_in_basis(smatrix, (port_modes1, port_modes2_sim_data), modes1=port_modes1)
    blocks = ("S11", "S12", "S21", "S22")
    for blk in blocks:
        assert np.all(np.isfinite(getattr(result, blk).values))
    # Port-2 self-block is not rebased -> passes through unchanged.
    assert np.allclose(result.S22.values, smatrix.S22.values)
    # The port-1 self-block is actually rebased (the change of basis took effect).
    assert not np.allclose(result.S11.values, smatrix.S11.values)
    # The untouched port's *type* must not leak into the rebased result: passing the
    # raw ModeSolverData instead of the ModeSimulationData yields the same answer.
    result_raw_port = sim.smatrix_in_basis(smatrix, (port_modes1, port_modes2), modes1=port_modes1)
    for blk in blocks:
        assert np.allclose(getattr(result, blk).values, getattr(result_raw_port, blk).values)


def make_local_eme_sim(
    num_cells=3, num_modes=4, sweep_spec=None, constraint="passive", num_pml=(6, 6)
):
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
            mode_spec=td.EMEModeSpec(num_modes=num_modes, num_pml=num_pml),
        ),
        freqs=[freq0],
        sweep_spec=sweep_spec,
        constraint=constraint,
    )


@pytest.mark.numerical
def test_eme_propagate_warns_when_diagnostics_requested():
    """One-shot local propagation cannot return diagnostic data."""
    sim = make_local_eme_sim(num_cells=2, num_modes=1, num_pml=(0, 0)).updated_copy(
        eme_diagnostics=True
    )
    mode_data = [ms.run_local() for ms in sim.mode_simulations]
    with AssertLogLevel("WARNING", contains_str="eme_diagnostics=True"):
        smatrix = sim.propagate(mode_data, progress=False)
    assert isinstance(smatrix, td.EMESMatrixDataset)


@pytest.mark.numerical
def test_compute_interface_diagnostics_end_to_end_real_solve():
    """End-to-end smoke test: real mode solve → cell overlaps → interface overlaps with
    diagnostic metrics → interface S-matrix → flux-weighted ``power_defect``.

    Verifies the modal-flux plumbing flows correctly from the solver through to
    :func:`compute_interface_diagnostics`. The flux-weighting formula itself is
    discriminated in the lower-level synthetic test
    ``test_compute_interface_diagnostics_power_defect_is_flux_weighted``; this test
    only verifies the public-API integration produces finite, sensible diagnostics
    on a real solve.
    """
    sim = make_local_eme_sim(num_cells=2, num_modes=3).updated_copy(eme_diagnostics=True)
    mode_data = [ms.run_local() for ms in sim.mode_simulations]

    cell_modes = [sim.stage_cell_modes(md, cell_index=i) for i, md in enumerate(mode_data)]
    cell_overlaps = [sim.compute_cell_overlap(cm) for cm in cell_modes]
    iface_overlap = sim.compute_interface_overlap(
        cell_modes[0],
        cell_modes[1],
        cell_overlaps[0],
        cell_overlaps[1],
    )
    # Diagnostic-path fields were stamped end-to-end.
    assert iface_overlap.electric_field_metric is not None
    assert iface_overlap.magnetic_field_metric is not None
    assert iface_overlap.aperture_electric_field_metric is not None
    assert iface_overlap.aperture_magnetic_field_metric is not None

    iface_smatrix = sim.compute_interface_smatrix(cell_overlaps[0], cell_overlaps[1], iface_overlap)
    diag = sim.compute_interface_diagnostics(
        cell_overlaps[0], cell_overlaps[1], iface_overlap, iface_smatrix
    )

    # All diagnostics finite, non-negative, with the canonical 5-D shape.
    power_defects = diag.power_defect.values
    e_res = diag.normalized_tangential_E_residual.values
    h_res = diag.normalized_tangential_H_residual.values
    aperture_e_res = diag.normalized_aperture_tangential_E_residual.values
    aperture_h_res = diag.normalized_aperture_tangential_H_residual.values
    for arr in (power_defects, e_res, h_res, aperture_e_res, aperture_h_res):
        assert arr.ndim == 5
        finite_mask = np.isfinite(arr)
        assert finite_mask.any()
        assert np.all(arr[finite_mask] >= 0.0)

    # Both cells come from a uniform multimode dielectric guide ⇒ the interface
    # S-matrix is the identity ⇒ all three diagnostics should be near zero for
    # every populated mode at every populated port.
    finite_pd = power_defects[np.isfinite(power_defects)]
    np.testing.assert_array_less(finite_pd, 1e-6)
    finite_e = e_res[np.isfinite(e_res)]
    finite_h = h_res[np.isfinite(h_res)]
    finite_aperture_e = aperture_e_res[np.isfinite(aperture_e_res)]
    finite_aperture_h = aperture_h_res[np.isfinite(aperture_h_res)]
    np.testing.assert_array_less(finite_e, 1e-6)
    np.testing.assert_array_less(finite_h, 1e-6)
    np.testing.assert_array_less(finite_aperture_e, 1e-6)
    np.testing.assert_array_less(finite_aperture_h, 1e-6)


def test_compute_interface_overlap_default_diagnostic_metrics_follows_sim_flag():
    """Manual staged overlap defaults should not pay diagnostic cost unless enabled."""
    from tidy3d.packaging import check_tidy3d_extras_licensed_feature

    try:
        check_tidy3d_extras_licensed_feature("local_eme", quiet=True)
    except Tidy3dImportError as exc:
        pytest.skip(f"tidy3d-extras local EME is unavailable: {exc}")

    sim = make_local_eme_sim(num_cells=2, num_modes=1, num_pml=(0, 0))
    left = sim.stage_cell_modes(_local_eme_basis_modes(sim, 0, "left", num_modes=1), cell_index=0)
    right = sim.stage_cell_modes(_local_eme_basis_modes(sim, 1, "right", num_modes=1), cell_index=1)
    left_overlap = sim.compute_cell_overlap(left)
    right_overlap = sim.compute_cell_overlap(right)

    default_overlap = sim.compute_interface_overlap(left, right, left_overlap, right_overlap)
    assert default_overlap.electric_field_metric is None
    assert default_overlap.magnetic_field_metric is None
    assert default_overlap.aperture_electric_field_metric is None
    assert default_overlap.aperture_magnetic_field_metric is None

    diagnostic_sim = sim.updated_copy(eme_diagnostics=True)
    diagnostic_default = diagnostic_sim.compute_interface_overlap(
        left, right, left_overlap, right_overlap
    )
    assert diagnostic_default.electric_field_metric is not None
    assert diagnostic_default.magnetic_field_metric is not None
    assert diagnostic_default.aperture_electric_field_metric is not None
    assert diagnostic_default.aperture_magnetic_field_metric is not None

    explicit_overlap = sim.compute_interface_overlap(
        left, right, left_overlap, right_overlap, include_diagnostic_metrics=True
    )
    assert explicit_overlap.electric_field_metric is not None


def test_compute_interface_diagnostics_rejects_cell_overlap_freq_mismatch():
    """Diagnostics read cell-overlap flux arrays and must validate their frequency grids."""
    from tidy3d.packaging import check_tidy3d_extras_licensed_feature

    try:
        check_tidy3d_extras_licensed_feature("local_eme", quiet=True)
    except Tidy3dImportError as exc:
        pytest.skip(f"tidy3d-extras local EME is unavailable: {exc}")

    sim = make_local_eme_sim(num_cells=2, num_modes=1)
    freq = sim.freqs[0]
    bad_freq = 1.01 * freq

    def cell_overlap(cell_index, freqs):
        return td.EMEStageCellOverlap(
            cell_index=cell_index,
            n_complex=td.ModeIndexDataArray(
                np.ones((1, 1), dtype=complex),
                coords={"f": freqs, "mode_index": [0]},
            ),
            complex_flux=td.FreqModeDataArray(
                np.ones((1, 1), dtype=complex),
                coords={"f": freqs, "mode_index": [0]},
            ),
            self_overlap=_mda(np.ones((1, 1, 1), dtype=complex), freqs, 1, 1),
        )

    interface_overlap = td.EMEStageInterfaceOverlap(
        cell_index=0,
        right_cell_index=1,
        O12=_mda(np.ones((1, 1, 1), dtype=complex), [freq], 1, 1),
        O21=_mda(np.ones((1, 1, 1), dtype=complex), [freq], 1, 1),
        electric_field_metric=_trace_metric([freq], 1, 1),
        magnetic_field_metric=_trace_metric([freq], 1, 1),
        aperture_electric_field_metric=_trace_metric([freq], 1, 1),
        aperture_magnetic_field_metric=_trace_metric([freq], 1, 1),
    )
    interface_smatrix = td.EMEStageInterfaceSMatrix(
        cell_index=0,
        right_cell_index=1,
        sweep_index=0,
        S11=_mda(np.zeros((1, 1, 1), dtype=complex), [freq], 1, 1),
        S12=_mda(np.ones((1, 1, 1), dtype=complex), [freq], 1, 1),
        S21=_mda(np.ones((1, 1, 1), dtype=complex), [freq], 1, 1),
        S22=_mda(np.zeros((1, 1, 1), dtype=complex), [freq], 1, 1),
    )

    with pytest.raises(ValidationError, match="left cell overlap"):
        sim.compute_interface_diagnostics(
            cell_overlap(0, [bad_freq]),
            cell_overlap(1, [freq]),
            interface_overlap,
            interface_smatrix,
        )


def test_stage_cell_modes_rejects_pre_truncated_basis():
    """stage_cell_modes() requires the full internal basis from `_internal_mode_spec`.

    The interface matching equations rely on every sorted mode being available
    as a test row; truncating with `sort_spec.keep_modes` before staging would
    silently fall back to a smaller test basis and lose the convergence
    guarantee. Synthesize an under-sized ModeSolverData with the right plane
    and frequencies (no real solve needed) and verify the validation fires.
    """
    from tidy3d.packaging import check_tidy3d_extras_licensed_feature

    try:
        check_tidy3d_extras_licensed_feature("local_eme", quiet=True)
    except Tidy3dImportError as exc:
        pytest.skip(f"tidy3d-extras local EME is unavailable: {exc}")

    # _get_mode_solver_data places the synthetic monitor at center=(0,0,-1)
    # size=(inf,inf,0), freqs=[td.C_0]; build a sim whose cell 0 plane matches.
    truncated_modes = 8
    expected_num_modes = truncated_modes + 2
    sim = td.EMESimulation(
        size=(2, 2, 3),
        axis=2,
        eme_grid_spec=td.EMEUniformGrid(
            num_cells=3,
            mode_spec=td.EMEModeSpec(num_modes=expected_num_modes, num_pml=(0, 0)),
        ),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        freqs=[td.C_0, 3e14],
    )
    assert sim._internal_mode_spec(0).num_modes == expected_num_modes

    truncated = _get_mode_solver_data(modes_out=False, num_modes=truncated_modes)

    with pytest.raises(ValidationError, match=f"at least {expected_num_modes} are required"):
        sim.stage_cell_modes(truncated, cell_index=0)


def test_cell_overlap_reconstructs_missing_trial_filter_mask():
    """Cached cell overlaps without a mask must not bypass current propagation filters."""
    freqs = [td.C_0]
    mode_index = [0, 1, 2]
    sort_spec = td.ModeSortSpec(
        filter_key="n_eff",
        filter_reference=1.5,
        keep_modes="filtered",
    )
    sim = td.EMESimulation(
        size=(1, 1, 1),
        axis=2,
        freqs=freqs,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        eme_grid_spec=td.EMEUniformGrid(
            num_cells=1,
            mode_spec=td.EMEModeSpec(
                num_modes=3,
                num_pml=(0, 0),
                sort_spec=sort_spec,
                increasing_mode_tolerance=1e-6,
            ),
        ),
    )
    cell_overlap = td.EMEStageCellOverlap(
        cell_index=0,
        n_complex=td.ModeIndexDataArray(
            np.array([[1.0, 2.0, 3.0 - 2e-6j]], dtype=complex),
            coords={"f": freqs, "mode_index": mode_index},
        ),
        complex_flux=td.FreqModeDataArray(
            np.ones((1, 3), dtype=complex),
            coords={"f": freqs, "mode_index": mode_index},
        ),
        self_overlap=_mda(np.eye(3, dtype=complex)[None, :, :], freqs, 3, 3),
    )

    reconstructed = sim._cell_overlap_with_trial_filter_mask(cell_overlap)

    assert reconstructed.filter_mask == (False, True, False)

    from tidy3d.packaging import check_tidy3d_extras_licensed_feature

    try:
        check_tidy3d_extras_licensed_feature("local_eme", quiet=True)
    except Tidy3dImportError as exc:
        pytest.skip(f"tidy3d-extras local EME is unavailable: {exc}")

    cell_smatrix = sim.compute_cell_smatrix(cell_overlap)
    np.testing.assert_array_equal(cell_smatrix.S11.mode_index_in.values, [1])


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
    """Check local smatrix_in_basis against the general Gram-inverse contraction.

    The contract is

        S_new[a, b] = G_new_a^{-1} . O_a . S_old[a, b] . G_port_b^{-T} . O_b^T

    where ``O_a = <new_a | port_a>``, ``G_new_a = <new_a | new_a>``, and
    ``G_port_a = <port_a | port_a>`` -- all in the bidirectional
    non-conjugated ``outer_dot`` inner product. Reduces to the previous
    ``O S O^T`` form when both bases happen to be biorthonormal.
    """
    port_cell_1 = sim.eme_grid_spec.virtual_cell_indices[0]
    port_cell_2 = sim.eme_grid_spec.virtual_cell_indices[-1]
    port_modes1 = sim.stage_cell_modes(port_modes[0], cell_index=port_cell_1).modes
    port_modes2 = sim.stage_cell_modes(port_modes[1], cell_index=port_cell_2).modes

    freqs = smatrix.S11.f.to_numpy()
    mi1 = smatrix.S11.mode_index_in.to_numpy()
    mi2 = smatrix.S22.mode_index_in.to_numpy()
    O1 = _overlap_block(modes1, port_modes1, freqs, mi1)
    O2 = _overlap_block(modes2, port_modes2, freqs, mi2)

    def _self_gram_block(modes, freqs_, mode_index):
        """Self-Gram of ``modes`` over ``mode_index`` at ``freqs_`` via outer_dot."""
        g = modes.outer_dot(modes, conjugate=False).sel(f=freqs_)
        if "mode_index_0" not in g.dims:
            g = g.expand_dims(dim={"mode_index_0": [0]}, axis=1)
        if "mode_index_1" not in g.dims:
            g = g.expand_dims(dim={"mode_index_1": [0]}, axis=2)
        if mode_index is not None:
            g = g.sel(mode_index_1=list(mode_index)).sel(mode_index_0=list(mode_index))
        return g.transpose("f", "mode_index_0", "mode_index_1").to_numpy()

    mi1_new = modes1.field_components[list(modes1.field_components)[0]].coords.get("mode_index")
    mi2_new = modes2.field_components[list(modes2.field_components)[0]].coords.get("mode_index")
    mi1_new_vals = None if mi1_new is None else mi1_new.values
    mi2_new_vals = None if mi2_new is None else mi2_new.values
    G_new1 = _self_gram_block(modes1, freqs, mi1_new_vals)
    G_new2 = _self_gram_block(modes2, freqs, mi2_new_vals)
    G_port1 = _self_gram_block(port_modes1, freqs, mi1)
    G_port2 = _self_gram_block(port_modes2, freqs, mi2)

    def _inv_stack(g):
        out = np.empty_like(g)
        for fi in range(g.shape[0]):
            try:
                out[fi] = np.linalg.inv(g[fi])
            except np.linalg.LinAlgError:
                out[fi] = np.linalg.pinv(g[fi])
        return out

    G_new1_inv = _inv_stack(G_new1)
    G_new2_inv = _inv_stack(G_new2)
    G_port1_inv = _inv_stack(G_port1)
    G_port2_inv = _inv_stack(G_port2)

    # L_a = G_new_a^{-1} O_a, Rt_b = G_port_b^{-1} O_b^T (G_port symmetric so
    # G_port^{-T} = G_port^{-1}); the block contraction is L_a . S_ab . Rt_b.
    L1 = np.einsum("fab,fbp->fap", G_new1_inv, O1)
    L2 = np.einsum("fab,fbp->fap", G_new2_inv, O2)
    Rt1 = np.einsum("fab,fcb->fac", G_port1_inv, O1)
    Rt2 = np.einsum("fab,fcb->fac", G_port2_inv, O2)

    S11 = _smatrix_block_with_sweep(smatrix.S11)
    S12 = _smatrix_block_with_sweep(smatrix.S12)
    S21 = _smatrix_block_with_sweep(smatrix.S21)
    S22 = _smatrix_block_with_sweep(smatrix.S22)

    expected11 = np.einsum("fap,fspq,fqb->fsab", L1, S11, Rt1)
    expected12 = np.einsum("fap,fspq,fqb->fsab", L1, S12, Rt2)
    expected21 = np.einsum("fap,fspq,fqb->fsab", L2, S21, Rt1)
    expected22 = np.einsum("fap,fspq,fqb->fsab", L2, S22, Rt2)

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
    io = td.EMEStageInterfaceOverlap(
        cell_index=0,
        right_cell_index=1,
        O12=S,
        O21=S,
        electric_field_metric=_trace_metric([2e14], 2, 2),
        magnetic_field_metric=_trace_metric([2e14], 2, 2),
        aperture_electric_field_metric=_trace_metric([2e14], 2, 2),
        aperture_magnetic_field_metric=_trace_metric([2e14], 2, 2),
    )
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

    io = td.EMEStageInterfaceOverlap(
        cell_index=0,
        right_cell_index=1,
        O12=S,
        O21=S,
        electric_field_metric=_trace_metric([2e14], 2, 2),
        magnetic_field_metric=_trace_metric([2e14], 2, 2),
        aperture_electric_field_metric=_trace_metric([2e14], 2, 2),
        aperture_magnetic_field_metric=_trace_metric([2e14], 2, 2),
    )
    io2 = _roundtrip(io, td.EMEStageInterfaceOverlap)
    assert io2.cell_index == 0
    assert io2.right_cell_index == 1
    np.testing.assert_allclose(
        io2.electric_field_metric.values,
        io.electric_field_metric.values,
    )
    np.testing.assert_allclose(
        io2.aperture_electric_field_metric.values,
        io.aperture_electric_field_metric.values,
    )

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


def test_eme_compute_smatrix_selects_port_flux_by_trial_label():
    """Flux normalization follows trial mode labels, not positional columns."""
    from tidy3d.components.eme.data.stage import (
        EMEStageCellOverlap,
        EMEStageCellSMatrix,
        EMEStageInterfaceSMatrix,
    )
    from tidy3d.packaging import check_tidy3d_extras_licensed_feature

    try:
        check_tidy3d_extras_licensed_feature("local_eme", quiet=True)
    except Tidy3dImportError as exc:
        pytest.skip(f"tidy3d-extras local EME is unavailable: {exc}")

    sim = make_local_eme_sim(num_cells=2, num_modes=4)
    freqs = list(sim.freqs)
    full_mi = np.arange(4)
    trial_mi = np.array([0, 2])

    def _overlap(cell_index):
        n_complex = td.ModeIndexDataArray(
            np.full((1, 4), 1.5 + 0j), coords={"f": freqs, "mode_index": full_mi}
        )
        flux_values = (
            np.array([[1.0, 10.0, 4.0, 10.0]], dtype=complex)
            if cell_index == 0
            else np.array([[9.0, 10.0, 16.0, 10.0]], dtype=complex)
        )
        flux = td.FreqModeDataArray(
            flux_values,
            coords={"f": freqs, "mode_index": full_mi},
        )
        so = td.EMESMatrixDataArray(
            np.eye(4, dtype=complex).reshape(1, 1, 4, 4),
            coords={
                "f": freqs,
                "sweep_index": [0],
                "mode_index_out": full_mi,
                "mode_index_in": full_mi,
            },
        )
        return EMEStageCellOverlap(
            cell_index=cell_index,
            n_complex=n_complex,
            complex_flux=flux,
            self_overlap=so,
        )

    def _smatrix_stage(cls, cell_index, right_cell_index=None):
        zero_block = np.zeros((1, 1, 2, 2), dtype=complex)
        pass_block = np.eye(2, dtype=complex).reshape(1, 1, 2, 2)
        coords = {
            "f": freqs,
            "sweep_index": [0],
            "mode_index_out": trial_mi,
            "mode_index_in": trial_mi,
        }
        zero = td.EMESMatrixDataArray(zero_block, coords=coords)
        passthrough = td.EMESMatrixDataArray(pass_block, coords=coords)
        kwargs = {
            "sweep_index": 0,
            "S11": zero,
            "S12": passthrough,
            "S21": passthrough,
            "S22": zero,
        }
        if right_cell_index is None:
            return cls(cell_index=cell_index, **kwargs)
        return cls(cell_index=cell_index, right_cell_index=right_cell_index, **kwargs)

    smatrix = sim.updated_copy(normalize=True).compute_smatrix(
        cell_overlaps=[_overlap(0), _overlap(1)],
        cell_smatrices=[
            _smatrix_stage(EMEStageCellSMatrix, 0),
            _smatrix_stage(EMEStageCellSMatrix, 1),
        ],
        interface_smatrices=[
            _smatrix_stage(EMEStageInterfaceSMatrix, 0, right_cell_index=1),
        ],
        sweep_index=0,
    )

    np.testing.assert_allclose(smatrix.S21.values.squeeze(), np.diag([3.0, 2.0]))
    np.testing.assert_allclose(smatrix.S12.values.squeeze(), np.diag([1.0 / 3.0, 1.0 / 2.0]))


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

    filtered_sort = td.ModeSortSpec(filter_key="n_eff", filter_reference=0.0, keep_modes="filtered")
    sim_filtered = sim_sweep.updated_copy(
        eme_grid_spec=td.EMEUniformGrid(
            num_cells=2,
            mode_spec=td.EMEModeSpec(num_modes=4, num_pml=(6, 6), sort_spec=filtered_sort),
        ),
    )
    for ms in sim_filtered.mode_simulations:
        assert ms.mode_spec.sort_spec.keep_modes == "all"
        assert ms.mode_spec.sort_spec.filter_key == "n_eff"

    int_sort = td.ModeSortSpec(keep_modes=2)
    with pytest.raises(pd.ValidationError):
        _ = sim_sweep.updated_copy(
            eme_grid_spec=td.EMEUniformGrid(
                num_cells=2,
                mode_spec=td.EMEModeSpec(num_modes=4, num_pml=(6, 6), sort_spec=int_sort),
            ),
        )

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
            sim_with_mnt.compute_interface_overlap(
                cell_modes[li],
                cell_modes[ri],
                cell_overlaps[li],
                cell_overlaps[ri],
            )
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
        sim.compute_interface_overlap(
            cell_modes[li],
            cell_modes[ri],
            cell_overlaps[li],
            cell_overlaps[ri],
        )
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
    # Finite port-mode fields: the NaN-padded *S-matrix* axes are what this
    # test exercises; the fixture's NaN-padded kept-mode field is unrelated and
    # would otherwise trip the non-finite-Gram guard.
    port_modes_raw = _get_eme_port_modes_finite(num_sweep=2)
    sim_data = td.EMESimulationData(
        simulation=sim,
        data=[],
        smatrix=smatrix,
        port_modes_raw=port_modes_raw,
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
    # Use finite port-mode fields: the synthetic fixture NaN-pads a kept
    # mode's field, which is unrelated to the ragged-sweep S-diagonal
    # pattern under test and would otherwise trip the non-finite-Gram guard.
    port_modes_raw = _get_eme_port_modes_finite(num_sweep=2)
    sim_data = td.EMESimulationData(
        simulation=sim,
        data=[],
        smatrix=smatrix,
        port_modes_raw=port_modes_raw,
    )
    modes1 = _get_mode_solver_data(num_modes=1)

    rebased = sim_data.smatrix_in_basis(modes1=modes1)
    mode_spec1 = modes1.monitor.mode_spec
    interp_spec1 = mode_spec1.interp_spec if mode_spec1 is not None else None
    freqs = rebased.S11.f.values

    for sweep_index in rebased.S11.sweep_index.values:
        # Port modes only vary across sweeps when the sweep type changes
        # the modes (e.g. EMEPeriodicitySweep). EMEModeSweep does not,
        # so port_modes_list_sweep has a single entry. For sweep types
        # that do vary modes, this pulls the sweep-specific basis as the
        # review follow-up suggested.
        port_modes_index = sweep_index if sim_data.simulation._sweep_modes else 0
        port_modes1 = sim_data.port_modes_list_sweep[port_modes_index][0]
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

        # New contract: S_new[a, b] = G_new_a^{-1} . O_a . S_old . G_port_b^{-T} . O_b^T.
        # Build G_new and G_port over active port-1 modes only (port 2 is
        # pass-through here -- no basis change requested).
        G_new1 = modes1.outer_dot(modes1, conjugate=False)
        if interp_spec1 is not None:
            G_new1 = modes1._interp_dataarray_in_freq(
                G_new1, freqs=freqs, method=interp_spec1.method
            )
        G_new1 = G_new1.sel(f=freqs).to_numpy()
        G_port1 = (
            port_modes1.outer_dot(port_modes1, conjugate=False)
            .sel(f=freqs, mode_index_0=keep_mode_inds1, mode_index_1=keep_mode_inds1)
            .to_numpy()
        )

        # Deliberately more tolerant than the real ``_per_freq_inverse``, which
        # *raises* on non-finite Gram entries (it does inv -> pinv only). Synthetic
        # fixtures can place NaN in the Gram, so this oracle zeros non-finite
        # entries and falls back inv -> pinv -> identity. The implementation only
        # ever sees finite active-mode Grams (NaN-padded dropped modes are excluded
        # upstream), so the two agree on the active modes compared below.
        def _robust_per_freq_inverse(g):
            g_clean = np.where(np.isfinite(g), g, 0.0).astype(g.dtype)
            n = g_clean.shape[-1]
            out = np.empty_like(g_clean)
            for fi in range(g_clean.shape[0]):
                try:
                    out[fi] = np.linalg.inv(g_clean[fi])
                except np.linalg.LinAlgError:
                    try:
                        out[fi] = np.linalg.pinv(g_clean[fi])
                    except np.linalg.LinAlgError:
                        out[fi] = np.eye(n, dtype=g_clean.dtype)
            return out

        G_new1_inv = _robust_per_freq_inverse(G_new1)
        G_port1_inv = _robust_per_freq_inverse(G_port1)
        O1_np = O1.transpose("f", "mode_index_0", "mode_index_1").to_numpy()
        # Left factor for OUT port 1: G_new^{-1} O, shape (nf, n_new, n_port_active).
        L1 = np.einsum("fab,fbp->fap", G_new1_inv, O1_np)
        # Right-side full chain for IN port 1: G_port^{-T} O^T (since G_port is
        # symmetric, this equals (O @ G_port^{-1})^T); store transposed for
        # convenient block einsum below. Shape (nf, n_port_active, n_new).
        Rt1 = np.einsum("fab,fcb->fac", G_port1_inv, O1_np)

        S11_np = S11.transpose("f", "mode_index_out", "mode_index_in").to_numpy()
        S12_np = S12.transpose("f", "mode_index_out", "mode_index_in").to_numpy()
        S21_np = S21.transpose("f", "mode_index_out", "mode_index_in").to_numpy()
        S22_np = S22.transpose("f", "mode_index_out", "mode_index_in").to_numpy()

        # S_new[a, b] = L_a . S_ab . Rt_b for OUT a, IN b.
        # Port 2 has no basis change here, so its in/out factors are absent.
        expected11_np = np.einsum("fap,fpq,fqb->fab", L1, S11_np, Rt1)
        expected12_np = np.einsum("fap,fpq->faq", L1, S12_np)
        expected21_np = np.einsum("frq,fqb->frb", S21_np, Rt1)
        expected22_np = S22_np

        out_mode_index_1 = rebased.S11.mode_index_out.values
        in_mode_index_1 = rebased.S11.mode_index_in.values
        out_mode_index_2 = rebased.S22.mode_index_out.values
        in_mode_index_2 = rebased.S22.mode_index_in.values

        def _as_da(arr, out_idx, in_idx):
            import xarray as _xr

            return _xr.DataArray(
                arr,
                coords={"f": freqs, "mode_index_out": out_idx, "mode_index_in": in_idx},
                dims=("f", "mode_index_out", "mode_index_in"),
            )

        expected11 = _as_da(expected11_np, out_mode_index_1, in_mode_index_1)
        expected12 = _as_da(expected12_np, out_mode_index_1, in_mode_index_2)
        expected21 = _as_da(expected21_np, out_mode_index_2, in_mode_index_1)
        expected22 = _as_da(expected22_np, out_mode_index_2, in_mode_index_2)

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


def _randomize_fields(mode_data, seed):
    """Replace field components with seeded non-zero values.

    The shared synthetic mode fixtures carry identically zero field data, so their
    ``outer_dot`` overlaps and self-Grams vanish and any basis change collapses to
    ``0 == 0``. Non-degenerate fields make the overlaps full-rank, so the
    contraction (and the Gram inverse) is genuinely exercised."""
    rng = np.random.default_rng(seed)
    randomized = {
        name: field.copy(
            data=rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape)
        )
        for name, field in mode_data.field_components.items()
    }
    return mode_data.updated_copy(**randomized)


def test_eme_sim_data_smatrix_in_basis_skip_gram_reproduces_legacy_contraction():
    """Non-vacuous public-path oracle for both contractions: with
    ``skip_gram_normalization=True`` the pure-Python ``smatrix_in_basis`` is the
    legacy ``O S O^T``; by default it is ``G_new^{-1} O S G_port^{-1} O^T``. The two
    differ for a non-orthonormal basis. Guards the public path against drift
    independently of the core ``tidy3d_extras`` coverage."""
    sim = make_eme_sim().updated_copy(monitors=[])
    smatrix = _get_eme_smatrix_dataset(num_modes_1=5, num_modes_2=5)
    # Inject non-degenerate fields on both the port basis and the new basis (the
    # shared fixtures are zero-field, which would make the contraction vacuous).
    port_modes_raw = _randomize_fields(_get_eme_port_modes_finite(), seed=1)
    sim_data = td.EMESimulationData(
        simulation=sim, data=[], smatrix=smatrix, port_modes_raw=port_modes_raw
    )
    modes1 = _randomize_fields(_get_mode_solver_data(modes_out=False, num_modes=3), seed=2)

    corrected = sim_data.smatrix_in_basis(modes1=modes1)
    raw = sim_data.smatrix_in_basis(modes1=modes1, skip_gram_normalization=True)

    # Form the overlap and self-Grams exactly as the implementation does (same
    # per-basis frequency-interpolation), so the only thing under test is the
    # contraction. Port 2 is pass-through here, so only the S11 block is rebased.
    freqs = corrected.S11.f.values
    port_modes1 = sim_data.port_modes_list_sweep[0][0]
    interp1 = modes1.monitor.mode_spec.interp_spec
    p_interp1 = getattr(port_modes1.monitor.mode_spec, "interp_spec", None)

    def _interp_np(modes, da, interp):
        if interp is not None:
            da = modes._interp_dataarray_in_freq(da, freqs=freqs, method=interp.method)
        return da.sel(f=freqs).transpose("f", "mode_index_0", "mode_index_1").to_numpy()

    O1 = _interp_np(modes1, modes1.outer_dot(port_modes1, conjugate=False), interp1)
    G_new1 = _interp_np(modes1, modes1.outer_dot(modes1, conjugate=False), interp1)
    G_port1 = _interp_np(
        port_modes1, port_modes1.outer_dot(port_modes1, conjugate=False), p_interp1
    )
    S11 = (
        smatrix.S11.sel(f=freqs, sweep_index=0)
        .transpose("f", "mode_index_out", "mode_index_in")
        .to_numpy()
    )

    def _S11(dataset):
        return (
            dataset.S11.sel(sweep_index=0)
            .transpose("f", "mode_index_out", "mode_index_in")
            .to_numpy()
        )

    # skip_gram_normalization=True -> plain legacy contraction O S O^T.
    np.testing.assert_allclose(
        _S11(raw), np.einsum("fap,fpq,fcq->fac", O1, S11, O1), rtol=1e-9, atol=1e-12
    )
    # Default -> G_new^{-1} O S G_port^{-1} O^T (Grams symmetric -> ^{-T} = ^{-1}).
    # Plain inv here: a singular Gram would send the impl to its pinv fallback and
    # break this match, so the assertion also confirms the bases are non-degenerate.
    left = np.einsum("fij,fjp->fip", np.linalg.inv(G_new1), O1)
    right = np.einsum("fqr,fcr->fqc", np.linalg.inv(G_port1), O1)
    np.testing.assert_allclose(
        _S11(corrected), np.einsum("fip,fpq,fqc->fic", left, S11, right), rtol=1e-9, atol=1e-12
    )
    # The opt-out is not a no-op: the two contractions differ for this basis.
    assert not np.allclose(_S11(raw), _S11(corrected))


def test_eme_sim_data_smatrix_in_basis_raises_on_non_finite_gram():
    """``EMESimulationData.smatrix_in_basis`` must raise ``SetupError`` when
    the basis-change Gram or overlap has non-finite entries -- on both the
    default (Gram-normalized) and ``skip_gram_normalization=True`` paths.

    Locks in the contract that the public path fails explicitly rather than
    silently propagating NaN to produce an all-NaN S-matrix. The opt-out skips
    the Gram inverse, so the overlap finiteness check is what guards it there.
    Uses the shared synthetic ``_get_eme_port_modes`` fixture as-is -- it
    intentionally NaN-pads some port-mode fields, which makes both the overlap
    and the port self-Gram non-finite for any sweep that keeps those modes.
    """
    sim = make_eme_sim().updated_copy(sweep_spec=td.EMEModeSweep(num_modes=[2, 5]), monitors=[])
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
    sim_data = td.EMESimulationData(
        simulation=sim,
        data=[],
        smatrix=smatrix,
        port_modes_raw=_get_eme_port_modes(num_sweep=2),
    )
    modes1 = _get_mode_solver_data(num_modes=1)

    from tidy3d.exceptions import SetupError

    with pytest.raises(SetupError, match="non-finite"):
        sim_data.smatrix_in_basis(modes1=modes1)

    # The 'skip_gram_normalization' opt-out must not revive the silent-NaN
    # path: the overlap finiteness check fires even though the Gram inverse
    # (the other non-finite gate) is skipped.
    with pytest.raises(SetupError, match="non-finite"):
        sim_data.smatrix_in_basis(modes1=modes1, skip_gram_normalization=True)


def test_eme_sim_data_smatrix_in_basis_masks_nan_padded_dropped_port_modes():
    """A *dropped* port mode (NaN S-matrix diagonal) whose fields are also NaN-padded
    is masked out of the overlap finiteness check, so the public ``smatrix_in_basis``
    does NOT raise -- the client-side mirror of the extras regression. (Active
    non-finite data still raises; see ``..._raises_on_non_finite_gram``.) The shared
    fixture NaN-pads port mode 1, so dropping mode 1 exercises the masked path while
    the remaining (finite) port modes are kept."""
    sim = make_eme_sim().updated_copy(sweep_spec=td.EMEModeSweep(num_modes=[2, 5]), monitors=[])
    template = _get_eme_smatrix_dataset(num_modes_1=5, num_modes_2=5, num_sweep=2)
    nan = complex(np.nan, np.nan)

    def _block_drop_mode1(block, offset):
        values = offset + np.arange(block.size, dtype=float).reshape(block.shape)
        data = (values + 1j * (values + 0.25)).astype(complex)
        data[:, :, 1, :] = nan  # drop mode 1 on the output axis
        data[:, :, :, 1] = nan  # and the input axis (NaN S-diagonal drop sentinel)
        return block.copy(data=data)

    smatrix = td.EMESMatrixDataset(
        S11=_block_drop_mode1(template.S11, 100),
        S12=_block_drop_mode1(template.S12, 200),
        S21=_block_drop_mode1(template.S21, 300),
        S22=_block_drop_mode1(template.S22, 400),
    )
    sim_data = td.EMESimulationData(
        simulation=sim,
        data=[],
        smatrix=smatrix,
        port_modes_raw=_get_eme_port_modes(num_sweep=2),  # NaN-pads port mode 1
    )
    modes1 = _get_mode_solver_data(num_modes=1)
    # Must NOT raise: the dropped (NaN-padded) mode 1 is masked out of the guard on
    # both the default and skip-Gram paths.
    sim_data.smatrix_in_basis(modes1=modes1)
    sim_data.smatrix_in_basis(modes1=modes1, skip_gram_normalization=True)


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


@pytest.mark.numerical
def test_eme_local_propagate_progress_default_renders(monkeypatch):
    """Default ``progress=True``: ``propagate`` renders a bar for every phase
    of both ``compute_overlaps`` and ``propagate_from_overlaps`` via the real
    logger console. A regression that flipped the default to ``False`` or
    broke ``propagate``'s kwarg forwarding to either lower method would lose
    one of the six phase labels."""
    buf = io.StringIO()
    captured = Console(file=buf, force_terminal=True, width=80)
    monkeypatch.setitem(log.handlers, "console", LogHandler(captured, "WARNING"))

    sim = make_local_eme_sim(num_cells=2, num_modes=2)
    mode_data = [ms.run_local() for ms in sim.mode_simulations]
    sim.propagate(mode_data)  # default progress kwarg

    out = buf.getvalue()
    for phase in (
        "stage_cell_modes",
        "compute_cell_overlap",
        "compute_interface_overlap",
        "compute_cell_smatrix",
        "compute_interface_smatrix",
        "compute_smatrix",
    ):
        assert phase in out, f"missing phase {phase!r} in captured progress output"


@pytest.mark.numerical
def test_eme_local_propagate_progress_false_leaves_handlers_untouched(monkeypatch):
    """``progress=False`` does not auto-install the default console handler.

    Regression for a previous bug where ``_ProgressContext`` called
    ``get_logging_console()`` even on the disabled path, silently
    re-enabling stdout logging after an explicit opt-out. Covers both
    ``compute_overlaps`` and ``propagate_from_overlaps`` because
    ``propagate`` forwards the kwarg to each.
    """
    monkeypatch.delitem(log.handlers, "console", raising=False)

    sim = make_local_eme_sim(num_cells=2, num_modes=2)
    mode_data = [ms.run_local() for ms in sim.mode_simulations]
    sim.propagate(mode_data, progress=False)

    assert "console" not in log.handlers
