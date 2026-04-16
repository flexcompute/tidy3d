from __future__ import annotations

import warnings
from typing import get_args

import matplotlib.pyplot as plt
import numpy as np
import pydantic as pd
import pytest
import responses

import tidy3d as td
import tidy3d.plugins.mode.web as msweb
from tidy3d import Coords, Grid, ModeIndexDataArray, ScalarFieldDataArray, ScalarModeFieldDataArray
from tidy3d.components.data.monitor_data import ModeSolverData
from tidy3d.components.mode.data.sim_data import ModeSimulationData
from tidy3d.components.mode.derivatives import create_d_matrices, create_sfactor_b, create_sfactor_f
from tidy3d.components.mode.solver import TOL_DEGENERATE_CANDIDATE, EigSolver
from tidy3d.components.mode_spec import MODE_DATA_KEYS
from tidy3d.exceptions import DataError, SetupError, ValidationError
from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins.mode.mode_solver import MODE_MONITOR_NAME
from tidy3d.web.core.environment import Env

from ..utils import AssertLogLevel, cartesian_to_unstructured


def assert_property_vs_runtime(mode_solver, mode_data):
    """Assert that _is_tensorial and _has_complex_eps never under-predict the runtime solver path.

    The runtime ``eps_spec`` per frequency tells us what the solver actually observed:
      - "diagonal"           → scalar (not tensorial)
      - "tensorial_real"     → tensorial, real matrix
      - "tensorial_complex"  → tensorial, complex matrix

    The frontend properties must be at least as conservative as the runtime to avoid
    memory under-allocation.  Over-prediction (wasteful but safe) is acceptable.
    """
    eps_spec = mode_data.eps_spec
    if eps_spec is None:
        return

    runtime_tensorial = any(s != "diagonal" for s in eps_spec)
    runtime_complex = any(s == "tensorial_complex" for s in eps_spec)

    if runtime_tensorial:
        assert mode_solver._is_tensorial, (
            f"Runtime solved tensorial (eps_spec={eps_spec}) but "
            f"_is_tensorial={mode_solver._is_tensorial} — would under-allocate memory"
        )

    if runtime_complex:
        assert mode_solver._has_complex_eps or mode_solver._is_tensorial, (
            f"Runtime solved complex (eps_spec={eps_spec}) but "
            f"_has_complex_eps={mode_solver._has_complex_eps}, "
            f"_is_tensorial={mode_solver._is_tensorial} — would under-allocate memory"
        )


WG_MEDIUM = td.Medium(permittivity=4.0, conductivity=1e-4)
WAVEGUIDE = td.Structure(geometry=td.Box(size=(1.5, 100, 1)), medium=WG_MEDIUM)
PLANE = td.Box(center=(0, 0, 0), size=(5, 0, 5))
SIM_SIZE = (4, 3, 3)
SRC = td.PointDipole(
    center=(0, 0, 0), source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13), polarization="Ex"
)

PROJECT_NAME = "Mode Solver"
TASK_NAME = "Untitled"
MODESOLVER_NAME = "mode_solver"
PROJECT_ID = "Project-ID"
TASK_ID = "Task-ID"
SOLVER_ID = "Solver-ID"


def make_fill_fraction_mode_data():
    freq = np.array([2e14])
    mode_spec = td.ModeSpec(num_modes=2)
    monitor = td.ModeSolverMonitor(
        size=(3.0, 0.0, 3.0),
        center=(0.0, 0.0, 0.0),
        freqs=freq,
        mode_spec=mode_spec,
        name="fill_fraction",
    )

    grid = Grid(
        boundaries=Coords(
            x=np.array([-1.5, -0.5, 0.5, 1.5]),
            y=np.array([-0.5, 0.5]),
            z=np.array([-1.5, -0.5, 0.5, 1.5]),
        )
    )

    coords = {
        "x": np.array([-1.0, 0.0, 1.0]),
        "y": np.array([0.0]),
        "z": np.array([-1.0, 0.0, 1.0]),
        "f": freq,
        "mode_index": np.arange(2),
    }
    shape = (3, 1, 3, 1, 2)

    ex_data = np.zeros(shape, dtype=complex)
    ex_data[1, 0, 1, 0, 0] = 2.0
    for ix in (0, 2):
        for iz in (0, 2):
            ex_data[ix, 0, iz, 0, 1] = 1.0

    zero_data = np.zeros(shape, dtype=complex)

    fields = {
        "Ex": ScalarModeFieldDataArray(ex_data, coords=coords),
        "Ey": ScalarModeFieldDataArray(np.copy(zero_data), coords=coords),
        "Ez": ScalarModeFieldDataArray(np.copy(zero_data), coords=coords),
        "Hx": ScalarModeFieldDataArray(np.copy(zero_data), coords=coords),
        "Hy": ScalarModeFieldDataArray(np.copy(zero_data), coords=coords),
        "Hz": ScalarModeFieldDataArray(np.copy(zero_data), coords=coords),
    }

    n_complex = ModeIndexDataArray(
        np.array([[1.6 + 0.0j, 1.3 + 0.0j]]),
        coords={"f": freq, "mode_index": np.arange(2)},
    )

    data = ModeSolverData(
        monitor=monitor,
        symmetry=(0, 0, 0),
        symmetry_center=(0.0, 0.0, 0.0),
        grid_expanded=grid,
        n_complex=n_complex,
        **fields,
    )

    bounding_box = td.Box(center=(0.0, 0.0, 0.0), size=(1.0, 2.0, 1.0))
    return data, bounding_box


@pytest.fixture
def mock_remote_api(monkeypatch):
    def void(*args, **kwargs):
        return None

    def mock_download(resource_id, remote_filename, to_file, *args, **kwargs):
        simulation = td.Simulation(
            size=SIM_SIZE,
            grid_spec=td.GridSpec(wavelength=1.0),
            structures=(WAVEGUIDE,),
            run_time=1e-12,
            symmetry=(1, 0, -1),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
            sources=(SRC,),
        )
        mode_spec = td.ModeSpec(
            num_modes=3,
            target_neff=2.0,
            filter_pol="tm",
            precision="double",
            track_freq="lowest",
        )
        ms = ModeSolver(
            simulation=simulation,
            plane=PLANE,
            mode_spec=mode_spec,
            freqs=[td.C_0 / 1.0],
        )
        ms.data_raw.to_file(to_file)

    from tidy3d.web.core import http_util as httputil

    monkeypatch.setattr(httputil, "api_key", lambda: "api_key")
    monkeypatch.setattr(httputil, "get_version", lambda: td.version.__version__)
    monkeypatch.setattr("tidy3d.web.api.mode.upload_file", void)
    monkeypatch.setattr("tidy3d.web.api.mode.download_gz_file", mock_download)
    monkeypatch.setattr("tidy3d.web.api.mode.download_file", mock_download)

    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        match=[responses.matchers.query_param_matcher({"projectName": PROJECT_NAME})],
        json={"data": {"projectId": PROJECT_ID, "projectName": PROJECT_NAME}},
        status=200,
    )

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/modesolver/py",
        match=[
            responses.matchers.json_params_matcher(
                {
                    "projectId": PROJECT_ID,
                    "taskName": TASK_NAME,
                    "modeSolverName": MODESOLVER_NAME,
                    "fileType": "Gz",
                    "source": "Python",
                    "protocolVersion": td.version.__version__,
                }
            )
        ],
        json={
            "data": {
                "refId": TASK_ID,
                "id": SOLVER_ID,
                "status": "draft",
                "createdAt": "2023-05-19T16:47:57.190Z",
                "charge": 0,
                "fileType": "Gz",
            }
        },
        status=200,
    )

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/modesolver/py",
        match=[
            responses.matchers.json_params_matcher(
                {
                    "projectId": PROJECT_ID,
                    "taskName": "BatchModeSolver_0",
                    "modeSolverName": MODESOLVER_NAME + "_batch_0",
                    "fileType": "Gz",
                    "source": "Python",
                    "protocolVersion": td.version.__version__,
                }
            )
        ],
        json={
            "data": {
                "refId": TASK_ID,
                "id": SOLVER_ID,
                "status": "draft",
                "createdAt": "2023-05-19T16:47:57.190Z",
                "charge": 0,
                "fileType": "Gz",
            }
        },
        status=200,
    )

    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/modesolver/py/{TASK_ID}/{SOLVER_ID}",
        json={
            "data": {
                "refId": TASK_ID,
                "id": SOLVER_ID,
                "status": "success",
                "createdAt": "2023-05-19T16:47:57.190Z",
                "charge": 0,
                "fileType": "Json",
            }
        },
        status=200,
    )

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/modesolver/py/{TASK_ID}/{SOLVER_ID}/run",
        json={
            "data": {
                "refId": TASK_ID,
                "id": SOLVER_ID,
                "status": "queued",
                "createdAt": "2023-05-19T16:47:57.190Z",
                "charge": 0,
                "fileType": "Gz",
            }
        },
        status=200,
    )


def compare_colocation(ms):
    """Compare mode-solver fields with colocation applied during run or post-run."""
    data_col = ms.solve()
    assert_property_vs_runtime(ms, data_col)
    ms_nocol = ms.updated_copy(colocate=False)
    data = ms_nocol.solve()
    data_at_boundaries = ms_nocol.sim_data.at_boundaries(MODE_MONITOR_NAME)

    for key, field in data_col.field_components.items():
        # Normalize both fields per-mode and per-frequency to the same peak value
        # (colocate=True and colocate=False may have different normalizations)
        field_at_boundaries = data_at_boundaries[key]
        # Get dims to reduce over (all except mode_index and f)
        reduce_dims = [d for d in field.dims if d not in ("mode_index", "f")]
        max_field = np.abs(field).max(dim=reduce_dims)
        max_at_boundaries = np.abs(field_at_boundaries).max(dim=reduce_dims)
        field_normalized = field / max_field
        field_at_boundaries_normalized = field_at_boundaries / max_at_boundaries
        assert np.allclose(field_at_boundaries_normalized, field_normalized, atol=1e-7)

        # Also check coordinates
        for dim, coords1 in field.coords.items():
            # Check that noncolocated data has one extra coordinate in the plane dimensions
            if coords1.size > 1 and dim in "xyz":
                coords2 = data.field_components[key].coords[dim]
                assert coords1.size == coords2.size - 1

            # Check that colocated coords are the same
            assert np.allclose(coords1, data_at_boundaries[key].coords[dim])


def verify_pol_fraction(ms):
    """Verify that polarization fraction was successfully filtered."""
    pol_frac = ms.data.pol_fraction
    pol_frac_wg = ms.data.pol_fraction_waveguide
    filter_pol = ms.mode_spec.filter_pol

    # print(pol_frac.isel(mode_index=0))
    # print(pol_frac_wg.isel(mode_index=0))
    # import matplotlib.pyplot as plt

    # f, ax = plt.subplots(3, 3, tight_layout=True, figsize=(10, 6))
    # for mode_index in range(3):
    #     ms.plot_field("Ex", "abs", mode_index=mode_index, f=ms.freqs[0], ax=ax[mode_index, 0])
    #     ms.plot_field("Ey", "abs", mode_index=mode_index, f=ms.freqs[0], ax=ax[mode_index, 1])
    #     ms.plot_field("Ez", "abs", mode_index=mode_index, f=ms.freqs[0], ax=ax[mode_index, 2])
    # plt.show()

    if filter_pol is not None:
        assert np.all(pol_frac[filter_pol].isel(mode_index=0) > 0.5)
        other_pol = "te" if filter_pol == "tm" else "tm"
        # There is no guarantee that the waveguide polarization fraction is also predominantly
        # the same as the standard definition, but it is true in the cases we test here
        assert np.all(
            pol_frac_wg[filter_pol].isel(mode_index=0).values
            > pol_frac_wg[other_pol].isel(mode_index=0).values
        )


def verify_dtype(ms):
    """Verify that the returned fields have the correct dtype w.r.t. the specified precision."""

    dtype = np.complex64 if ms.mode_spec.precision == "single" else np.complex128
    for field in ms.data.field_components.values():
        print(dtype, field.dtype, type(field.dtype))
        assert dtype == field.dtype


def check_ms_reduction(ms):
    ms_red = ms.reduced_simulation_copy
    grids_1d = ms._solver_grid.boundaries
    grids_1d_red = ms_red._solver_grid.boundaries
    assert np.allclose(grids_1d.x, grids_1d_red.x)
    assert np.allclose(grids_1d.y, grids_1d_red.y)
    assert np.allclose(grids_1d.z, grids_1d_red.z)
    modes_red = ms.solve()
    assert_property_vs_runtime(ms, modes_red)
    assert np.allclose(ms.data.n_eff.values, modes_red.n_eff.values)
    assert len(ms_red.simulation.sources) == 0
    assert len(ms_red.simulation.internal_absorbers) == 0


def test_mode_solver_validation():
    """Test invalidate mode solver setups."""

    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        run_time=1e-12,
    )
    mode_spec = td.ModeSpec(
        num_modes=1,
    )

    # frequency is too low
    with pytest.raises(pd.ValidationError):
        ms = ModeSolver(
            simulation=simulation,
            plane=PLANE,
            mode_spec=mode_spec,
            freqs=[1.1],
            direction="+",
        )

    # frequency not too low
    ms = ModeSolver(
        simulation=simulation,
        plane=PLANE,
        mode_spec=mode_spec,
        freqs=[1e12],
        direction="+",
    )

    # num of modes * plane grid points too large
    # 1) number of modes too big
    with pytest.raises(pd.ValidationError):
        ms = ModeSolver(
            simulation=simulation,
            plane=PLANE,
            mode_spec=mode_spec.updated_copy(num_modes=2**32),
            freqs=[1e12],
            direction="+",
        )
    # 2) number of grid points too big
    with pytest.raises(pd.ValidationError):
        ms = ModeSolver(
            simulation=simulation.updated_copy(grid_spec=td.GridSpec.uniform(dl=0.0001)),
            plane=PLANE,
            mode_spec=mode_spec,
            freqs=[1e12],
            direction="+",
        )

    # mode data too large
    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec.uniform(dl=0.001),
        run_time=1e-12,
    )
    ms = ms.updated_copy(simulation=simulation, freqs=np.linspace(1e12, 2e12, 50))

    with pytest.raises(SetupError):
        ms.validate_pre_upload()


@pytest.mark.parametrize("group_index_step, log_level", ((1e-7, "WARNING"), (1e-5, None)))
def test_mode_solver_group_index_warning(group_index_step, log_level):
    """Test mode solver setups issuing warnings."""

    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        run_time=1e-12,
    )

    with AssertLogLevel(log_level):
        mode_spec = td.ModeSpec(
            num_modes=1,
            group_index_step=group_index_step,
            precision="auto",
        )

    _ = ModeSolver(
        simulation=simulation,
        plane=PLANE,
        mode_spec=mode_spec,
        freqs=[1e12],
        direction="+",
    )


def test_mode_solver_fields():
    """Test that fields can be excluded and that errors are raised in methods that need them."""
    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        run_time=1e-12,
    )
    mode_spec = td.ModeSpec(num_modes=1)
    ms = ModeSolver(
        simulation=simulation,
        plane=PLANE,
        mode_spec=mode_spec,
        freqs=[1e12],
        direction="+",
        fields=["Ex", "Hz"],
    )
    mode_data = ms.solve()
    assert_property_vs_runtime(ms, mode_data)
    components = mode_data.field_components.keys()
    for comp in ["Ex", "Hz"]:
        assert comp in components
    for comp in ["Ey", "Ez", "Hx", "Hy"]:
        assert comp not in components

    with pytest.raises(DataError):
        mode_data.dot(mode_data)
    with pytest.raises(DataError):
        mode_data.mode_area
    with pytest.raises(DataError):
        mode_data.pol_fraction


@pytest.mark.parametrize("local", [True, False])
@responses.activate
def test_mode_solver_simple(mock_remote_api, local, tmp_path):
    """Simple mode solver run (with symmetry)"""

    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=(WAVEGUIDE,),
        run_time=1e-12,
        symmetry=(0, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=(SRC,),
    )
    mode_spec = td.ModeSpec(
        num_modes=3,
        target_neff=2.0,
        filter_pol="tm",
        precision="double" if local else "single",
        track_freq="lowest",
    )
    if local:
        freqs = [td.C_0 / 0.9, td.C_0 / 1.0, td.C_0 / 1.1]
    else:
        freqs = [td.C_0 / 1.0]
    ms = ModeSolver(
        simulation=simulation,
        plane=PLANE,
        mode_spec=mode_spec,
        freqs=freqs,
        direction="-",
    )

    if local:
        compare_colocation(ms)
        verify_pol_fraction(ms)
        verify_dtype(ms)
        _ = ms.data.to_dataframe()
        check_ms_reduction(ms)

    else:
        _ = msweb.run(ms, results_file=tmp_path / "tmp.hdf5")

    # Testing issue 807 functions
    freq0 = td.C_0 / 1.55
    source_time = td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10)
    nS_add_source = ms.sim_with_source(mode_index=0, direction="+", source_time=source_time)
    nS_add_monitor = ms.sim_with_monitor(freqs=freqs, name="mode monitor")
    nS_add_mode_solver_monitor = ms.sim_with_mode_solver_monitor(name="mode solver monitor")
    assert len(nS_add_source.sources) == len(simulation.sources) + 1
    assert len(nS_add_monitor.monitors) == len(simulation.monitors) + 1
    assert len(nS_add_mode_solver_monitor.monitors) == len(simulation.monitors) + 1


@responses.activate
def test_mode_solver_remote_after_local(mock_remote_api, tmp_path):
    """Test that running a remote solver after a local one modifies the stored data. This is to
    catch a bug if ``_cached_properties["data"]`` is inadvertently used."""

    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=(WAVEGUIDE,),
        run_time=1e-12,
        symmetry=(0, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=(SRC,),
    )
    mode_spec = td.ModeSpec(
        num_modes=3,
        target_neff=2.0,
        filter_pol="tm",
        track_freq="lowest",
    )

    ms = ModeSolver(
        simulation=simulation,
        plane=PLANE,
        mode_spec=mode_spec,
        freqs=[td.C_0 / 1.0],
        direction="-",
    )
    data_local = ms.data

    data_remote = msweb.run(ms, results_file=tmp_path / "ms_remote.hdf5")

    assert np.all(data_local.n_eff != data_remote.n_eff)


@pytest.mark.parametrize("local", [True, False])
@responses.activate
def test_mode_solver_custom_medium(mock_remote_api, local, tmp_path):
    """Test mode solver can work with custom medium. Consider a waveguide with varying
    permittivity along x-direction. The value of n_eff at different x position should be
    different.
    """

    # waveguide made of custom medium
    x_custom = np.linspace(-0.6, 0.6, 2)
    y_custom = [0]
    z_custom = [0]
    freq0 = td.C_0 / 1.0
    n = np.array([1.5, 5])
    n = n[:, None, None, None]
    n_data = ScalarFieldDataArray(
        n, coords={"x": x_custom, "y": y_custom, "z": z_custom, "f": [freq0]}
    )
    mat_custom = td.CustomMedium.from_nk(n_data, interp_method="nearest")

    waveguide = td.Structure(geometry=td.Box(size=(100, 0.5, 0.5)), medium=mat_custom)
    simulation = td.Simulation(
        size=(2, 2, 2),
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=(waveguide,),
        run_time=1e-12,
    )
    mode_spec = td.ModeSpec(
        num_modes=1,
        precision="double" if local else "single",
    )

    plane_left = td.Box(center=(-0.5, 0, 0), size=(0, 0.9, 0.9))
    plane_right = td.Box(center=(0.5, 0, 0), size=(0, 0.9, 0.9))

    n_eff = []
    for plane in [plane_left, plane_right]:
        ms = ModeSolver(
            simulation=simulation,
            plane=plane,
            mode_spec=mode_spec,
            freqs=[freq0],
            direction="+",
        )
        modes = (
            ms.solve() if local else msweb.run(ms, results_file=tmp_path / "ms_custom_medium.hdf5")
        )
        if local:
            assert_property_vs_runtime(ms, modes)
        n_eff.append(modes.n_eff.values)

        if local:
            check_ms_reduction(ms)

        fname = str(tmp_path / "ms_custom_medium.hdf5")
        ms.to_file(fname)
        m2 = ModeSolver.from_file(fname)
        assert m2 == ms

    if local:
        assert n_eff[0] < 1.5
        assert n_eff[1] > 4
        assert n_eff[1] < 5


@pytest.mark.parametrize("interp,tol", [("linear", 1e-3), ("nearest", 1e-3)])
@pytest.mark.parametrize("cond_factor", [0, 0.01])
@pytest.mark.parametrize("nx", [1, 3])
def test_mode_solver_unstructured_custom_medium(nx, cond_factor, interp, tol, tmp_path):
    """Test mode solver can work with unstructured custom medium. We compare mode solver results
    with unstructured custom medium to the results with usual Cartesian custom medium.
    """

    freq0 = td.C_0 / 1.0

    # Cartesian
    x_custom = np.linspace(-0.6, 0.6, nx)
    y_custom = np.linspace(-0.3, 0.3, 21)
    z_custom = np.linspace(-0.3, 0.3, 22)
    n = 2.5 + (x_custom[:, None, None] + 0.6) / 1.2 * np.sin(y_custom[None, :, None]) * np.cos(
        z_custom[None, None, :]
    )
    n_data = td.SpatialDataArray(n, coords={"x": x_custom, "y": y_custom, "z": z_custom})

    # unperturbed unstructured grid
    n_data_u = cartesian_to_unstructured(n_data, pert=0, seed=987, method="direct")

    # more perturbed unstructured grid
    n_data_up = cartesian_to_unstructured(n_data, pert=0.15, seed=987)

    md = []

    for n_arr in [n_data, n_data_u, n_data_up]:
        mat_custom = td.CustomMedium.from_nk(
            n=n_arr, k=cond_factor * n_arr, freq=freq0, interp_method=interp
        )
        waveguide = td.Structure(geometry=td.Box(size=(100, 0.5, 0.5)), medium=mat_custom)
        simulation = td.Simulation(
            size=(2, 2, 2),
            grid_spec=td.GridSpec(wavelength=1.0),
            structures=(waveguide,),
            run_time=1e-12,
        )
        mode_spec = td.ModeSpec(num_modes=1)

        plane = td.Box(center=(0, 0, 0), size=(0.0, 0.9, 0.9))
        ms = ModeSolver(
            simulation=simulation,
            plane=plane,
            mode_spec=mode_spec,
            freqs=[freq0],
            direction="+",
        )
        modes = ms.solve()
        assert_property_vs_runtime(ms, modes)
        md.append(modes)

    # ms.plot_field(mode_index=0, f=freq0, field_name="Ez")
    # plt.show()

    error_u = np.abs(md[0].n_eff - md[1].n_eff).values.item()
    error_up = np.abs(md[0].n_eff - md[2].n_eff).values.item()

    print(nx, cond_factor, interp, tol, error_u, error_up)

    assert error_u < 5e-5
    assert error_up < tol


def test_mode_bend_radius():
    """Test that the bend radius is correctly applied to the center of the mode plane in the case
    of an auto-grid that is not symmetric w.r.t. that center, and that nominally identical
    waveguides produce the same modes when the bend center if shifted away from the mode plane
    center, after a post-processing transformation to correct for that."""

    simulation = td.Simulation(
        size=(10, 10, 10),
        grid_spec=td.GridSpec(wavelength=1.0),
        # grid_spec=td.GridSpec.uniform(dl=0.04),
        structures=(WAVEGUIDE,),
        run_time=1e-12,
    )
    mode_spec1 = td.ModeSpec(
        num_modes=3,
        bend_radius=5,
        bend_axis=1,
    )

    # plane centered on the waveguide center
    plane1 = td.Box(center=(0, 0, 0), size=(3, 0, 2))

    # plane centered away from the waveguide center
    plane2 = td.Box(center=(0.5, 0, 0), size=(4, 0, 2))
    # mode spec with a radius such that the radius w.r.t. the waveguide center should be the same
    mode_spec2 = mode_spec1.updated_copy(bend_radius=5.5)

    ms1 = ModeSolver(
        simulation=simulation,
        plane=plane1,
        mode_spec=mode_spec1,
        freqs=[td.C_0 / 1.0],
    )
    ms2 = ms1.updated_copy(plane=plane2, mode_spec=mode_spec2)
    data1 = ms1.solve()
    assert_property_vs_runtime(ms1, data1)
    data2 = ms2.solve()
    assert_property_vs_runtime(ms2, data2)

    print(data1.n_complex)
    print(data2.n_complex * 5 / 5.5)

    # Plot fields
    # _, ax = plt.subplots(3, 2)
    # for mode_index in range(3):
    #     ms1.plot_field("Ex", ax=ax[mode_index, 0], mode_index=mode_index)
    #     ms2.plot_field("Ex", ax=ax[mode_index, 1], mode_index=mode_index)
    # plt.show()

    # The mode field dependence is E0 * exp(1j * n * R * k0 * phi) so to switch from one radius
    # to another we have n' * R' = n * R -> n' = n * R / R'
    assert np.allclose(data1.n_complex, data2.n_complex * 5.5 / 5.0)


def test_mode_solver_2D():
    """Run mode solver in 2D simulations."""
    mode_spec = td.ModeSpec(
        num_modes=3,
        filter_pol="te",
        precision="double",
        num_pml=(0, 10),
        track_freq="central",
    )
    simulation = td.Simulation(
        size=(0, SIM_SIZE[1], SIM_SIZE[2]),
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=(WAVEGUIDE,),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=(SRC,),
    )
    ms = ModeSolver(
        simulation=simulation, plane=PLANE, mode_spec=mode_spec, freqs=[td.C_0 / 1.0], direction="-"
    )
    compare_colocation(ms)
    verify_pol_fraction(ms)
    verify_dtype(ms)
    _ = ms.data.to_dataframe()
    check_ms_reduction(ms)

    mode_spec = td.ModeSpec(
        num_modes=3,
        filter_pol="te",
        precision="double",
        num_pml=(10, 0),
    )
    simulation = td.Simulation(
        size=(SIM_SIZE[0], SIM_SIZE[1], 0),
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=(WAVEGUIDE,),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.pml(z=False),
        sources=(SRC,),
    )
    ms = ModeSolver(
        simulation=simulation, plane=PLANE, mode_spec=mode_spec, freqs=[td.C_0 / 1.0], direction="+"
    )
    compare_colocation(ms)
    # verify_pol_fraction(ms)
    _ = ms.data.to_dataframe()
    check_ms_reduction(ms)

    # The simulation and the mode plane are both 0D along the same dimension
    simulation = td.Simulation(
        size=PLANE.size,
        grid_spec=td.GridSpec(wavelength=1.0),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=(SRC,),
    )
    ms = ModeSolver(simulation=simulation, plane=PLANE, mode_spec=mode_spec, freqs=[td.C_0 / 1.0])
    compare_colocation(ms)
    verify_pol_fraction(ms)
    check_ms_reduction(ms)


@pytest.mark.parametrize("local", [True, False])
@responses.activate
@td.packaging.disable_local_subpixel
def test_group_index(mock_remote_api, local, tmp_path):
    """Test group index and dispersion calculation"""

    simulation = td.Simulation(
        size=(5, 5, 1),
        grid_spec=td.GridSpec(wavelength=1.55),
        structures=(
            td.Structure(
                geometry=td.Box(size=(0.5, 0.22, td.inf)), medium=td.Medium(permittivity=3.48**2)
            ),
        ),
        medium=td.Medium(permittivity=1.44**2),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=(SRC,),
    )
    mode_spec = td.ModeSpec(
        num_modes=2,
        target_neff=3.0,
        precision="double" if local else "single",
        sort_spec=td.ModeSortSpec(track_freq="central"),
    )

    if local:
        freqs = [td.C_0 / 1.54, td.C_0 / 1.55, td.C_0 / 1.56]
    else:
        freqs = [td.C_0 / 1.0]

    # No group index calculation by default
    ms = ModeSolver(
        simulation=simulation,
        plane=td.Box(size=(td.inf, td.inf, 0)),
        mode_spec=mode_spec,
        freqs=freqs,
        direction="-",
    )

    modes = ms.solve() if local else msweb.run(ms, results_file=tmp_path / "ms_remote.hdf5")

    if local:
        with AssertLogLevel("WARNING", contains_str="ModeSpec") as ctx:
            assert modes.n_group is None
            assert ctx.num_records == 1
        with AssertLogLevel("WARNING", contains_str="ModeSpec") as ctx:
            assert modes.dispersion is None
            assert ctx.num_records == 1
        check_ms_reduction(ms)

    # Group index calculated
    ms = ModeSolver(
        simulation=simulation,
        plane=td.Box(size=(td.inf, td.inf, 0)),
        mode_spec=mode_spec.copy(update={"group_index_step": True}),
        freqs=freqs,
    )
    modes = ms.solve() if local else msweb.run(ms, results_file=tmp_path / "tmp.hdf5")
    if local:
        assert (modes.n_group.sel(mode_index=0).values > 3.9).all()
        assert (modes.n_group.sel(mode_index=0).values < 4.2).all()
        assert (modes.n_group.sel(mode_index=1).values > 3.7).all()
        assert (modes.n_group.sel(mode_index=1).values < 4.0).all()
        assert (modes.dispersion.sel(mode_index=0).values > 1400).all()
        assert (modes.dispersion.sel(mode_index=0).values < 1500).all()
        assert (modes.dispersion.sel(mode_index=1).values > -16500).all()
        assert (modes.dispersion.sel(mode_index=1).values < -15000).all()
        check_ms_reduction(ms)


def test_pml_params():
    """Test that mode solver pml parameters are computed correctly.
    Profiles start with H-field locations on both sides. On the max side, they also terminate with
    an H-field location, i.e. the last E-field parameter is missing.
    """
    omega = 1
    N = 100
    dls = np.ones((N,))
    n_pml = 12

    # Normalized target is just third power scaling with position
    # E-field locations for backward derivatives
    target_profile = (np.arange(1, n_pml + 1) / n_pml) ** 3
    target_profile = target_profile / target_profile[0]
    sf_b = create_sfactor_b(omega, dls, N, n_pml, dmin_pml=True)
    assert np.allclose(sf_b[:n_pml] / sf_b[n_pml - 1], target_profile[::-1])
    assert np.allclose(sf_b[N - n_pml + 1 :] / sf_b[N - n_pml + 1], target_profile[:-1])

    # H-field locations for backward derivatives
    target_profile = (np.arange(0.5, n_pml + 0.5, 1) / n_pml) ** 3
    target_profile = target_profile / target_profile[0]
    sf_f = create_sfactor_f(omega, dls, N, n_pml, dmin_pml=True)
    assert np.allclose(sf_f[:n_pml] / sf_f[n_pml - 1], target_profile[::-1])
    assert np.allclose(sf_f[N - n_pml :] / sf_f[N - n_pml], target_profile)


@pytest.mark.parametrize("dmin_pmc", [(False, False), (False, True), (True, False), (True, True)])
def test_derivative_matrices_do_not_warn_and_preserve_values(dmin_pmc):
    """Derivative matrix construction should stay warning-free and numerically unchanged."""

    def expected_dx_matrix(dls, shape, pmc, backward):
        Nx, Ny = shape
        matrix = np.diag(np.ones(Nx))
        if backward:
            matrix += np.diag(-np.ones(Nx - 1), k=-1)
            matrix[0, 0] = 2.0 if pmc else 0.0
        else:
            matrix *= -1.0
            matrix += np.diag(np.ones(Nx - 1), k=1)
            if not pmc:
                matrix[0, 0] = 0.0
        return np.kron(np.diag(1 / dls) @ matrix, np.eye(Ny))

    def expected_dy_matrix(dls, shape, pmc, backward):
        Nx, Ny = shape
        matrix = np.diag(np.ones(Ny))
        if backward:
            matrix += np.diag(-np.ones(Ny - 1), k=-1)
            matrix[0, 0] = 2.0 if pmc else 0.0
        else:
            matrix *= -1.0
            matrix += np.diag(np.ones(Ny - 1), k=1)
            if not pmc:
                matrix[0, 0] = 0.0
        return np.kron(np.eye(Nx), np.diag(1 / dls) @ matrix)

    shape = (2, 3)
    dlf = (np.array([2.0, 4.0]), np.array([5.0, 10.0, 20.0]))
    dlb = (np.array([3.0, 6.0]), np.array([7.0, 14.0, 28.0]))

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        dxf, dxb, dyf, dyb = create_d_matrices(shape, (dlf, dlb), dmin_pmc=dmin_pmc)

    assert np.allclose(
        dxf.toarray(), expected_dx_matrix(dlf[0], shape, dmin_pmc[0], backward=False)
    )
    assert np.allclose(dxb.toarray(), expected_dx_matrix(dlb[0], shape, dmin_pmc[0], backward=True))
    assert np.allclose(
        dyf.toarray(), expected_dy_matrix(dlf[1], shape, dmin_pmc[1], backward=False)
    )
    assert np.allclose(dyb.toarray(), expected_dy_matrix(dlb[1], shape, dmin_pmc[1], backward=True))


@td.packaging.disable_local_subpixel
def test_mode_solver_nan_pol_fraction():
    """Test mode solver when eigensolver returns 0 for some modes."""
    wg = td.Structure(geometry=td.Box(size=(0.5, 100, 0.22)), medium=td.Medium(permittivity=12))

    simulation = td.Simulation(
        medium=td.Medium(permittivity=2),
        size=SIM_SIZE,
        grid_spec=td.GridSpec.auto(wavelength=1.55, min_steps_per_wvl=15),
        structures=(wg,),
        run_time=1e-12,
        symmetry=(0, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=(SRC,),
    )

    mode_spec = td.ModeSpec(
        num_modes=10,
        target_neff=3.48,
        filter_pol="tm",
        precision="single",
        track_freq="central",
    )

    freqs = [td.C_0 / 1.55]

    ms = ModeSolver(
        simulation=simulation,
        plane=td.Box(center=(0, 0, 0), size=(2, 0, 1.1)),
        mode_spec=mode_spec,
        freqs=freqs,
        direction="-",
    )

    md = ms.solve()
    assert_property_vs_runtime(ms, md)
    check_ms_reduction(ms)
    # Inject NaN at mode_index=5 for selected field components
    nan_fields = {}
    for field_name in ["Ex", "Ez", "Hx", "Hz"]:
        field = getattr(md, field_name)
        data = field.values.copy()
        data[..., 5] = np.nan
        nan_fields[field_name] = field.copy(data=data)

    md = md.updated_copy(**nan_fields)
    md = ms._filter_polarization(md)
    assert list(np.where(np.isnan(md.pol_fraction.te))[1]) == [9]


def test_mode_solver_method_defaults():
    """Test that changes to mode solver default values in methods work."""

    simulation = td.Simulation(
        medium=td.Medium(permittivity=2),
        size=SIM_SIZE,
        grid_spec=td.GridSpec.auto(wavelength=1.55, min_steps_per_wvl=15),
        run_time=1e-12,
        symmetry=(0, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=(SRC,),
    )

    mode_spec = td.ModeSpec(
        num_modes=10,
        target_neff=3.48,
        filter_pol="tm",
        precision="single",
        track_freq="central",
    )

    freqs = [td.C_0 / 1.55]

    ms = ModeSolver(
        simulation=simulation,
        plane=td.Box(center=(0, 0, 0), size=(2, 0, 1.1)),
        mode_spec=mode_spec,
        freqs=freqs,
        direction="-",
    )

    # test defaults
    st = td.GaussianPulse(freq0=1.0e12, fwidth=1.0e12)

    src = ms.to_source(source_time=st)
    assert src.direction == ms.direction

    src = ms.to_source(source_time=st, direction="+")
    assert src.direction != ms.direction

    mnt = ms.to_monitor(name="mode_mnt")
    assert np.allclose(mnt.freqs, ms.freqs)

    mnt = ms.to_monitor(name="mode_mnt", freqs=[2e14])
    assert not np.allclose(mnt.freqs, ms.freqs)

    sim = ms.sim_with_source(source_time=st)
    assert sim.sources[-1].direction == ms.direction

    sim = ms.sim_with_monitor(name="test")
    assert np.allclose(sim.monitors[-1].freqs, ms.freqs)


@responses.activate
def test_mode_solver_web_run_batch(mock_remote_api, tmp_path):
    """Testing run_batch function for the web mode solver."""

    wav = 1.5
    wav_min = 1.4
    wav_max = 1.5
    num_freqs = 1
    num_of_sims = 1
    freqs = np.linspace(td.C_0 / wav_min, td.C_0 / wav_max, num_freqs)

    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=wav),
        structures=(WAVEGUIDE,),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    )

    # create a list of mode solvers
    mode_solver_list = [None] * num_of_sims

    # create three different mode solvers with different number of modes specifications
    for i in range(num_of_sims):
        mode_solver_list[i] = ModeSolver(
            simulation=simulation,
            plane=PLANE,
            mode_spec=td.ModeSpec(
                num_modes=i + 1,
                target_neff=2.0,
            ),
            freqs=freqs,
            direction="+",
        )

    # Run mode solver one at a time
    results_files = [tmp_path / f"ms_batch_{i}.hdf5" for i in range(num_of_sims)]
    results = msweb.run_batch(
        mode_solver_list,
        verbose=False,
        folder_name="Mode Solver",
        results_files=results_files,
    )
    print(*results, sep="\n")
    assert all(isinstance(x, ModeSolverData) for x in results)
    assert (results[i].n_eff.shape == (num_freqs, i + 1) for i in range(num_of_sims))


def test_mode_solver_relative():
    """Relative mode solver"""

    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=(WAVEGUIDE,),
        run_time=1e-12,
        symmetry=(0, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=(SRC,),
    )
    mode_spec = td.ModeSpec(
        num_modes=3,
        target_neff=2.0,
        filter_pol="tm",
        precision="double",
        track_freq="lowest",
    )
    freqs = [td.C_0 / 0.9, td.C_0 / 1.0, td.C_0 / 1.1]
    ms = ModeSolver(
        simulation=simulation,
        plane=PLANE,
        mode_spec=mode_spec,
        freqs=freqs,
        direction="-",
        colocate=False,
    )
    basis = ms.data_raw
    assert_property_vs_runtime(ms, basis)
    new_freqs = np.array(freqs) * 1.01
    ms = ms.updated_copy(freqs=new_freqs)
    _ = ms._data_on_yee_grid_relative(basis=basis)


def test_mode_solver_plot():
    """Test mode plane plotting functions"""

    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=(WAVEGUIDE,),
        run_time=1e-12,
        symmetry=(0, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=(SRC,),
    )
    mode_spec = td.ModeSpec(
        num_modes=3,
        target_neff=2.0,
        num_pml=[8, 4],
    )
    freqs = [td.C_0 / 0.9, td.C_0 / 1.0, td.C_0 / 1.1]
    ms = ModeSolver(
        simulation=simulation,
        plane=PLANE,
        mode_spec=mode_spec,
        freqs=freqs,
        direction="-",
        colocate=False,
    )
    _, ax = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
    ms.plot(ax=ax[0, 0])
    ms.plot_eps(freq=200e14, alpha=0.7, ax=ax[0, 1])
    ms.plot_structures_eps(freq=200e14, alpha=0.8, cbar=True, reverse=False, ax=ax[1, 0])
    ms.plot_grid(linewidth=0.3, ax=ax[1, 0])
    ms.plot(ax=ax[1, 1])
    ms.plot_pml(ax=ax[1, 1])
    ms.plot_grid(linewidth=0.3, ax=ax[1, 1])
    plt.close()


def make_test_mode_solver(freqs: list[float] | None = None, plane: td.Box = PLANE) -> ModeSolver:
    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=(WAVEGUIDE,),
        run_time=1e-12,
        symmetry=(0, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=(SRC,),
    )
    mode_spec = td.ModeSpec(
        num_modes=3,
        target_neff=2.0,
        num_pml=[8, 4],
    )
    if freqs is None:
        freqs = [td.C_0 / 0.9, td.C_0 / 1.0, td.C_0 / 1.1]
    return ModeSolver(
        simulation=simulation,
        plane=plane,
        mode_spec=mode_spec,
        freqs=freqs,
        direction="-",
        colocate=False,
    )


def test_mode_solver_plot_field_components():
    ms = make_test_mode_solver()
    freq = ms.freqs[0]

    fig, axs = ms.plot_field_components(
        field_names=("Ex", "Ey"),
        mode_indices=(0, 1),
        show_n_eff=True,
        f=freq,
    )

    assert axs.shape == (2, 2)
    assert axs[0, 0].get_title().startswith("Ex, mode_index=0")
    assert "n_eff=" in axs[0, 0].get_title()
    assert axs[1, 1].get_title().startswith("Ey, mode_index=1")
    assert len(fig.axes) > axs.size
    plt.close(fig)


def test_mode_solver_plot_field_components_default_figsize_scales_with_span_ratio():
    freq = td.C_0 / 1.0

    widths = []
    for plane_size in ((0, 1, 3), (0, 1, 1), (0, 3, 1)):
        ms = make_test_mode_solver(freqs=[freq], plane=td.Box(center=(0, 0, 0), size=plane_size))
        fig, _ = ms.plot_field_components(
            field_names=("Ex",),
            mode_indices=(0,),
            show_n_eff=False,
            f=freq,
        )
        widths.append(float(fig.get_size_inches()[0]))
        plt.close(fig)

    assert widths[0] < widths[1] < widths[2]


def test_mode_solver_plot_field_components_forwards_to_mode_sim_data(monkeypatch):
    freq = td.C_0 / 1.0
    ms = make_test_mode_solver(freqs=[freq])

    fig, axs = plt.subplots(1, 1, squeeze=False)
    calls = {}

    def fake_plot_field_components(self, **kwargs):
        calls["self"] = self
        calls["kwargs"] = kwargs
        return fig, axs

    monkeypatch.setattr(ModeSimulationData, "plot_field_components", fake_plot_field_components)

    fig_out, axs_out = ms.plot_field_components(
        field_names=("Ex",),
        mode_indices=(0,),
        show_n_eff=True,
        f=freq,
    )

    assert fig_out is fig
    assert axs_out is axs
    assert isinstance(calls["self"], ModeSimulationData)
    assert calls["self"].modes_raw.monitor.name == ms.data_raw.monitor.name
    assert calls["kwargs"]["field_names"] == ("Ex",)
    assert calls["kwargs"]["mode_indices"] == (0,)
    assert calls["kwargs"]["show_n_eff"] is True
    assert calls["kwargs"]["f"] == freq
    plt.close(fig)


@pytest.mark.parametrize("local", [True, False])
@responses.activate
def test_modes_eme_sim(mock_remote_api, local, tmp_path):
    lambda0 = 1
    freq0 = td.C_0 / lambda0
    sim_size = (1, 1, 1)
    mode_spec = td.EMEModeSpec(num_modes=10)
    eme_grid_spec = td.EMEUniformGrid(num_cells=2, mode_spec=mode_spec)
    sim = td.EMESimulation(size=sim_size, freqs=[freq0], axis=2, eme_grid_spec=eme_grid_spec)
    solver = ModeSolver(
        simulation=sim,
        freqs=[freq0],
        mode_spec=td.ModeSpec(num_modes=2),
        plane=sim.eme_grid.mode_planes[0],
    )
    if local:
        data = solver.data
        assert_property_vs_runtime(solver, data)
    else:
        with pytest.raises(SetupError):
            _ = msweb.run(solver, results_file=tmp_path / "eme_solver_remote.hdf5")
        _ = msweb.run(
            solver.to_fdtd_mode_solver(), results_file=tmp_path / "eme_solver_fdtd_remote.hdf5"
        )

    _ = solver.reduced_simulation_copy


def test_mode_small_bend_radius_fail():
    """Test that small bend radius fails."""
    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[WAVEGUIDE],
        run_time=1e-12,
        symmetry=(1, 0, -1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
    )
    with pytest.raises(ValueError):
        ms = ModeSolver(
            plane=PLANE,
            freqs=np.linspace(1e14, 2e14, 100),
            mode_spec=td.ModeSpec(num_modes=1, bend_radius=1, bend_axis=0),
            simulation=simulation,
        )
    # also error when bend radius is exactly aligned with simulation boundary
    with pytest.raises(ValueError):
        ms = ModeSolver(
            plane=PLANE,
            freqs=np.linspace(1e14, 2e14, 100),
            mode_spec=td.ModeSpec(num_modes=1, bend_radius=1.5, bend_axis=0),
            simulation=simulation,
        )
    # should work for infinite mode plane
    ms = ModeSolver(
        plane=td.Box(center=(0, 0, 0), size=(td.inf, 0, td.inf)),
        freqs=np.linspace(1e14, 2e14, 100),
        mode_spec=td.ModeSpec(num_modes=1, bend_radius=10000, bend_axis=0),
        simulation=simulation,
    )


def make_high_order_mode_solver(sign, dim=3):
    waveguide = td.Structure(
        geometry=td.Box(size=(td.inf, 0.6, 0.2)),
        medium=td.Medium(permittivity=3.47**2),
    )

    refine_box = td.MeshOverrideStructure(
        geometry=td.Box(center=(0, sign * 0.3, 0), size=(td.inf, 0.1, 0.3)),
        dl=[None, 0.02, 0.02],
    )

    pml = td.Boundary(plus=td.PML(), minus=td.PML())
    periodic = td.Boundary(plus=td.Periodic(), minus=td.Periodic())

    sim = td.Simulation(
        size=(10, 2.5, 1.5 if dim == 3 else 0),
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=20, wavelength=1.55, override_structures=[refine_box]
        ),
        structures=(waveguide,),
        medium=td.Medium(permittivity=1.44**2),
        boundary_spec=td.BoundarySpec(x=pml, y=pml, z=pml if dim == 3 else periodic),
        run_time=1e-12,
    )

    plane = td.Box(center=(0, 0, 0), size=(0, 2.5, 1.5))
    freq0 = td.C_0 / 1.55
    num_modes = 3

    mode_spec = td.ModeSpec(
        num_modes=num_modes,
        target_neff=3.47,
        group_index_step=False,
    )

    return ModeSolver(
        simulation=sim,
        plane=plane,
        mode_spec=mode_spec,
        freqs=[freq0],
    )


def test_high_order_mode_normalization():
    # 3D simulation
    ms1 = make_high_order_mode_solver(1)
    ms2 = make_high_order_mode_solver(-1)
    assert_property_vs_runtime(ms1, ms1.data)
    assert_property_vs_runtime(ms2, ms2.data)
    overlap = ms1.data.outer_dot(ms2.data).isel(mode_index_0=2, mode_index_1=2).values.item()
    assert abs(1 - overlap) < 1e-3

    # 2D simulation
    ms1 = make_high_order_mode_solver(1, 2)
    values = ms1.data.Ez.isel(mode_index=2).values.squeeze().real
    assert (values[: values.size // 3] > 0).all()

    ms2 = make_high_order_mode_solver(-1, 2)
    values = ms2.data.Ez.isel(mode_index=2).values.squeeze().real
    assert (values[: values.size // 3] > 0).all()


def test_gauge_robustness():
    array = np.zeros((5, 5), dtype=float)
    ij = np.arange(5)
    assert ModeSolver._weighted_coord_max(array, ij, ij) == (0, 0)

    array[1, -1] = np.nan
    assert ModeSolver._weighted_coord_max(array, ij, ij) == (0, 0)


def test_translated_dot():
    sim_size = (5, 5, 5)
    lambda0 = 1.55
    freq0 = td.C_0 / lambda0
    si = td.material_library["cSi"]["Li1993_293K"]
    sio2 = td.material_library["SiO2"]["Horiba"]
    wg = td.Structure(geometry=td.Box(size=(0.22, 0.5, td.inf)), medium=si)
    mode_spec = td.ModeSpec(num_modes=3)
    grid_spec = td.GridSpec.auto(wavelength=lambda0, min_steps_per_wvl=20)

    sim = td.Simulation(
        size=sim_size, medium=sio2, structures=(wg,), grid_spec=grid_spec, run_time=1e-30
    )
    mode_plane = td.Box(size=(3, 3, 0))
    mode_solver = ModeSolver(simulation=sim, plane=mode_plane, mode_spec=mode_spec, freqs=[freq0])

    data = mode_solver.data_raw
    assert_property_vs_runtime(mode_solver, data)

    # now create a translated copy
    vector = (0.5, 0, 0)
    mode_solver2 = mode_solver.updated_copy(center=vector, path="simulation/structures/0/geometry")

    data2 = mode_solver2.data_raw

    # self-overlaps are close to 1, others are close to 0
    atol = 1e-2

    # just make sure the mode overlaps in the translated waveguide are the same
    assert np.allclose(data.dot(data), data2.dot(data2), atol=atol)
    assert np.allclose(data.outer_dot(data), data2.outer_dot(data2), atol=atol)

    # now translate the data, and check that its overlaps with the modes
    # of the translated waveguide agree with the self-overlaps of those modes
    data_translated = data.translated_copy(vector)

    assert np.allclose(data2.dot(data_translated), data2.dot(data2), atol=atol)
    assert np.allclose(data_translated.dot(data2), data2.dot(data2), atol=atol)

    assert np.allclose(data2.outer_dot(data_translated), data2.outer_dot(data2), atol=atol)
    assert np.allclose(data_translated.outer_dot(data2), data2.outer_dot(data2), atol=atol)


def test_mode_spec_filter_pol_sort_spec_exclusive():
    """Ensure ModeSpec errors when both filter_pol and sort_spec are set."""
    # Using a non-default sort_key triggers the exclusivity check
    with pytest.raises(pd.ValidationError, match="simultaneously"):
        _ = td.ModeSpec(num_modes=1, filter_pol="te", sort_spec=td.ModeSortSpec(sort_key="k_eff"))
    # Using a sort_reference also triggers the exclusivity check
    with pytest.raises(pd.ValidationError, match="simultaneously"):
        _ = td.ModeSpec(num_modes=1, filter_pol="te", sort_spec=td.ModeSortSpec(sort_reference=1.5))
    # Using a filter_key also triggers the exclusivity check
    with pytest.raises(pd.ValidationError, match="simultaneously"):
        _ = td.ModeSpec(
            num_modes=1, filter_pol="te", sort_spec=td.ModeSortSpec(filter_key="TE_fraction")
        )


def test_modes_filter_sort():
    """Test the filtering and sorting of modes."""
    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[WAVEGUIDE],
        run_time=1e-12,
        symmetry=(0, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
    )
    # turn off track_freq so sorting is exact at all freqs
    mode_spec = td.ModeSpec(
        num_modes=5,
        target_neff=2.0,
        sort_spec=td.ModeSortSpec(sort_key="n_eff", sort_order="ascending", track_freq=None),
        num_pml=(10, 10),
    )
    ms = ModeSolver(
        simulation=simulation,
        plane=PLANE,
        mode_spec=mode_spec,
        freqs=[td.C_0 / 1.0, td.C_0 / 2.0],
        direction="-",
    )
    modes = ms.solve()
    assert_property_vs_runtime(ms, modes)
    n_eff = modes.n_eff
    print(n_eff.diff(dim="mode_index"))
    assert np.all(n_eff.diff(dim="mode_index") >= 0)

    for key in get_args(MODE_DATA_KEYS):
        print(key)
        # Test ascending
        sort_kwargs = {
            "sort_key": key,
            "sort_order": "ascending",
            "track_freq": None,
        }
        if key == "fill_fraction_box":
            sort_kwargs["bounding_box"] = td.Box(center=PLANE.center, size=(5.0, 4.0, 5.0))
        sort_spec = td.ModeSortSpec(**sort_kwargs)
        # just check it works without sort_spec
        _ = modes.sort_modes(track_freq="central")
        modes = modes.sort_modes(sort_spec)
        metric = getattr(modes, key)
        assert np.all(metric.diff(dim="mode_index") >= 0)

        # Test descending
        sort_kwargs = {
            "sort_key": key,
            "sort_order": "descending",
            "track_freq": None,
        }
        if key == "fill_fraction_box":
            sort_kwargs["bounding_box"] = td.Box(center=PLANE.center, size=(5.0, 4.0, 5.0))
        sort_spec = td.ModeSortSpec(**sort_kwargs)
        modes = modes.sort_modes(sort_spec)
        metric = getattr(modes, key)
        assert np.all(metric.diff(dim="mode_index") <= 0)

        # Test descending with a large reference value should be the same as ascending
        sort_kwargs = {
            "sort_key": key,
            "sort_order": "descending",
            "sort_reference": 100,
            "track_freq": None,
        }
        if key == "fill_fraction_box":
            sort_kwargs["bounding_box"] = td.Box(center=PLANE.center, size=(5.0, 4.0, 5.0))
        sort_spec = td.ModeSortSpec(**sort_kwargs)
        modes = modes.sort_modes(sort_spec)
        metric = getattr(modes, key)
        assert np.all(metric.diff(dim="mode_index") >= 0)

    # Test filter + sort within groups using n_eff at first frequency
    ms = ms.updated_copy(mode_spec=mode_spec)
    metric = modes.n_eff.isel(f=0)
    thresh = float(np.median(metric.values))

    # Test filter by n_eff and sort by k_eff
    sort_spec = td.ModeSortSpec(
        filter_key="n_eff",
        filter_reference=thresh,
        filter_order="over",
        sort_key="k_eff",
        sort_order="ascending",
        track_freq=None,
    )
    modes = modes.sort_modes(sort_spec)
    for ifreq in range(len(ms.freqs)):
        metric_filtered = modes.n_eff.isel(f=ifreq)
        metric_sorted = modes.k_eff.isel(f=ifreq)
        first_group_size = int(np.sum(metric_filtered.values >= thresh))
        # first group satisfies filter
        assert np.all(metric_filtered.isel(mode_index=slice(0, first_group_size)).values >= thresh)
        # and is sorted ascending within the group
        assert np.all(
            metric_sorted.isel(mode_index=slice(0, first_group_size)).diff(dim="mode_index") >= 0
        )
        # second group satisfies filter
        assert np.all(
            metric_filtered.isel(mode_index=slice(first_group_size, None)).values < thresh
        )
        # and is also sorted ascending within the group
        assert np.all(
            metric_sorted.isel(mode_index=slice(first_group_size, None)).diff(dim="mode_index") >= 0
        )

    # Test filter only with filter_order="under", and no sorting defined
    ms = ms.updated_copy(
        mode_spec=mode_spec.updated_copy(
            sort_spec=td.ModeSortSpec(
                filter_key="TE_fraction",
                filter_reference=0.5,
                filter_order="under",
            )
        )
    )
    # need to solve again because so that the default sorting from the solver will apply, as
    # the modes had been reordered previously, and sort_val is None
    modes = ms.solve()
    for ifreq in range(len(ms.freqs)):
        metric_filtered = modes.TE_fraction.isel(f=ifreq)
        metric_sorted = modes.n_eff.isel(f=ifreq)  # defaults to n_eff in descending order
        first_group_size = int(np.sum(metric_filtered.values <= 0.5))

        # print(metric_filtered.values)
        # print(metric_sorted.values)

        # first group satisfies filter
        assert np.all(metric_filtered.isel(mode_index=slice(0, first_group_size)).values <= 0.5)
        # and is sorted
        assert np.all(
            metric_sorted.isel(mode_index=slice(0, first_group_size)).diff(dim="mode_index") <= 0
        )
        # second group satisfies filter
        assert np.all(metric_filtered.isel(mode_index=slice(first_group_size, None)).values > 0.5)
        # and is also sorted
        assert np.all(
            metric_sorted.isel(mode_index=slice(first_group_size, None)).diff(dim="mode_index") <= 0
        )

    # Test that if we now reorder based on a track_freq, the original order is preserved at
    # the track_freq, but not at the other one
    sort_spec = td.ModeSortSpec(
        sort_key="k_eff",
        sort_order="ascending",
        track_freq="lowest",
    )
    modes = modes.sort_modes(sort_spec=sort_spec)
    assert np.all(np.diff(modes.k_eff.isel(f=0)) >= 0)
    assert not np.all(np.diff(modes.k_eff.isel(f=-1)) >= 0)


def test_sort_spec_track_freq():
    """Test various ways to sort and track that should result in the same final modes."""
    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[WAVEGUIDE],
        run_time=1e-12,
        symmetry=(0, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
    )
    sort_spec = td.ModeSortSpec(sort_key="TE_fraction")
    mode_spec = td.ModeSpec(
        num_modes=5,
        target_neff=2.0,
        sort_spec=sort_spec.updated_copy(track_freq="lowest"),
        num_pml=(10, 10),
        group_index_step=True,
    )
    ms = ModeSolver(
        simulation=simulation,
        plane=PLANE,
        mode_spec=mode_spec,
        freqs=[td.C_0 / 0.5, td.C_0 / 1.0, td.C_0 / 2.0],
        direction="-",
    )
    modes_lowest = ms.solve()

    # TODO remove this when track_freq is removed
    mode_spec = td.ModeSpec(
        num_modes=5,
        target_neff=2.0,
        sort_spec=sort_spec,
        track_freq="lowest",
        num_pml=(10, 10),
        group_index_step=True,
    )
    ms = ms.updated_copy(mode_spec=mode_spec)
    modes_lowest_legacy = ms.solve()

    assert modes_lowest == modes_lowest_legacy

    mode_spec = td.ModeSpec(
        num_modes=5,
        target_neff=2.0,
        sort_spec=sort_spec.updated_copy(track_freq=None),
        num_pml=(10, 10),
        group_index_step=True,
    )
    ms = ms.updated_copy(mode_spec=mode_spec)
    modes_untracked = ms.solve()

    assert not np.all(modes_lowest.n_eff == modes_untracked.n_eff)

    modes_lowest_retracked = modes_untracked.overlap_sort(track_freq="lowest")

    # The field modes come out with a different phase so the datas are not equivalent but we can
    # check that everything matches
    assert np.allclose(modes_lowest.Ex.abs, modes_lowest_retracked.Ex.abs)
    assert np.all(modes_lowest.n_eff == modes_lowest_retracked.n_eff)
    assert np.all(modes_lowest.n_group == modes_lowest_retracked.n_group)


def test_degenerate_mode_processing():
    """Ensure degenerate modes returned by mode solver are bi-orthogonal."""
    freq0 = td.C_0
    sim_size = (0, 2, 2)
    inf = 10
    W1 = 0.3
    n = 1.5
    num_modes = 4
    mode_spec = td.ModeSpec(num_modes=num_modes)
    medium = td.Medium(permittivity=n**2)
    geom1 = td.Box.from_bounds((-inf, -W1 / 2, -W1 / 2), (inf, W1 / 2, W1 / 2))
    wg1 = td.Structure(geometry=geom1, medium=medium)

    grid_spec = td.GridSpec.uniform(dl=0.2)

    sim = td.Simulation(
        size=sim_size,
        structures=[wg1],
        grid_spec=grid_spec,
        run_time=10 / freq0,
    )

    ms = ModeSolver(
        simulation=sim,
        plane=sim.geometry,
        mode_spec=mode_spec,
        freqs=[freq0],
        direction="+",
        colocate=False,
        use_colocated_integration=False,
        conjugated_dot_product=False,
    )

    mode_data = ms.data_raw
    assert_property_vs_runtime(ms, mode_data)

    degen_sets = EigSolver._identify_degenerate_modes(
        mode_data.n_complex.values[0, :], TOL_DEGENERATE_CANDIDATE
    )
    assert len(degen_sets) == 1

    S = mode_data.outer_dot(mode_data, conjugate=False).isel(f=0).values
    # The orthogonalization is numerically stable, but the residual off-diagonal
    # overlap can drift slightly above 1e-9 on some CI platforms.
    threshold = 2e-9
    off_diag_mask = ~np.eye(S.shape[0], dtype=bool)
    large_vals = np.abs(S) > threshold
    problem_mask = off_diag_mask & large_vals

    indices = np.argwhere(problem_mask)
    msg = f"Found {len(indices)} off-diagonal values > {threshold}:\n"
    msg += "\n".join(f"  |S[{i},{j}]| = {np.abs(S[i, j]):.4e}" for i, j in indices)
    assert not np.any(problem_mask), msg


def test_mode_sort_spec_drop_modes_reduces_modes():
    freqs = np.array([2e14, 4e14])
    mode_spec = td.ModeSpec(num_modes=3)
    monitor = td.ModeSolverMonitor(
        size=(1.0, 0.0, 1.0),
        center=(0.0, 0.0, 0.0),
        freqs=freqs,
        mode_spec=mode_spec,
        name="drop_modes",
    )
    n_complex = ModeIndexDataArray(
        np.array(
            [
                [1.6 + 0.6j, 1.5 + 0.2j, 1.1 + 0.5j],
                [1.7 + 0.4j, 1.4 + 0.3j, 1.0 + 0.1j],
            ]
        ),
        coords={"f": freqs, "mode_index": np.arange(3)},
    )
    data = ModeSolverData(monitor=monitor, n_complex=n_complex)

    sort_spec = td.ModeSortSpec(
        filter_key="n_eff",
        filter_reference=1.3,
        filter_order="over",
        sort_key="k_eff",
        sort_order="ascending",
        keep_modes="filtered",
    )

    sorted_data = data.sort_modes(sort_spec)

    assert sorted_data.n_eff.sizes["mode_index"] == 2
    assert np.allclose(sorted_data.n_eff.isel(f=0).values, [1.5, 1.6])
    assert np.allclose(sorted_data.n_eff.isel(f=1).values, [1.4, 1.7])
    assert sorted_data.monitor.mode_spec.num_modes == 2
    assert sorted_data.monitor.mode_spec.sort_spec.keep_modes == "filtered"


@pytest.mark.parametrize("keep_modes", (1, 3))
def test_mode_sort_spec_keep_modes_integer(keep_modes):
    freqs = np.array([2e14, 4e14])
    mode_spec = td.ModeSpec(num_modes=4)
    monitor = td.ModeSolverMonitor(
        size=(1.0, 0.0, 1.0),
        center=(0.0, 0.0, 0.0),
        freqs=freqs,
        mode_spec=mode_spec,
        name="drop_modes",
    )
    n_complex = ModeIndexDataArray(
        np.array(
            [
                [1.6 + 0.6j, 1.5 + 0.2j, 1.1 + 0.5j, 1.05 + 0.2j],
                [1.7 + 0.4j, 1.4 + 0.3j, 1.07 + 0.1j, 1.02 + 0.3j],
            ]
        ),
        coords={"f": freqs, "mode_index": np.arange(4)},
    )
    data = ModeSolverData(monitor=monitor, n_complex=n_complex)

    sort_spec = td.ModeSortSpec(
        filter_key="n_eff",
        filter_reference=1.3,
        filter_order="over",
        sort_key="k_eff",
        sort_order="ascending",
        keep_modes=keep_modes,
    )

    if keep_modes == 1:
        with AssertLogLevel(None):
            sorted_data = data.sort_modes(sort_spec)
    else:
        with AssertLogLevel("WARNING", contains_str="filter"):
            sorted_data = data.sort_modes(sort_spec)

    assert sorted_data.n_eff.sizes["mode_index"] == keep_modes
    if keep_modes == 1:
        assert np.allclose(sorted_data.n_eff.isel(f=0).values, [1.5])
        assert np.allclose(sorted_data.n_eff.isel(f=1).values, [1.4])
    else:
        assert np.allclose(sorted_data.n_eff.isel(f=0).values, [1.5, 1.6, 1.05])
        assert np.allclose(sorted_data.n_eff.isel(f=1).values, [1.4, 1.7, 1.07])
    assert sorted_data.monitor.mode_spec.num_modes == keep_modes
    assert sorted_data.monitor.mode_spec.sort_spec.keep_modes == keep_modes


def test_mode_sort_spec_drop_modes_all_filtered():
    freqs = np.array([2e14, 4e14])
    mode_spec = td.ModeSpec(num_modes=3)
    monitor = td.ModeSolverMonitor(
        size=(1.0, 0.0, 1.0),
        center=(0.0, 0.0, 0.0),
        freqs=freqs,
        mode_spec=mode_spec,
        name="drop_all",
    )
    n_complex = ModeIndexDataArray(
        np.array(
            [
                [1.1 + 0.1j, 1.05 + 0.05j, 1.0 + 0.01j],
                [1.1 + 0.1j, 1.05 + 0.05j, 1.0 + 0.01j],
            ]
        ),
        coords={"f": freqs, "mode_index": np.arange(3)},
    )
    data = ModeSolverData(monitor=monitor, n_complex=n_complex)

    sort_spec = td.ModeSortSpec(
        filter_key="n_eff",
        filter_reference=2.0,
        keep_modes="filtered",
    )

    with pytest.raises(ValidationError):
        _ = data.sort_modes(sort_spec)


def test_mode_sort_spec_drop_modes_requires_filter():
    with pytest.raises(pd.ValidationError):
        td.ModeSortSpec(keep_modes="filtered")


def test_mode_sort_spec_keep_modes_at_most_num_modes():
    sort_spec = td.ModeSortSpec(keep_modes=4)
    with pytest.raises(pd.ValidationError):
        _ = td.ModeSpec(num_modes=2, sort_spec=sort_spec)


def test_mode_sort_spec_fill_fraction_box_filter_drops_modes():
    data, bounding_box = make_fill_fraction_mode_data()

    sort_spec = td.ModeSortSpec(
        filter_key="fill_fraction_box",
        filter_reference=0.5,
        filter_order="over",
        keep_modes="filtered",
        bounding_box=bounding_box,
    )

    filtered = data.sort_modes(sort_spec)

    assert filtered.n_eff.sizes["mode_index"] == 1
    assert filtered.monitor.mode_spec.num_modes == 1

    fills = data.fill_fraction(bounding_box)
    assert np.isclose(fills.isel(mode_index=0, f=0).item(), 1.0)
    assert np.isclose(fills.isel(mode_index=1, f=0).item(), 0.0)


def test_mode_sort_spec_fill_fraction_box_requires_bounding_box():
    with pytest.raises(pd.ValidationError):
        td.ModeSortSpec(filter_key="fill_fraction_box")


def test_mode_data_fill_fraction_box_requires_intersection():
    data, _ = make_fill_fraction_mode_data()
    with pytest.raises(ValidationError):
        data.fill_fraction(td.Box(center=(0.0, 2.0, 0.0), size=(1.0, 1.0, 1.0)))
