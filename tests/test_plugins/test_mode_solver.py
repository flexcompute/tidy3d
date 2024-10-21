import matplotlib.pyplot as plt
import numpy as np
import pydantic.v1 as pydantic
import pytest
import responses
import tidy3d as td
import tidy3d.plugins.mode.web as msweb
from tidy3d import ScalarFieldDataArray
from tidy3d.components.data.monitor_data import ModeSolverData
from tidy3d.exceptions import SetupError
from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins.mode.derivatives import create_sfactor_b, create_sfactor_f
from tidy3d.plugins.mode.mode_solver import MODE_MONITOR_NAME
from tidy3d.plugins.mode.solver import compute_modes
from tidy3d.web.core.environment import Env

from ..utils import assert_log_level, cartesian_to_unstructured

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


@pytest.fixture
def mock_remote_api(monkeypatch):
    def void(*args, **kwargs):
        return None

    def mock_download(resource_id, remote_filename, to_file, *args, **kwargs):
        simulation = td.Simulation(
            size=SIM_SIZE,
            grid_spec=td.GridSpec(wavelength=1.0),
            structures=[WAVEGUIDE],
            run_time=1e-12,
            symmetry=(1, 0, -1),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
            sources=[SRC],
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


def test_compute_modes():
    """Test direct call to `compute_modes`."""
    eps_cross = np.random.rand(10, 10)
    coords = np.arange(11)
    mode_spec = td.ModeSpec(num_modes=3, target_neff=2.0)
    _ = compute_modes(
        eps_cross=[eps_cross] * 9,
        coords=[coords, coords],
        freq=td.C_0 / 1.0,
        mode_spec=mode_spec,
        direction="-",
        precision="single",
    )


def compare_colocation(ms):
    """Compare mode-solver fields with colocation applied during run or post-run."""
    data_col = ms.solve()
    ms_nocol = ms.updated_copy(colocate=False)
    data = ms_nocol.solve()
    data_at_boundaries = ms_nocol.sim_data.at_boundaries(MODE_MONITOR_NAME)

    for key, field in data_col.field_components.items():
        # Check the colocated data is the same
        assert np.allclose(data_at_boundaries[key], field, atol=1e-7)

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
    assert np.allclose(ms.data.n_eff.values, modes_red.n_eff.values)


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
    with pytest.raises(pydantic.ValidationError):
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
def test_mode_solver_group_index_warning(group_index_step, log_level, log_capture):
    """Test mode solver setups issuing warnings."""

    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        run_time=1e-12,
    )
    mode_spec = td.ModeSpec(
        num_modes=1,
        group_index_step=group_index_step,
    )

    _ = ModeSolver(
        simulation=simulation,
        plane=PLANE,
        mode_spec=mode_spec,
        freqs=[1e12],
        direction="+",
    )
    assert_log_level(log_capture, log_level)


@pytest.mark.parametrize("local", [True, False])
@responses.activate
def test_mode_solver_simple(mock_remote_api, local):
    """Simple mode solver run (with symmetry)"""

    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[WAVEGUIDE],
        run_time=1e-12,
        symmetry=(0, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
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
        _ = msweb.run(ms)

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
def test_mode_solver_remote_after_local(mock_remote_api):
    """Test that running a remote solver after a local one modifies the stored data. This is to
    catch a bug if ``_cached_properties["data"]`` is inadvertently used."""

    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[WAVEGUIDE],
        run_time=1e-12,
        symmetry=(0, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
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
    data_remote = msweb.run(ms)
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
    n_data = ScalarFieldDataArray(n, coords=dict(x=x_custom, y=y_custom, z=z_custom, f=[freq0]))
    mat_custom = td.CustomMedium.from_nk(n_data, interp_method="nearest")

    waveguide = td.Structure(geometry=td.Box(size=(100, 0.5, 0.5)), medium=mat_custom)
    simulation = td.Simulation(
        size=(2, 2, 2),
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[waveguide],
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
        modes = ms.solve() if local else msweb.run(ms)
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
    n_data = td.SpatialDataArray(n, coords=dict(x=x_custom, y=y_custom, z=z_custom))

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
            structures=[waveguide],
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
        md.append(modes)

    # ms.plot_field(mode_index=0, f=freq0, field_name="Ez")
    # plt.show()

    error_u = np.abs(md[0].n_eff - md[1].n_eff).values.item()
    error_up = np.abs(md[0].n_eff - md[2].n_eff).values.item()

    print(nx, cond_factor, interp, tol, error_u, error_up)

    assert error_u < 5e-5
    assert error_up < tol


def test_mode_solver_straight_vs_angled():
    """Compare results for a straight and angled nominally identical waveguides.
    Note: results do not match perfectly because of the numerical grid.
    """
    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec.auto(wavelength=1.0, min_steps_per_wvl=16),
        structures=[WAVEGUIDE],
        run_time=1e-12,
        symmetry=(0, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
    )
    mode_spec = td.ModeSpec(num_modes=5, group_index_step=True)
    freqs = [td.C_0 / 0.9, td.C_0 / 1.0, td.C_0 / 1.1]
    ms = ModeSolver(
        simulation=simulation,
        plane=PLANE,
        mode_spec=mode_spec,
        freqs=freqs,
        direction="-",
    )

    angle = np.pi / 6
    width, height = WAVEGUIDE.geometry.size[0], WAVEGUIDE.geometry.size[2]
    vertices = np.array(
        [[-width / 2, -100, 0], [width / 2, -100, 0], [width / 2, 100, 0], [-width / 2, 100, 0]]
    )
    vertices = PLANE.rotate_points(vertices.T, axis=[0, 0, 1], angle=-angle).T
    vertices = [verts[:2] for verts in vertices]
    wg_angled = td.Structure(
        geometry=td.PolySlab(vertices=vertices, slab_bounds=(-height / 2, height / 2)),
        medium=WG_MEDIUM,
    )
    mode_spec_angled = mode_spec.updated_copy(angle_theta=angle)
    src_angled = td.ModeSource(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        center=PLANE.center,
        size=PLANE.size,
        mode_spec=mode_spec_angled,
        direction="-",
        mode_index=0,
    )
    sim_angled = simulation.updated_copy(structures=[wg_angled], sources=[src_angled])
    # sim_angled.plot(z=0)
    # plt.show()

    ms_angled = ModeSolver(
        simulation=sim_angled,
        plane=PLANE,
        mode_spec=mode_spec_angled,
        freqs=freqs,
        direction="-",
    )

    check_ms_reduction(ms)
    check_ms_reduction(ms_angled)

    for key, val in ms.data.modes_info.items():
        tol = 1e-2
        if key == "TE (Ex) fraction":
            tol = 0.1
        elif key == "wg TE fraction":
            tol = 1.3e-2
        elif key == "mode area":
            tol = 2.1e-2
        elif key == "dispersion (ps/(nm km))":
            tol = 0.7
        # print(
        #     key,
        #     (np.abs(val - ms_angled.data.modes_info[key]) / np.abs(val)).values.max(),
        #     (np.abs(val - ms_angled.data.modes_info[key]) / np.abs(ms_angled.data.modes_info[key])).values.max(),
        # )
        assert np.allclose(val, ms_angled.data.modes_info[key], rtol=tol)


def test_mode_solver_angle_bend():
    """Run mode solver with angle and bend and symmetry"""
    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[WAVEGUIDE],
        run_time=1e-12,
        symmetry=(-1, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
    )
    mode_spec = td.ModeSpec(
        num_modes=3,
        target_neff=2.0,
        bend_radius=3,
        bend_axis=0,
        angle_theta=np.pi / 3,
        angle_phi=np.pi,
        track_freq="highest",
    )
    # put plane entirely in the symmetry quadrant rather than sitting on its center
    plane = td.Box(center=(0, 0.5, 0), size=(1, 0, 1))
    ms = ModeSolver(
        simulation=simulation, plane=plane, mode_spec=mode_spec, freqs=[td.C_0 / 1.0], direction="-"
    )
    compare_colocation(ms)
    verify_pol_fraction(ms)
    verify_dtype(ms)
    _ = ms.data.to_dataframe()
    check_ms_reduction(ms)

    # Plot field
    _, ax = plt.subplots(1)
    ms.plot_field("Ex", ax=ax, mode_index=1)
    plt.close()

    # Create source and monitor
    st = td.GaussianPulse(freq0=1.0e12, fwidth=1.0e12)
    _ = ms.to_source(source_time=st, direction="-")
    _ = ms.to_monitor(freqs=np.array([1.0, 2.0]) * 1e12, name="mode_mnt")


def test_mode_bend_radius():
    """Test that the bend radius is correctly applied to the center of the mode plane in the case
    of an auto-grid that is not symmetric w.r.t. that center, and that nominally identical
    waveguides produce the same modes when the bend center if shifted away from the mode plane
    center, after a post-processing transformation to correct for that."""

    simulation = td.Simulation(
        size=(10, 10, 10),
        grid_spec=td.GridSpec(wavelength=1.0),
        # grid_spec=td.GridSpec.uniform(dl=0.04),
        structures=[WAVEGUIDE],
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
    data2 = ms2.solve()

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
        structures=[WAVEGUIDE],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
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
        structures=[WAVEGUIDE],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.pml(z=False),
        sources=[SRC],
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
        sources=[SRC],
    )
    ms = ModeSolver(simulation=simulation, plane=PLANE, mode_spec=mode_spec, freqs=[td.C_0 / 1.0])
    compare_colocation(ms)
    verify_pol_fraction(ms)
    check_ms_reduction(ms)


@pytest.mark.parametrize("local", [True, False])
@responses.activate
def test_group_index(mock_remote_api, log_capture, local):
    """Test group index and dispersion calculation"""

    simulation = td.Simulation(
        size=(5, 5, 1),
        grid_spec=td.GridSpec(wavelength=1.55),
        structures=[
            td.Structure(
                geometry=td.Box(size=(0.5, 0.22, td.inf)), medium=td.Medium(permittivity=3.48**2)
            )
        ],
        medium=td.Medium(permittivity=1.44**2),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
    )
    mode_spec = td.ModeSpec(
        num_modes=2,
        target_neff=3.0,
        precision="double" if local else "single",
        track_freq="central",
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
    modes = ms.solve() if local else msweb.run(ms)
    if local:
        assert modes.n_group is None
        assert len(log_capture) == 1
        assert modes.dispersion is None
        assert len(log_capture) == 2
        for log_msg in log_capture:
            assert log_msg[0] == 30
            assert "ModeSpec" in log_msg[1]
        _ = modes.n_group
        assert len(log_capture) == 2
        _ = modes.dispersion
        assert len(log_capture) == 2
        check_ms_reduction(ms)

    # Group index calculated
    ms = ModeSolver(
        simulation=simulation,
        plane=td.Box(size=(td.inf, td.inf, 0)),
        mode_spec=mode_spec.copy(update={"group_index_step": True}),
        freqs=freqs,
    )
    modes = ms.solve() if local else msweb.run(ms)
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


def test_mode_solver_nan_pol_fraction():
    """Test mode solver when eigensolver returns 0 for some modes."""
    wg = td.Structure(geometry=td.Box(size=(0.5, 100, 0.22)), medium=td.Medium(permittivity=12))

    simulation = td.Simulation(
        medium=td.Medium(permittivity=2),
        size=SIM_SIZE,
        grid_spec=td.GridSpec.auto(wavelength=1.55, min_steps_per_wvl=15),
        structures=[wg],
        run_time=1e-12,
        symmetry=(0, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
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
    check_ms_reduction(ms)

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
        sources=[SRC],
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
def test_mode_solver_web_run_batch(mock_remote_api):
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
        structures=[WAVEGUIDE],
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
    results = msweb.run_batch(mode_solver_list, verbose=False, folder_name="Mode Solver")
    print(*results, sep="\n")
    assert all(isinstance(x, ModeSolverData) for x in results)
    assert (results[i].n_eff.shape == (num_freqs, i + 1) for i in range(num_of_sims))


def test_mode_solver_relative():
    """Relative mode solver"""

    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[WAVEGUIDE],
        run_time=1e-12,
        symmetry=(0, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
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
    new_freqs = np.array(freqs) * 1.01
    ms = ms.updated_copy(freqs=new_freqs)
    _ = ms._data_on_yee_grid_relative(basis=basis)


def test_mode_solver_plot():
    """Test mode plane plotting functions"""

    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[WAVEGUIDE],
        run_time=1e-12,
        symmetry=(0, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
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


@pytest.mark.parametrize("local", [True, False])
@responses.activate
def test_modes_eme_sim(mock_remote_api, local):
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
        _ = solver.data
    else:
        with pytest.raises(SetupError):
            _ = msweb.run(solver)
        _ = msweb.run(solver.to_fdtd_mode_solver())

    _ = solver.reduced_simulation_copy
