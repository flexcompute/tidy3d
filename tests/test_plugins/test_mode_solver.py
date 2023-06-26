import pytest
import responses
import numpy as np
import matplotlib.pyplot as plt

import tidy3d as td

import tidy3d.plugins.mode.web as msweb
from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins.mode.derivatives import create_sfactor_b, create_sfactor_f
from tidy3d.plugins.mode.solver import compute_modes
from tidy3d import ScalarFieldDataArray
from tidy3d.web.environment import Env


WAVEGUIDE = td.Structure(geometry=td.Box(size=(100, 0.5, 0.5)), medium=td.Medium(permittivity=4.0))
PLANE = td.Box(center=(0, 0, 0), size=(5, 0, 5))
SIM_SIZE = (5, 5, 5)
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

    def mock_download(task_id, remote_path, to_file, *args, **kwargs):
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
            num_modes=1,
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

    monkeypatch.setattr(td.web.http_management, "api_key", lambda: "api_key")
    monkeypatch.setattr("tidy3d.plugins.mode.web.upload_file", void)
    monkeypatch.setattr("tidy3d.plugins.mode.web.upload_string", void)
    monkeypatch.setattr("tidy3d.plugins.mode.web.download_file", mock_download)

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
                    "fileType": "Json",
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
                "fileType": "Json",
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
                    "taskName": TASK_NAME,
                    "modeSolverName": MODESOLVER_NAME,
                    "fileType": "Hdf5",
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
                "fileType": "Hdf5",
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
                "fileType": "Json",
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
    )


@pytest.mark.parametrize("local", [True, False])
@responses.activate
def test_mode_solver_simple(mock_remote_api, local):
    """Simple mode solver run (with symmetry)"""

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
    _ = ms.solve() if local else msweb.run(ms)


@pytest.mark.parametrize("local", [True, False])
@responses.activate
def test_mode_solver_custom_medium(mock_remote_api, local):
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

    plane_left = td.Box(center=(-0.5, 0, 0), size=(1, 0, 1))
    plane_right = td.Box(center=(0.5, 0, 0), size=(1, 0, 1))

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
        assert n_eff[0] < 1.5
        assert n_eff[1] > 4
        assert n_eff[1] < 5


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
    _ = ms.solve()
    # Plot field
    _, ax = plt.subplots(1)
    ms.plot_field("Ex", ax=ax, mode_index=1)
    plt.close()

    # Create source and monitor
    st = td.GaussianPulse(freq0=1.0, fwidth=1.0)
    _ = ms.to_source(source_time=st, direction="-")
    _ = ms.to_monitor(freqs=[1.0, 2.0], name="mode_mnt")


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
    _ = ms.solve()

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
        sources=[SRC],
    )
    ms = ModeSolver(
        simulation=simulation, plane=PLANE, mode_spec=mode_spec, freqs=[td.C_0 / 1.0], direction="+"
    )
    _ = ms.solve()

    # The simulation and the mode plane are both 0D along the same dimension
    simulation = td.Simulation(
        size=PLANE.size,
        grid_spec=td.GridSpec(wavelength=1.0),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
    )
    ms = ModeSolver(simulation=simulation, plane=PLANE, mode_spec=mode_spec, freqs=[td.C_0 / 1.0])
    _ = ms.solve()


@pytest.mark.parametrize("local", [True, False])
@responses.activate
def test_group_index(mock_remote_api, local):
    """Test group index calculation"""

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
