import pytest
import responses
import numpy as np
import matplotlib.pyplot as plt
import pydantic.v1 as pydantic

import tidy3d as td

import tidy3d.plugins.mode.web as msweb
from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins.mode.mode_solver import MODE_MONITOR_NAME
from tidy3d.plugins.mode.derivatives import create_sfactor_b, create_sfactor_f
from tidy3d.plugins.mode.solver import compute_modes
from tidy3d.exceptions import SetupError
from ..utils import assert_log_level, log_capture
from tidy3d import ScalarFieldDataArray
from tidy3d.web.core.environment import Env


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


@pytest.mark.parametrize("group_index_step, log_level", ((1e-7, "WARNING"), (1e-5, "INFO")))
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

    ms = ModeSolver(
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
        dataframe = ms.data.to_dataframe()

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

    plane_left = td.Box(center=(-0.5, 0, 0), size=(0.9, 0, 0.9))
    plane_right = td.Box(center=(0.5, 0, 0), size=(0.9, 0, 0.9))

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

        fname = str(tmp_path / "ms_custom_medium.hdf5")
        ms.to_file(fname)
        m2 = ModeSolver.from_file(fname)
        assert m2 == ms

    if local:
        assert n_eff[0] < 1.5
        assert n_eff[1] > 4
        assert n_eff[1] < 5


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

    for key, val in ms.data.modes_info.items():
        tol = 1e-2
        if key == "mode area":
            tol = 2.5e-2  # mode area has higher error
        # print(val, ms_angled.data.modes_info[key])
        print(np.amax(np.abs(val - ms_angled.data.modes_info[key])))
        assert np.allclose(val, ms_angled.data.modes_info[key], rtol=tol, atol=tol)


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
    dataframe = ms.data.to_dataframe()

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
    compare_colocation(ms)
    verify_pol_fraction(ms)
    verify_dtype(ms)
    dataframe = ms.data.to_dataframe()

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
    dataframe = ms.data.to_dataframe()

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


@pytest.mark.parametrize("local", [True, False])
@responses.activate
def test_group_index(mock_remote_api, log_capture, local):
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
        assert len(log_capture) == 1
        assert log_capture[0][0] == 30
        assert "ModeSpec" in log_capture[0][1]
        _ = modes.n_group
        assert len(log_capture) == 1

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

    assert list(np.where(np.isnan(md.pol_fraction.te))[1]) == [8, 9]


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
    st = td.GaussianPulse(freq0=1.0, fwidth=1.0)

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
