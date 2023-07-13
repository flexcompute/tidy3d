"""Test near field to far field transformations."""
import numpy as np
import tidy3d as td
import pytest

from tidy3d.exceptions import SetupError, DataError
from ..utils import clear_tmp


MEDIUM = td.Medium(permittivity=3)
WAVELENGTH = 1
F0 = td.C_0 / WAVELENGTH / np.sqrt(MEDIUM.permittivity)
R_FAR = 50 * WAVELENGTH
MAKE_PLOTS = False


def make_proj_monitors(center, size, freqs):
    """Helper function to make near-to-far monitors."""
    Ntheta = 40
    Nphi = 36
    thetas = np.linspace(0, np.pi, Ntheta)
    phis = np.linspace(0, 2 * np.pi, Nphi)

    far_size = 10 * WAVELENGTH
    Nx = 40
    Ny = 36
    xs = np.linspace(-far_size / 2, far_size / 2, Nx)
    ys = np.linspace(-far_size / 2, far_size / 2, Ny)
    z = R_FAR

    Nux = 40
    Nuy = 36
    uxs = np.linspace(-0.3, 0.3, Nux)
    uys = np.linspace(-0.4, 0.4, Nuy)

    exclude_surfaces = None
    if size.count(0.0) == 0:
        exclude_surfaces = ["x+", "y-"]

    n2f_angle_monitor = td.FieldProjectionAngleMonitor(
        center=center,
        size=size,
        freqs=freqs,
        name="n2f_angle",
        custom_origin=center,
        phi=list(phis),
        theta=list(thetas),
        normal_dir="+",
        exclude_surfaces=exclude_surfaces,
    )

    proj_axis = 0
    n2f_cart_monitor = td.FieldProjectionCartesianMonitor(
        center=center,
        size=size,
        freqs=freqs,
        name="n2f_cart",
        custom_origin=center,
        x=list(xs),
        y=list(ys),
        proj_axis=proj_axis,
        proj_distance=z,
        normal_dir="+",
        exclude_surfaces=exclude_surfaces,
    )

    proj_axis = 0
    n2f_ksp_monitor = td.FieldProjectionKSpaceMonitor(
        center=center,
        size=size,
        freqs=freqs,
        name="n2f_ksp",
        custom_origin=center,
        ux=list(uxs),
        uy=list(uys),
        proj_axis=proj_axis,
        normal_dir="+",
        exclude_surfaces=exclude_surfaces,
    )

    exact_cart_monitor = td.FieldProjectionCartesianMonitor(
        center=center,
        size=size,
        freqs=freqs,
        name="exact_cart",
        custom_origin=center,
        x=list(xs),
        y=list(ys),
        proj_axis=proj_axis,
        proj_distance=z,
        normal_dir="+",
        exclude_surfaces=exclude_surfaces,
        far_field_approx=False,
    )

    return n2f_angle_monitor, n2f_cart_monitor, n2f_ksp_monitor, exact_cart_monitor


def test_proj_monitors():
    """Make sure all the near-to-far monitors can be created."""

    dipole_center = [0, 0, 0]
    domain_size = 5 * WAVELENGTH  # domain size
    buffer_mon = 1 * WAVELENGTH  # buffer between the dipole and the monitors

    grid_spec = td.GridSpec.auto(min_steps_per_wvl=20)
    boundary_spec = td.BoundarySpec.all_sides(boundary=td.PML())
    sim_size = (domain_size, domain_size, domain_size)

    # source
    fwidth = F0 / 10.0
    offset = 4.0
    gaussian = td.GaussianPulse(freq0=F0, fwidth=fwidth, offset=offset)
    source = td.PointDipole(center=dipole_center, source_time=gaussian, polarization="Ez")
    run_time = 40 / fwidth
    freqs = [(0.9 * F0), F0, (1.1 * F0)]

    # make monitors
    mon_size = [buffer_mon] * 3
    proj_monitors = make_proj_monitors(dipole_center, mon_size, freqs)

    near_monitors = td.FieldMonitor.surfaces(
        center=dipole_center, size=mon_size, freqs=freqs, name="near"
    )

    all_monitors = near_monitors + list(proj_monitors)

    sim = td.Simulation(
        size=sim_size,
        grid_spec=grid_spec,
        structures=[],
        sources=[source],
        monitors=all_monitors,
        run_time=run_time,
        boundary_spec=boundary_spec,
        medium=MEDIUM,
    )


@clear_tmp
def test_proj_data():
    """Make sure all the near-to-far data structures can be created."""

    f = np.linspace(1e14, 2e14, 10)
    r = np.atleast_1d(5)
    theta = np.linspace(0, np.pi, 10)
    phi = np.linspace(0, 2 * np.pi, 20)
    coords_tp = dict(r=r, theta=theta, phi=phi, f=f)
    values_tp = (1 + 1j) * np.random.random((len(r), len(theta), len(phi), len(f)))
    scalar_field_tp = td.FieldProjectionAngleDataArray(values_tp, coords=coords_tp)
    monitor_tp = td.FieldProjectionAngleMonitor(
        center=(1, 2, 3), size=(2, 2, 2), freqs=f, name="n2f_monitor_tp", phi=phi, theta=theta
    )
    data_tp = td.FieldProjectionAngleData(
        monitor=monitor_tp,
        projection_surfaces=monitor_tp.projection_surfaces,
        Er=scalar_field_tp,
        Etheta=scalar_field_tp,
        Ephi=scalar_field_tp,
        Hr=scalar_field_tp,
        Htheta=scalar_field_tp,
        Hphi=scalar_field_tp,
    )

    x = np.linspace(0, 5, 10)
    y = np.linspace(0, 10, 20)
    z = np.atleast_1d(5)
    coords_xy = dict(x=x, y=y, z=z, f=f)
    values_xy = (1 + 1j) * np.random.random((len(x), len(y), len(z), len(f)))
    scalar_field_xy = td.FieldProjectionCartesianDataArray(values_xy, coords=coords_xy)
    monitor_xy = td.FieldProjectionCartesianMonitor(
        center=(1, 2, 3),
        size=(2, 2, 2),
        freqs=f,
        name="n2f_monitor_xy",
        x=x,
        y=y,
        proj_axis=2,
        proj_distance=50,
    )
    data_xy = td.FieldProjectionCartesianData(
        monitor=monitor_xy,
        projection_surfaces=monitor_xy.projection_surfaces,
        Er=scalar_field_xy,
        Etheta=scalar_field_xy,
        Ephi=scalar_field_xy,
        Hr=scalar_field_xy,
        Htheta=scalar_field_xy,
        Hphi=scalar_field_xy,
    )

    ux = np.linspace(0, 0.4, 10)
    uy = np.linspace(0, 0.6, 20)
    r = np.atleast_1d(5)
    coords_u = dict(ux=ux, uy=uy, r=r, f=f)
    values_u = (1 + 1j) * np.random.random((len(ux), len(uy), len(r), len(f)))
    scalar_field_u = td.FieldProjectionKSpaceDataArray(values_u, coords=coords_u)
    monitor_u = td.FieldProjectionKSpaceMonitor(
        center=(1, 2, 3), size=(2, 2, 2), freqs=f, name="n2f_monitor_u", ux=ux, uy=uy, proj_axis=2
    )
    data_u = td.FieldProjectionKSpaceData(
        monitor=monitor_u,
        projection_surfaces=monitor_u.projection_surfaces,
        Er=scalar_field_u,
        Etheta=scalar_field_u,
        Ephi=scalar_field_u,
        Hr=scalar_field_u,
        Htheta=scalar_field_u,
        Hphi=scalar_field_u,
    )

    sim = td.Simulation(
        size=(7, 7, 9),
        grid_spec=td.GridSpec.auto(wavelength=5.0),
        monitors=[monitor_xy, monitor_u, monitor_tp],
        run_time=1e-12,
    )

    sim_data = td.SimulationData(simulation=sim, data=(data_xy, data_u, data_tp))
    sim_data[monitor_xy.name]
    sim_data.to_file("tests/tmp/sim_data_n2f.hdf5")
    sim_data = td.SimulationData.from_file("tests/tmp/sim_data_n2f.hdf5")

    x = np.linspace(0, 5, 10)
    y = np.linspace(0, 10, 20)
    z = np.atleast_1d(5)
    coords_xy = dict(x=x, y=y, z=z, f=f)
    values_xy = (1 + 1j) * np.random.random((len(x), len(y), len(z), len(f)))
    scalar_field_xy = td.FieldProjectionCartesianDataArray(values_xy, coords=coords_xy)
    monitor_xy_exact = td.FieldProjectionCartesianMonitor(
        center=(1, 2, 3),
        size=(2, 2, 2),
        freqs=f,
        name="exact_monitor_xy",
        x=x,
        y=y,
        proj_axis=2,
        proj_distance=50,
        far_field_approx=False,
    )
    data_xy_exact = td.FieldProjectionCartesianData(
        monitor=monitor_xy,
        projection_surfaces=monitor_xy.projection_surfaces,
        Er=scalar_field_xy,
        Etheta=scalar_field_xy,
        Ephi=scalar_field_xy,
        Hr=scalar_field_xy,
        Htheta=scalar_field_xy,
        Hphi=scalar_field_xy,
    )


def test_proj_clientside():
    """Make sure the client-side near-to-far class can be created."""

    center = (0, 0, 0)
    size = (2, 2, 0)
    f0 = 1
    monitor = td.FieldMonitor(size=size, center=center, freqs=[f0], name="near_field")

    sim_size = (5, 5, 5)
    sim = td.Simulation(
        size=sim_size,
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / f0),
        monitors=[monitor],
        run_time=1e-12,
    )

    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    z = np.array([0.0])
    f = [f0]
    coords = dict(x=x, y=y, z=z, f=f)
    scalar_field = td.ScalarFieldDataArray(
        (1 + 1j) * np.random.random((10, 10, 1, 1)), coords=coords
    )
    data = td.FieldData(
        monitor=monitor,
        Ex=scalar_field,
        Ey=scalar_field,
        Ez=scalar_field,
        Hx=scalar_field,
        Hy=scalar_field,
        Hz=scalar_field,
        symmetry=sim.symmetry,
        symmetry_center=sim.center,
        grid_expanded=sim.discretize(monitor, extend=True),
    )

    sim_data = td.SimulationData(simulation=sim, data=(data,))

    proj = td.FieldProjector.from_near_field_monitors(
        sim_data=sim_data, near_monitors=[monitor], normal_dirs=["+"]
    )

    # make near-to-far monitors
    n2f_angle_monitor, n2f_cart_monitor, n2f_ksp_monitor, exact_cart_monitor = make_proj_monitors(
        center, size, [f0]
    )

    far_fields_angular = proj.project_fields(n2f_angle_monitor)
    far_fields_cartesian = proj.project_fields(n2f_cart_monitor)
    far_fields_kspace = proj.project_fields(n2f_ksp_monitor)
    exact_fields_cartesian = proj.project_fields(exact_cart_monitor)

    # compute far field quantities
    far_fields_angular.r
    far_fields_angular.theta
    far_fields_angular.phi
    far_fields_angular.fields_spherical
    far_fields_angular.fields_cartesian
    far_fields_angular.radar_cross_section
    far_fields_angular.power
    for key, val in far_fields_angular.field_components.items():
        val.sel(f=f0)
    far_fields_angular.renormalize_fields(proj_distance=5e6)

    far_fields_cartesian.x
    far_fields_cartesian.y
    far_fields_cartesian.z
    far_fields_cartesian.fields_spherical
    far_fields_cartesian.fields_cartesian
    far_fields_cartesian.radar_cross_section
    far_fields_cartesian.power
    for key, val in far_fields_cartesian.field_components.items():
        val.sel(f=f0)
    far_fields_cartesian.renormalize_fields(proj_distance=5e6)

    far_fields_kspace.ux
    far_fields_kspace.uy
    far_fields_kspace.r
    far_fields_kspace.fields_spherical
    far_fields_kspace.fields_cartesian
    far_fields_kspace.radar_cross_section
    far_fields_kspace.power
    for key, val in far_fields_kspace.field_components.items():
        val.sel(f=f0)
    far_fields_kspace.renormalize_fields(proj_distance=5e6)

    exact_fields_cartesian.x
    exact_fields_cartesian.y
    exact_fields_cartesian.z
    exact_fields_cartesian.fields_spherical
    exact_fields_cartesian.fields_cartesian
    exact_fields_cartesian.radar_cross_section
    exact_fields_cartesian.power
    for key, val in exact_fields_cartesian.field_components.items():
        val.sel(f=f0)
    with pytest.raises(DataError):
        exact_fields_cartesian.renormalize_fields(proj_distance=5e6)
