"""Test near-to-far field transformations.
"""
import numpy as np
import tidy3d as td
import pytest

from tidy3d.log import SetupError

# Settings

MEDIUM = td.Medium(permittivity=3)
WAVELENGTH = 1
F0 = td.C_0 / WAVELENGTH / np.sqrt(MEDIUM.permittivity)
R_FAR = 50 * WAVELENGTH
MAKE_PLOTS = False


def make_n2f_monitors(center, size, freqs):
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

    n2f_angle_monitor = td.Near2FarAngleMonitor(
        center=center,
        size=size,
        freqs=freqs,
        name="n2f_angle",
        custom_origin=center,
        phi=list(phis),
        theta=list(thetas),
        medium=MEDIUM,
        normal_dir="+",
        exclude_surfaces=exclude_surfaces,
    )

    plane_axis = 0
    n2f_cart_monitor = td.Near2FarCartesianMonitor(
        center=center,
        size=size,
        freqs=freqs,
        name="n2f_cart",
        custom_origin=center,
        x=list(xs),
        y=list(ys),
        plane_axis=plane_axis,
        plane_distance=z,
        medium=MEDIUM,
        normal_dir="+",
        exclude_surfaces=exclude_surfaces,
    )

    u_axis = 0
    n2f_ksp_monitor = td.Near2FarKSpaceMonitor(
        center=center,
        size=size,
        freqs=freqs,
        name="n2f_ksp",
        custom_origin=center,
        ux=list(uxs),
        uy=list(uys),
        u_axis=u_axis,
        medium=MEDIUM,
        normal_dir="+",
        exclude_surfaces=exclude_surfaces,
    )
    return n2f_angle_monitor, n2f_cart_monitor, n2f_ksp_monitor


def test_n2f_monitors():
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
    n2f_monitors = make_n2f_monitors(dipole_center, mon_size, freqs)

    near_monitors = td.FieldMonitor.surfaces(
        center=dipole_center, size=mon_size, freqs=freqs, name="near"
    )

    all_monitors = near_monitors + list(n2f_monitors)

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

    # Make sure server-side n2f monitors raise an error in the presence of symmetry
    with pytest.raises(SetupError):
        sim = td.Simulation(
            size=sim_size,
            grid_spec=grid_spec,
            structures=[],
            sources=[source],
            monitors=all_monitors,
            run_time=run_time,
            boundary_spec=boundary_spec,
            medium=MEDIUM,
            symmetry=[1, 0, 0],
        )


def test_n2f_data():
    """Make sure all the near-to-far data structures can be created."""

    f = np.linspace(1e14, 2e14, 10)
    theta = np.linspace(0, np.pi, 10)
    phi = np.linspace(0, 2 * np.pi, 20)
    coords_tp = dict(f=f, theta=theta, phi=phi)
    values_tp = (1 + 1j) * np.random.random((len(theta), len(phi), len(f)))
    scalar_field_tp = td.Near2FarAngleDataArray(values_tp, coords=coords_tp)
    monitor_tp = td.Near2FarAngleMonitor(
        center=(1, 2, 3), size=(2, 2, 2), freqs=f, name="n2f_monitor", phi=phi, theta=theta
    )
    data_tp = td.Near2FarAngleData(
        monitor=monitor_tp,
        Ntheta=scalar_field_tp,
        Nphi=scalar_field_tp,
        Ltheta=scalar_field_tp,
        Lphi=scalar_field_tp,
    )

    x = np.linspace(0, 5, 10)
    y = np.linspace(0, 10, 20)
    coords_xy = dict(f=f, x=x, y=y)
    values_xy = (1 + 1j) * np.random.random((len(x), len(y), len(f)))
    scalar_field_xy = td.Near2FarCartesianDataArray(values_xy, coords=coords_xy)
    monitor_xy = td.Near2FarCartesianMonitor(
        center=(1, 2, 3),
        size=(2, 2, 2),
        freqs=f,
        name="n2f_monitor",
        x=x,
        y=y,
        plane_axis=2,
        plane_distance=50,
    )
    data_xy = td.Near2FarCartesianData(
        monitor=monitor_xy,
        Ntheta=scalar_field_xy,
        Nphi=scalar_field_xy,
        Ltheta=scalar_field_xy,
        Lphi=scalar_field_xy,
    )

    ux = np.linspace(0, 5, 10)
    uy = np.linspace(0, 10, 20)
    coords_u = dict(f=f, ux=ux, uy=uy)
    values_u = (1 + 1j) * np.random.random((len(ux), len(uy), len(f)))
    scalar_field_u = td.Near2FarKSpaceDataArray(values_u, coords=coords_u)
    monitor_u = td.Near2FarKSpaceMonitor(
        center=(1, 2, 3), size=(2, 2, 2), freqs=f, name="n2f_monitor", ux=ux, uy=uy, u_axis=2
    )
    data_u = td.Near2FarKSpaceData(
        monitor=monitor_u,
        Ntheta=scalar_field_u,
        Nphi=scalar_field_u,
        Ltheta=scalar_field_u,
        Lphi=scalar_field_u,
    )


def test_n2f_clientside():
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
    )

    sim_data = td.SimulationData(simulation=sim, monitor_data={"near_field": data})

    n2f = td.RadiationVectors.from_near_field_monitors(
        sim_data=sim_data, near_monitors=[monitor], normal_dirs=["+"]
    )

    # make near-to-far monitors
    n2f_angle_monitor, n2f_cart_monitor, n2f_ksp_monitor = make_n2f_monitors(center, size, [f0])

    rad_vecs_angular = n2f.radiation_vectors(n2f_angle_monitor)
    rad_vecs_cartesian = n2f.radiation_vectors(n2f_cart_monitor)
    rad_vecs_kspace = n2f.radiation_vectors(n2f_ksp_monitor)

    # compute far field quantities
    rad_vecs_angular.fields()
    rad_vecs_angular.fields(r=20)
    rad_vecs_angular.radar_cross_section()
    rad_vecs_angular.power(r=20)

    rad_vecs_cartesian.fields()
    rad_vecs_cartesian.power()

    rad_vecs_kspace.fields()
    rad_vecs_kspace.power()
