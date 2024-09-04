"""Test near field to far field transformations."""

import numpy as np
import pytest
import tidy3d as td
from tidy3d.components.field_projection import FieldProjector
from tidy3d.exceptions import DataError

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

    downsampled_cart_monitor = td.FieldProjectionCartesianMonitor(
        center=center,
        size=size,
        freqs=freqs,
        name="downsampled_cart",
        custom_origin=center,
        x=list(xs),
        y=list(ys),
        proj_axis=proj_axis,
        proj_distance=z,
        normal_dir="+",
        exclude_surfaces=exclude_surfaces,
        interval_space=(1, 2, 3),
    )

    return (
        n2f_angle_monitor,
        n2f_cart_monitor,
        n2f_ksp_monitor,
        exact_cart_monitor,
        downsampled_cart_monitor,
    )


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

    _ = td.Simulation(
        size=sim_size,
        grid_spec=grid_spec,
        structures=[],
        sources=[source],
        monitors=all_monitors,
        run_time=run_time,
        boundary_spec=boundary_spec,
        medium=MEDIUM,
    )


def test_proj_data(tmp_path):
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
    sim_data.to_file(str(tmp_path / "sim_data_n2f.hdf5"))
    sim_data = td.SimulationData.from_file(str(tmp_path / "sim_data_n2f.hdf5"))

    x = np.linspace(0, 5, 10)
    y = np.linspace(0, 10, 20)
    z = np.atleast_1d(5)
    coords_xy = dict(x=x, y=y, z=z, f=f)
    values_xy = (1 + 1j) * np.random.random((len(x), len(y), len(z), len(f)))
    scalar_field_xy = td.FieldProjectionCartesianDataArray(values_xy, coords=coords_xy)
    _ = td.FieldProjectionCartesianMonitor(
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
    _ = td.FieldProjectionCartesianData(
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
    f0 = 1e13
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
        grid_expanded=sim.discretize_monitor(monitor),
    )

    sim_data = td.SimulationData(simulation=sim, data=(data,))

    proj = td.FieldProjector.from_near_field_monitors(
        sim_data=sim_data, near_monitors=[monitor], normal_dirs=["+"]
    )

    # make near-to-far monitors
    (
        n2f_angle_monitor,
        n2f_cart_monitor,
        n2f_ksp_monitor,
        exact_cart_monitor,
        _,
    ) = make_proj_monitors(center, size, [f0])

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
    for val in far_fields_angular.field_components.values():
        val.sel(f=f0)
    far_fields_angular.renormalize_fields(proj_distance=5e6)

    far_fields_cartesian.x
    far_fields_cartesian.y
    far_fields_cartesian.z
    far_fields_cartesian.fields_spherical
    far_fields_cartesian.fields_cartesian
    far_fields_cartesian.radar_cross_section
    far_fields_cartesian.power
    far_fields_cartesian.poynting
    far_fields_cartesian.flux
    for val in far_fields_cartesian.field_components.values():
        val.sel(f=f0)
    far_fields_cartesian.renormalize_fields(proj_distance=5e6)

    far_fields_kspace.ux
    far_fields_kspace.uy
    far_fields_kspace.r
    far_fields_kspace.fields_spherical
    far_fields_kspace.fields_cartesian
    far_fields_kspace.radar_cross_section
    far_fields_kspace.power
    for val in far_fields_kspace.field_components.values():
        val.sel(f=f0)
    far_fields_kspace.renormalize_fields(proj_distance=5e6)

    exact_fields_cartesian.x
    exact_fields_cartesian.y
    exact_fields_cartesian.z
    exact_fields_cartesian.fields_spherical
    exact_fields_cartesian.fields_cartesian
    exact_fields_cartesian.radar_cross_section
    exact_fields_cartesian.power
    exact_fields_cartesian.poynting
    exact_fields_cartesian.flux
    for val in exact_fields_cartesian.field_components.values():
        val.sel(f=f0)
    with pytest.raises(DataError):
        exact_fields_cartesian.renormalize_fields(proj_distance=5e6)


def make_2d_proj_monitors(center, size, freqs, plane):
    """Helper function to make near-to-far monitors in 2D simulations."""

    if plane == "xy":
        thetas = [np.pi / 2]
        phis = np.linspace(0, 2 * np.pi, 100)
        far_size = 10 * WAVELENGTH
        Ns = 40
        xs = np.linspace(-far_size, far_size, Ns)
        ys = [0]
        kx = np.linspace(-0.7, 0.7, Ns)
        ky = [0]
        projection_axis = 0
    elif plane == "yz":
        thetas = np.linspace(0, np.pi, 1)
        phis = [np.pi / 2]
        far_size = 10 * WAVELENGTH
        Ns = 40
        xs = [0]
        ys = np.linspace(-far_size, far_size, Ns)
        kx = [0]
        ky = np.linspace(-0.7, 0.7, Ns)
        projection_axis = 1
    elif plane == "xz":
        thetas = np.linspace(0, np.pi, 100)
        phis = [0]
        far_size = 10 * WAVELENGTH
        Ns = 40
        xs = [0]
        ys = np.linspace(-far_size, far_size, Ns)
        kx = [0]
        ky = np.linspace(-0.7, 0.7, Ns)
        projection_axis = 0
    else:
        raise ValueError("Invalid plane. Use 'xy', 'yz', or 'xz'.")

    n2f_angle_monitor_2d = td.FieldProjectionAngleMonitor(
        center=center,
        size=size,
        freqs=freqs,
        name="far_field_angle",
        phi=list(phis),
        theta=list(thetas),
        proj_distance=R_FAR,
        far_field_approx=True,  # Fields are far enough for geometric far field approximations
    )

    n2f_car_monitor_2d = td.FieldProjectionCartesianMonitor(
        center=center,
        size=size,
        freqs=freqs,
        name="far_field_cartesian",
        x=list(xs),
        y=list(ys),
        proj_axis=projection_axis,
        proj_distance=R_FAR,
        far_field_approx=True,  # Fields are far enough for geometric far field approximations
    )

    n2f_k_monitor_2d = td.FieldProjectionKSpaceMonitor(
        center=center,
        size=size,
        freqs=freqs,
        name="far_field_kspace",
        ux=list(kx),
        uy=list(ky),
        proj_axis=projection_axis,
        proj_distance=R_FAR,
        far_field_approx=True,  # Fields are far enough for geometric far field approximations
    )

    return (n2f_angle_monitor_2d, n2f_car_monitor_2d, n2f_k_monitor_2d)


def make_2d_proj(plane):
    center = (0, 0, 0)
    f0 = 1e13

    if plane == "xy":
        sim_size = (5, 5, 0)
        monitor_size = (0, 2, td.inf)
        # boundary conditions
        boundary_conds = td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml(),
            z=td.Boundary.periodic(),
        )
        # data coordinates
        x = np.array([0.0])
        y = np.linspace(-1, 1, 10)
        z = np.array([0.0])
        coords = dict(x=x, y=y, z=z, f=[f0])
        scalar_field = td.ScalarFieldDataArray(
            (1 + 1j) * np.random.random((1, 10, 1, 1)), coords=coords
        )
    elif plane == "yz":
        sim_size = (0, 5, 5)
        monitor_size = (td.inf, 0, 2)
        # boundary conditions
        boundary_conds = td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.pml(),
            z=td.Boundary.pml(),
        )
        # data coordinates
        x = np.array([0.0])
        y = np.array([0.0])
        z = np.linspace(-1, 1, 10)
        coords = dict(x=x, y=y, z=z, f=[f0])
        scalar_field = td.ScalarFieldDataArray(
            (1 + 1j) * np.random.random((1, 1, 10, 1)), coords=coords
        )
    elif plane == "xz":
        sim_size = (5, 0, 5)
        monitor_size = (0, td.inf, 2)
        # boundary conditions
        boundary_conds = td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.periodic(),
            z=td.Boundary.pml(),
        )
        # data coordinates
        x = np.array([0.0])
        y = np.array([0.0])
        z = np.linspace(-1, 1, 10)
        coords = dict(x=x, y=y, z=z, f=[f0])
        scalar_field = td.ScalarFieldDataArray(
            (1 + 1j) * np.random.random((1, 1, 10, 1)), coords=coords
        )
    else:
        raise ValueError("Invalid plane. Use 'xy', 'yz', or 'xz'.")

    monitor = td.FieldMonitor(
        center=center, size=monitor_size, freqs=[f0], name="near_field", colocate=False
    )

    # Set up the simulation
    sim = td.Simulation(
        size=sim_size,
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / f0),
        boundary_spec=boundary_conds,
        monitors=[monitor],
        run_time=1e-12,
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
        grid_expanded=sim.discretize_monitor(monitor),
    )

    sim_data = td.SimulationData(simulation=sim, data=(data,))

    proj = td.FieldProjector.from_near_field_monitors(
        sim_data=sim_data,
        near_monitors=[monitor],
        normal_dirs=["+"],
    )

    # make near-to-far monitors
    (
        n2f_angle_monitor_2d,
        n2f_cart_monitor_2d,
        n2f_kspace_monitor_2d,
    ) = make_2d_proj_monitors(center, monitor_size, [f0], plane)

    far_fields_angular_2d = proj.project_fields(n2f_angle_monitor_2d)
    far_fields_cartesian_2d = proj.project_fields(n2f_cart_monitor_2d)
    far_fields_kspace_2d = proj.project_fields(n2f_kspace_monitor_2d)

    # compute far field quantities
    far_fields_angular_2d.r
    far_fields_angular_2d.theta
    far_fields_angular_2d.phi
    far_fields_angular_2d.fields_spherical
    far_fields_angular_2d.fields_cartesian
    far_fields_angular_2d.radar_cross_section
    far_fields_angular_2d.power
    for val in far_fields_angular_2d.field_components.values():
        val.sel(f=f0)
    far_fields_angular_2d.renormalize_fields(proj_distance=5e6)

    far_fields_cartesian_2d.x
    far_fields_cartesian_2d.y
    far_fields_cartesian_2d.z
    far_fields_cartesian_2d.fields_spherical
    far_fields_cartesian_2d.fields_cartesian
    far_fields_cartesian_2d.radar_cross_section
    far_fields_cartesian_2d.power
    for val in far_fields_cartesian_2d.field_components.values():
        val.sel(f=f0)
    far_fields_cartesian_2d.renormalize_fields(proj_distance=5e6)

    far_fields_kspace_2d.ux
    far_fields_kspace_2d.uy
    far_fields_kspace_2d.r
    far_fields_kspace_2d.fields_spherical
    far_fields_kspace_2d.fields_cartesian
    far_fields_kspace_2d.radar_cross_section
    far_fields_kspace_2d.power
    for val in far_fields_kspace_2d.field_components.values():
        val.sel(f=f0)
    far_fields_kspace_2d.renormalize_fields(proj_distance=5e6)


def test_2d_proj_clientside():
    # Run simulations and tests for all three planes
    planes = ["xy", "yz", "xz"]

    for plane in planes:
        make_2d_proj(plane)


@pytest.mark.parametrize(
    "array, pts, axes, expected",
    [
        # 1D array, integrate over axis 0
        (np.array([1, 2, 3]), np.array([0, 1, 2]), 0, 4.0),
        # 2D array, integrate over axis 0
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([0, 1]), 0, np.array([2.5, 3.5, 4.5])),
        # 2D array, integrate over axis 1
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1]), 1, np.array([1.5, 3.5, 5.5])),
        # 3D array, integrate over axes 0 and 1
        (np.ones((2, 2, 2)), [np.array([0, 1]), np.array([0, 1])], [0, 1], np.array([1.0, 1.0])),
        # one element along integration axis but two points in pts
        (np.array([[1, 1], [2, 2], [3, 3]]), np.array([0, 1]), 1, np.array([1.0, 2.0, 3.0])),
        # 2D array of shape (1, 3), integrate over both axes
        (np.array([[1, 2, 3]]), [np.array([0]), np.array([0, 1, 2])], [0, 1], 4.0),
    ],
)
def test_trapezoid(array, pts, axes, expected):
    result = FieldProjector.trapezoid(array, pts, axes)
    assert np.allclose(result, expected)
