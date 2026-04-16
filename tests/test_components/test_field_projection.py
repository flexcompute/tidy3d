"""Test near field to far field transformations."""

from __future__ import annotations

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import make_vjp
from pydantic import ValidationError

import tidy3d as td
import tidy3d.components.field_projection.common as field_projection_common
import tidy3d.components.field_projection.exact as field_projection_exact
from tidy3d.components.field_projection import FieldProjector
from tidy3d.components.field_projection.common import (
    _far_field_integral,
    _far_field_integral_pairs,
    _FarFieldIntegralSpec,
    _trapz_weights_1d,
)
from tidy3d.exceptions import DataError, SetupError

from ..utils import run_emulated

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
    coords_tp = {"r": r, "theta": theta, "phi": phi, "f": f}
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
    coords_xy = {"x": x, "y": y, "z": z, "f": f}
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
    coords_u = {"ux": ux, "uy": uy, "r": r, "f": f}
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
    coords_xy = {"x": x, "y": y, "z": z, "f": f}
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


def make_clientside_projection_inputs(center, size, freqs, num_points=10, seed: int | None = None):
    """Build the synthetic near-field setup used by client-side projection tests."""
    freqs = np.atleast_1d(freqs)
    monitor = td.FieldMonitor(size=size, center=center, freqs=list(freqs), name="near_field")

    sim_size = (5, 5, 5)
    sim = td.Simulation(
        size=sim_size,
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / freqs[0]),
        monitors=(monitor,),
        run_time=1e-12,
    )

    x = np.linspace(-1, 1, num_points)
    y = np.linspace(-1, 1, num_points)
    z = np.array([0.0])
    coords = {"x": x, "y": y, "z": z, "f": freqs}
    rng = np.random.default_rng(seed) if seed is not None else None
    values = (
        (1 + 1j) * rng.random((num_points, num_points, 1, len(freqs)))
        if rng is not None
        else (1 + 1j) * np.random.random((num_points, num_points, 1, len(freqs)))
    )
    scalar_field = td.ScalarFieldDataArray(values, coords=coords)
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
    return sim_data, monitor, data


def make_clientside_projector(center, size, freqs, num_points=10, seed: int | None = None):
    """Helper to build a client-side field projector from the shared synthetic setup."""

    sim_data, monitor, _ = make_clientside_projection_inputs(center, size, freqs, num_points, seed)
    return td.FieldProjector.from_near_field_monitors(
        sim_data=sim_data,
        near_monitors=[monitor],
        normal_dirs=["+"],
    )


def make_single_point_cart_monitor(center, size, f0, name):
    """Helper to make a one-point Cartesian projection monitor."""

    return td.FieldProjectionCartesianMonitor(
        center=center,
        size=size,
        freqs=[f0],
        name=name,
        custom_origin=center,
        x=[0.0],
        y=[0.0],
        proj_axis=0,
        proj_distance=R_FAR,
        normal_dir="+",
    )


def test_proj_clientside():
    """Make sure the client-side near-to-far class can be created."""

    center = (0, 0, 0)
    size = (2, 2, 0)
    f0 = 1e13
    proj = make_clientside_projector(center, size, f0)

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


def test_proj_clientside_from_near_field_data():
    center = (0, 0, 0)
    size = (2, 2, 0)
    f0 = 1e13

    sim_data, _, field_data = make_clientside_projection_inputs(center, size, f0, seed=0)
    phase = np.exp(1j * np.pi / 7)
    modified_data = field_data.copy(
        update={
            "Ex": 2.0 * field_data.Ex,
            "Hy": phase * field_data.Hy,
            "grid_primal_correction": 2.0,
            "grid_dual_correction": 3.0,
        }
    )

    projector_manual = td.FieldProjector.from_near_field_monitors(
        sim_data=td.SimulationData(simulation=sim_data.simulation, data=(modified_data,)),
        near_monitors=[modified_data.monitor],
        normal_dirs=["+"],
    )

    projector_from_data, proj_monitor = td.FieldProjector.from_near_field_data(
        near_field_data=modified_data,
        medium=projector_manual.medium,
        name="n2f_cart_direct",
        x=[0.0],
        y=[0.0],
        proj_axis=0,
        proj_distance=R_FAR,
        normal_dir="+",
    )

    projected_from_data = projector_from_data.project_fields(proj_monitor)
    projected_manual = projector_manual.project_fields(
        make_single_point_cart_monitor(center, size, f0, "n2f_cart_manual")
    )

    for field_name, field_data_from_data in projected_from_data.field_components.items():
        np.testing.assert_allclose(
            field_data_from_data.values,
            projected_manual.field_components[field_name].values,
        )


def test_proj_clientside_from_custom_near_field_data():
    center = (0, 0, 0)
    size = (2, 2, 0)
    f0 = 1e13
    coords = {
        "x": np.linspace(-1, 1, 10),
        "y": np.linspace(-1, 1, 10),
        "z": np.array([0.0]),
        "f": [f0],
    }
    scalar_field = td.ScalarFieldDataArray(np.ones((10, 10, 1, 1), dtype=complex), coords=coords)
    near_field_data = td.FieldData(
        monitor=td.FieldMonitor(size=size, center=center, freqs=[f0], name="raw_near_field"),
        Ex=scalar_field,
        Ey=scalar_field,
        Ez=scalar_field,
        Hx=scalar_field,
        Hy=scalar_field,
        Hz=scalar_field,
    )

    projector, proj_monitor = td.FieldProjector.from_near_field_data(
        near_field_data=near_field_data,
        medium=td.Medium(),
        name="raw_proj",
        x=[0.0],
        y=[0.0],
        proj_axis=0,
        proj_distance=R_FAR,
        normal_dir="+",
    )

    projected = projector.project_fields(proj_monitor)
    for field in projected.field_components.values():
        field.sel(f=f0)


def test_proj_clientside_from_near_field_data_without_resampling_keeps_finite_fields():
    f0 = 1e13
    center = (1.0, 1.5, 0.0)
    size = (2.0, 1.0, 0.0)
    monitor = td.FieldMonitor(size=size, center=center, freqs=[f0], name="raw_near_field_staggered")

    coords_xy0 = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 2.0]),
        "z": np.array([0.0]),
        "f": [f0],
    }
    coords_xy1 = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([1.0, 3.0]),
        "z": np.array([0.0]),
        "f": [f0],
    }
    field_xy0 = td.ScalarFieldDataArray(np.ones((3, 2, 1, 1), dtype=complex), coords=coords_xy0)
    field_xy1 = td.ScalarFieldDataArray(
        2.0 * np.ones((3, 2, 1, 1), dtype=complex), coords=coords_xy1
    )
    near_field_data = td.FieldData(
        monitor=monitor,
        Ex=field_xy1,
        Ey=field_xy0,
        Ez=field_xy0,
        Hx=field_xy0,
        Hy=field_xy1,
        Hz=field_xy0,
    )

    projector, proj_monitor = td.FieldProjector.from_near_field_data(
        near_field_data=near_field_data,
        medium=td.Medium(),
        name="raw_proj_no_resample",
        x=[0.0],
        y=[0.0],
        proj_axis=0,
        proj_distance=R_FAR,
        normal_dir="+",
        pts_per_wavelength=None,
    )

    projected = projector.project_fields(proj_monitor)
    for field_name, field in projected.field_components.items():
        assert np.all(np.isfinite(np.asarray(field.data))), field_name


def test_from_near_field_data_validates_monitor_span():
    center = (0, 0, 0)
    f0 = 1e13
    monitor = td.FieldMonitor(size=(4, 4, 0), center=center, freqs=[f0], name="near_field")
    coords = {
        "x": np.linspace(-1, 1, 5),
        "y": np.linspace(-1, 1, 5),
        "z": np.array([0.0]),
        "f": [f0],
    }
    scalar_field = td.ScalarFieldDataArray(np.ones((5, 5, 1, 1), dtype=complex), coords=coords)
    near_field_data = td.FieldData(
        monitor=monitor,
        Ex=scalar_field,
        Ey=scalar_field,
        Ez=scalar_field,
        Hx=scalar_field,
        Hy=scalar_field,
        Hz=scalar_field,
    )

    with pytest.raises(SetupError, match="does not cover the full monitor span"):
        td.FieldProjector.from_near_field_data(
            near_field_data=near_field_data,
            medium=td.Medium(),
            name="invalid_proj",
            x=[0.0],
            y=[0.0],
            proj_axis=0,
            proj_distance=R_FAR,
            normal_dir="+",
        )


def test_from_near_field_data_validates_component_span_intersection():
    center = (0, 0, 0)
    f0 = 1e13
    monitor = td.FieldMonitor(size=(4, 4, 0), center=center, freqs=[f0], name="near_field")
    full_coords = {
        "x": np.linspace(-2, 2, 5),
        "y": np.linspace(-2, 2, 5),
        "z": np.array([0.0]),
        "f": [f0],
    }
    narrow_x_coords = {
        "x": np.linspace(-1, 1, 3),
        "y": np.linspace(-2, 2, 5),
        "z": np.array([0.0]),
        "f": [f0],
    }
    scalar_field_full = td.ScalarFieldDataArray(
        np.ones((5, 5, 1, 1), dtype=complex), coords=full_coords
    )
    scalar_field_narrow_x = td.ScalarFieldDataArray(
        np.ones((3, 5, 1, 1), dtype=complex), coords=narrow_x_coords
    )
    near_field_data = td.FieldData(
        monitor=monitor,
        Ex=scalar_field_full,
        Ey=scalar_field_full,
        Ez=scalar_field_full,
        Hx=scalar_field_full,
        Hy=scalar_field_narrow_x,
        Hz=scalar_field_full,
    )

    with pytest.raises(SetupError, match="does not cover the full monitor span"):
        td.FieldProjector.from_near_field_data(
            near_field_data=near_field_data,
            medium=td.Medium(),
            name="invalid_proj_intersection",
            x=[0.0],
            y=[0.0],
            proj_axis=0,
            proj_distance=R_FAR,
            normal_dir="+",
        )


def test_from_near_field_data_validates_single_source_plane():
    center = (0, 0, 0)
    f0 = 1e13
    monitor = td.FieldMonitor(size=(4, 4, 0), center=center, freqs=[f0], name="near_field")
    coords = {
        "x": np.linspace(-2, 2, 5),
        "y": np.linspace(-2, 2, 5),
        "z": np.array([-0.1, 0.1]),
        "f": [f0],
    }
    scalar_field = td.ScalarFieldDataArray(np.ones((5, 5, 2, 1), dtype=complex), coords=coords)
    near_field_data = td.FieldData(
        monitor=monitor,
        Ex=scalar_field,
        Ey=scalar_field,
        Ez=scalar_field,
        Hx=scalar_field,
        Hy=scalar_field,
        Hz=scalar_field,
    )

    with pytest.raises(SetupError, match="single monitor plane"):
        td.FieldProjector.from_near_field_data(
            near_field_data=near_field_data,
            medium=td.Medium(),
            name="invalid_proj_plane",
            x=[0.0],
            y=[0.0],
            proj_axis=0,
            proj_distance=R_FAR,
            normal_dir="+",
        )


def test_field_projector_direct_construction_requires_sim_data():
    _, monitor, _ = make_clientside_projection_inputs((0, 0, 0), (2, 2, 0), 1e13)
    surface = td.FieldProjectionSurface(monitor=monitor, normal_dir="+")

    with pytest.raises(ValidationError, match="Field required"):
        td.FieldProjector(surfaces=(surface,))


def test_from_near_field_data_uses_standard_field_validation():
    center = (0, 0, 0)
    size = (2, 2, 0)
    f0 = 1e13
    _, _, field_data = make_clientside_projection_inputs(center, size, f0, seed=1)

    with pytest.raises(ValidationError, match="valid integer"):
        td.FieldProjector.from_near_field_data(
            near_field_data=field_data,
            medium=td.Medium(),
            name="n2f_cart_invalid_ppw",
            x=[0.0],
            y=[0.0],
            proj_axis=0,
            proj_distance=R_FAR,
            normal_dir="+",
            pts_per_wavelength="bad",
        )


def test_from_near_field_data_with_direct_medium_matches_2d_monitor_path():
    plane = "xz"
    projector_manual, center, monitor_size, f0 = make_2d_projector(plane)
    field_data = projector_manual.sim_data.monitor_data["near_field"]
    proj_monitor_manual = make_2d_proj_monitors(center, monitor_size, [f0], plane)[1]

    projector_from_data, proj_monitor = td.FieldProjector.from_near_field_data(
        near_field_data=field_data,
        medium=projector_manual.medium,
        name="n2f_cart_direct_2d_structures",
        x=list(proj_monitor_manual.x),
        y=list(proj_monitor_manual.y),
        proj_axis=proj_monitor_manual.proj_axis,
        proj_distance=proj_monitor_manual.proj_distance,
        normal_dir="+",
        dimensionality="auto",
    )

    projected_from_data = projector_from_data.project_fields(proj_monitor)
    projected_manual = projector_manual.project_fields(proj_monitor_manual)

    for field_name, field_data_from_data in projected_from_data.field_components.items():
        np.testing.assert_allclose(
            field_data_from_data.values,
            projected_manual.field_components[field_name].values,
        )


def test_proj_clientside_from_near_field_data_auto_infers_2d():
    plane = "xz"
    projector_manual, center, monitor_size, f0 = make_2d_projector(plane)
    field_data = projector_manual.sim_data.monitor_data["near_field"]
    field_data = field_data.copy(
        update={"grid_primal_correction": 2.0, "grid_dual_correction": 3.0}
    )
    projector_manual = td.FieldProjector.from_near_field_monitors(
        sim_data=td.SimulationData(
            simulation=projector_manual.sim_data.simulation, data=(field_data,)
        ),
        near_monitors=[field_data.monitor],
        normal_dirs=["+"],
    )
    proj_monitor_manual = make_2d_proj_monitors(center, monitor_size, [f0], plane)[1]

    projector_from_data, proj_monitor = td.FieldProjector.from_near_field_data(
        near_field_data=field_data,
        medium=projector_manual.medium,
        name="n2f_cart_direct_2d",
        x=list(proj_monitor_manual.x),
        y=list(proj_monitor_manual.y),
        proj_axis=proj_monitor_manual.proj_axis,
        proj_distance=proj_monitor_manual.proj_distance,
        normal_dir="+",
        dimensionality="auto",
    )

    projected_from_data = projector_from_data.project_fields(proj_monitor)
    projected_manual = projector_manual.project_fields(proj_monitor_manual)

    for field_name, field_data_from_data in projected_from_data.field_components.items():
        np.testing.assert_allclose(
            field_data_from_data.values,
            projected_manual.field_components[field_name].values,
        )


def test_from_near_field_data_rejects_line_like_source_as_3d():
    plane = "xz"
    projector_manual, _, _, _ = make_2d_projector(plane)
    field_data = projector_manual.sim_data.monitor_data["near_field"]

    with pytest.raises(SetupError, match='dimensionality="3D"'):
        td.FieldProjector.from_near_field_data(
            near_field_data=field_data,
            medium=projector_manual.medium,
            name="invalid_proj_dimensionality",
            x=[0.0],
            y=[0.0],
            proj_axis=0,
            proj_distance=R_FAR,
            normal_dir="+",
            dimensionality="3D",
        )


def test_proj_clientside_freq_chunk_size_matches_default():
    """Approximate multi-frequency client-side projections should be chunk-size invariant."""

    center = (0, 0, 0)
    size = (2, 2, 0)
    freqs = [0.9e13, 1e13, 1.1e13]
    proj = make_clientside_projector(center, size, freqs, seed=0)

    _, n2f_cart_monitor, n2f_ksp_monitor, _, _ = make_proj_monitors(center, size, freqs)

    far_fields_cartesian_default = proj.project_fields(n2f_cart_monitor, verbose=False)
    far_fields_cartesian_chunked = proj.project_fields(
        n2f_cart_monitor, verbose=False, freq_chunk_size=1
    )
    far_fields_kspace_default = proj.project_fields(n2f_ksp_monitor, verbose=False)
    far_fields_kspace_chunked = proj.project_fields(
        n2f_ksp_monitor, verbose=False, freq_chunk_size=1
    )

    for name in far_fields_cartesian_default.field_components:
        np.testing.assert_allclose(
            np.asarray(getattr(far_fields_cartesian_default, name).data),
            np.asarray(getattr(far_fields_cartesian_chunked, name).data),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(getattr(far_fields_kspace_default, name).data),
            np.asarray(getattr(far_fields_kspace_chunked, name).data),
            rtol=1e-12,
            atol=1e-12,
        )


def test_proj_clientside_exact_freq_chunk_size_matches_default():
    """Exact multi-frequency client-side projections should be chunk-size invariant."""

    center = (0, 0, 0)
    size = (2, 2, 0)
    freqs = np.linspace(0.9e13, 1.1e13, 9)
    proj = make_clientside_projector(center, size, freqs, num_points=4, seed=1)

    n2f_angle_monitor, _, n2f_ksp_monitor, exact_cart_monitor, _ = make_proj_monitors(
        center, size, freqs
    )
    exact_angle_monitor = n2f_angle_monitor.updated_copy(
        name="exact_angle_chunked",
        far_field_approx=False,
        theta=[float(n2f_angle_monitor.theta[0])],
        phi=[float(n2f_angle_monitor.phi[0])],
    )
    exact_cart_monitor = exact_cart_monitor.updated_copy(
        x=[float(exact_cart_monitor.x[0])],
        y=[float(exact_cart_monitor.y[0])],
    )
    exact_kspace_monitor = n2f_ksp_monitor.updated_copy(
        name="exact_ksp_chunked",
        far_field_approx=False,
        ux=[float(n2f_ksp_monitor.ux[0])],
        uy=[float(n2f_ksp_monitor.uy[0])],
    )

    exact_angle_default = proj.project_fields(exact_angle_monitor, verbose=False)
    exact_angle_chunked = proj.project_fields(exact_angle_monitor, verbose=False, freq_chunk_size=1)
    exact_cartesian_default = proj.project_fields(exact_cart_monitor, verbose=False)
    exact_cartesian_chunked = proj.project_fields(
        exact_cart_monitor, verbose=False, freq_chunk_size=1
    )
    exact_kspace_default = proj.project_fields(exact_kspace_monitor, verbose=False)
    exact_kspace_chunked = proj.project_fields(
        exact_kspace_monitor, verbose=False, freq_chunk_size=1
    )

    for name in exact_angle_default.field_components:
        np.testing.assert_allclose(
            np.asarray(getattr(exact_angle_default, name).data),
            np.asarray(getattr(exact_angle_chunked, name).data),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(getattr(exact_cartesian_default, name).data),
            np.asarray(getattr(exact_cartesian_chunked, name).data),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(getattr(exact_kspace_default, name).data),
            np.asarray(getattr(exact_kspace_chunked, name).data),
            rtol=1e-12,
            atol=1e-12,
        )


def test_proj_clientside_freq_chunk_size_validation():
    """Public projection entry point should reject nonpositive frequency chunk sizes."""

    center = (0, 0, 0)
    size = (2, 2, 0)
    f0 = 1e13
    projector = make_clientside_projector(center, size, f0, num_points=4)
    proj_monitor = make_single_point_cart_monitor(center, size, f0, name="n2f_cart_chunk_size")

    with pytest.raises(ValueError, match="freq_chunk_size >= 1"):
        projector.project_fields(proj_monitor, verbose=False, freq_chunk_size=0)


def test_resample_surface_currents_clips_to_data_bounds():
    """Surface-current colocation should stay within the available field-data bounds."""

    center = (0, 0, 0)
    size = (4, 4, 0)
    projector = make_clientside_projector(center, size, F0, num_points=4)
    surface = projector.surfaces[0]
    field_data = projector.sim_data.monitor_data[surface.monitor.name].symmetry_expanded.copy(
        update={"grid_expanded": None}
    )
    currents = FieldProjector._fields_to_currents(field_data, surface)

    resampled = FieldProjector._resample_surface_currents(
        currents,
        field_data,
        surface,
        projector.medium,
        projector.pts_per_wavelength,
    )

    for field_name, current_component in resampled.data_vars.items():
        assert np.all(np.isfinite(np.asarray(current_component.data))), field_name

    for coord_name in ("x", "y"):
        mins = [
            np.min(np.asarray(current_component.coords[coord_name].values))
            for current_component in currents.values()
        ]
        maxs = [
            np.max(np.asarray(current_component.coords[coord_name].values))
            for current_component in currents.values()
        ]
        coord_values = np.asarray(resampled.coords[coord_name].values)
        assert coord_values[0] >= max(mins)
        assert coord_values[-1] <= min(maxs)


def test_compute_surface_currents_applies_grid_correction():
    center = (0, 0, 0)
    size = (2, 2, 0)
    f0 = 1e13
    coords = {
        "x": np.linspace(-1, 1, 5),
        "y": np.linspace(-1, 1, 5),
        "z": np.array([0.0]),
        "f": [f0],
    }
    scalar_field = td.ScalarFieldDataArray(np.ones((5, 5, 1, 1), dtype=complex), coords=coords)
    near_field_data = td.FieldData(
        monitor=td.FieldMonitor(size=size, center=center, freqs=[f0], name="grid_corrected_field"),
        Ex=scalar_field,
        Ey=2.0 * scalar_field,
        Ez=scalar_field,
        Hx=3.0 * scalar_field,
        Hy=5.0 * scalar_field,
        Hz=scalar_field,
        grid_primal_correction=2.0,
        grid_dual_correction=3.0,
    )
    surface = td.FieldProjectionSurface(monitor=near_field_data.monitor, normal_dir="+")

    corrected_field_data = near_field_data.grid_corrected_copy
    expected_currents = FieldProjector._resample_surface_currents(
        FieldProjector._fields_to_currents(corrected_field_data, surface),
        corrected_field_data,
        surface,
        td.Medium(),
        None,
    )
    raw_currents = FieldProjector._resample_surface_currents(
        FieldProjector._fields_to_currents(near_field_data, surface),
        near_field_data,
        surface,
        td.Medium(),
        None,
    )
    actual_currents = FieldProjector.compute_surface_currents(
        near_field_data, surface, td.Medium(), None
    )

    for field_name in expected_currents.data_vars:
        np.testing.assert_allclose(
            np.asarray(actual_currents[field_name].data),
            np.asarray(expected_currents[field_name].data),
        )

    assert any(
        not np.allclose(
            np.asarray(actual_currents[field_name].data),
            np.asarray(raw_currents[field_name].data),
        )
        for field_name in actual_currents.data_vars
    )


def test_proj_clientside_verbose_flag(monkeypatch):
    """Make sure local projection progress output can be disabled."""

    center = (0, 0, 0)
    size = (2, 2, 0)
    f0 = 1e13
    projector = make_clientside_projector(center, size, f0, num_points=4)
    proj_monitor = make_single_point_cart_monitor(center, size, f0, name="n2f_cart_quiet")

    track_calls = []

    def fake_track(iterable, *args, **kwargs):
        track_calls.append(kwargs)
        return iterable

    monkeypatch.setattr(field_projection_common, "track", fake_track)

    projector.project_fields(proj_monitor)
    assert len(track_calls) == 1
    assert track_calls[0]["description"] == "Computing projected fields"
    assert track_calls[0]["total"] == 1

    projected_fields = projector.project_fields(proj_monitor, verbose=False)
    assert len(track_calls) == 1
    assert projected_fields.monitor.name == proj_monitor.name


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


def make_2d_projector(plane):
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
        coords = {"x": x, "y": y, "z": z, "f": [f0]}
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
        coords = {"x": x, "y": y, "z": z, "f": [f0]}
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
        coords = {"x": x, "y": y, "z": z, "f": [f0]}
        scalar_field = td.ScalarFieldDataArray(
            (1 + 1j) * np.random.random((1, 1, 10, 1)), coords=coords
        )
    else:
        raise ValueError("Invalid plane. Use 'xy', 'yz', or 'xz'.")

    monitor = td.FieldMonitor(
        center=center, size=monitor_size, freqs=[f0], name="near_field", colocate=False
    )

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

    return proj, center, monitor_size, f0


def make_2d_proj(plane):
    proj, center, monitor_size, f0 = make_2d_projector(plane)

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


def test_proj_clientside_homogeneous_clips_to_sim_bounds_2d():
    f0 = 1e13
    medium_bg = td.Medium(permittivity=2)
    medium_air = td.Medium(permittivity=1)

    def make_sim_data(structures):
        near_monitor = td.FieldMonitor(
            center=(0, 0, 0.5),
            size=(0, 0.1, 1.0),
            freqs=[f0],
            name="near_field",
            colocate=False,
        )
        sim = td.Simulation(
            size=(1, 1, 0),
            medium=medium_bg,
            structures=structures,
            grid_spec=td.GridSpec.auto(wavelength=td.C_0 / f0),
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(),
                y=td.Boundary.pml(),
                z=td.Boundary.periodic(),
            ),
            monitors=[near_monitor],
            run_time=1e-12,
        )

        coords = {
            "x": np.array([0.0]),
            "y": np.linspace(-0.05, 0.05, 5),
            "z": np.array([0.0]),
            "f": [f0],
        }
        scalar_field = td.ScalarFieldDataArray(np.ones((1, 5, 1, 1), dtype=complex), coords=coords)
        data = td.FieldData(
            monitor=near_monitor,
            Ex=scalar_field,
            Ey=scalar_field,
            Ez=scalar_field,
            Hx=scalar_field,
            Hy=scalar_field,
            Hz=scalar_field,
            symmetry=sim.symmetry,
            symmetry_center=sim.center,
            grid_expanded=sim.discretize_monitor(near_monitor),
        )
        return td.SimulationData(simulation=sim, data=(data,)), near_monitor

    box_outside_sim = td.Structure(
        geometry=td.Box(center=(0, 0.025, 1.5), size=(0.2, 0.05, 2.0)),
        medium=medium_air,
    )
    sim_data, near_monitor = make_sim_data((box_outside_sim,))
    projector = td.FieldProjector.from_near_field_monitors(
        sim_data=sim_data,
        near_monitors=[near_monitor],
        normal_dirs=["+"],
    )
    assert projector.medium == medium_bg

    box_in_2d_sim = td.Structure(
        geometry=td.Box(center=(0, 0.025, 0), size=(0.2, 0.05, 2.0)),
        medium=medium_air,
    )
    sim_data, near_monitor = make_sim_data((box_in_2d_sim,))
    with pytest.raises(ValidationError, match="Plane must be homogeneous"):
        td.FieldProjector.from_near_field_monitors(
            sim_data=sim_data,
            near_monitors=[near_monitor],
            normal_dirs=["+"],
        )


def test_2d_proj_clientside_cartesian_single_cell_dimension():
    freq0 = td.C_0 / 1.55
    sio2 = td.Medium(permittivity=1.44**2)
    si = td.Medium(permittivity=3.47**2)

    sim = td.Simulation(
        center=(0, 0, 0),
        size=(20, 0, 5),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=10, wavelength=1.55),
        structures=[
            td.Structure(
                geometry=td.Box.from_bounds((-td.inf, -td.inf, -1), (td.inf, td.inf, 0)),
                medium=si,
            ),
        ],
        sources=[
            td.ModeSource(
                center=(-8, 0, 0),
                size=(0, td.inf, 4),
                source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
                direction="+",
            ),
        ],
        monitors=[
            td.FieldMonitor(
                center=(0, 0, 1),
                size=(10, td.inf, 0),
                freqs=[freq0],
                name="nf",
                colocate=False,
            ),
        ],
        run_time=1e-12,
        medium=sio2,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.periodic(),
            z=td.Boundary.pml(),
        ),
    )

    sim_data = run_emulated(sim)
    near_monitor = sim_data.simulation.get_monitor_by_name("nf")
    projector = td.FieldProjector.from_near_field_monitors(
        sim_data=sim_data,
        near_monitors=[near_monitor],
        normal_dirs=["+"],
    )
    proj_monitor = td.FieldProjectionCartesianMonitor(
        center=near_monitor.center,
        size=near_monitor.size,
        freqs=[freq0],
        name="proj",
        proj_axis=2,
        proj_distance=50,
        x=list(range(-5, 6)),
        y=[0.0],
        far_field_approx=True,
    )

    projected = projector.project_fields(proj_monitor)
    for field in projected.field_components.values():
        field.sel(f=freq0)


@pytest.mark.parametrize("plane", ["xy", "yz", "xz"])
@pytest.mark.parametrize("monitor_index", [0, 1, 2])
def test_2d_proj_clientside_exact_not_supported(plane, monitor_index):
    proj, center, monitor_size, f0 = make_2d_projector(plane)
    monitor = make_2d_proj_monitors(center, monitor_size, [f0], plane)[monitor_index]
    monitor = monitor.updated_copy(proj_distance=R_FAR / 50, far_field_approx=False)

    with pytest.raises(
        SetupError,
        match="Exact far-field projection for 2D simulations is not yet available",
    ):
        proj.project_fields(monitor)


@pytest.mark.parametrize(
    "plane,monitor_index,update",
    [
        ("xy", 0, {"theta": [0]}),
        ("xy", 1, {"y": [1]}),
        ("xy", 2, {"uy": [0.1]}),
        ("yz", 0, {"phi": [0]}),
        ("yz", 1, {"x": [1]}),
        ("yz", 2, {"ux": [0.1]}),
        ("xz", 0, {"phi": [np.pi / 2]}),
        ("xz", 1, {"x": [1]}),
        ("xz", 2, {"ux": [0.1]}),
    ],
)
def test_2d_proj_clientside_invalid_monitor_settings(plane, monitor_index, update):
    proj, center, monitor_size, f0 = make_2d_projector(plane)
    monitor = make_2d_proj_monitors(center, monitor_size, [f0], plane)[monitor_index]
    monitor = monitor.updated_copy(**update)

    with pytest.raises(SetupError):
        proj.project_fields(monitor)


def test_2d_sim_with_proj_monitors_near():
    """Creates near-field projection monitors by modifying proj_distance and far_field_approx."""
    center = [0, 0, 0]
    freqs = 1e13
    monitor_size = (0, 2, td.inf)
    plane = "xy"
    f0 = 1e13
    sim_size = (5, 5, 0)
    # boundary conditions
    boundary_conds = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.pml(),
        z=td.Boundary.periodic(),
    )

    monitors = make_2d_proj_monitors(center, monitor_size, freqs, plane)

    # Modify only proj_distance and far_field_approx
    proj_monitors_near = [
        type(monitor)(
            proj_distance=R_FAR / 50,  # Adjust projection distance
            far_field_approx=False,  # Disable far-field approximation
            **{
                k: v
                for k, v in monitor.__dict__.items()
                if k not in ["proj_distance", "far_field_approx"]
            },
        )
        for monitor in monitors
    ]

    with pytest.raises(
        ValidationError,
        match="Exact far-field projection for 2D simulations is not yet available",
    ):
        _ = td.Simulation(
            size=sim_size,
            grid_spec=td.GridSpec.auto(wavelength=td.C_0 / f0),
            boundary_spec=boundary_conds,
            monitors=proj_monitors_near,
            run_time=1e-12,
        )


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


@pytest.mark.parametrize("idx_u, idx_v", [(0, 1), (0, 2), (1, 2)])
def test_far_field_integral_vjp_3d(idx_u, idx_v):
    rng = np.random.default_rng(0)
    n_x, n_y, n_z = 3, 4, 5
    n_theta, n_phi = 6, 7

    currents = rng.standard_normal((n_x, n_y, n_z)) + 1j * rng.standard_normal((n_x, n_y, n_z))
    pts = [
        np.cumsum(rng.random(n_x)),
        np.cumsum(rng.random(n_y)),
        np.cumsum(rng.random(n_z)),
    ]
    phase_0 = rng.standard_normal((n_x, n_theta, n_phi)) + 1j * rng.standard_normal(
        (n_x, n_theta, n_phi)
    )
    phase_1 = rng.standard_normal((n_y, n_theta, n_phi)) + 1j * rng.standard_normal(
        (n_y, n_theta, n_phi)
    )
    phase_2 = rng.standard_normal((n_z, n_theta)) + 1j * rng.standard_normal((n_z, n_theta))

    def reference(currents_in):
        chunk = anp.einsum("xtp,ytp,zt,xyz->xyztp", phase_0, phase_1, phase_2, currents_in)
        axes = tuple(sorted((idx_u, idx_v)))
        pts_int = tuple(pts[axis] for axis in axes)
        return FieldProjector.trapezoid(chunk, pts_int, axes)

    def primitive(currents_in):
        spec = _FarFieldIntegralSpec(
            weights=tuple(_trapz_weights_1d(pt) for pt in pts),
            idx_u=idx_u,
            idx_v=idx_v,
            is_2d=False,
            idx_integration_1d=None,
        )
        return _far_field_integral(
            currents_in,
            (phase_0, phase_1, phase_2),
            spec,
        )

    vjp_primitive, ans_primitive = make_vjp(primitive)(currents)
    vjp_reference, ans_reference = make_vjp(reference)(currents)

    np.testing.assert_allclose(
        np.asarray(ans_primitive), np.asarray(ans_reference), rtol=1e-12, atol=1e-12
    )

    g = rng.standard_normal(np.asarray(ans_reference).shape) + 1j * rng.standard_normal(
        np.asarray(ans_reference).shape
    )
    grad_primitive = np.asarray(vjp_primitive(g))
    grad_reference = np.asarray(vjp_reference(g))
    np.testing.assert_allclose(grad_primitive, grad_reference, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("idx_integration_1d", [0, 1, 2])
def test_far_field_integral_vjp_2d(idx_integration_1d):
    rng = np.random.default_rng(1)
    n_theta, n_phi = 5, 6

    n_x, n_y, n_z = 1, 1, 1
    if idx_integration_1d == 0:
        n_x = 4
    elif idx_integration_1d == 1:
        n_y = 4
    else:
        n_z = 4

    currents = rng.standard_normal((n_x, n_y, n_z)) + 1j * rng.standard_normal((n_x, n_y, n_z))
    pts = [
        np.cumsum(rng.random(n_x)) if n_x > 1 else np.zeros((n_x,)),
        np.cumsum(rng.random(n_y)) if n_y > 1 else np.zeros((n_y,)),
        np.cumsum(rng.random(n_z)) if n_z > 1 else np.zeros((n_z,)),
    ]
    phase_0 = rng.standard_normal((n_x, n_theta, n_phi)) + 1j * rng.standard_normal(
        (n_x, n_theta, n_phi)
    )
    phase_1 = rng.standard_normal((n_y, n_theta, n_phi)) + 1j * rng.standard_normal(
        (n_y, n_theta, n_phi)
    )
    phase_2 = rng.standard_normal((n_z, n_theta)) + 1j * rng.standard_normal((n_z, n_theta))

    def reference(currents_in):
        chunk = anp.einsum("xtp,ytp,zt,xyz->xyztp", phase_0, phase_1, phase_2, currents_in)
        return FieldProjector.trapezoid(chunk, pts[idx_integration_1d], idx_integration_1d)

    def primitive(currents_in):
        spec = _FarFieldIntegralSpec(
            weights=tuple(_trapz_weights_1d(pt) for pt in pts),
            idx_u=0,
            idx_v=1,
            is_2d=True,
            idx_integration_1d=idx_integration_1d,
        )
        return _far_field_integral(
            currents_in,
            (phase_0, phase_1, phase_2),
            spec,
        )

    vjp_primitive, ans_primitive = make_vjp(primitive)(currents)
    vjp_reference, ans_reference = make_vjp(reference)(currents)

    np.testing.assert_allclose(
        np.asarray(ans_primitive), np.asarray(ans_reference), rtol=1e-12, atol=1e-12
    )

    g = rng.standard_normal(np.asarray(ans_reference).shape) + 1j * rng.standard_normal(
        np.asarray(ans_reference).shape
    )
    grad_primitive = np.asarray(vjp_primitive(g))
    grad_reference = np.asarray(vjp_reference(g))
    np.testing.assert_allclose(grad_primitive, grad_reference, rtol=1e-10, atol=1e-10)


def test_far_field_integral_pairs_matches_reference():
    rng = np.random.default_rng(2)
    n_x, n_y, n_z = 3, 4, 5
    n_pairs = 6
    idx_u, idx_v = 0, 1

    currents = rng.standard_normal((n_x, n_y, n_z)) + 1j * rng.standard_normal((n_x, n_y, n_z))
    pts = [
        np.cumsum(rng.random(n_x)),
        np.cumsum(rng.random(n_y)),
        np.cumsum(rng.random(n_z)),
    ]
    phase_0 = rng.standard_normal((n_x, n_pairs)) + 1j * rng.standard_normal((n_x, n_pairs))
    phase_1 = rng.standard_normal((n_y, n_pairs)) + 1j * rng.standard_normal((n_y, n_pairs))
    phase_2 = rng.standard_normal((n_z, n_pairs)) + 1j * rng.standard_normal((n_z, n_pairs))
    spec = _FarFieldIntegralSpec(
        weights=tuple(_trapz_weights_1d(pt) for pt in pts),
        idx_u=idx_u,
        idx_v=idx_v,
        is_2d=False,
        idx_integration_1d=None,
    )

    actual = _far_field_integral_pairs(currents, (phase_0, phase_1, phase_2), spec)

    expected = np.empty((n_z, n_pairs), dtype=complex)
    for idx_pair in range(n_pairs):
        chunk = (
            phase_0[:, idx_pair][:, None, None]
            * phase_1[:, idx_pair][None, :, None]
            * phase_2[:, idx_pair][None, None, :]
            * currents
        )
        expected[:, idx_pair] = FieldProjector.trapezoid(chunk, (pts[0], pts[1]), (0, 1))

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("idx_integration_1d", [0, 1, 2])
def test_far_field_integral_pairs_matches_reference_2d(idx_integration_1d):
    rng = np.random.default_rng(3)
    n_pairs = 6

    n_x, n_y, n_z = 1, 1, 1
    if idx_integration_1d == 0:
        n_x = 4
    elif idx_integration_1d == 1:
        n_y = 4
    else:
        n_z = 4

    currents = rng.standard_normal((n_x, n_y, n_z)) + 1j * rng.standard_normal((n_x, n_y, n_z))
    pts = [
        np.cumsum(rng.random(n_x)) if n_x > 1 else np.zeros((n_x,)),
        np.cumsum(rng.random(n_y)) if n_y > 1 else np.zeros((n_y,)),
        np.cumsum(rng.random(n_z)) if n_z > 1 else np.zeros((n_z,)),
    ]
    phase_0 = rng.standard_normal((n_x, n_pairs)) + 1j * rng.standard_normal((n_x, n_pairs))
    phase_1 = rng.standard_normal((n_y, n_pairs)) + 1j * rng.standard_normal((n_y, n_pairs))
    phase_2 = rng.standard_normal((n_z, n_pairs)) + 1j * rng.standard_normal((n_z, n_pairs))
    spec = _FarFieldIntegralSpec(
        weights=tuple(_trapz_weights_1d(pt) for pt in pts),
        idx_u=0,
        idx_v=1,
        is_2d=True,
        idx_integration_1d=idx_integration_1d,
    )

    actual = _far_field_integral_pairs(currents, (phase_0, phase_1, phase_2), spec)

    expected = []
    for idx_pair in range(n_pairs):
        chunk = (
            phase_0[:, idx_pair][:, None, None]
            * phase_1[:, idx_pair][None, :, None]
            * phase_2[:, idx_pair][None, None, :]
            * currents
        )
        expected.append(
            FieldProjector.trapezoid(chunk, pts[idx_integration_1d], idx_integration_1d)
        )

    np.testing.assert_allclose(
        np.asarray(actual), np.stack(expected, axis=-1), rtol=1e-12, atol=1e-12
    )


def test_fields_for_surface_exact_kernel_vjp():
    center = (0.0, 0.0, 0.0)
    size = (2.0, 2.0, 0.0)
    projector = make_clientside_projector(center=center, size=size, freqs=F0, num_points=4)
    surface = projector.surfaces[0]
    currents = projector.currents[surface.monitor.name]
    point = (3.0, 0.4, -0.2)

    prepared = field_projection_exact._prepare_exact_surface_projection_point(
        x=point[0],
        y=point[1],
        z=point[2],
        prepared=field_projection_exact._prepare_exact_surface_projection_static(
            surface=surface,
            currents=currents,
            medium=projector.medium,
            frequencies=projector.frequencies,
        ),
    )
    components = field_projection_exact._prepare_exact_surface_currents(surface, currents)

    def reference(currents_in):
        return field_projection_exact._fields_for_surface_exact_impl(currents_in, prepared)

    def primitive(currents_in):
        return field_projection_exact._fields_for_surface_exact_primitive(currents_in, prepared)

    vjp_primitive, ans_primitive = make_vjp(primitive)(components)
    vjp_reference, ans_reference = make_vjp(reference)(components)

    np.testing.assert_allclose(
        np.asarray(ans_primitive), np.asarray(ans_reference), rtol=1e-12, atol=1e-12
    )

    rng = np.random.default_rng(2)
    g = rng.standard_normal(np.asarray(ans_reference).shape) + 1j * rng.standard_normal(
        np.asarray(ans_reference).shape
    )
    grad_primitive = np.asarray(vjp_primitive(g))
    grad_reference = np.asarray(vjp_reference(g))
    np.testing.assert_allclose(grad_primitive, grad_reference, rtol=1e-10, atol=1e-10)


def test_fields_for_surface_exact_batch_kernel_vjp():
    center = (0.0, 0.0, 0.0)
    size = (2.0, 2.0, 0.0)
    projector = make_clientside_projector(center=center, size=size, freqs=F0, num_points=4)
    surface = projector.surfaces[0]
    currents = projector.currents[surface.monitor.name]
    points = np.array(
        [
            (3.0, 0.4, -0.2),
            (2.8, -0.1, 0.3),
            (3.2, 0.2, 0.1),
        ]
    )

    prepared_static = field_projection_exact._prepare_exact_surface_projection_static(
        surface=surface,
        currents=currents,
        medium=projector.medium,
        frequencies=projector.frequencies,
    )
    prepared_batch = field_projection_exact._prepare_exact_surface_projection_batch(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        prepared=prepared_static,
    )
    components = field_projection_exact._prepare_exact_surface_currents(surface, currents)

    def reference(currents_in):
        return anp.stack(
            [
                field_projection_exact._fields_for_surface_exact_impl(
                    currents_in,
                    field_projection_exact._prepare_exact_surface_projection_point(
                        x=point[0], y=point[1], z=point[2], prepared=prepared_static
                    ),
                )
                for point in points
            ],
            axis=0,
        )

    def primitive(currents_in):
        return field_projection_exact._fields_for_surface_exact_batch_primitive(
            currents_in, prepared_batch
        )

    vjp_primitive, ans_primitive = make_vjp(primitive)(components)
    vjp_reference, ans_reference = make_vjp(reference)(components)

    np.testing.assert_allclose(
        np.asarray(ans_primitive), np.asarray(ans_reference), rtol=1e-12, atol=1e-12
    )

    rng = np.random.default_rng(3)
    g = rng.standard_normal(np.asarray(ans_reference).shape) + 1j * rng.standard_normal(
        np.asarray(ans_reference).shape
    )
    grad_primitive = np.asarray(vjp_primitive(g))
    grad_reference = np.asarray(vjp_reference(g))
    np.testing.assert_allclose(grad_primitive, grad_reference, rtol=1e-10, atol=1e-10)
