"""Tests monitors."""

from __future__ import annotations

import numpy as np
import pydantic as pd
import pytest

import tidy3d as td
from tidy3d.exceptions import SetupError, ValidationError

from ..utils import AssertLogLevel


def test_stop_start():
    with pytest.raises(pd.ValidationError):
        td.FluxTimeMonitor(size=(1, 1, 0), name="f", start=2, stop=1)


# interval, start, stop, log_out
time_sampling_tests = [
    (None, 0.0, None, "WARNING"),  # all defaults
    (1, 0.0, None, None),  # interval set (=1)
    (2, 0.0, None, None),  # interval set (=2)
    (None, 1e-12, None, None),  # start specified
    (None, 0.0, 5e-12, None),  # stop specified
]


@pytest.mark.parametrize("interval, start, stop, log_desired", time_sampling_tests)
def test_monitor_interval_warn(interval, start, stop, log_desired):
    """Assert time monitor interval warning handled as expected."""

    with AssertLogLevel(log_desired):
        mnt = td.FluxTimeMonitor(
            size=(1, 1, 0), name="f", interval=interval, stop=stop, start=start
        )

    # make sure it got set to either 1 (undefined) or the specified value
    mnt_interval = interval if interval else 1
    assert mnt.interval == mnt_interval


def test_time_inds():
    M = td.FluxTimeMonitor(size=(1, 1, 0), name="f", start=0, stop=1)
    assert M.time_inds(tmesh=[]) == (0, 0)

    M.time_inds(tmesh=[0.1, 0.2])

    DT = 1
    M = td.FluxTimeMonitor(size=(1, 1, 0), name="f", start=0, stop=DT / 2)
    M.time_inds(tmesh=[0, DT, 2 * DT])


def test_downsampled():
    M = td.FieldMonitor(size=(1, 1, 1), name="f", freqs=[1e12], interval_space=(1, 2, 3))
    num_cells = (10, 10, 10)
    downsampled_num_cells = a, b, c = M.downsampled_num_cells(num_cells=(10, 10, 10))
    assert downsampled_num_cells != num_cells


def test_excluded_surfaces_flat():
    with pytest.raises(pd.ValidationError):
        _ = td.FluxMonitor(size=(1, 1, 0), name="f", freqs=[1e12], exclude_surfaces=("x-",))


def test_fld_mnt_freqs_none():
    """Test that validation errors if freqs=[None]."""
    with pytest.raises(pd.ValidationError):
        td.FieldMonitor(center=(0, 0, 0), size=(0, 0, 0), freqs=[None], name="test")


def test_integration_surfaces():
    # test that integration surfaces are extracted correctly for surface and volume
    # integration monitors

    # surface monitor
    surfaces = td.FieldProjectionAngleMonitor(
        size=(2, 0, 2), theta=[1, 2], phi=[0], name="f", freqs=[2e12]
    ).integration_surfaces
    assert len(surfaces) == 1
    assert surfaces[0].normal_dir == "+"

    # surface monitor oppositely oriented
    surfaces = td.FieldProjectionAngleMonitor(
        size=(2, 0, 2), theta=[1, 2], phi=[0], name="f", freqs=[2e12], normal_dir="-"
    ).integration_surfaces
    assert len(surfaces) == 1
    assert surfaces[0].normal_dir == "-"

    # volume monitor
    surfaces = td.FieldProjectionAngleMonitor(
        size=(2, 2, 2), theta=[1, 2], phi=[0], name="f", freqs=[2e12]
    ).integration_surfaces
    assert len(surfaces) == 6
    for idx, surface in enumerate(surfaces):
        if np.mod(idx, 2) == 0:
            assert surface.normal_dir == "-"
            assert surface.name[-1] == "-"
        else:
            assert surface.normal_dir == "+"
            assert surface.name[-1] == "+"

    # volume monitor with excluded surfaces
    surfaces = td.FieldProjectionAngleMonitor(
        size=(2, 2, 2), theta=[1], phi=[0], name="f", freqs=[2e12], exclude_surfaces=["x-", "y+"]
    ).integration_surfaces
    assert len(surfaces) == 4
    expected_surfs = ["x+", "y-", "z-", "z+"]
    for idx, surface in enumerate(surfaces):
        assert surface.normal_dir == expected_surfs[idx][-1]
        assert surface.name[-2:] == expected_surfs[idx]

    # volume monitor with an infinite dimension
    surfaces = td.FieldProjectionAngleMonitor(
        size=(td.inf, 2, 2), theta=[1], phi=[0], name="f", freqs=[2e12]
    ).integration_surfaces
    assert len(surfaces) == 4
    expected_surfs = ["y-", "y+", "z-", "z+"]
    for idx, surface in enumerate(surfaces):
        assert surface.normal_dir == expected_surfs[idx][-1]
        assert surface.name[-2:] == expected_surfs[idx]

    # volume monitor with all infinite dimensions
    surfaces = td.FieldProjectionAngleMonitor(
        size=(td.inf, td.inf, td.inf), theta=[1], phi=[0], name="f", freqs=[2e12]
    ).integration_surfaces
    assert len(surfaces) == 0


def test_fieldproj_surfaces():
    # test the field projection surfaces are set correctly for projection monitors
    M = td.FieldProjectionAngleMonitor(
        size=(2, 0, 2), theta=[1, 2], phi=[0], name="f", freqs=[2e12]
    ).projection_surfaces
    assert len(M) == 1
    assert M[0].axis == 1

    M = td.FieldProjectionAngleMonitor(
        size=(2, 2, 2), theta=[1, 2], phi=[0], name="f", freqs=[2e12]
    ).projection_surfaces
    assert len(M) == 6

    M = td.FieldProjectionAngleMonitor(
        size=(2, 2, 2), theta=[1], phi=[0], name="f", freqs=[2e12], exclude_surfaces=["x-", "y+"]
    ).projection_surfaces
    assert len(M) == 4


def test_fieldproj_surfaces_in_simulaiton():
    # test error if all projection surfaces are outside the simulation domain
    M = td.FieldProjectionAngleMonitor(size=(3, 3, 3), theta=[1], phi=[0], name="f", freqs=[2e12])
    with pytest.raises(pd.ValidationError):
        _ = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            monitors=[M],
            grid_spec=td.GridSpec.uniform(0.1),
        )
    # no error when some surfaces are in
    M = M.updated_copy(size=(1, 3, 3))
    _ = td.Simulation(
        size=(2, 2, 2),
        run_time=1e-12,
        monitors=(M,),
        grid_spec=td.GridSpec.uniform(0.1),
    )

    # error when the surfaces that are in are excluded
    M = M.updated_copy(exclude_surfaces=("x-", "x+"))
    with pytest.raises(pd.ValidationError):
        _ = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            monitors=[M],
            grid_spec=td.GridSpec.uniform(0.1),
        )


def test_fieldproj_kspace_range():
    # make sure ux, uy are in [-1, 1] for k-space projection monitors
    with pytest.raises(pd.ValidationError):
        _ = td.FieldProjectionKSpaceMonitor(
            size=(2, 0, 2), ux=[0.1, 2], uy=[0], name="f", freqs=[2e12], proj_axis=1
        )
    with pytest.raises(pd.ValidationError):
        _ = td.FieldProjectionKSpaceMonitor(
            size=(2, 0, 2), ux=[0.1, 0.2], uy=[1.1], name="f", freqs=[2e12], proj_axis=1
        )
    _ = td.FieldProjectionKSpaceMonitor(
        size=(2, 0, 2), ux=[1, 0.2], uy=[1.0], name="f", freqs=[2e12], proj_axis=1
    )


def test_fieldproj_local_origin():
    M = td.FieldProjectionAngleMonitor(
        size=(2, 0, 2), theta=[1, 2], phi=[0], name="f", freqs=[2e12]
    )
    M.local_origin
    M = td.FieldProjectionAngleMonitor(
        size=(2, 0, 2), theta=[1, 2], phi=[0], name="f", freqs=[2e12], custom_origin=(1, 2, 3)
    )
    M.local_origin


def test_fieldproj_window():
    M = td.FieldProjectionAngleMonitor(
        size=(2, 0, 2), theta=[1, 2], phi=[0], name="f", freqs=[2e12], window_size=(0.2, 1)
    )
    window_size, window_minus, window_plus = M.window_parameters()
    window_size, window_minus, window_plus = M.window_parameters(M.bounds)
    points = np.linspace(0, 10, 100)
    _ = M.window_function(points, window_size, window_minus, window_plus, 2)
    # do not allow a window size larger than 1
    with pytest.raises(pd.ValidationError):
        _ = td.FieldProjectionAngleMonitor(
            size=(2, 0, 2), theta=[1, 2], phi=[0], name="f", freqs=[2e12], window_size=(0.2, 1.1)
        )
    # do not allow non-zero windows for volume monitors
    with pytest.raises(pd.ValidationError):
        _ = td.FieldProjectionAngleMonitor(
            size=(2, 1, 2), theta=[1, 2], phi=[0], name="f", freqs=[2e12], window_size=(0.2, 0)
        )


PROJ_MNTS = [
    td.FieldProjectionAngleMonitor(size=(2, 0, 2), theta=[1, 2], phi=[0], name="f", freqs=[2e12]),
    td.FieldProjectionCartesianMonitor(
        size=(2, 0, 2), x=[1, 2], y=[0], proj_distance=0, proj_axis=2, name="f", freqs=[2e12]
    ),
    td.FieldProjectionKSpaceMonitor(
        size=(2, 0, 2), ux=[1, 0.2], uy=[0], proj_axis=2, name="f", freqs=[2e12]
    ),
]


@pytest.mark.parametrize("proj_mnt", PROJ_MNTS)
def test_storage_sizes(proj_mnt):
    proj_mnt.storage_size(num_cells=100, tmesh=[1, 2, 3])


def test_monitor_freqs_empty():
    # errors when no frequencies supplied

    with pytest.raises(pd.ValidationError):
        _ = td.FieldMonitor(
            size=(td.inf, td.inf, td.inf),
            freqs=[],
            name="test",
            interval_space=(1, 1, 1),
        )


def test_monitor_colocate():
    """test default colocate value, and warning if not set"""

    with AssertLogLevel(None):
        monitor = td.FieldMonitor(
            size=(td.inf, td.inf, td.inf),
            freqs=np.linspace(1e12, 200e12, 1001),
            name="test",
            interval_space=(1, 2, 3),
        )
        assert monitor.colocate is True

    monitor = td.FieldMonitor(
        size=(td.inf, td.inf, td.inf),
        freqs=np.linspace(1e12, 200e12, 1001),
        name="test",
        interval_space=(1, 2, 3),
        colocate=False,
    )
    assert monitor.colocate is False


@pytest.mark.parametrize(
    "freqs, log_level", [(np.arange(1, 2500), "WARNING"), (np.arange(1, 100), None)]
)
def test_monitor_num_freqs(freqs, log_level):
    """test default colocate value, and warning if not set"""

    with AssertLogLevel(log_level):
        td.FieldMonitor(
            size=(td.inf, td.inf, td.inf),
            freqs=freqs * 1e12,
            name="test",
            colocate=True,
        )


@pytest.mark.parametrize("num_modes, log_level", [(101, "WARNING"), (100, None)])
def test_monitor_num_modes(num_modes, log_level):
    """test default colocate value, and warning if not set"""

    with AssertLogLevel(log_level):
        td.ModeMonitor(
            size=(td.inf, 0, td.inf),
            freqs=np.linspace(1e14, 2e14, 100),
            name="test",
            mode_spec=td.ModeSpec(num_modes=num_modes),
        )


def test_mode_bend_radius():
    """Test that small bend radius fails."""

    with pytest.raises(ValueError):
        mnt = td.ModeMonitor(
            size=(5, 0, 1),
            freqs=np.linspace(1e14, 2e14, 100),
            name="test",
            mode_spec=td.ModeSpec(num_modes=1, bend_radius=1, bend_axis=2),
        )
        _ = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            monitors=[mnt],
            grid_spec=td.GridSpec.uniform(dl=0.1),
        )


def test_diffraction_validators():
    # ensure error if boundaries are not periodic
    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.periodic(),
        z=td.Boundary.pml(),
    )
    with pytest.raises(pd.ValidationError):
        _ = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            structures=[td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium())],
            boundary_spec=boundary_spec,
            monitors=[td.DiffractionMonitor(size=[td.inf, td.inf, 0], freqs=[1e12], name="de")],
            grid_spec=td.GridSpec.uniform(dl=0.1),
        )

    # ensure error if monitor isn't infinite in two directions
    with pytest.raises(pd.ValidationError):
        _ = td.DiffractionMonitor(size=[td.inf, 4, 0], freqs=[1e12], name="de")


FREQS = np.array([1, 2, 3]) * 1e12


def test_gaussian_overlap_monitors_basic():
    g = td.GaussianOverlapMonitor(size=(1, 1, 0), name="g", freqs=FREQS)
    a = td.AstigmaticGaussianOverlapMonitor(size=(1, 1, 0), name="a", freqs=FREQS)
    for m in (g, a):
        s = m.storage_size(num_cells=10, tmesh=[0.0, 1.0])
        assert isinstance(s, int) and s > 0


def test_monitor():
    size = (1, 2, 3)
    center = (1, 2, 3)

    pd = np.atleast_1d(40000)
    thetas = np.linspace(0, 2 * np.pi, 100)
    phis = np.linspace(0, np.pi, 100)

    m1 = td.FieldMonitor(size=size, center=center, freqs=FREQS, name="test_monitor")
    _ = td.FieldMonitor.surfaces(size=size, center=center, freqs=FREQS, name="test_monitor")
    m2 = td.FieldTimeMonitor(size=size, center=center, name="test_mon")
    m3 = td.FluxMonitor(size=(1, 1, 0), center=center, freqs=FREQS, name="test_mon")
    m4 = td.FluxTimeMonitor(size=(1, 1, 0), center=center, name="test_mon")
    m5 = td.ModeMonitor(
        size=(1, 1, 0), center=center, mode_spec=td.ModeSpec(), freqs=FREQS, name="test_mon"
    )
    m6 = td.ModeMonitor(size=(1, 1, 0), center=center, freqs=FREQS, name="test_mon")
    m7 = td.ModeSolverMonitor(
        size=(1, 1, 0),
        center=center,
        freqs=FREQS,
        name="test_mon",
        direction="-",
    )
    m8 = td.PermittivityMonitor(size=size, center=center, freqs=FREQS, name="perm")
    m9 = td.DirectivityMonitor(
        size=size,
        center=center,
        theta=thetas,
        phi=phis,
        proj_distance=pd,
        freqs=FREQS,
        name="directivity",
    )
    m10 = td.PermittivityMonitor(size=size, center=center, freqs=FREQS, name="perm")
    m11 = td.AuxFieldTimeMonitor(size=size, center=center, name="aux_field_time", fields=("Nfx",))
    m12 = td.MediumMonitor(size=size, center=center, freqs=FREQS, name="mat")
    m13 = td.GaussianOverlapMonitor(size=(1, 1, 0), center=center, freqs=FREQS, name="gauss")
    m14 = td.AstigmaticGaussianOverlapMonitor(
        size=(1, 1, 0), center=center, freqs=FREQS, name="astigauss"
    )

    tmesh = np.linspace(0, 1, 10)

    for m in [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14]:
        m.storage_size(num_cells=100, tmesh=tmesh)

    for m in [m2, m4]:
        m.time_inds(tmesh=tmesh)
        m.num_steps(tmesh=tmesh)


def test_monitor_plane():
    # make sure flux, mode and diffraction monitors fail with non planar geometries
    for size in ((0, 0, 0), (1, 0, 0), (1, 1, 1)):
        with pytest.raises(pd.ValidationError):
            td.ModeMonitor(size=size, freqs=FREQS, modes=[])
        with pytest.raises(pd.ValidationError):
            td.ModeSolverMonitor(size=size, freqs=FREQS, modes=[])
        with pytest.raises(pd.ValidationError):
            td.DiffractionMonitor(size=size, freqs=FREQS, name="de")


def _test_freqs_nonempty():
    with pytest.raises(ValidationError):
        td.FieldMonitor(size=(1, 1, 1), freqs=[])


def test_monitor_surfaces_from_volume():
    center = (1, 2, 3)

    # make sure that monitors with zero volume raise an error (adapted from test_monitor_plane())
    for size in ((0, 0, 0), (1, 0, 0), (1, 1, 0)):
        with pytest.raises(SetupError):
            _ = td.FieldMonitor.surfaces(size=size, center=center, freqs=FREQS, name="test_monitor")

    # test that the surface monitors can be extracted from a volume monitor
    size = (1, 2, 3)
    monitor_surfaces = td.FieldMonitor.surfaces(
        size=size, center=center, freqs=FREQS, name="test_monitor"
    )

    # x- surface
    assert monitor_surfaces[0].center == (center[0] - size[0] / 2.0, center[1], center[2])
    assert monitor_surfaces[0].size == (0.0, size[1], size[2])

    # x+ surface
    assert monitor_surfaces[1].center == (center[0] + size[0] / 2.0, center[1], center[2])
    assert monitor_surfaces[1].size == (0.0, size[1], size[2])

    # y- surface
    assert monitor_surfaces[2].center == (center[0], center[1] - size[1] / 2.0, center[2])
    assert monitor_surfaces[2].size == (size[0], 0.0, size[2])

    # y+ surface
    assert monitor_surfaces[3].center == (center[0], center[1] + size[1] / 2.0, center[2])
    assert monitor_surfaces[3].size == (size[0], 0.0, size[2])

    # z- surface
    assert monitor_surfaces[4].center == (center[0], center[1], center[2] - size[2] / 2.0)
    assert monitor_surfaces[4].size == (size[0], size[1], 0.0)

    # z+ surface
    assert monitor_surfaces[5].center == (center[0], center[1], center[2] + size[2] / 2.0)
    assert monitor_surfaces[5].size == (size[0], size[1], 0.0)


def test_directivity_monitor():
    """Check validation of directivity monitor."""
    size = (1, 2, 3)
    center = (1, 2, 3)

    pd_arr = np.atleast_1d(40000)
    thetas = np.linspace(0, 2 * np.pi, 100)
    phis = np.linspace(0, np.pi, 100)

    # far_field_approx cannot be set to False
    with pytest.raises(pd.ValidationError):
        _ = td.DirectivityMonitor(
            size=size,
            center=center,
            theta=thetas,
            phi=phis,
            proj_distance=pd_arr,
            freqs=FREQS,
            name="directivity",
            far_field_approx=False,
        )


def test_surface_monitors():
    pec_sphere = td.Structure(geometry=td.Sphere(radius=0.5), medium=td.PECMedium())

    surf_mnt = td.SurfaceFieldMonitor(size=(1, 1, 1), freqs=[td.C_0], name="surface")

    _ = td.Simulation(
        size=(2, 2, 2),
        structures=[pec_sphere],
        monitors=[surf_mnt],
        run_time=1e-12,
        grid_spec=td.GridSpec.auto(wavelength=1),
    )

    # background is PEC with dielectric sphere
    _ = td.Simulation(
        size=(2, 2, 2),
        medium=td.PECMedium(),
        structures=[pec_sphere.updated_copy(medium=td.Medium())],
        monitors=[surf_mnt],
        run_time=1e-12,
        grid_spec=td.GridSpec.auto(wavelength=1),
    )

    # monitor doesn't overlap any pec structure
    with pytest.raises(
        pd.ValidationError,
        match="Surface monitor surface does not cross any PEC or LossyMetalMedium structures.",
    ):
        surf_mnt = td.SurfaceFieldMonitor(
            size=(0.2, 1, 1), center=(0.8, 0, 0), freqs=[td.C_0], name="surface"
        )

        _ = td.Simulation(
            size=(2, 2, 2),
            structures=[pec_sphere],
            monitors=[surf_mnt],
            run_time=1e-12,
            grid_spec=td.GridSpec.auto(wavelength=1),
        )

    # monitor must be volumetric
    with pytest.raises(pd.ValidationError, match="must be volumetric"):
        surf_mnt = td.SurfaceFieldMonitor(size=(1, 0, 1), freqs=[td.C_0], name="surface")

    # surface monitors are not allowed in 2D simulations
    with pytest.raises(
        pd.ValidationError,
        match="Simulation domain has size zero along at least one dimension; surface monitors are not allowed in this case.",
    ):
        surf_mnt = td.SurfaceFieldMonitor(size=(1, 1, 1), freqs=[td.C_0], name="surface")
        _ = td.Simulation(
            size=(1, 1, 0),
            structures=[pec_sphere],
            monitors=[surf_mnt],
            run_time=1e-12,
            grid_spec=td.GridSpec.auto(wavelength=1),
        )


def test_use_colocated_integration_requires_colocate_false():
    """use_colocated_integration=False requires colocate=False."""
    # Valid: colocate=False, use_colocated_integration=False
    td.FieldMonitor(
        size=(1, 1, 0), freqs=[1e14], name="valid", colocate=False, use_colocated_integration=False
    )

    # Invalid: colocate=True, use_colocated_integration=False
    with pytest.raises(pd.ValidationError):
        td.FieldMonitor(
            size=(1, 1, 0), freqs=[1e14], name="m", colocate=True, use_colocated_integration=False
        )

    with pytest.raises(pd.ValidationError):
        td.ModeMonitor(
            size=(1, 1, 0),
            freqs=[1e14],
            name="m",
            mode_spec=td.ModeSpec(),
            colocate=True,
            use_colocated_integration=False,
        )
