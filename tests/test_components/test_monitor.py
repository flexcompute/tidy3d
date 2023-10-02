"""Tests monitors."""
import pytest
import pydantic.v1 as pydantic
import numpy as np
import tidy3d as td
from tidy3d.exceptions import SetupError, ValidationError
from ..utils import assert_log_level, log_capture


def test_stop_start():
    with pytest.raises(pydantic.ValidationError):
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
def test_monitor_interval_warn(log_capture, interval, start, stop, log_desired):
    """Assert time monitor interval warning handled as expected."""

    mnt = td.FluxTimeMonitor(size=(1, 1, 0), name="f", interval=interval, stop=stop, start=start)
    assert_log_level(log_capture, log_desired)

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

    with pytest.raises(pydantic.ValidationError):
        _ = td.FluxMonitor(size=(1, 1, 0), name="f", freqs=[1e12], exclude_surfaces=("x-",))


def test_fld_mnt_freqs_none():
    """Test that validation errors if freqs=[None]."""
    with pytest.raises(pydantic.ValidationError):
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
    with pytest.raises(pydantic.ValidationError):
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
        monitors=[M],
        grid_spec=td.GridSpec.uniform(0.1),
    )

    # error when the surfaces that are in are excluded
    M = M.updated_copy(exclude_surfaces=["x-", "x+"])
    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            monitors=[M],
            grid_spec=td.GridSpec.uniform(0.1),
        )


def test_fieldproj_kspace_range():
    # make sure ux, uy are in [-1, 1] for k-space projection monitors
    with pytest.raises(pydantic.ValidationError):
        _ = td.FieldProjectionKSpaceMonitor(
            size=(2, 0, 2), ux=[0.1, 2], uy=[0], name="f", freqs=[2e12], proj_axis=1
        )
    with pytest.raises(pydantic.ValidationError):
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

    with pytest.raises(pydantic.ValidationError):
        _ = td.FieldMonitor(
            size=(td.inf, td.inf, td.inf),
            freqs=[],
            name="test",
            interval_space=(1, 1, 1),
        )


def test_monitor_colocate(log_capture):
    """test default colocate value, and warning if not set"""

    monitor = td.FieldMonitor(
        size=(td.inf, td.inf, td.inf),
        freqs=np.linspace(0, 200e12, 1001),
        name="test",
        interval_space=(1, 2, 3),
    )
    assert monitor.colocate is True
    assert_log_level(log_capture, "WARNING")

    monitor = td.FieldMonitor(
        size=(td.inf, td.inf, td.inf),
        freqs=np.linspace(0, 200e12, 1001),
        name="test",
        interval_space=(1, 2, 3),
        colocate=False,
    )
    assert monitor.colocate is False


@pytest.mark.parametrize("freqs, log_level", [(np.arange(2500), "WARNING"), (np.arange(100), None)])
def test_monitor_num_freqs(log_capture, freqs, log_level):
    """test default colocate value, and warning if not set"""

    monitor = td.FieldMonitor(
        size=(td.inf, td.inf, td.inf),
        freqs=freqs,
        name="test",
        colocate=True,
    )
    assert_log_level(log_capture, log_level)


@pytest.mark.parametrize("num_modes, log_level", [(101, "WARNING"), (100, None)])
def test_monitor_num_modes(log_capture, num_modes, log_level):
    """test default colocate value, and warning if not set"""

    monitor = td.ModeMonitor(
        size=(td.inf, 0, td.inf),
        freqs=np.linspace(1e14, 2e14, 100),
        name="test",
        mode_spec=td.ModeSpec(num_modes=num_modes),
    )
    assert_log_level(log_capture, log_level)


def test_diffraction_validators():

    # ensure error if boundaries are not periodic
    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.periodic(),
        z=td.Boundary.pml(),
    )
    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            structures=[td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium())],
            boundary_spec=boundary_spec,
            monitors=[td.DiffractionMonitor(size=[td.inf, td.inf, 0], freqs=[1e12], name="de")],
            grid_spec=td.GridSpec.uniform(dl=0.1),
        )

    # ensure error if monitor isn't infinite in two directions
    with pytest.raises(pydantic.ValidationError):
        _ = td.DiffractionMonitor(size=[td.inf, 4, 0], freqs=[1e12], name="de")


def test_monitor():

    size = (1, 2, 3)
    center = (1, 2, 3)

    m1 = td.FieldMonitor(size=size, center=center, freqs=[1, 2, 3], name="test_monitor")
    _ = td.FieldMonitor.surfaces(size=size, center=center, freqs=[1, 2, 3], name="test_monitor")
    m2 = td.FieldTimeMonitor(size=size, center=center, name="test_mon")
    m3 = td.FluxMonitor(size=(1, 1, 0), center=center, freqs=[1, 2, 3], name="test_mon")
    m4 = td.FluxTimeMonitor(size=(1, 1, 0), center=center, name="test_mon")
    m5 = td.ModeMonitor(
        size=(1, 1, 0), center=center, mode_spec=td.ModeSpec(), freqs=[1, 2, 3], name="test_mon"
    )
    m6 = td.ModeSolverMonitor(
        size=(1, 1, 0),
        center=center,
        mode_spec=td.ModeSpec(),
        freqs=[1, 2, 3],
        name="test_mon",
        direction="-",
    )
    m7 = td.PermittivityMonitor(size=size, center=center, freqs=[1, 2, 3], name="perm")

    tmesh = np.linspace(0, 1, 10)

    for m in [m1, m2, m3, m4, m5, m6, m7]:
        # m.plot(y=2)
        # plt.close()
        m.storage_size(num_cells=100, tmesh=tmesh)

    for m in [m2, m4]:
        m.time_inds(tmesh=tmesh)
        m.num_steps(tmesh=tmesh)


def test_monitor_plane():

    freqs = [1, 2, 3]

    # make sure flux, mode and diffraction monitors fail with non planar geometries
    for size in ((0, 0, 0), (1, 0, 0), (1, 1, 1)):
        with pytest.raises(pydantic.ValidationError):
            td.ModeMonitor(size=size, freqs=freqs, modes=[])
        with pytest.raises(pydantic.ValidationError):
            td.ModeSolverMonitor(size=size, freqs=freqs, modes=[])
        with pytest.raises(pydantic.ValidationError):
            td.DiffractionMonitor(size=size, freqs=freqs, name="de")


def _test_freqs_nonempty():
    with pytest.raises(ValidationError):
        td.FieldMonitor(size=(1, 1, 1), freqs=[])


def test_monitor_surfaces_from_volume():

    center = (1, 2, 3)

    # make sure that monitors with zero volume raise an error (adapted from test_monitor_plane())
    for size in ((0, 0, 0), (1, 0, 0), (1, 1, 0)):
        with pytest.raises(SetupError):
            _ = td.FieldMonitor.surfaces(
                size=size, center=center, freqs=[1, 2, 3], name="test_monitor"
            )

    # test that the surface monitors can be extracted from a volume monitor
    size = (1, 2, 3)
    monitor_surfaces = td.FieldMonitor.surfaces(
        size=size, center=center, freqs=[1, 2, 3], name="test_monitor"
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
