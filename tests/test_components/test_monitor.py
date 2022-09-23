"""Tests monitors."""
import pytest
import numpy as np
import tidy3d as td
from tidy3d.log import SetupError, DataError, ValidationError


def test_stop_start():
    with pytest.raises(SetupError):
        td.FluxTimeMonitor(size=(1, 1, 0), name="f", start=2, stop=1)


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

    with pytest.raises(SetupError):
        M = td.FluxMonitor(size=(1, 1, 0), name="f", freqs=[1e12], exclude_surfaces=("x-",))


def test_near2far_monitor_axis_volumous():

    M = td.Near2FarAngleMonitor(size=(2, 0, 2), theta=[1, 2], phi=[0], name="f", freqs=[2e12])
    assert M.axis == 1

    M = td.Near2FarAngleMonitor(size=(2, 2, 2), theta=[1, 2], phi=[0], name="f", freqs=[2e12])
    with pytest.raises(SetupError):
        M.axis


def test_near2far_local_origin():
    M = td.Near2FarAngleMonitor(size=(2, 0, 2), theta=[1, 2], phi=[0], name="f", freqs=[2e12])
    M.local_origin
    M = td.Near2FarAngleMonitor(
        size=(2, 0, 2), theta=[1, 2], phi=[0], name="f", freqs=[2e12], custom_origin=(1, 2, 3)
    )
    M.local_origin


N2F_MNTS = [
    td.Near2FarAngleMonitor(size=(2, 0, 2), theta=[1, 2], phi=[0], name="f", freqs=[2e12]),
    td.Near2FarCartesianMonitor(
        size=(2, 0, 2), x=[1, 2], y=[0], plane_distance=0, plane_axis=2, name="f", freqs=[2e12]
    ),
    td.Near2FarKSpaceMonitor(size=(2, 0, 2), ux=[1, 2], uy=[0], u_axis=2, name="f", freqs=[2e12]),
]


@pytest.mark.parametrize("n2f_mnt", N2F_MNTS)
def test_storage_sizes(n2f_mnt):
    n2f_mnt.storage_size(num_cells=100, tmesh=[1, 2, 3])


def test_monitor_freqs_empty():
    # errors when no frequencies supplied

    with pytest.raises(ValidationError):
        monitor = td.FieldMonitor(
            size=(td.inf, td.inf, td.inf),
            freqs=[],
            name="test",
            interval_space=(1, 1, 1),
        )


def test_monitor_downsampling():
    # test that the downsampling parameters can be set and the colocation validator works

    monitor = td.FieldMonitor(
        size=(td.inf, td.inf, td.inf),
        freqs=np.linspace(0, 200e12, 10001),
        name="test",
        interval_space=(1, 1, 1),
    )
    assert monitor.colocate is False

    monitor = td.FieldMonitor(
        size=(td.inf, td.inf, td.inf),
        freqs=np.linspace(0, 200e12, 10001),
        name="test",
        interval_space=(1, 2, 3),
    )
    assert monitor.colocate is True

    monitor = td.FieldMonitor(
        size=(td.inf, td.inf, td.inf),
        freqs=np.linspace(0, 200e12, 10001),
        name="test",
        interval_space=(1, 2, 3),
        colocate=False,
    )
    assert monitor.colocate is False


def test_diffraction_validators():

    # ensure error if boundaries are not periodic
    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.periodic(),
        z=td.Boundary.pml(),
    )
    with pytest.raises(SetupError) as e_info:
        sim = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            structures=[td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium())],
            boundary_spec=boundary_spec,
            monitors=[td.DiffractionMonitor(size=[td.inf, td.inf, 0], freqs=[1e12], name="de")],
        )

    # ensure error if monitor isn't infinite in two directions
    with pytest.raises(SetupError) as e_info:
        monitor = td.DiffractionMonitor(size=[td.inf, 4, 0], freqs=[1e12], name="de")

    # ensure error if orders are non-unique
    with pytest.raises(SetupError) as e_info:
        monitor = td.DiffractionMonitor(
            size=[td.inf, td.inf, 0], freqs=[1e12], name="de", orders_x=[0, 1, 1]
        )
    with pytest.raises(SetupError) as e_info:
        monitor = td.DiffractionMonitor(
            size=[td.inf, td.inf, 0], freqs=[1e12], name="de", orders_y=[0, 1, 1]
        )


def test_monitor():

    size = (1, 2, 3)
    center = (1, 2, 3)

    m1 = td.FieldMonitor(size=size, center=center, freqs=[1, 2, 3], name="test_monitor")
    m1s = td.FieldMonitor.surfaces(size=size, center=center, freqs=[1, 2, 3], name="test_monitor")
    m2 = td.FieldTimeMonitor(size=size, center=center, name="test_mon")
    m3 = td.FluxMonitor(size=(1, 1, 0), center=center, freqs=[1, 2, 3], name="test_mon")
    m4 = td.FluxTimeMonitor(size=(1, 1, 0), center=center, name="test_mon")
    m5 = td.ModeMonitor(
        size=(1, 1, 0), center=center, mode_spec=td.ModeSpec(), freqs=[1, 2, 3], name="test_mon"
    )
    m6 = td.ModeSolverMonitor(
        size=(1, 1, 0), center=center, mode_spec=td.ModeSpec(), freqs=[1, 2, 3], name="test_mon"
    )
    m7 = td.PermittivityMonitor(size=size, center=center, freqs=[1, 2, 3], name="perm")

    tmesh = np.linspace(0, 1, 10)

    for m in [m1, m2, m3, m4, m5, m6, m7]:
        # m.plot(y=2)
        m.storage_size(num_cells=100, tmesh=tmesh)

    for m in [m2, m4]:
        m.time_inds(tmesh=tmesh)
        m.num_steps(tmesh=tmesh)


def test_monitor_plane():

    freqs = [1, 2, 3]

    # make sure flux, mode and diffraction monitors fail with non planar geometries
    for size in ((0, 0, 0), (1, 0, 0), (1, 1, 1)):
        with pytest.raises(ValidationError) as e_info:
            td.ModeMonitor(size=size, freqs=freqs, modes=[])
        with pytest.raises(ValidationError) as e_info:
            td.ModeSolverMonitor(size=size, freqs=freqs, modes=[])
        with pytest.raises(ValidationError) as e_info:
            td.DiffractionMonitor(size=size, freqs=freqs, name="de")


def _test_freqs_nonempty():
    with pytest.raises(ValidationError) as e_info:
        td.FieldMonitor(size=(1, 1, 1), freqs=[])


def test_monitor_surfaces_from_volume():

    center = (1, 2, 3)

    # make sure that monitors with zero volume raise an error (adapted from test_monitor_plane())
    for size in ((0, 0, 0), (1, 0, 0), (1, 1, 0)):
        with pytest.raises(SetupError) as e_info:
            mon_surfaces = td.FieldMonitor.surfaces(
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
