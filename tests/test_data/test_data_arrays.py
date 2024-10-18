"""Tests tidy3d/components/data/data_array.py"""

from typing import List, Tuple

import numpy as np
import pytest
import tidy3d as td
import xarray.testing as xrt
from tidy3d.exceptions import DataError

np.random.seed(4)

STRUCTURES = [
    td.Structure(
        geometry=td.Box(size=(1, td.inf, 1)), medium=td.material_library["cSi"]["SalzbergVilla1957"]
    )
]
SIZE_3D = (2, 4, 5)
SIZE_2D = list(SIZE_3D)
SIZE_2D[1] = 0
MODE_SPEC = td.ModeSpec(num_modes=4)
FREQS = [1e14, 2e14]
SOURCES = [
    td.PointDipole(source_time=td.GaussianPulse(freq0=FREQS[0], fwidth=1e14), polarization="Ex"),
    td.ModeSource(
        size=SIZE_2D,
        mode_spec=MODE_SPEC,
        source_time=td.GaussianPulse(freq0=FREQS[1], fwidth=1e14),
        direction="+",
    ),
]
FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hz")
INTERVAL = 2
ORDERS_X = list(range(-1, 2))
ORDERS_Y = list(range(-2, 3))

FS11 = np.linspace(1e14, 2e14, 11)
# unused
# TS = np.linspace(0, 1e-12, 11)
# MODE_INDICES = np.arange(0, 3)
# DIRECTIONS = ["+", "-"]
FS = np.linspace(1e14, 2e14, 5)
TS = np.linspace(0, 1e-12, 4)
MODE_INDICES = np.arange(0, 4)
DIRECTIONS = ["+", "-"]

FIELD_MONITOR = td.FieldMonitor(size=SIZE_3D, fields=FIELDS, name="field", freqs=FREQS)
FIELD_TIME_MONITOR = td.FieldTimeMonitor(
    size=SIZE_3D, fields=FIELDS, name="field_time", interval=INTERVAL
)
FIELD_MONITOR_2D = td.FieldMonitor(size=SIZE_2D, fields=FIELDS, name="field_2d", freqs=FREQS)
FIELD_TIME_MONITOR_2D = td.FieldTimeMonitor(
    size=SIZE_2D, fields=FIELDS, name="field_time_2d", interval=INTERVAL
)
PERMITTIVITY_MONITOR = td.PermittivityMonitor(size=SIZE_3D, name="permittivity", freqs=FREQS)
MODE_MONITOR = td.ModeMonitor(size=SIZE_2D, name="mode", mode_spec=MODE_SPEC, freqs=FREQS)
MODE_MONITOR_WITH_FIELDS = td.ModeMonitor(
    size=SIZE_2D, name="mode_solver", mode_spec=MODE_SPEC, freqs=FS, store_fields_direction="+"
)
FLUX_MONITOR = td.FluxMonitor(size=SIZE_2D, freqs=FREQS, name="flux")
FLUX_TIME_MONITOR = td.FluxTimeMonitor(size=SIZE_2D, interval=INTERVAL, name="flux_time")
DIFFRACTION_MONITOR = td.DiffractionMonitor(
    center=(0, 0, 2),
    size=(td.inf, td.inf, 0),
    freqs=FS,
    name="diffraction",
)

MONITORS = [
    FIELD_MONITOR,
    FIELD_TIME_MONITOR,
    MODE_MONITOR_WITH_FIELDS,
    PERMITTIVITY_MONITOR,
    MODE_MONITOR,
    FLUX_MONITOR,
    FLUX_TIME_MONITOR,
    DIFFRACTION_MONITOR,
]

GRID_SPEC = td.GridSpec(wavelength=2.0)
RUN_TIME = 1e-12

SIM_SYM = td.Simulation(
    size=SIZE_3D,
    run_time=RUN_TIME,
    grid_spec=GRID_SPEC,
    symmetry=(1, -1, 1),
    sources=SOURCES,
    monitors=MONITORS,
    structures=STRUCTURES,
    boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
)

SIM = td.Simulation(
    size=SIZE_3D,
    run_time=RUN_TIME,
    grid_spec=GRID_SPEC,
    symmetry=(0, 0, 0),
    sources=SOURCES,
    monitors=MONITORS,
    structures=STRUCTURES,
    boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
)

""" Generate the data arrays (used in other test files) """


def get_xyz(
    monitor: td.components.monitor.MonitorType, grid_key: str, symmetry: bool
) -> Tuple[List[float], List[float], List[float]]:
    if symmetry:
        grid = SIM_SYM.discretize_monitor(monitor)
        x, y, z = grid[grid_key].to_list
        x = [_x for _x in x if _x >= 0]
        y = [_y for _y in y if _y >= 0]
        z = [_z for _z in z if _z >= 0]
    else:
        grid = SIM.discretize_monitor(monitor)
        x, y, z = grid[grid_key].to_list
    return x, y, z


def make_scalar_field_data_array(grid_key: str, symmetry=True):
    XS, YS, ZS = get_xyz(FIELD_MONITOR, grid_key, symmetry)
    values = (1 + 1j) * np.random.random((len(XS), len(YS), len(ZS), len(FS)))
    return td.ScalarFieldDataArray(values, coords=dict(x=XS, y=YS, z=ZS, f=FS))


def make_scalar_field_time_data_array(grid_key: str, symmetry=True):
    XS, YS, ZS = get_xyz(FIELD_TIME_MONITOR, grid_key, symmetry)
    values = np.random.random((len(XS), len(YS), len(ZS), len(TS)))
    return td.ScalarFieldTimeDataArray(values, coords=dict(x=XS, y=YS, z=ZS, t=TS))


def make_scalar_mode_field_data_array(grid_key: str, symmetry=True):
    XS, YS, ZS = get_xyz(MODE_MONITOR_WITH_FIELDS, grid_key, symmetry)
    values = (1 + 0.1j) * np.random.random((len(XS), 1, len(ZS), len(FS), len(MODE_INDICES)))

    return td.ScalarModeFieldDataArray(
        values, coords=dict(x=XS, y=[0.0], z=ZS, f=FS, mode_index=MODE_INDICES)
    )


def make_scalar_mode_field_data_array_smooth(grid_key: str, symmetry=True, rot: float = 0):
    XS, YS, ZS = get_xyz(MODE_MONITOR_WITH_FIELDS, grid_key, symmetry)

    values = np.array([1 + 0.1j])[None, :, None, None, None] * np.sin(
        0.5
        * np.pi
        * (MODE_INDICES[None, None, None, None, :] + 1)
        * (1.0 + 3e-15 * (FS[None, None, None, :, None] - FS[0]))
        * (
            np.cos(rot) * np.array(XS)[:, None, None, None, None]
            + np.sin(rot) * np.array(ZS)[None, None, :, None, None]
        )
    )

    return td.ScalarModeFieldDataArray(
        values, coords=dict(x=XS, y=[0.0], z=ZS, f=FS, mode_index=MODE_INDICES)
    )


def make_mode_amps_data_array():
    values = (1 + 1j) * np.random.random((len(DIRECTIONS), len(MODE_INDICES), len(FS)))
    return td.ModeAmpsDataArray(
        values, coords=dict(direction=DIRECTIONS, mode_index=MODE_INDICES, f=FS)
    )


def make_mode_index_data_array():
    values = (1 + 0.1j) * np.random.random((len(FS), len(MODE_INDICES)))
    return td.ModeIndexDataArray(values, coords=dict(f=FS, mode_index=MODE_INDICES))


def make_flux_data_array():
    values = np.random.random(len(FS))
    return td.FluxDataArray(values, coords=dict(f=FS))


def make_flux_time_data_array():
    values = np.random.random(len(TS))
    return td.FluxTimeDataArray(values, coords=dict(t=TS))


def make_diffraction_data_array():
    values = (1 + 1j) * np.random.random((len(ORDERS_X), len(ORDERS_Y), len(FS)))
    return (
        [SIZE_2D[0], SIZE_2D[2]],
        [1.0, 2.0],
        td.DiffractionDataArray(values, coords=dict(orders_x=ORDERS_X, orders_y=ORDERS_Y, f=FS)),
    )


""" Test that they work """


def test_scalar_field_data_array():
    for grid_key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        data = make_scalar_field_data_array(grid_key)
        data = data.interp(f=1.5e14)
        _ = data.isel(y=2)


def test_scalar_field_time_data_array():
    for grid_key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        data = make_scalar_field_time_data_array(grid_key)
        data = data.interp(t=1e-13)
        _ = data.isel(y=2)


def test_scalar_mode_field_data_array():
    for grid_key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        data = make_scalar_mode_field_data_array(grid_key)
        data = data.interp(f=1.5e14)
        data = data.isel(x=2)
        _ = data.sel(mode_index=2)


def test_mode_amps_data_array():
    data = make_mode_amps_data_array()
    data = data.interp(f=1.5e14)
    data = data.isel(direction=0)
    _ = data.sel(mode_index=1)


def test_mode_index_data_array():
    data = make_mode_index_data_array()
    data = data.interp(f=1.5e14)
    _ = data.sel(mode_index=1)


def test_flux_data_array():
    data = make_flux_data_array()
    data = data.interp(f=1.5e14)


def test_flux_time_data_array():
    data = make_flux_time_data_array()
    data = data.interp(t=1e-13)


def test_diffraction_data_array():
    _, _, data = make_diffraction_data_array()
    data = data.interp(f=1.5e14)


def test_ops():
    data1 = make_flux_data_array()
    data2 = make_flux_data_array()
    data1.data = np.ones_like(data1.data)
    data2.data = np.ones_like(data2.data)
    data3 = make_flux_time_data_array()
    assert np.all(data1 == data2), "identical data are not equal"
    data1.data[0] = 1e12
    assert not np.all(data1 == data2), "different data are equal"
    assert not np.all(data1 == data3), "different data are equal"


def test_empty_field_time():
    _ = td.ScalarFieldTimeDataArray(
        np.random.rand(5, 5, 5, 0),
        coords=dict(x=np.arange(5), y=np.arange(5), z=np.arange(5), t=[]),
    )
    _ = td.ScalarFieldTimeDataArray(
        np.random.rand(5, 5, 5, 0),
        coords=dict(x=np.arange(5), y=np.arange(5), z=np.arange(5), t=[]),
    )


def test_abs():
    data = make_mode_amps_data_array()
    _ = data.abs


def test_heat_data_array():
    T = [0, 1e-12, 2e-12]
    _ = td.HeatDataArray((1 + 1j) * np.random.random((3,)), coords=dict(T=T))


def test_charge_data_array():
    n = [0, 1e-12, 2e-12]
    p = [0, 3e-12, 4e-12]
    _ = td.ChargeDataArray((1 + 1j) * np.random.random((3, 3)), coords=dict(n=n, p=p))


def test_point_data_array():
    _ = td.PointDataArray(
        np.random.rand(2, 3),
        coords=dict(index=np.arange(2), axis=np.arange(3)),
    )


def test_cell_data_array():
    _ = td.CellDataArray(
        [[0, 1, 2], [1, 2, 3]],
        coords=dict(cell_index=np.arange(2), vertex_index=np.arange(3)),
    )


def test_indexed_data_array():
    _ = td.IndexedDataArray(
        np.random.rand(10),
        coords=dict(index=np.arange(10)),
    )


def test_spatial_data_array():
    arr = td.SpatialDataArray(
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
        coords=dict(x=[0, 1], y=[1, 2], z=[2, 3]),
    )

    # make it non sorted
    arr = arr.isel(x=[1, 0], z=[1, 0])

    # reflection with a gap
    reflected = arr.reflect(axis=0, center=-0.5)

    reflected_expected = td.SpatialDataArray(
        [[[4, 5], [6, 7]], [[0, 1], [2, 3]], [[0, 1], [2, 3]], [[4, 5], [6, 7]]],
        coords=dict(x=[-2, -1, 0, 1], y=[1, 2], z=[2, 3]),
    )

    assert reflected == reflected_expected

    # reflection with a gap, only reflection
    reflected = arr.reflect(axis=0, center=-0.5, reflection_only=True)

    reflected_expected = td.SpatialDataArray(
        [[[4, 5], [6, 7]], [[0, 1], [2, 3]]],
        coords=dict(x=[-2, -1], y=[1, 2], z=[2, 3]),
    )

    assert reflected == reflected_expected

    # reflection with no gap
    reflected = arr.reflect(axis=1, center=1)

    reflected_expected = td.SpatialDataArray(
        [[[2, 3], [0, 1], [2, 3]], [[6, 7], [4, 5], [6, 7]]],
        coords=dict(x=[0, 1], y=[0, 1, 2], z=[2, 3]),
    )

    assert reflected == reflected_expected

    # reflection with no gap, only reflection
    reflected = arr.reflect(axis=1, center=1, reflection_only=True)

    reflected_expected = td.SpatialDataArray(
        [[[2, 3], [0, 1]], [[6, 7], [4, 5]]],
        coords=dict(x=[0, 1], y=[0, 1], z=[2, 3]),
    )

    assert reflected == reflected_expected

    with pytest.raises(DataError):
        reflected = arr.reflect(axis=2, center=2.5)


@pytest.mark.parametrize("nx", [10, 1])
def test_sel_inside(nx):
    ny = 11
    nz = 12
    arr = td.SpatialDataArray(
        np.random.random((nx, ny, nz)),
        coords=dict(
            x=np.linspace(0, 1, nx),
            y=np.linspace(2, 3, ny),
            z=np.linspace(0, 2, nz),
        ),
    )

    bounds_small = [[0.1, 2, 2], [1, 2.5, 2]]
    bounds_large = [[0.1, 2, 2], [1, 4, 2]]
    assert arr.does_cover(bounds_small)
    assert not arr.does_cover(bounds_large)

    arr_selected = arr.sel_inside(bounds_small)
    assert arr_selected.does_cover(bounds_small)

    arr_selected = arr.sel_inside(bounds_large)
    assert not arr_selected.does_cover(bounds_large)

    with pytest.raises(DataError):
        _ = arr.does_cover([[0.1, 3, 2], [1, 2.5, 2]])


def test_uniform_check():
    """check if each element in the array is of equal value."""
    arr = td.SpatialDataArray(
        np.ones((2, 2, 2), dtype=np.complex128),
        coords=dict(x=[0, 1], y=[1, 2], z=[2, 3]),
    )
    assert arr.is_uniform

    # small variation is still considered as uniform
    arr = td.SpatialDataArray(
        np.ones((2, 2, 2)) + np.random.random((2, 2, 2)) * 1e-6,
        coords=dict(x=[0, 1], y=[1, 2], z=[2, 3]),
    )
    assert arr.is_uniform

    arr = td.SpatialDataArray(
        np.ones((2, 2, 2)) + np.random.random((2, 2, 2)) * 1e-4,
        coords=dict(x=[0, 1], y=[1, 2], z=[2, 3]),
    )
    assert not arr.is_uniform


@pytest.mark.parametrize("method", ["nearest", "linear"])
@pytest.mark.parametrize("scalar_index", [True, False])
def test_interp(method, scalar_index):
    data = make_scalar_field_data_array("Ex")

    f = 1.5e14
    if not scalar_index:
        f = [f]

    xr_interp = data.interp(f=f)
    ag_interp = data._ag_interp(f=f)
    xrt.assert_allclose(xr_interp, ag_interp)
