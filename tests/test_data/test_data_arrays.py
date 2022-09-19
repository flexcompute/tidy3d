"""Tests tidy3d/components/data/data_array.py"""
import pytest
import numpy as np
from typing import Tuple, List
import xarray as xr

from tidy3d.components.data.data_array import ScalarFieldDataArray, ScalarFieldTimeDataArray
from tidy3d.components.data.data_array import ScalarModeFieldDataArray
from tidy3d.components.data.data_array import ModeAmpsDataArray, ModeIndexDataArray
from tidy3d.components.data.data_array import FluxDataArray, FluxTimeDataArray
from tidy3d.components.data.data_array import Near2FarAngleDataArray, Near2FarKSpaceDataArray
from tidy3d.components.data.data_array import Near2FarCartesianDataArray
from tidy3d.components.source import PointDipole, GaussianPulse, ModeSource
from tidy3d.components.simulation import Simulation
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d.components.mode import ModeSpec
from tidy3d.components.monitor import FieldMonitor, FieldTimeMonitor, PermittivityMonitor
from tidy3d.components.monitor import ModeSolverMonitor, ModeMonitor
from tidy3d.components.monitor import FluxMonitor, FluxTimeMonitor, Near2FarKSpaceMonitor
from tidy3d.components.monitor import Near2FarCartesianMonitor, Near2FarAngleMonitor
from tidy3d.components.monitor import MonitorType
from tidy3d.components.structure import Structure
from tidy3d.components.geometry import Box
from tidy3d.material_library import material_library
from tidy3d.constants import inf

from ..utils import clear_tmp

STRUCTURES = [
    Structure(geometry=Box(size=(1, inf, 1)), medium=material_library["cSi"]["SalzbergVilla1957"])
]
SIZE_3D = (2, 4, 5)
SIZE_2D = list(SIZE_3D)
SIZE_2D[1] = 0
MODE_SPEC = ModeSpec(num_modes=4)
FREQS = np.linspace(1e14, 2e14, 4)
SOURCES = [
    PointDipole(source_time=GaussianPulse(freq0=FREQS[0], fwidth=1e14), polarization="Ex"),
    ModeSource(
        size=SIZE_2D,
        mode_spec=MODE_SPEC,
        source_time=GaussianPulse(freq0=FREQS[1], fwidth=1e14),
        direction="+",
    ),
]
FIELDS = ("Ex", "Ey", "Ez", "Hz")
INTERVAL = 2
FIELD_MONITOR = FieldMonitor(size=SIZE_3D, fields=FIELDS, name="field", freqs=FREQS)
FIELD_TIME_MONITOR = FieldTimeMonitor(
    size=SIZE_3D, fields=FIELDS, name="field_time", interval=INTERVAL
)
MODE_SOLVE_MONITOR = ModeSolverMonitor(
    size=SIZE_2D, name="mode_solver", mode_spec=MODE_SPEC, freqs=FREQS
)
PERMITTIVITY_MONITOR = PermittivityMonitor(size=SIZE_3D, name="permittivity", freqs=FREQS)
MODE_MONITOR = ModeMonitor(size=SIZE_2D, name="mode", mode_spec=MODE_SPEC, freqs=FREQS)
FLUX_MONITOR = FluxMonitor(size=SIZE_2D, freqs=FREQS, name="flux")
FLUX_TIME_MONITOR = FluxTimeMonitor(size=SIZE_2D, interval=INTERVAL, name="flux_time")

YS = np.linspace(0, 5, 6)
XS = np.linspace(0, 10, 4)
THETA = np.linspace(0, np.pi, 3)
PHI = np.linspace(0, 2 * np.pi, 7)
UX = np.linspace(0, 5, 2)
UY = np.linspace(0, 10, 4)
N2F_CARTESIAN_MONITOR = Near2FarCartesianMonitor(
    size=SIZE_2D, freqs=FREQS, plane_axis=2, plane_distance=10, x=XS, y=YS, name="n2f_cart"
)
N2F_ANGLE_MONITOR = Near2FarAngleMonitor(
    size=SIZE_2D, freqs=FREQS, theta=THETA, phi=PHI, name="n2f_angle"
)
N2F_KSPACE_MONITOR = Near2FarKSpaceMonitor(
    size=SIZE_2D, freqs=FREQS, u_axis=2, ux=UX, uy=UY, name="n2f_kspace"
)

MONITORS = [
    FIELD_MONITOR,
    FIELD_TIME_MONITOR,
    PERMITTIVITY_MONITOR,
    MODE_SOLVE_MONITOR,
    MODE_MONITOR,
    FLUX_MONITOR,
    FLUX_TIME_MONITOR,
    N2F_CARTESIAN_MONITOR,
    N2F_ANGLE_MONITOR,
    N2F_KSPACE_MONITOR,
]

GRID_SPEC = GridSpec(wavelength=4.0)
RUN_TIME = 1e-12

SIM_SYM = Simulation(
    size=SIZE_3D,
    run_time=RUN_TIME,
    grid_spec=GRID_SPEC,
    symmetry=(1, -1, 1),
    sources=SOURCES,
    monitors=MONITORS,
    structures=STRUCTURES,
)

SIM = Simulation(
    size=SIZE_3D,
    run_time=RUN_TIME,
    grid_spec=GRID_SPEC,
    symmetry=(0, 0, 0),
    sources=SOURCES,
    monitors=MONITORS,
    structures=STRUCTURES,
)

TS = np.linspace(0, 1e-12, 11)
MODE_INDICES = np.arange(0, 3)
DIRECTIONS = ["+", "-"]

""" Generate the data arrays (used in other test files) """


def get_xyz(
    monitor: MonitorType, grid_key: str, symmetry: bool
) -> Tuple[List[float], List[float], List[float]]:
    if symmetry:
        grid = SIM_SYM.discretize(monitor, extend=True)
    else:
        grid = SIM.discretize(monitor, extend=True)
    x, y, z = grid[grid_key].to_list
    x = [_x for _x in x if _x >= 0]
    y = [_y for _y in y if _y >= 0]
    z = [_z for _z in z if _z >= 0]
    return x, y, z


def make_scalar_field_data_array(grid_key: str, symmetry=True):
    XS, YS, ZS = get_xyz(FIELD_MONITOR, grid_key, symmetry)
    values = (1 + 1j) * np.random.random((len(XS), len(YS), len(ZS), len(FREQS)))
    data_array = xr.DataArray(values, coords=dict(x=XS, y=YS, z=ZS, f=FREQS))
    return ScalarFieldDataArray(data=data_array)


def make_scalar_field_time_data_array(grid_key: str, symmetry=True):
    XS, YS, ZS = get_xyz(FIELD_TIME_MONITOR, grid_key, symmetry)
    values = np.random.random((len(XS), len(YS), len(ZS), len(TS)))
    data_array = xr.DataArray(values, coords=dict(x=XS, y=YS, z=ZS, t=TS))
    return ScalarFieldTimeDataArray(data=data_array)


def make_scalar_mode_field_data_array(grid_key: str, symmetry=True):
    XS, YS, ZS = get_xyz(MODE_SOLVE_MONITOR, grid_key, symmetry)
    values = np.random.random((len(XS), 1, len(ZS), len(FREQS), len(MODE_INDICES)))

    data_array = xr.DataArray(
        values, coords=dict(x=XS, y=[0.0], z=ZS, f=FREQS, mode_index=MODE_INDICES)
    )
    return ScalarModeFieldDataArray(data=data_array)


def make_mode_amps_data_array():
    values = (1 + 1j) * np.random.random((len(DIRECTIONS), len(FREQS), len(MODE_INDICES)))
    data_array = xr.DataArray(
        values, coords=dict(direction=DIRECTIONS, f=FREQS, mode_index=MODE_INDICES)
    )
    return ModeAmpsDataArray(data=data_array)


def make_mode_index_data_array():
    f = np.linspace(2e14, 3e14, 1001)
    values = (1 + 1j) * np.random.random((len(FREQS), len(MODE_INDICES)))
    data_array = xr.DataArray(values, coords=dict(f=FREQS, mode_index=MODE_INDICES))
    return ModeIndexDataArray(data=data_array)


def make_flux_data_array():
    values = np.random.random(len(FREQS))
    data_array = xr.DataArray(values, coords=dict(f=FREQS))
    return FluxDataArray(data=data_array)


def make_flux_time_data_array():
    values = np.random.random(len(TS))
    data_array = xr.DataArray(values, coords=dict(t=TS))
    return FluxTimeDataArray(data=data_array)


def make_n2f_angle_data_array():
    coords_tp = dict(theta=THETA, phi=PHI, f=FREQS)
    values_tp = (1 + 1j) * np.random.random((len(THETA), len(PHI), len(FREQS)))
    data_array = xr.DataArray(values_tp, coords=coords_tp)
    return Near2FarAngleDataArray(data=data_array)


def make_n2f_cartesian_data_array():
    coords_xy = dict(x=XS, y=YS, f=FREQS)
    values_xy = (1 + 1j) * np.random.random((len(XS), len(YS), len(FREQS)))
    data_array = xr.DataArray(values_xy, coords=coords_xy)
    return Near2FarCartesianDataArray(data=data_array)


def make_n2f_kspace_data_array():
    coords_u = dict(ux=UX, uy=UY, f=FREQS)
    values_u = (1 + 1j) * np.random.random((len(UX), len(UY), len(FREQS)))
    data_array = xr.DataArray(values_u, coords=coords_u)
    return Near2FarKSpaceDataArray(data=data_array)


ALL_DATA_ARRAYS = [
    make_scalar_field_data_array("Ex"),
    make_scalar_field_time_data_array("Ex"),
    make_scalar_mode_field_data_array("Ex"),
    make_mode_amps_data_array(),
    make_mode_index_data_array(),
    make_flux_data_array(),
    make_flux_time_data_array(),
    make_n2f_angle_data_array(),
    make_n2f_cartesian_data_array(),
    make_n2f_kspace_data_array(),
]

""" Test that they work """


def test_scalar_field_data_array():
    for grid_key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        data = make_scalar_field_data_array(grid_key)
        data = data.data.interp(f=1.5e14)
        _ = data.isel(y=2)


def test_scalar_field_time_data_array():
    for grid_key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        data = make_scalar_field_time_data_array(grid_key)
        data = data.data.interp(t=1e-13)
        _ = data.isel(y=2)


def test_scalar_mode_field_data_array():
    for grid_key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        data = make_scalar_mode_field_data_array(grid_key)
        data = data.data.interp(f=1.5e14)
        data = data.isel(x=2)
        _ = data.sel(mode_index=2)


def test_mode_amps_data_array():
    data = make_mode_amps_data_array()
    data = data.data.interp(f=1.5e14)
    data = data.isel(direction=0)
    _ = data.sel(mode_index=1)


def test_mode_index_data_array():
    data = make_mode_index_data_array()
    data = data.data.interp(f=1.5e14)
    _ = data.sel(mode_index=1)


def test_flux_data_array():
    data = make_flux_data_array()
    _ = data.data.interp(f=1.5e14)


def test_flux_time_data_array():
    data = make_flux_time_data_array()
    _ = data.data.interp(t=1e-13)


def test_n2f_angle_data_array():
    data = make_n2f_angle_data_array()
    _ = data.data.interp(phi=0.0)


def test_n2f_cartesian_data_array():
    data = make_n2f_cartesian_data_array()
    _ = data.data.interp(x=0.0)


def test_n2f_kspace_data_array():
    data = make_n2f_kspace_data_array()
    _ = data.data.interp(ux=0.0)


def test_attrs():
    data = make_flux_data_array()
    assert data.data.attrs, "data has no attrs"
    assert data.data.f.attrs, "data coordinates have no attrs"


def test_ops():
    data1 = make_flux_data_array()
    data2 = make_flux_data_array()
    data1.data.data = np.ones_like(data1.data)
    data2.data.data = np.ones_like(data2.data)
    data3 = make_flux_time_data_array()
    assert np.all(data1 == data2), "identical data are not equal"
    data1.data[0] = 1e12
    assert not np.all(data1.data == data2.data), "different data are equal"
    assert not np.all(data1.data == data3.data), "different data are equal"


def test_empty_field_time():
    coords = coords = dict(x=np.arange(5), y=np.arange(5), z=np.arange(5), t=[])
    data_array = xr.DataArray(np.random.rand(5, 5, 5, 0), coords=coords)
    data = ScalarFieldTimeDataArray(data=data_array)


def test_abs():
    data = make_mode_amps_data_array()
    dabs = abs(data.data)


@pytest.mark.parametrize("data_array", ALL_DATA_ARRAYS)
def test_json(data_array):

    FNAME = "tests/tmp/data_array.json"
    data_array.to_file(FNAME)
    da2 = data_array.from_file(FNAME)
    # if data_array != da2:
    # import pdb; pdb.set_trace()
    assert data_array == da2


@clear_tmp
def test_json_clear_tmp():
    """Clear tmp after above function runs (decorators dont play well together)"""
    pass
