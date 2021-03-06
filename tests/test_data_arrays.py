"""Tests tidy3d/components/data/data_array.py"""
import pytest
import numpy as np
from typing import Tuple, List

from tidy3d.components.data.data_array import ScalarFieldDataArray, ScalarFieldTimeDataArray
from tidy3d.components.data.data_array import ScalarModeFieldDataArray
from tidy3d.components.data.data_array import ModeAmpsDataArray, ModeIndexDataArray
from tidy3d.components.data.data_array import FluxDataArray, FluxTimeDataArray
from tidy3d.components.source import PointDipole, GaussianPulse
from tidy3d.components.simulation import Simulation
from tidy3d.components.grid import GridSpec
from tidy3d.components.mode import ModeSpec
from tidy3d.components.monitor import FieldMonitor, FieldTimeMonitor, PermittivityMonitor
from tidy3d.components.monitor import ModeSolverMonitor, ModeMonitor
from tidy3d.components.monitor import FluxMonitor, FluxTimeMonitor
from tidy3d.components.monitor import MonitorType
from tidy3d.components.structure import Structure
from tidy3d.components.geometry import Box
from tidy3d.material_library import material_library
from tidy3d.constants import inf

from .utils import clear_tmp

STRUCTURES = [
    Structure(geometry=Box(size=(1, inf, 1)), medium=material_library["cSi"]["SalzbergVilla1957"])
]
SOURCES = [PointDipole(source_time=GaussianPulse(freq0=1e14, fwidth=1e14), polarization="Ex")]
GRID_SPEC = GridSpec(wavelength=1.0)
RUN_TIME = 1e-12
SIZE_3D = (2, 4, 5)

SIZE_2D = list(SIZE_3D)
SIZE_2D[1] = 0
MODE_SPEC = ModeSpec(num_modes=4)
FREQS = [1e14, 2e14]
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

MONITORS = [
    FIELD_MONITOR,
    FIELD_TIME_MONITOR,
    MODE_SOLVE_MONITOR,
    PERMITTIVITY_MONITOR,
    MODE_MONITOR,
    FLUX_MONITOR,
    FLUX_TIME_MONITOR,
]

SIM = Simulation(
    size=SIZE_3D,
    run_time=RUN_TIME,
    grid_spec=GRID_SPEC,
    symmetry=(1, -1, 1),
    sources=SOURCES,
    monitors=MONITORS,
    structures=STRUCTURES,
)

FS = np.linspace(1e14, 2e14, 11)
TS = np.linspace(0, 1e-12, 11)
MODE_INDICES = np.arange(0, 3)
DIRECTIONS = ["+", "-"]

""" Generate the data arrays (used in other test files) """


def get_xyz(monitor: MonitorType, grid_key: str) -> Tuple[List[float], List[float], List[float]]:
    grid = SIM.discretize(monitor, extend=True)
    x, y, z = grid[grid_key].to_list
    x = [_x for _x in x if _x >= 0]
    y = [_y for _y in y if _y >= 0]
    z = [_z for _z in z if _z >= 0]
    return x, y, z


def make_scalar_field_data_array(grid_key: str):
    XS, YS, ZS = get_xyz(FIELD_MONITOR, grid_key)
    values = (1 + 1j) * np.random.random((len(XS), len(YS), len(ZS), len(FS)))
    return ScalarFieldDataArray(values, coords=dict(x=XS, y=YS, z=ZS, f=FS))


def make_scalar_field_time_data_array(grid_key: str):
    XS, YS, ZS = get_xyz(FIELD_TIME_MONITOR, grid_key)
    values = np.random.random((len(XS), len(YS), len(ZS), len(TS)))
    return ScalarFieldTimeDataArray(values, coords=dict(x=XS, y=YS, z=ZS, t=TS))


def make_scalar_mode_field_data_array(grid_key: str):
    XS, YS, ZS = get_xyz(MODE_SOLVE_MONITOR, grid_key)
    values = np.random.random((len(XS), 1, len(ZS), len(FS), len(MODE_INDICES)))

    return ScalarModeFieldDataArray(
        values, coords=dict(x=XS, y=[0.0], z=ZS, f=FS, mode_index=MODE_INDICES)
    )


def make_mode_amps_data_array():
    values = (1 + 1j) * np.random.random((len(DIRECTIONS), len(FS), len(MODE_INDICES)))
    return ModeAmpsDataArray(
        values, coords=dict(direction=DIRECTIONS, mode_index=MODE_INDICES, f=FS)
    )


def make_mode_index_data_array():
    f = np.linspace(2e14, 3e14, 1001)
    values = (1 + 1j) * np.random.random((len(FS), len(MODE_INDICES)))
    return ModeIndexDataArray(values, coords=dict(f=FS, mode_index=MODE_INDICES))


def make_flux_data_array():
    values = np.random.random(len(FS))
    return FluxDataArray(values, coords=dict(f=FS))


def make_flux_time_data_array():
    values = np.random.random(len(TS))
    return FluxTimeDataArray(values, coords=dict(t=TS))


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


def test_attrs():
    data = make_flux_data_array()
    assert data.attrs, "data has no attrs"
    assert data.f.attrs, "data coordinates have no attrs"


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
    data = ScalarFieldTimeDataArray(
        np.random.rand(5, 5, 5, 0),
        coords=dict(x=np.arange(5), y=np.arange(5), z=np.arange(5), t=[]),
    )
    data = ScalarFieldTimeDataArray(
        np.random.rand(5, 5, 5, 0),
        coords=dict(x=np.arange(5), y=np.arange(5), z=np.arange(5), t=[]),
    )
