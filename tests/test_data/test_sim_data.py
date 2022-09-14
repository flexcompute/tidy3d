"""Tests SimulationData"""
import pytest
import numpy as np
import xarray as xr

import tidy3d as td
from tidy3d.log import DataError

from tidy3d.components.simulation import Simulation
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d.components.data.data_array import ScalarFieldTimeDataArray
from tidy3d.components.data.dataset import FieldTimeData
from tidy3d.components.data.monitor_data import FieldTimeMonitorData
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.monitor import FieldMonitor, FieldTimeMonitor, ModeSolverMonitor
from tidy3d.components.source import GaussianPulse, PointDipole

from .test_monitor_data import make_field_monitor_data, make_field_time_monitor_data
from .test_monitor_data import make_permittivity_monitor_data
from .test_monitor_data import make_mode_monitor_data, make_mode_solver_monitor_data
from .test_monitor_data import make_flux_monitor_data, make_flux_time_monitor_data

from .test_data_arrays import FIELD_MONITOR, FIELD_TIME_MONITOR, MODE_SOLVE_MONITOR
from .test_data_arrays import MODE_MONITOR, PERMITTIVITY_MONITOR, FLUX_MONITOR, FLUX_TIME_MONITOR
from .test_data_arrays import SIM, SIM_SYM

from ..utils import clear_tmp

# monitor data instances

FIELD_SYM = make_field_monitor_data()
FIELD = make_field_monitor_data(symmetry=False)
FIELD_TIME_SYM = make_field_time_monitor_data()
FIELD_TIME = make_field_time_monitor_data(symmetry=False)
PERMITTIVITY_SYM = make_permittivity_monitor_data()
PERMITTIVITY = make_permittivity_monitor_data(symmetry=False)
MODE = make_mode_monitor_data()
MODE_SOLVER = make_mode_solver_monitor_data()
FLUX = make_flux_monitor_data()
FLUX_TIME = make_flux_time_monitor_data()

# for constructing SimulationData
MONITOR_DATA = (FIELD, FIELD_TIME, MODE_SOLVER, PERMITTIVITY, MODE, FLUX, FLUX_TIME)
MONITOR_DATA_SYM = (FIELD_SYM, FIELD_TIME_SYM, MODE_SOLVER, PERMITTIVITY_SYM, MODE, FLUX, FLUX_TIME)


def make_sim_data(symmetry: bool = True):
    if symmetry:
        simulation = SIM_SYM
        data = MONITOR_DATA_SYM
    else:
        simulation = SIM
        data = MONITOR_DATA
    return SimulationData(
        simulation=simulation,
        data=data,
        log="- Time step    827 / time 4.13e-14s (  4 % done), field decay: 0.110e+00",
    )


def test_sim_data():
    sim_data = make_sim_data()


def test_apply_symmetry():
    sim_data = make_sim_data()
    for monitor_data in sim_data.data:
        _ = sim_data.apply_symmetry(monitor_data)


def test_apply_symmetry2():
    sim_data = make_sim_data()
    eps_raw = sim_data.monitor_data["permittivity"]
    shape_raw = eps_raw.dataset.eps_xx.data.shape

    eps_ret = sim_data["permittivity"]
    shape_ret = eps_ret.dataset.eps_xx.data.shape
    assert shape_raw != shape_ret


def test_apply_symmetry3():
    sim_data = make_sim_data()
    Ex_raw = sim_data.monitor_data["field"]
    shape_raw = Ex_raw.dataset.Ex.data.shape

    Ex_ret = sim_data["field"]
    shape_ret = Ex_ret.dataset.Ex.data.shape
    assert shape_raw != shape_ret


def test_no_symmetry():
    sim_data = make_sim_data(symmetry=False)
    Ex_raw = sim_data.monitor_data["field"].dataset.Ex.data
    Ex_ret = sim_data["field"].dataset.Ex.data
    assert np.allclose(Ex_raw, Ex_ret)


def test_normalize():
    sim_data_norm0 = make_sim_data()
    sim_data_norm_none = sim_data_norm0.renormalize(normalize_index=None)
    sim_data_norm1 = sim_data_norm_none.renormalize(normalize_index=1)
    sim_data_renorm0 = sim_data_norm1.renormalize(normalize_index=0)
    name = FIELD_MONITOR.name
    assert np.allclose(sim_data_norm0[name].dataset.Ex.data, sim_data_renorm0[name].dataset.Ex.data)
    assert not np.allclose(
        sim_data_norm0[name].dataset.Ex.data, sim_data_norm_none[name].dataset.Ex.data
    )
    assert not np.allclose(
        sim_data_norm0[name].dataset.Ex.data, sim_data_norm1[name].dataset.Ex.data
    )
    assert not np.allclose(
        sim_data_norm_none[name].dataset.Ex.data, sim_data_norm1[name].dataset.Ex.data
    )


def test_getitem():
    sim_data = make_sim_data()
    for mon in sim_data.simulation.monitors:
        data = sim_data[mon.name]


def test_centers():
    sim_data = make_sim_data()
    for mon in sim_data.simulation.monitors:
        if isinstance(mon, (FieldMonitor, FieldTimeMonitor, ModeSolverMonitor)):
            data = sim_data.at_centers(mon.name)


def test_plot():
    sim_data = make_sim_data()
    ax = None

    # plot regular field data
    for field_cmp in sim_data.simulation.get_monitor_by_name("field").fields:
        field_data = sim_data["field"].dataset.field_components[field_cmp]
        for axis_name in "xyz":
            xyz_kwargs = {axis_name: field_data.data.coords[axis_name][0]}
            ax = sim_data.plot_field("field", field_cmp, val="real", f=1e14, ax=ax, **xyz_kwargs)
    for axis_name in "xyz":
        xyz_kwargs = {axis_name: 0}
        ax = sim_data.plot_field("field", "int", f=1e14, ax=ax, **xyz_kwargs)

    # plot field time data
    for field_cmp in sim_data.simulation.get_monitor_by_name("field_time").fields:
        field_data = sim_data["field_time"].dataset.field_components[field_cmp]
        for axis_name in "xyz":
            xyz_kwargs = {axis_name: field_data.data.coords[axis_name][0]}
            ax = sim_data.plot_field(
                "field_time", field_cmp, val="real", t=0.0, ax=ax, **xyz_kwargs
            )
    for axis_name in "xyz":
        xyz_kwargs = {axis_name: 0}
        ax = sim_data.plot_field("field_time", "int", t=0.0, ax=ax, **xyz_kwargs)

    # plot mode field data
    for field_cmp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        ax = sim_data.plot_field("mode_solver", field_cmp, val="real", f=1e14, mode_index=1, ax=ax)
    ax = sim_data.plot_field("mode_solver", "int", f=1e14, mode_index=1, ax=ax)


def test_intensity():
    sim_data = make_sim_data()
    for monitor_name in ["field", "field_time", "mode_solver"]:
        _ = sim_data.get_intensity(monitor_name)


def test_final_decay():
    sim_data = make_sim_data()
    dv = sim_data.final_decay_value
    assert dv == 0.11


def test_to_dict():
    sim_data = make_sim_data()
    j = sim_data.dict()
    sim_data2 = SimulationData(**j)
    assert np.all(sim_data == sim_data2)


@clear_tmp
def test_to_json():
    sim_data = make_sim_data()
    FNAME = "tests/tmp/sim_data_refactor.json"
    DATA_FILE = "tests/tmp/sim_extra_data.hdf5"
    sim_data.to_file(fname=FNAME, data_file=DATA_FILE)
    sim_data2 = SimulationData.from_file(fname=FNAME)
    assert sim_data == sim_data2


def test_sel_kwarg_freq():
    """Use freq in sel_kwarg, should still work (but warning) for 1.6.x"""
    sim_data = make_sim_data()
    sim_data.plot_field("mode_solver", "Ex", y=0.0, val="real", freq=1e14, mode_index=1)


def test_sel_kwarg_time():
    """Use time in sel_kwarg, should still work (but warning) for 1.6.x"""
    sim_data = make_sim_data()
    sim_data.plot_field("field_time", "Ex", y=0.0, val="real", time=1e-12)


def test_sel_kwarg_len1():
    sim_data = make_sim_data()

    # data has no y dimension (only exists at y=0)

    # passing y=0 sel kwarg should still work
    sim_data.plot_field("mode_solver", "Ex", y=0.0, val="real", f=1e14, mode_index=1)

    # passing y=1 sel kwarg should error
    with pytest.raises(KeyError):
        sim_data.plot_field("mode_solver", "Ex", y=1.0, val="real", f=1e14, mode_index=1)


@clear_tmp
def test_to_hdf5():
    sim_data = make_sim_data()
    FNAME = "tests/tmp/sim_data_refactor.hdf5"
    sim_data.to_file(fname=FNAME)
    sim_data2 = SimulationData.from_file(fname=FNAME)
    assert sim_data == sim_data2


# @clear_tmp
def test_empty_io():
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    data_array = xr.DataArray(np.random.rand(10, 10, 10, 0), coords=coords)
    fields = {"Ex": td.ScalarFieldTimeDataArray(data=data_array)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), name="test", fields=["Ex"])
    dataset = td.FieldTimeData(**fields)
    field_data = td.FieldTimeMonitorData(monitor=monitor, dataset=dataset)
    sim = td.Simulation(
        size=(1, 1, 1),
        monitors=(monitor,),
        run_time=1e-12,
        grid_spec=td.GridSpec(wavelength=1.0),
        normalize_index=0,
    )
    sim_data = SimulationData(simulation=sim, data=[field_data])
    sim_data.to_file("tests/tmp/sim_data_empty.hdf5")
    sim_data = SimulationData.from_file("tests/tmp/sim_data_empty.hdf5")
    field_data = sim_data["test"]
    Ex = field_data.dataset.Ex.data
    assert Ex.size == 0


@clear_tmp
def test_run_time_lt_start():

    # Point source inside a box
    box = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)),
        medium=td.Medium(permittivity=10, conductivity=0.0),
    )

    tmnt = td.FieldTimeMonitor(
        center=(0.0, 0.0, 0.1),
        size=(1.2, 1.2, 0.0),
        name="tmnt",
        start=1e-13,
        stop=None,
        interval=1,
        fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        interval_space=(1, 1, 1),
        colocate=False,
    )

    sim = td.Simulation(
        size=(2, 2, 2),
        run_time=1e-50,
        grid_spec=td.GridSpec(
            grid_x=td.UniformGrid(dl=1 / 20),
            grid_y=td.UniformGrid(dl=1 / 22),
            grid_z=td.UniformGrid(dl=1 / 24),
        ),
        structures=(box,),
        monitors=(tmnt,),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        normalize_index=None,
    )

    coords = dict(
        x=np.linspace(-0.6, 0.6, 10),
        y=np.linspace(-0.6, 0.6, 10),
        z=[0.1],
        t=[],
    )

    data_array = xr.DataArray(np.zeros((10, 10, 1, 0)), coords=coords)

    field_components = {
        field_name: ScalarFieldTimeDataArray(data=data_array) for field_name in tmnt.fields
    }

    field_data = FieldTimeData(**field_components)
    field_mnt_data = FieldTimeMonitorData(monitor=tmnt, dataset=field_data)

    sim_data = SimulationData(
        simulation=sim,
        data=[field_mnt_data],
    )

    sim_data.renormalize(0).to_file("tests/tmp/sim_data_empty.hdf5")
    sim_data = SimulationData.from_file("tests/tmp/sim_data_empty.hdf5")
    tmnt_data = sim_data.monitor_data[tmnt.name]
    tmnt_data = sim_data[tmnt.name]


def test_plot_field_title():
    sim_data = make_sim_data()
    ax = sim_data.plot_field("field", "Ey", "real", f=2e14, z=0.10)
    assert "z=0.10" in ax.title.get_text(), "title rendered incorrectly."
