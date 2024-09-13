"""Tests SimulationData"""

import matplotlib.pyplot as plt
import numpy as np
import pydantic.v1 as pydantic
import pytest
import tidy3d as td
from tidy3d.components.data.data_array import ScalarFieldTimeDataArray
from tidy3d.components.data.monitor_data import FieldTimeData
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.file_util import replace_values
from tidy3d.components.monitor import FieldMonitor, FieldTimeMonitor, ModeMonitor
from tidy3d.exceptions import DataError, Tidy3dKeyError

from ..utils import get_nested_shape
from .test_data_arrays import FIELD_MONITOR, SIM, SIM_SYM
from .test_monitor_data import (
    make_diffraction_data,
    make_field_data,
    make_field_time_data,
    make_flux_data,
    make_flux_time_data,
    make_mode_data,
    make_mode_solver_data,
    make_permittivity_data,
)

# monitor data instances

FIELD_SYM = make_field_data()
FIELD = make_field_data(symmetry=False)
FIELD_TIME_SYM = make_field_time_data()
FIELD_TIME = make_field_time_data(symmetry=False)
PERMITTIVITY_SYM = make_permittivity_data()
PERMITTIVITY = make_permittivity_data(symmetry=False)
MODE = make_mode_data()
MODE_SOLVER = make_mode_solver_data()
FLUX = make_flux_data()
FLUX_TIME = make_flux_time_data()
DIFFRACTION = make_diffraction_data()

# for constructing SimulationData
MONITOR_DATA = (FIELD, FIELD_TIME, MODE_SOLVER, PERMITTIVITY, MODE, FLUX, FLUX_TIME, DIFFRACTION)
MONITOR_DATA_SYM = (
    FIELD_SYM,
    FIELD_TIME_SYM,
    MODE_SOLVER,
    PERMITTIVITY_SYM,
    MODE,
    FLUX,
    FLUX_TIME,
    DIFFRACTION,
)
MONITOR_DATA_DICT = {data.monitor.name: data for data in MONITOR_DATA}
MONITOR_DATA_DICT_SYM = {data.monitor.name: data for data in MONITOR_DATA_SYM}


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
    _ = make_sim_data()


def test_apply_symmetry2():
    sim_data = make_sim_data()
    eps_raw = sim_data.monitor_data["permittivity"]
    shape_raw = eps_raw.eps_xx.shape

    eps_ret = sim_data["permittivity"]
    shape_ret = eps_ret.eps_xx.shape
    assert shape_raw != shape_ret


def test_apply_symmetry3():
    sim_data = make_sim_data()
    Ex_raw = sim_data.monitor_data["field"]
    shape_raw = Ex_raw.Ex.shape

    Ex_ret = sim_data["field"]
    shape_ret = Ex_ret.Ex.shape
    assert shape_raw != shape_ret


def test_no_symmetry():
    sim_data = make_sim_data(symmetry=False)
    Ex_raw = sim_data.monitor_data["field"].Ex
    Ex_ret = sim_data["field"].Ex
    assert np.allclose(Ex_raw, Ex_ret)


def test_normalize():
    sim_data_norm0 = make_sim_data()
    sim_data_norm_none = sim_data_norm0.renormalize(normalize_index=None)
    sim_data_norm1 = sim_data_norm_none.renormalize(normalize_index=1)
    sim_data_renorm0 = sim_data_norm1.renormalize(normalize_index=0)
    name = FIELD_MONITOR.name
    assert np.allclose(sim_data_norm0[name].Ex, sim_data_renorm0[name].Ex)
    assert not np.allclose(sim_data_norm0[name].Ex, sim_data_norm_none[name].Ex)
    assert not np.allclose(sim_data_norm0[name].Ex, sim_data_norm1[name].Ex)
    assert not np.allclose(sim_data_norm_none[name].Ex, sim_data_norm1[name].Ex)


def test_getitem():
    sim_data = make_sim_data()
    for mon in sim_data.simulation.monitors:
        _ = sim_data[mon.name]


def test_centers():
    sim_data = make_sim_data()
    for mon in sim_data.simulation.monitors:
        if isinstance(mon, (FieldMonitor, FieldTimeMonitor)) or (
            isinstance(mon, ModeMonitor) and mon.store_fields_direction
        ):
            _ = sim_data.at_centers(mon.name)


@pytest.mark.parametrize("phase", [0, 1.0])
def test_plot(phase):
    sim_data = make_sim_data()

    # plot regular field data
    for field_cmp in sim_data.simulation.get_monitor_by_name("field").fields:
        field_data = sim_data["field"].field_components[field_cmp]
        for axis_name in "xyz":
            xyz_kwargs = {axis_name: field_data.coords[axis_name][0]}
            _ = sim_data.plot_field(
                "field", field_cmp, val="imag", f=1e14, phase=phase, **xyz_kwargs
            )
            plt.close()
    for shading in ["gouraud", "nearest", "auto"]:
        _ = sim_data.plot_field(
            "field", field_cmp, val="imag", f=1e14, phase=phase, shading=shading, **xyz_kwargs
        )
    for axis_name in "xyz":
        xyz_kwargs = {axis_name: 0}
        _ = sim_data.plot_field("field", "int", f=1e14, phase=phase, **xyz_kwargs)
        plt.close()

    # plot field time data
    for field_cmp in sim_data.simulation.get_monitor_by_name("field_time").fields:
        field_data = sim_data["field_time"].field_components[field_cmp]
        for axis_name in "xyz":
            xyz_kwargs = {axis_name: field_data.coords[axis_name][0]}
            _ = sim_data.plot_field(
                "field_time", field_cmp, val="real", phase=phase, t=0.0, **xyz_kwargs
            )
            plt.close()
    for axis_name in "xyz":
        xyz_kwargs = {axis_name: 0}
        _ = sim_data.plot_field("field_time", "int", t=0.0, phase=phase, **xyz_kwargs)
        plt.close()

    # plot mode field data
    for field_cmp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        _ = sim_data.plot_field(
            "mode_solver", field_cmp, val="real", f=1e14, mode_index=1, phase=phase
        )
        plt.close()
    _ = sim_data.plot_field("mode_solver", "int", f=1e14, mode_index=1, phase=phase)
    plt.close()


def test_plot_field_missing_derived_data():
    sim_data = make_sim_data()
    with pytest.raises(Tidy3dKeyError):
        sim_data.plot_field(field_monitor_name="field_time", field_name="E", val="int")


def test_plot_field_missing_field_value():
    sim_data = make_sim_data()
    with pytest.raises(Tidy3dKeyError):
        sim_data.plot_field(field_monitor_name="field", field_name="Ex", val="test")


@pytest.mark.parametrize("monitor_name", ["field", "field_time", "mode_solver"])
def test_intensity(monitor_name):
    sim_data = make_sim_data()
    _ = sim_data.get_intensity(monitor_name)


@pytest.mark.parametrize("monitor_name", ["field", "field_time", "mode_solver"])
def test_poynting(monitor_name):
    sim_data = make_sim_data()
    mnt_data = sim_data[monitor_name]
    poynting = sim_data.get_poynting_vector(monitor_name)
    zero_dims = mnt_data.monitor.zero_dims
    if len(zero_dims) == 1:
        comp = f"S{'xyz'[zero_dims[0]]}"
        assert np.allclose(np.squeeze(poynting[comp].real), mnt_data.poynting)


def test_final_decay():
    sim_data = make_sim_data()
    dv = sim_data.final_decay_value
    assert dv == 0.11


def test_to_dict():
    sim_data = make_sim_data()
    j = sim_data.dict()
    sim_data2 = SimulationData(**j)
    assert sim_data == sim_data2


def test_to_json(tmp_path):
    sim_data = make_sim_data()
    FNAME = str(tmp_path / "sim_data_refactor.json")
    sim_data.to_file(fname=FNAME)

    # saving to json does not store data, so trying to load from file will trigger custom error.
    with pytest.raises(pydantic.ValidationError):
        _ = SimulationData.from_file(fname=FNAME)


@pytest.mark.filterwarnings("ignore:log10")
@pytest.mark.parametrize("field_name", ["Ex", "Ey", "Ez", "E", "Hx", "Hz", "Sy"])
@pytest.mark.parametrize("val", ["real", "re", "imag", "im", "abs", "phase"])
def test_derived_components(field_name, val):
    sim_data = make_sim_data()
    if len(field_name) == 1 and val == "phase":
        with pytest.raises(Tidy3dKeyError):
            sim_data.plot_field(
                "field_time",
                field_name=field_name,
                val=val,
                y=0.0,
                time=1e-12,
            )
    else:
        sim_data.plot_field(
            "field_time",
            field_name=field_name,
            val=val,
            y=0.0,
            time=1e-12,
        )
    plt.close()


def test_logscale():
    sim_data = make_sim_data()
    sim_data.plot_field("field_time", "Ex", val="real", scale="dB", y=0.0, time=1e-12)
    plt.close()


def test_sel_kwarg_freq():
    """Use freq in sel_kwarg, should still work (but warning) for 1.6.x"""
    sim_data = make_sim_data()
    sim_data.plot_field("mode_solver", "Ex", y=0.0, val="real", freq=1e14, mode_index=1)
    plt.close()


def test_sel_kwarg_time():
    """Use time in sel_kwarg, should still work (but warning) for 1.6.x"""
    sim_data = make_sim_data()
    sim_data.plot_field("field_time", "Ex", y=0.0, val="real", time=1e-12)
    plt.close()


def test_sel_kwarg_len1():
    sim_data = make_sim_data()

    # data has no y dimension (only exists at y=0)

    # passing y=0 sel kwarg should still work
    sim_data.plot_field("mode_solver", "Ex", y=0.0, val="real", f=1e14, mode_index=1)
    plt.close()

    # passing y=1 sel kwarg should error
    with pytest.raises(KeyError):
        sim_data.plot_field("mode_solver", "Ex", y=-1.0, val="real", f=1e14, mode_index=1)
        plt.close()


def test_to_hdf5(tmp_path):
    sim_data = make_sim_data()
    FNAME = str(tmp_path / "sim_data_refactor.hdf5")
    sim_data.to_file(fname=FNAME)
    sim_data2 = SimulationData.from_file(fname=FNAME)
    # Make sure that opening sim_data2 didn't lock the hdf5 file
    sim_data.to_file(fname=FNAME)
    # Make sure data is loaded as Tidy3dDataArray and not xarray DataArray
    for data, data2 in zip(sim_data.data, sim_data2.data):
        assert type(data) is type(data2)
    assert sim_data == sim_data2


def test_from_hdf5_group_path(tmp_path):
    """Tests that individual monitor data can be loaded from a SimulationData hdf5."""

    data = make_sim_data()
    FNAME = str(tmp_path / "sim_data.hdf5")
    data.to_file(fname=FNAME)
    for i, monitor_data in enumerate(data.data):
        group_name = data.get_tuple_group_name(index=i)
        group_path = f"data/{group_name}"
        loaded_data = type(monitor_data).from_file(fname=FNAME, group_path=group_path)
        assert loaded_data == monitor_data


def test_empty_io(tmp_path):
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    fields = {"Ex": td.ScalarFieldTimeDataArray(np.random.rand(10, 10, 10, 0), coords=coords)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), name="test", fields=["Ex"])
    sim = td.Simulation(
        size=(1, 1, 1),
        monitors=(monitor,),
        run_time=1e-12,
        grid_spec=td.GridSpec(wavelength=1.0),
        normalize_index=0,
    )
    field_data = td.FieldTimeData(
        monitor=monitor,
        symmetry=sim.symmetry,
        symmetry_center=sim.center,
        grid_expanded=sim.discretize_monitor(monitor),
        **fields,
    )
    sim_data = SimulationData(simulation=sim, data=(field_data,))
    sim_data.to_file(str(tmp_path / "sim_data_empty.hdf5"))
    sim_data = SimulationData.from_file(str(tmp_path / "sim_data_empty.hdf5"))
    field_data = sim_data[monitor.name]
    Ex = field_data.Ex
    assert Ex.size == 0


def test_run_time_lt_start(tmp_path):
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

    field_components = {
        field_name: ScalarFieldTimeDataArray(np.zeros((10, 10, 1, 0)), coords=coords)
        for field_name in tmnt.fields
    }

    field_data = FieldTimeData(
        monitor=tmnt,
        symmetry=sim.symmetry,
        symmetry_center=sim.center,
        grid_expanded=sim.discretize(tmnt, extend=True),
        **field_components,
    )

    sim_data = SimulationData(
        simulation=sim,
        data=(field_data,),
    )

    sim_data.renormalize(0).to_file(str(tmp_path / "sim_data_empty.hdf5"))
    sim_data = SimulationData.from_file(str(tmp_path / "sim_data_empty.hdf5"))
    _ = sim_data.monitor_data[tmnt.name]
    _ = sim_data[tmnt.name]


def test_plot_field_title():
    sim_data = make_sim_data()
    ax = sim_data.plot_field("field", "Ey", "real", f=2e14, z=0.10)
    assert "z=0.10" in ax.title.get_text(), "title rendered incorrectly."
    plt.close()


def test_missing_monitor():
    sim_data = make_sim_data()
    new_monitors = list(sim_data.simulation.monitors)[:-1]
    new_sim = sim_data.simulation.copy(update=dict(monitors=new_monitors))
    with pytest.raises(pydantic.ValidationError):
        _ = sim_data.copy(update=dict(simulation=new_sim))


def test_loading_non_field_data():
    sim_data = make_sim_data()
    with pytest.raises(DataError):
        sim_data.load_field_monitor("flux")


def test_replace_values_dict():
    """
    Test that replacing values in initial dict hasn't changed the shape and that values have been correctly replaced.
    Used in simData.to_mat() method.
    """
    # Create data for test
    test_data = make_sim_data()
    test_data_dict = test_data.dict()
    test_data_dict["none_test"] = None  # Check that replace works at top level of nested dict

    # Get the original shape of nested dict
    original_shape = get_nested_shape(test_data_dict)

    # Modify dict and get shape of dict
    new_data = replace_values(test_data_dict, None, [])
    new_shape = get_nested_shape(new_data)

    assert (
        original_shape == new_shape
        and new_data["simulation"]["medium"]["name"] == []
        and new_data["none_test"] == []
    )


def test_replace_values_list():
    """
    Test that replacing values in initial list hasn't changed the shape and that values have been correctly replaced.
    """
    # Create data for test
    test_list = [
        {"nest1": {"nest2": {"nest3": None}}},
        None,
        [5, "asdf", 12.34, None, False, [3.14, None]],
        3.14,
        (None, None, "notNone"),
    ]
    original_shape = get_nested_shape(test_list)

    # Modify list and get shape
    new_list = replace_values(test_list, 3.14, [])
    new_shape = get_nested_shape(new_list)

    assert original_shape == new_shape and new_list[3] == [] and new_list[2][5][0] == []


def test_to_mat_file(tmp_path):
    """
    Test output of ``.mat`` file completes without error.
    """
    sim_data = make_sim_data()
    path = str(tmp_path / "test.mat")
    sim_data.to_mat_file(path)
