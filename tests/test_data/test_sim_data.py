"""Tests SimulationData"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import colors as mcolors
from pydantic import ValidationError

import tidy3d as td
from tidy3d.components.data.data_array import ScalarFieldTimeDataArray
from tidy3d.components.data.monitor_data import FieldTimeData
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.file_util import replace_values
from tidy3d.components.monitor import FieldMonitor, FieldTimeMonitor, ModeMonitor
from tidy3d.exceptions import DataError, SetupError, Tidy3dKeyError

from ..test_components.test_mode import get_mode_sim_data
from ..utils import get_nested_shape
from .test_data_arrays import FIELD_MONITOR, SIM, SIM_SYM
from .test_monitor_data import (
    make_aux_field_time_data,
    make_diffraction_data,
    make_field_data,
    make_field_time_data,
    make_flux_data,
    make_flux_time_data,
    make_mode_data,
    make_mode_data_with_fields,
    make_mode_solver_data,
    make_permittivity_data,
)

# monitor data instances

FIELD_SYM = make_field_data()
FIELD = make_field_data(symmetry=False)
FIELD_TIME_SYM = make_field_time_data()
FIELD_TIME = make_field_time_data(symmetry=False)
AUX_FIELD_TIME_SYM = make_aux_field_time_data()
AUX_FIELD_TIME = make_aux_field_time_data(symmetry=False)
PERMITTIVITY_SYM = make_permittivity_data()
PERMITTIVITY = make_permittivity_data(symmetry=False)
MODE = make_mode_data()
MODE_WITH_FIELDS = make_mode_data_with_fields()
MODE_SOLVER = make_mode_solver_data()

FLUX = make_flux_data()
FLUX_TIME = make_flux_time_data()
DIFFRACTION = make_diffraction_data()

# for constructing SimulationData
MONITOR_DATA = (
    FIELD,
    FIELD_TIME,
    AUX_FIELD_TIME,
    MODE_WITH_FIELDS,
    PERMITTIVITY,
    MODE,
    FLUX,
    FLUX_TIME,
    DIFFRACTION,
    MODE_SOLVER,
)
MONITOR_DATA_SYM = (
    FIELD_SYM,
    FIELD_TIME_SYM,
    AUX_FIELD_TIME_SYM,
    MODE_WITH_FIELDS,
    PERMITTIVITY_SYM,
    MODE,
    FLUX,
    FLUX_TIME,
    DIFFRACTION,
    MODE_SOLVER,
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


def make_heat_charge_sim_data():
    """Create a simple HeatChargeSimulationData for testing."""
    temp_mnt = td.TemperatureMonitor(size=(1, 2, 3), name="temperature")

    tet_grid_points = td.PointDataArray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dims=("index", "axis"),
    )
    tet_grid_cells = td.CellDataArray(
        [[0, 1, 2, 4], [1, 2, 3, 4]],
        dims=("cell_index", "vertex_index"),
    )
    tet_grid_values = td.IndexedDataArray(
        np.linspace(300, 350, tet_grid_points.shape[0]),
        dims=("index",),
        name="T",
    )

    tet_grid = td.TetrahedralGridDataset(
        points=tet_grid_points,
        cells=tet_grid_cells,
        values=tet_grid_values,
    )

    temp_data = td.TemperatureData(monitor=temp_mnt, temperature=tet_grid)

    heat_sim = td.HeatChargeSimulation(
        size=(3.0, 3.0, 3.0),
        structures=[
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
                medium=td.Medium(
                    permittivity=2.0,
                    heat_spec=td.SolidSpec(conductivity=1, capacity=1),
                ),
                name="box",
            ),
        ],
        medium=td.Medium(permittivity=3.0, heat_spec=td.FluidSpec()),
        grid_spec=td.UniformUnstructuredGrid(dl=0.1),
        sources=[td.HeatSource(rate=1, structures=["box"])],
        boundary_spec=[
            td.HeatChargeBoundarySpec(
                placement=td.StructureBoundary(structure="box"),
                condition=td.TemperatureBC(temperature=500),
            )
        ],
        monitors=[temp_mnt],
    )

    return td.HeatChargeSimulationData(simulation=heat_sim, data=(temp_data,))


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
            "mode_with_fields", field_cmp, val="real", f=1e14, mode_index=1, phase=phase
        )
        plt.close()
    _ = sim_data.plot_field("mode_with_fields", "int", f=1e14, mode_index=1, phase=phase)
    plt.close()


def test_plot_field_custom_cmap():
    sim_data = make_sim_data()
    _ = sim_data.plot_field("field", "Ex", val="real", f=2e14, z=0.10, cmap="viridis")
    plt.close()
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("two", ["black", "white"])
    _ = sim_data.plot_field(
        "field",
        "Ez",
        val="imag",
        f=2e14,
        z=0.10,
        cmap=custom_cmap,
    )
    plt.close()


def test_plot_field_missing_derived_data():
    sim_data = make_sim_data()
    with pytest.raises(Tidy3dKeyError):
        sim_data.plot_field(field_monitor_name="field_time", field_name="E", val="int")


def test_plot_field_missing_field_value():
    sim_data = make_sim_data()
    with pytest.raises(Tidy3dKeyError):
        sim_data.plot_field(field_monitor_name="field", field_name="Ex", val="test")


@pytest.mark.parametrize("monitor_name", ["field", "field_time", "mode_with_fields"])
def test_intensity(monitor_name):
    sim_data = make_sim_data()
    _ = sim_data.get_intensity(monitor_name)


@pytest.mark.parametrize("monitor_name", ["field", "field_time", "mode_with_fields"])
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


def test_field_decay_array():
    sim_data = make_sim_data()
    log_extended = (
        sim_data.log + "\n- Time step   1392 / time 1.33e-13s ( 19 % done), field decay: 6.58e-01"
    )
    sim_data = sim_data.copy(update={"log": log_extended})

    fd = sim_data.field_decay
    steps = fd.coords["t"].values
    decay_values = fd.values

    assert len(fd) == 2
    assert int(steps[0]) == 827
    assert int(steps[1]) == 1392
    assert float(decay_values[0]) == 0.11
    assert float(decay_values[1]) == 0.658


def test_decay_missing_in_log():
    sim_data = make_sim_data()
    sim_data = sim_data.copy(update={"log": "no regex matches in this log"})
    dv = sim_data.final_decay_value
    assert dv == 1.0


def test_field_decay_log_none():
    sim_data = make_sim_data()
    sim_data = sim_data.copy(update={"log": None})
    with pytest.raises(DataError):
        _ = sim_data.field_decay


def test_to_dict():
    sim_data = make_sim_data()
    j = sim_data.model_dump()
    sim_data2 = SimulationData(**j)
    assert sim_data == sim_data2


def test_to_json(tmp_path):
    sim_data = make_sim_data()
    FNAME = str(tmp_path / "sim_data_refactor.json")
    sim_data.to_file(fname=FNAME)

    # saving to json does not store data, so trying to load from file will trigger custom error.
    with pytest.raises(ValidationError):
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
    sim_data.plot_field("mode_with_fields", "Ex", y=0.0, val="real", freq=1e14, mode_index=1)
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
    sim_data.plot_field("mode_with_fields", "Ex", y=0.0, val="real", f=1e14, mode_index=1)
    plt.close()

    # passing y=1 sel kwarg should error
    with pytest.raises(KeyError):
        sim_data.plot_field("mode_with_fields", "Ex", y=-1.0, val="real", f=1e14, mode_index=1)
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

    coords = {
        "x": np.linspace(-0.6, 0.6, 10),
        "y": np.linspace(-0.6, 0.6, 10),
        "z": [0.1],
        "t": [],
    }

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
    new_monitors = tuple(sim_data.simulation.monitors)[:-1]
    new_sim = sim_data.simulation.copy(update={"monitors": new_monitors})
    with pytest.raises(ValidationError):
        _ = sim_data.copy(update={"simulation": new_sim})


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
    test_data_dict = test_data.model_dump()
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


@pytest.mark.parametrize(
    "make_sim_data_fn",
    [make_sim_data, get_mode_sim_data, make_heat_charge_sim_data],
    ids=["SimulationData", "ModeSimulationData", "HeatChargeSimulationData"],
)
def test_to_mat_file(tmp_path, make_sim_data_fn):
    """
    Test output of ``.mat`` file completes without error for all simulation data types.
    """
    sim_data = make_sim_data_fn()
    path = str(tmp_path / "test.mat")
    sim_data.to_mat_file(path)


def test_plot_field_monitor_data_unsupported_scale():
    """Test plot_field_monitor_data with unsupported scale to trigger SetupError."""
    sim_data = make_sim_data()

    # Test with unsupported scale
    with pytest.raises(
        SetupError, match="The scale 'invalid' is not supported for plotting field data"
    ):
        sim_data.plot_field_monitor_data(
            field_monitor_data=sim_data.monitor_data["field"],
            field_name="Ex",
            val="real",
            scale="invalid",
        )


def test_plot_field_with_zeros_db_scale():
    """Test that plotting field data with zeros using dB scale doesn't produce warnings."""
    sim_data = make_sim_data()

    # Get the existing field data and modify it to include zeros
    field_monitor_data = sim_data["field"]
    ex_data = field_monitor_data.Ex

    # Create modified data with some zeros
    values = ex_data.values.copy()
    values[np.abs(values) < np.abs(values).max() * 0.3] = 0  # Set small values to zero
    ex_with_zeros = ex_data.copy(data=values)

    # Create new field data with zeros
    field_data_with_zeros = field_monitor_data.updated_copy(Ex=ex_with_zeros)
    f_sel = ex_data.f.values[0]
    x_sel = ex_data.x.values[0]

    # Plot with dB scale - this should not produce divide by zero warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="divide by zero")
        sim_data.plot_field_monitor_data(
            field_monitor_data=field_data_with_zeros,
            field_name="Ex",
            val="abs",
            scale="dB",
            f=f_sel,
            x=x_sel,
        )
        plt.close()

    # Also test with vmin specified (zeros replaced with floor value)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="divide by zero")
        sim_data.plot_field_monitor_data(
            field_monitor_data=field_data_with_zeros,
            field_name="Ex",
            val="abs",
            scale="dB",
            f=f_sel,
            x=x_sel,
            vmin=-50,
        )
        plt.close()
