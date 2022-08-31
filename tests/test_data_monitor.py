"""Tests tidy3d/components/data/monitor_data.py"""
import numpy as np
import pytest

import tidy3d as td

from tidy3d.components.monitor import FieldMonitor, FieldTimeMonitor, PermittivityMonitor
from tidy3d.components.monitor import ModeSolverMonitor, ModeMonitor
from tidy3d.components.monitor import FluxMonitor, FluxTimeMonitor
from tidy3d.components.mode import ModeSpec
from tidy3d.log import DataError

from tidy3d.components.data.monitor_data import FieldData, FieldTimeData, PermittivityData
from tidy3d.components.data.monitor_data import ModeSolverData, ModeData
from tidy3d.components.data.monitor_data import FluxData, FluxTimeData, DiffractionData

from .test_data_arrays import make_scalar_field_data_array, make_scalar_field_time_data_array
from .test_data_arrays import make_scalar_mode_field_data_array
from .test_data_arrays import make_flux_data_array, make_flux_time_data_array
from .test_data_arrays import make_mode_amps_data_array, make_mode_index_data_array
from .test_data_arrays import make_diffraction_data_array
from .test_data_arrays import FIELD_MONITOR, FIELD_TIME_MONITOR, MODE_SOLVE_MONITOR
from .test_data_arrays import MODE_MONITOR, PERMITTIVITY_MONITOR, FLUX_MONITOR, FLUX_TIME_MONITOR
from .test_data_arrays import DIFFRACTION_MONITOR
from .utils import clear_tmp

# data array instances
AMPS = make_mode_amps_data_array()
N_COMPLEX = make_mode_index_data_array()
FLUX = make_flux_data_array()
FLUX_TIME = make_flux_time_data_array()

""" Make the montor data """


def make_field_data(symmetry: bool = True):
    return FieldData(
        monitor=FIELD_MONITOR,
        Ex=make_scalar_field_data_array("Ex", symmetry),
        Ey=make_scalar_field_data_array("Ey", symmetry),
        Ez=make_scalar_field_data_array("Ez", symmetry),
        Hz=make_scalar_field_data_array("Hz", symmetry),
    )


def make_field_time_data(symmetry: bool = True):
    return FieldTimeData(
        monitor=FIELD_TIME_MONITOR,
        Ex=make_scalar_field_time_data_array("Ex", symmetry),
        Ey=make_scalar_field_time_data_array("Ey", symmetry),
        Ez=make_scalar_field_time_data_array("Ez", symmetry),
        Hz=make_scalar_field_time_data_array("Ez", symmetry),
    )


def make_mode_solver_data():
    return ModeSolverData(
        monitor=MODE_SOLVE_MONITOR,
        Ex=make_scalar_mode_field_data_array("Ex"),
        Ey=make_scalar_mode_field_data_array("Ey"),
        Ez=make_scalar_mode_field_data_array("Ez"),
        Hx=make_scalar_mode_field_data_array("Hx"),
        Hy=make_scalar_mode_field_data_array("Hy"),
        Hz=make_scalar_mode_field_data_array("Hz"),
        n_complex=N_COMPLEX.copy(),
    )


def make_permittivity_data(symmetry: bool = True):
    return PermittivityData(
        monitor=PERMITTIVITY_MONITOR,
        eps_xx=make_scalar_field_data_array("Ex", symmetry),
        eps_yy=make_scalar_field_data_array("Ey", symmetry),
        eps_zz=make_scalar_field_data_array("Ez", symmetry),
    )


def make_mode_data():
    return ModeData(monitor=MODE_MONITOR, amps=AMPS.copy(), n_complex=N_COMPLEX.copy())


def make_flux_data():
    return FluxData(monitor=FLUX_MONITOR, flux=FLUX.copy())


def make_flux_time_data():
    return FluxTimeData(monitor=FLUX_TIME_MONITOR, flux=FLUX_TIME.copy())


def make_diffraction_data():
    sim_size, bloch_vecs, data = make_diffraction_data_array()
    return DiffractionData(
        monitor=DIFFRACTION_MONITOR,
        L=data,
        N=data,
        sim_size=sim_size,
        bloch_vecs=bloch_vecs,
    )


""" Test them out """


def test_field_data():
    data = make_field_data()
    for field in FIELD_MONITOR.fields:
        _ = getattr(data, field)


def test_field_time_data():
    data = make_field_time_data()
    for field in FIELD_TIME_MONITOR.fields:
        _ = getattr(data, field)


def test_mode_field_data():
    data = make_mode_solver_data()
    for field in "EH":
        for component in "xyz":
            _ = getattr(data, field + component)


def test_permittivity_data():
    data = make_permittivity_data()
    for comp in "xyz":
        _ = getattr(data, "eps_" + comp + comp)


def test_mode_data():
    data = make_mode_data()
    _ = data.amps
    _ = data.n_complex


def test_flux_data():
    data = make_flux_data()
    _ = data.flux


def test_flux_time_data():
    data = make_flux_time_data()
    _ = data.flux


def test_diffraction_data():
    data = make_diffraction_data()
    _ = data.L
    _ = data.N
    _ = data.orders_x
    _ = data.orders_y
    _ = data.frequencies
    _ = data.wavelength
    _ = data.wavenumber
    _ = data.ux
    _ = data.uy
    _ = data.angles
    _ = data.sim_size
    _ = data.bloch_vecs
    _ = data.power
    _ = data.amps
    _ = data.E_sph
    _ = data.H_sph
    _ = data.E_car
    _ = data.H_car
    _ = data.L_sph
    _ = data.N_sph


def test_colocate():
    # TODO: can we colocate into regions where we dont store fields due to symmetry?
    # regular colocate
    data = make_field_data()
    _ = data.colocate(x=[+0.1, 0.5], y=[+0.1, 0.5], z=[+0.1, 0.5])

    # ignore coordinate
    _ = data.colocate(x=[+0.1, 0.5], y=None, z=[+0.1, 0.5])

    # data outside range of len(coord)==1 dimension
    data = make_mode_solver_data()
    with pytest.raises(DataError):
        _ = data.colocate(x=[+0.1, 0.5], y=1.0, z=[+0.1, 0.5])

    with pytest.raises(DataError):
        _ = data.colocate(x=[+0.1, 0.5], y=[1.0, 2.0], z=[+0.1, 0.5])


def test_sel_mode_index():

    data = make_mode_solver_data()
    field_data = data.sel_mode_index(mode_index=0)
    assert isinstance(field_data, FieldData), "ModeSolverData wasnt converted to FieldData."
    assert isinstance(
        field_data.monitor, FieldMonitor
    ), "ModeSolverMonitor wasnt converted to FieldMonitor."
    for _, scalar_field in field_data.field_components.items():
        assert "mode_index" not in scalar_field.coords, "mode_index coordinate remained in data."


def _test_eq():
    data1 = make_flux_data()
    data2 = make_flux_data()
    data1.flux.data = np.ones_like(data1.flux.data)
    data2.flux.data = np.ones_like(data2.flux.data)
    data3 = make_flux_time_data_array()
    assert data1 == data2, "same data are not equal"
    data1.flux.data[0] = 1e12
    assert data1 != data2, "different data are equal"
    assert data1 != data3, "different data are equal"


def test_empty_array():
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    fields = {"Ex": td.ScalarFieldTimeDataArray(np.random.rand(10, 10, 10, 0), coords=coords)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), fields=["Ex"], name="test")
    field_data = td.FieldTimeData(monitor=monitor, **fields)


def test_empty_list():
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    fields = {"Ex": td.ScalarFieldTimeDataArray([], coords=coords)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), fields=["Ex"], name="test")
    field_data = td.FieldTimeData(monitor=monitor, **fields)


def test_empty_tuple():
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    fields = {"Ex": td.ScalarFieldTimeDataArray((), coords=coords)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), fields=["Ex"], name="test")
    field_data = td.FieldTimeData(monitor=monitor, **fields)


@clear_tmp
def test_empty_io():
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    fields = {"Ex": td.ScalarFieldTimeDataArray(np.random.rand(10, 10, 10, 0), coords=coords)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), name="test", fields=["Ex"])
    field_data = td.FieldTimeData(monitor=monitor, **fields)
    field_data.to_file("tests/tmp/field_data.hdf5")
    field_data = td.FieldTimeData.from_file("tests/tmp/field_data.hdf5")
    assert field_data.Ex.size == 0


def test_mode_solver_plot_field():
    """Ensure we get a helpful error if trying to .plot_field with a ModeSolverData."""
    ms_data = make_mode_solver_data()
    with pytest.raises(DeprecationWarning):
        ms_data.plot_field(1, 2, 3, z=5, b=True)
