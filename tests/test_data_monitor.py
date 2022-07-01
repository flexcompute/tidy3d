"""Tests data/monitor_data.py"""
import numpy as np
import pytest

from tidy3d.components.monitor import FieldMonitor, FieldTimeMonitor, PermittivityMonitor
from tidy3d.components.monitor import ModeFieldMonitor, ModeMonitor
from tidy3d.components.monitor import FluxMonitor, FluxTimeMonitor
from tidy3d.components.mode import ModeSpec
from tidy3d.log import DataError

from tidy3d.components.data.monitor_data import FieldData, FieldTimeData, PermittivityData
from tidy3d.components.data.monitor_data import ModeFieldData, ModeData
from tidy3d.components.data.monitor_data import FluxData, FluxTimeData

from .test_data_arrays import make_scalar_field_data_array, make_scalar_field_time_data_array
from .test_data_arrays import make_scalar_mode_field_data_array
from .test_data_arrays import make_flux_data_array, make_flux_time_data_array
from .test_data_arrays import make_mode_amps_data_array, make_mode_index_data_array

# data array instances
FIELD = make_scalar_field_data_array()
FIELD_TIME = make_scalar_field_time_data_array()
MODE_FIELD = make_scalar_mode_field_data_array()
AMPS = make_mode_amps_data_array()
N_COMPLEX = make_mode_index_data_array()
FLUX = make_flux_data_array()
FLUX_TIME = make_flux_time_data_array()

# monitor inputs
SIZE_3D = (1, 1, 1)
SIZE_2D = (1, 0, 1)
MODE_SPEC = ModeSpec(num_modes=4)
FREQS = [1e14, 2e14]
FIELDS = ("Ex", "Ey", "Hz")
INTERVAL = 2

""" Make the montor data """


def make_field_data():
    monitor = FieldMonitor(size=SIZE_3D, fields=FIELDS, name="field", freqs=FREQS)
    return FieldData(monitor=monitor, Ex=FIELD.copy(), Ey=FIELD.copy(), Hz=FIELD.copy())


def make_field_time_data():
    monitor = FieldTimeMonitor(size=SIZE_3D, fields=FIELDS, name="field_time", interval=INTERVAL)
    return FieldTimeData(
        monitor=monitor, Ex=FIELD_TIME.copy(), Ey=FIELD_TIME.copy(), Hz=FIELD_TIME.copy()
    )


def make_mode_field_data():
    monitor = ModeFieldMonitor(size=SIZE_2D, name="mode_field", mode_spec=MODE_SPEC, freqs=FREQS)
    return ModeFieldData(
        monitor=monitor,
        Ex=MODE_FIELD.copy(),
        Ey=MODE_FIELD.copy(),
        Ez=MODE_FIELD.copy(),
        Hx=MODE_FIELD.copy(),
        Hy=MODE_FIELD.copy(),
        Hz=MODE_FIELD.copy(),
    )


def make_permittivity_data():
    monitor = PermittivityMonitor(size=SIZE_3D, name="permittivity", freqs=FREQS)
    return PermittivityData(
        monitor=monitor, eps_xx=FIELD.copy(), eps_yy=FIELD.copy(), eps_zz=FIELD.copy()
    )


def make_mode_data():
    monitor = ModeMonitor(size=SIZE_2D, name="mode", mode_spec=MODE_SPEC, freqs=FREQS)
    return ModeData(monitor=monitor, amps=AMPS.copy(), n_complex=N_COMPLEX.copy())


def make_flux_data():
    monitor = FluxMonitor(size=SIZE_2D, freqs=FREQS, name="flux")
    return FluxData(monitor=monitor, flux=FLUX.copy())


def make_flux_time_data():
    monitor = FluxTimeMonitor(size=SIZE_2D, interval=INTERVAL, name="flux_time")
    return FluxTimeData(monitor=monitor, flux=FLUX_TIME.copy())


""" Test them out """


def test_field_data():
    data = make_field_data()
    for field in FIELDS:
        _ = getattr(data, field)


def test_field_time_data():
    data = make_field_time_data()
    for field in FIELDS:
        _ = getattr(data, field)


def test_mode_field_data():
    data = make_mode_field_data()
    for field in "EH":
        for comp in "xyz":
            _ = getattr(data, field + comp)


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


def test_colocate():

    # regular colocate
    data = make_field_data()
    _ = data.colocate(x=[-0.5, 0.5], y=[-0.5, 0.5], z=[-0.5, 0.5])

    # select len(coord) == 1 at the exact position (z=0)
    data = make_mode_field_data()
    _ = data.colocate(x=[-0.5, 0.5], y=[-0.5, 0.5], z=0.0)

    # ignore coordinate
    _ = data.colocate(x=[-0.5, 0.5], y=[-0.5, 0.5], z=None)

    # data outside range of len(coord)==1 dimension
    with pytest.raises(DataError):
        _ = data.colocate(x=[-0.5, 0.5], y=[-0.5, 0.5], z=1.0)


def test_sel_mode_index():

    data = make_mode_field_data()
    field_data = data.sel_mode_index(mode_index=0)
    assert isinstance(field_data, FieldData), "ModeFieldData wasnt converted to FieldData."
    assert isinstance(
        field_data.monitor, FieldMonitor
    ), "ModeFieldMonitor wasnt converted to FieldMonitor."
    for _, (scalar_field, _, _) in field_data.field_components.items():
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
