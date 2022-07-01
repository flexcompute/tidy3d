"""Tests data/data_array.py"""

import pytest
import numpy as np

from tidy3d.components.data.base import Tidy3dData
from tidy3d.components.data.data_array import ScalarFieldDataArray, ScalarFieldTimeDataArray
from tidy3d.components.data.data_array import ScalarModeFieldDataArray
from tidy3d.components.data.data_array import ModeAmpsDataArray, ModeIndexDataArray
from tidy3d.components.data.data_array import FluxDataArray, FluxTimeDataArray
from .utils import clear_tmp

FS = np.linspace(1e14, 2e14, 1001)
TS = np.linspace(0, 1e-12, 1001)
XS = np.linspace(-1, 1, 10)
YS = np.linspace(-2, 2, 20)
ZS = np.linspace(-3, 3, 30)
MODE_INDICES = np.arange(0, 3)
DIRECTIONS = ["+", "-"]

""" Generate the data arrays (used in other test files) """


def make_scalar_field_data_array():
    values = (1 + 1j) * np.random.random((len(XS), len(YS), len(ZS), len(FS)))
    return ScalarFieldDataArray(values, coords=dict(x=XS, y=YS, z=ZS, f=FS))


def make_scalar_field_time_data_array():
    values = np.random.random((len(XS), len(YS), len(ZS), len(TS)))
    return ScalarFieldTimeDataArray(values, coords=dict(x=XS, y=YS, z=ZS, t=TS))


def make_scalar_mode_field_data_array():
    values = np.random.random((len(XS), len(YS), 1, len(FS), len(MODE_INDICES)))
    return ScalarModeFieldDataArray(
        values, coords=dict(x=XS, y=YS, z=[0.0], f=FS, mode_index=MODE_INDICES)
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
    data = make_scalar_field_data_array()
    data = data.interp(f=1.5e14)
    data = data.sel(x=-1)
    _ = data.isel(y=2)


def test_scalar_field_time_data_array():
    data = make_scalar_field_time_data_array()
    data = data.interp(t=1e-13)
    data = data.sel(x=-1)
    _ = data.isel(y=2)


def test_scalar_field_time_data_array():
    data = make_scalar_mode_field_data_array()
    data = data.interp(f=1.5e14)
    data = data.sel(x=-1)
    data = data.isel(y=2)
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
