"""Tests tidy3d/components/data/monitor_data.py"""
import numpy as np
import pytest
import xarray as xr
import tidy3d as td

from tidy3d.components.monitor import FieldMonitor, FieldTimeMonitor, PermittivityMonitor
from tidy3d.components.monitor import ModeSolverMonitor, ModeMonitor
from tidy3d.components.monitor import FluxMonitor, FluxTimeMonitor
from tidy3d.components.mode import ModeSpec
from tidy3d.log import DataError

from tidy3d.components.data.dataset import FieldData, FieldTimeData, PermittivityData
from tidy3d.components.data.dataset import ModeSolverData, ModeData
from tidy3d.components.data.dataset import FluxData, FluxTimeData
from tidy3d.components.data.dataset import Near2FarAngleData, Near2FarKSpaceData
from tidy3d.components.data.dataset import Near2FarCartesianData

from .test_data_arrays import make_scalar_field_data_array, make_scalar_field_time_data_array
from .test_data_arrays import make_scalar_mode_field_data_array
from .test_data_arrays import make_flux_data_array, make_flux_time_data_array
from .test_data_arrays import make_mode_amps_data_array, make_mode_index_data_array
from .test_data_arrays import make_n2f_angle_data_array, make_n2f_kspace_data_array
from .test_data_arrays import make_n2f_cartesian_data_array
from ..utils import clear_tmp

# data array instances
AMPS = make_mode_amps_data_array()
N_COMPLEX = make_mode_index_data_array()
FLUX = make_flux_data_array()
FLUX_TIME = make_flux_time_data_array()

""" Make the montor data """


def make_field_data(symmetry: bool = True):
    return FieldData(
        Ex=make_scalar_field_data_array("Ex", symmetry),
        Ey=make_scalar_field_data_array("Ey", symmetry),
        Ez=make_scalar_field_data_array("Ez", symmetry),
        Hz=make_scalar_field_data_array("Hz", symmetry),
    )


def make_field_time_data(symmetry: bool = True):
    return FieldTimeData(
        Ex=make_scalar_field_time_data_array("Ex", symmetry),
        Ey=make_scalar_field_time_data_array("Ey", symmetry),
        Ez=make_scalar_field_time_data_array("Ez", symmetry),
        Hz=make_scalar_field_time_data_array("Ez", symmetry),
    )


def make_mode_solver_data():
    return ModeSolverData(
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
        eps_xx=make_scalar_field_data_array("Ex", symmetry),
        eps_yy=make_scalar_field_data_array("Ey", symmetry),
        eps_zz=make_scalar_field_data_array("Ez", symmetry),
    )


def make_mode_data():
    return ModeData(amps=AMPS.copy(), n_complex=N_COMPLEX.copy())


def make_flux_data():
    return FluxData(flux=FLUX.copy())


def make_flux_time_data():
    return FluxTimeData(flux=FLUX_TIME.copy())


def make_n2f_angle_data():
    """Make sure all the near-to-far data structures can be created."""

    scalar_field_tp = make_n2f_angle_data_array()
    return Near2FarAngleData(
        Ntheta=scalar_field_tp,
        Nphi=scalar_field_tp,
        Ltheta=scalar_field_tp,
        Lphi=scalar_field_tp,
    )


def make_n2f_cartesian_data():

    scalar_field_xy = make_n2f_cartesian_data_array()
    return Near2FarCartesianData(
        Ntheta=scalar_field_xy,
        Nphi=scalar_field_xy,
        Ltheta=scalar_field_xy,
        Lphi=scalar_field_xy,
    )


def make_n2f_kspace_data():

    scalar_field_u = make_n2f_kspace_data_array()
    return Near2FarKSpaceData(
        Ntheta=scalar_field_u,
        Nphi=scalar_field_u,
        Ltheta=scalar_field_u,
        Lphi=scalar_field_u,
    )


""" Test them out """


def test_field_data():
    data = make_field_data()
    for field in "EH":
        for component in "xyz":
            _ = getattr(data, field + component)


def test_field_time_data():
    data = make_field_time_data()
    for field in "EH":
        for component in "xyz":
            _ = getattr(data, field + component)


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


def test_n2f_angle_data():
    data = make_n2f_angle_data()
    _ = data.Lphi


def test_n2f_kspace_data():
    data = make_n2f_kspace_data()
    _ = data.Lphi


def test_n2f_cartesian_data():
    data = make_n2f_cartesian_data()
    _ = data.Lphi


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
    for _, scalar_field in field_data.field_components.items():
        assert (
            "mode_index" not in scalar_field.data.coords
        ), "mode_index coordinate remained in data."


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
    data_array = xr.DataArray(np.random.rand(10, 10, 10, 0), coords=coords)
    fields = {"Ex": td.ScalarFieldTimeDataArray(data=data_array)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), fields=["Ex"], name="test")
    field_data = td.FieldTimeData(**fields)


def test_empty_list():
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    data_array = xr.DataArray(np.random.random((10, 10, 10, 0)), coords=coords)
    fields = {"Ex": td.ScalarFieldTimeDataArray(data=data_array)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), fields=["Ex"], name="test")
    field_data = td.FieldTimeData(**fields)


def test_empty_tuple():
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    data_array = xr.DataArray(np.random.random((10, 10, 10, 0)), coords=coords)
    fields = {"Ex": td.ScalarFieldTimeDataArray(data=data_array)}
    field_data = td.FieldTimeData(**fields)


@clear_tmp
def test_empty_io():
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    data_array = xr.DataArray(np.random.rand(10, 10, 10, 0), coords=coords)
    fields = {"Ex": td.ScalarFieldTimeDataArray(data=data_array)}
    field_data = td.FieldTimeData(**fields)
    field_data.to_file("tests/tmp/field_data.hdf5")
    field_data = td.FieldTimeData.from_file("tests/tmp/field_data.hdf5")
    assert field_data.Ex.data.size == 0


def test_mode_solver_plot_field():
    """Ensure we get a helpful error if trying to .plot_field with a ModeSolverData."""
    ms_data = make_mode_solver_data()
    with pytest.raises(DeprecationWarning):
        ms_data.plot_field(1, 2, 3, z=5, b=True)
