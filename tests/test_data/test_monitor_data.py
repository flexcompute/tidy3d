"""Tests tidy3d/components/data/monitor_data.py"""
import numpy as np
import pytest

import tidy3d as td

from tidy3d.components.monitor import FieldMonitor, FieldTimeMonitor, PermittivityMonitor
from tidy3d.components.monitor import ModeSolverMonitor, ModeMonitor
from tidy3d.components.monitor import FluxMonitor, FluxTimeMonitor
from tidy3d.components.monitor import Near2FarAngleMonitor, Near2FarKSpaceMonitor
from tidy3d.components.monitor import Near2FarCartesianMonitor
from tidy3d.components.mode import ModeSpec
from tidy3d.log import DataError

from tidy3d.components.data.monitor_data import FieldMonitorData, FieldTimeMonitorData
from tidy3d.components.data.monitor_data import PermittivityMonitorData
from tidy3d.components.data.monitor_data import ModeSolverMonitorData, ModeMonitorData
from tidy3d.components.data.monitor_data import FluxMonitorData, FluxTimeMonitorData
from tidy3d.components.data.monitor_data import MonitorDataTypes
from tidy3d.components.data.monitor_data import Near2FarAngleMonitorData, Near2FarKSpaceMonitorData
from tidy3d.components.data.monitor_data import Near2FarCartesianMonitorData
from tidy3d.components.data.data_array import DataArray

from .test_dataset import make_field_data, make_field_time_data, make_mode_solver_data
from .test_dataset import make_permittivity_data, make_mode_data
from .test_dataset import make_flux_data, make_flux_time_data
from .test_dataset import make_n2f_angle_data, make_n2f_kspace_data
from .test_dataset import make_n2f_cartesian_data

from .test_data_arrays import FIELD_MONITOR, FIELD_TIME_MONITOR, MODE_SOLVE_MONITOR
from .test_data_arrays import MODE_MONITOR, PERMITTIVITY_MONITOR, FLUX_MONITOR, FLUX_TIME_MONITOR
from .test_data_arrays import N2F_CARTESIAN_MONITOR, N2F_ANGLE_MONITOR, N2F_KSPACE_MONITOR


from .test_data_arrays import SIM_SYM
from ..utils import clear_tmp

""" Make the montor data """


def make_field_monitor_data(symmetry: bool = True):
    return FieldMonitorData(monitor=FIELD_MONITOR, dataset=make_field_data(symmetry))


def make_field_time_monitor_data(symmetry: bool = True):
    return FieldTimeMonitorData(monitor=FIELD_TIME_MONITOR, dataset=make_field_time_data(symmetry))


def make_mode_solver_monitor_data():
    return ModeSolverMonitorData(monitor=MODE_SOLVE_MONITOR, dataset=make_mode_solver_data())


def make_permittivity_monitor_data(symmetry: bool = True):
    return PermittivityMonitorData(
        monitor=PERMITTIVITY_MONITOR, dataset=make_permittivity_data(symmetry)
    )


def make_mode_monitor_data():
    return ModeMonitorData(monitor=MODE_MONITOR, dataset=make_mode_data())


def make_flux_monitor_data():
    return FluxMonitorData(monitor=FLUX_MONITOR, dataset=make_flux_data())


def make_flux_time_monitor_data():
    return FluxTimeMonitorData(monitor=FLUX_TIME_MONITOR, dataset=make_flux_time_data())


def make_n2f_angle_monitor_data():
    return Near2FarAngleMonitorData(monitor=N2F_ANGLE_MONITOR, dataset=make_n2f_angle_data())


def make_n2f_cartesian_monitor_data():
    return Near2FarCartesianMonitorData(
        monitor=N2F_CARTESIAN_MONITOR, dataset=make_n2f_cartesian_data()
    )


def make_n2f_kspace_monitor_data():
    return Near2FarKSpaceMonitorData(monitor=N2F_KSPACE_MONITOR, dataset=make_n2f_kspace_data())


""" Test them out """


def test_field_monitor_data():
    data = make_field_monitor_data()


def test_field_time_monitor_data():
    data = make_field_time_monitor_data()


def test_mode_field_monitor_data():
    data = make_mode_solver_monitor_data()


def test_permittivity_monitor_data():
    data = make_permittivity_monitor_data()


def test_mode_monitor_data():
    data = make_mode_monitor_data()


def test_flux_monitor_data():
    data = make_flux_monitor_data()


def test_flux_time_monitor_data():
    data = make_flux_time_monitor_data()


def test_n2f_angle_monitor_data():
    data = make_n2f_angle_monitor_data()


def test_n2f_cartesian_monitor_data():
    data = make_n2f_cartesian_monitor_data()


def test_n2f_kspace_monitor_data():
    data = make_n2f_kspace_monitor_data()


def test_symmetry():
    data = make_field_monitor_data(symmetry=False)

    # make sure we cant interpolate into the region that is defined by symmetry only
    with pytest.raises(ValueError):
        data.dataset.Ex.data.interp(x=-0.5, kwargs={"bounds_error": True})

    # make sure we can interpolate into the region that was extrapolated by symmetry
    data_sym = data.apply_symmetry(
        symmetry=(1, 1, 1), symmetry_center=(0, 0, 0), grid_expanded=SIM_SYM.grid
    )
    data_sym.dataset.Ex.data.interp(x=-0.5, kwargs={"bounds_error": True})


def test_colocate():
    # TODO: can we colocate into regions where we dont store fields due to symmetry?
    # regular colocate
    data = make_field_monitor_data()
    _ = data.dataset.colocate(x=[+0.1, 0.5], y=[+0.1, 0.5], z=[+0.1, 0.5])

    # ignore coordinate
    _ = data.dataset.colocate(x=[+0.1, 0.5], y=None, z=[+0.1, 0.5])

    # data outside range of len(coord)==1 dimension
    data = make_mode_solver_monitor_data()
    with pytest.raises(DataError):
        _ = data.dataset.colocate(x=[+0.1, 0.5], y=1.0, z=[+0.1, 0.5])

    with pytest.raises(DataError):
        _ = data.dataset.colocate(x=[+0.1, 0.5], y=[1.0, 2.0], z=[+0.1, 0.5])


def test_mode_solver_plot_field():
    """Ensure we get a helpful error if trying to .plot_field with a ModeSolverData."""
    ms_data = make_mode_solver_data()
    with pytest.raises(DeprecationWarning):
        ms_data.plot_field(1, 2, 3, z=5, b=True)


# prepare the monitor data and all their properties for checking in the test below.
mnt_data_makers = [
    make_field_monitor_data,
    make_field_time_monitor_data,
    make_mode_solver_monitor_data,
    make_permittivity_monitor_data,
    make_flux_monitor_data,
    make_flux_time_monitor_data,
    make_mode_monitor_data,
    make_n2f_angle_monitor_data,
    make_n2f_cartesian_monitor_data,
    make_n2f_kspace_monitor_data,
]
MNT_DATA_LIST = []
PROPERTY_NAMES = []
for i, mnt_data_maker in enumerate(mnt_data_makers):
    mnt_data = mnt_data_maker()
    MNT_DATA_LIST.append(mnt_data)

    dataset_type = type(mnt_data).__fields__["dataset"].type_

    PROPERTY_NAMES.append([])

    for field_name, model_field in dataset_type.__fields__.items():
        field_type = model_field.outer_type_
        if isinstance(field_type, type) and issubclass(field_type, DataArray):
            PROPERTY_NAMES[i].append(field_name)


@pytest.mark.parametrize("mnt_data, property_names", zip(MNT_DATA_LIST, PROPERTY_NAMES))
def test_field_name_property(mnt_data, property_names):
    """Ensure that the property linking of MonitorData.{field_name} works propertly."""

    for property_name in property_names:
        # print(f"checking: {mnt_data.__class__.__name__}.{property_name}")
        xr_data_array_direct = getattr(mnt_data, property_name)
        data_array_explicit = getattr(mnt_data.dataset, property_name)
        if data_array_explicit is None and xr_data_array_direct is None:
            continue
        xr_data_array_explicit = data_array_explicit.data
        assert np.all(xr_data_array_explicit == xr_data_array_direct)
