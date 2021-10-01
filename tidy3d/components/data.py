""" Classes for Storing Monitor and Simulation Data """

from abc import ABC, abstractmethod
from typing import Dict

import xarray as xr
import numpy as np
import holoviews as hv
import h5py

from .simulation import Simulation
from .geometry import Box
from .monitor import (
    FluxMonitor,
    FluxTimeMonitor,
    FieldMonitor,
    FieldTimeMonitor,
    ModeMonitor,
    PermittivityMonitor,
)
from .monitor import AbstractFieldMonitor, AbstractFluxMonitor, FreqMonitor, TimeMonitor
from .base import Tidy3dBaseModel
from .types import AxesSubplot, Axis, Numpy
from .viz import add_ax_if_none, SimDataGeoParams


class Tidy3dData(Tidy3dBaseModel):
    """base class for data associated with a specific task."""

    class Config:  # pylint: disable=too-few-public-methods
        """sets config for all Tidy3dBaseModel objects"""

        validate_all = True  # validate default values too
        extra = "allow"  # allow extra kwargs not specified in model (like dir=['+', '-'])
        validate_assignment = True  # validate when attributes are set after initialization
        arbitrary_types_allowed = True


class MonitorData(Tidy3dData, ABC):
    """Stores data from a Monitor"""

    monitor_name: str
    values: Numpy
    data: xr.DataArray = None

    def __init__(self, **kwargs):
        """compute xarray and add to monitor after init"""
        super().__init__(**kwargs)
        self.data = self._make_xarray()

    def __eq__(self, other):
        """check equality against another MonitorData instance"""
        assert isinstance(other, MonitorData), "can only check eqality on two monitor data objects"
        return np.all(self.values == self.values)

    def plot(self) -> AxesSubplot:
        """make static plot"""

    # @abstractmethod
    def visualize(self) -> None:
        """make interactive plot (impement in subclasses)"""

    @abstractmethod
    def _make_xarray(self) -> xr.DataArray:
        """returns an xarray representation of data"""

    def export(self, fname: str) -> None:
        """Export MonitorData's xarray to hdf5 file"""
        self.data.to_netcdf(fname, engine="h5netcdf", invalid_netcdf=True)

    @classmethod
    def load(cls, fname: str):
        """Load MonitorData from .hdf5 file containing xarray"""

        # open from file
        data_array = xr.open_dataarray(fname, engine="h5netcdf")

        # kwargs that all MonitorData instances have
        kwargs = {
            "monitor_name": data_array.name,
            "values": data_array.values,
        }

        # get other kwargs from the data array, allow extras
        for name, val in data_array.coords.items():
            if name not in kwargs:
                kwargs[name] = np.array(val)

        return cls(**kwargs)


class FreqData(MonitorData, ABC):
    """stores data in frequency domain"""

    f: Numpy


class TimeData(MonitorData, ABC):
    """stores data in time domain"""

    t: Numpy


class AbstractFieldData(MonitorData, ABC):
    """stores data as a function of x,y,z"""

    x: Numpy  # (Nx,)
    y: Numpy  # (Ny,)
    z: Numpy  # (Nz,)

    @property
    def geometry(self):
        """return Box representation of field data"""
        size_x = np.ptp(self.x)
        size_y = np.ptp(self.y)
        size_z = np.ptp(self.z)
        center_x = np.min(self.x) + size_x / 2.0
        center_y = np.min(self.y) + size_y / 2.0
        center_z = np.min(self.z) + size_z / 2.0
        return Box(center=(center_x, center_y, center_z), size=(size_x, size_y, size_z))


class AbstractFluxData(MonitorData, ABC):
    """stores flux data through a surface"""


""" usable monitors """


class FieldData(AbstractFieldData, FreqData):
    """Stores Electric and Magnetic fields from a FieldMonitor"""

    # values.shape = (2, 3, Nx, Ny, Nz, Nf)

    def _make_xarray(self):
        """returns an xarray representation of data"""
        coords = {
            "field": ["E", "H"],
            "component": ["x", "y", "z"],
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "f": self.f,
        }
        return xr.DataArray(self.values, coords=coords, name=self.monitor_name)

    @add_ax_if_none
    def plot(
        self,
        field_component: str,
        freq: float,
        position: float,
        axis: Axis,
        ax: AxesSubplot = None,
    ) -> AxesSubplot:
        """make plot of field data along plane"""
        field, component = field_component
        field_data = self.data.sel(field=field, component=component, f=freq)
        interp_kwargs = {"xyz"[axis]: position}
        data_plane = field_data.interp(**interp_kwargs)
        data_plane.real.plot.imshow(ax=ax)
        ax = self.geometry._add_ax_labels_lims(axis=axis, ax=ax, buffer=0.0)
        return ax

    def visualize(self):
        """make interactive plot"""
        hv_ds = hv.Dataset(np.abs(self.data.copy()))
        image = hv_ds.to(hv.Image, kdims=["x", "y"], dynamic=True)
        return image.options(cmap="magma", colorbar=True, aspect="equal")


class FieldTimeData(AbstractFieldData, TimeData):
    """Stores Electric and Magnetic fields from a FieldTimeMonitor"""

    # values.shape = (2, 3, Nx, Ny, Nz, Nt)

    def _make_xarray(self):
        """returns an xarray representation of data"""
        coords = {
            "field": ["E", "H"],
            "component": ["x", "y", "z"],
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "t": self.t,
        }
        return xr.DataArray(self.values, coords=coords, name=self.monitor_name)


class PermittivityData(AbstractFieldData, FreqData):
    """Stores Reltive Permittivity from a FieldMonitor"""

    # values.shape = (3, Nx, Ny, Nz, Nf)

    def _make_xarray(self):
        """returns an xarray representation of data"""
        coords = {
            "component": ["x", "y", "z"],
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "f": self.f,
        }
        return xr.DataArray(self.values, coords=coords, name=self.monitor_name)

    def visualize(self):
        """make interactive plot"""
        hv_ds = hv.Dataset(self.data.real.copy())
        image = hv_ds.to(hv.Image, kdims=["x", "y"], dynamic=True)
        return image.options(cmap="RdBu", colorbar=True, aspect="equal")


class FluxData(AbstractFluxData, FreqData):
    """Stores power flux data through a planar FluxMonitor"""

    # values.shape = (Nt,)

    def _make_xarray(self):
        """returns an xarray representation of data"""
        coords = {"f": self.f}
        return xr.DataArray(self.values, coords=coords, name=self.monitor_name)

    def visualize(self):
        """make interactive plot"""
        hv.extension("bokeh")
        hv_ds = hv.Dataset(self.data.copy())
        image = hv_ds.to(hv.Curve, "f")
        return image


class FluxTimeData(AbstractFluxData, TimeData):
    """Stores power flux data through a planar FluxMonitor"""

    # values.shape = (Nt,)

    def _make_xarray(self):
        """returns an xarray representation of data"""
        coords = {"t": self.t}
        return xr.DataArray(self.values, coords=coords, name=self.monitor_name)

    def visualize(self):
        """make interactive plot"""
        hv.extension("bokeh")
        hv_ds = hv.Dataset(self.data.copy())
        image = hv_ds.to(hv.Curve, "t")
        return image


class ModeData(FreqData):
    """Stores modal amplitdudes from a ModeMonitor"""

    mode_index: Numpy  # (Nm,)
    # values.shape = (Nm, Ns)

    def _make_xarray(self):
        """returns an xarray representation of data"""
        coords = {
            "direction": ["+", "-"],
            "mode_index": self.mode_index,
            "f": self.f,
        }
        return xr.DataArray(self.values, coords=coords, name=self.monitor_name)

    def visualize(self):
        """make interactive plot"""
        hv_ds = hv.Dataset(self.data.real.copy())
        image = hv_ds.to(hv.Curve, "f", dynamic=True)
        return image


# maps monitor type to corresponding data type
monitor_data_map = {
    FieldMonitor: FieldData,
    FieldTimeMonitor: FieldTimeData,
    PermittivityMonitor: PermittivityData,
    FluxMonitor: FluxData,
    FluxTimeMonitor: FluxTimeData,
    ModeMonitor: ModeData,
    AbstractFieldMonitor: AbstractFieldData,
    AbstractFluxMonitor: AbstractFluxData,
    FreqMonitor: FreqData,
    TimeMonitor: TimeData,
}


class SimulationData(Tidy3dData):
    """holds simulation and its monitors' data."""

    simulation: Simulation
    monitor_data: Dict[str, MonitorData]

    def __eq__(self, other):
        """check equality against another SimulationData instance"""

        if self.simulation != other.simulation:
            return False
        for mon_name, mon_data in self.monitor_data.items():
            other_data = other.monitor_data.get(mon_name)
            if other_data is None:
                return False
            if mon_data != other.monitor_data[mon_name]:
                return False
        return True

    def plot(self, field_mon_name: str, ax: AxesSubplot = None, **plot_params: dict) -> AxesSubplot:
        """plot the monitor with simulation object overlay"""

        monitor_data = self.monitor_data[field_mon_name]
        # plot_params_new = SimDataGeoParams().update_params(**plot_params)
        # ax = self.simulation.plot_structures_eps(
        #     position=position, axis=axis, ax=ax, cbar=False, **plot_params_new
        # )
        ax = monitor_data.plot(ax=ax, **plot_params)
        return ax

    def export(self, fname: str) -> None:
        """Export all data to an hdf5"""

        with h5py.File(fname, "a") as f_handle:

            # save json string as an attribute
            json_string = self.simulation.json()
            f_handle.attrs["json_string"] = json_string

            # make a group for monitor_data
            mon_data_grp = f_handle.create_group("monitor_data")
            for mon_name, mon_data in self.monitor_data.items():

                # for each monitor, make new group with the same name
                mon_grp = mon_data_grp.create_group(mon_name)

                # for each attribute in MonitorData
                for name, value in mon_data.dict().items():

                    # ignore data
                    if name == "data":
                        continue

                    # add dataset to hdf5
                    mon_grp.create_dataset(name, data=value)

    @staticmethod
    def _load_monitor_data(sim: Simulation, mon_name: str, mon_data: Numpy) -> MonitorData:
        """load the solver data for a monitor into a MonitorData instance"""

        # get info about the original monitor
        monitor = sim.monitors.get(mon_name)
        assert monitor is not None, "monitor not found in original simulation"
        mon_type = type(monitor)
        mon_data_type = monitor_data_map[mon_type]

        # construct kwarg dict from hdf5 data group for monitor
        kwargs = {}
        for data_name, data_value in mon_data.items():
            kwargs[data_name] = np.array(data_value)

        kwargs["monitor_name"] = str(mon_name)

        # construct MonitorData and return
        monitor_data_instance = mon_data_type(**kwargs)
        return monitor_data_instance

    @classmethod
    def load(cls, fname: str):
        """Load SimulationData from files"""

        # read from file at fname
        with h5py.File(fname, "r") as f_handle:

            # construct the original simulation from the json string
            json_string = f_handle.attrs["json_string"]
            sim = Simulation.parse_raw(json_string)

            # loop through monitor dataset and create all MonitorData instances
            monitor_data = f_handle["monitor_data"]
            monitor_data_dict = {}
            for mon_name, mon_data in monitor_data.items():

                # load this monitor data, add to dict
                monitor_data_instance = cls._load_monitor_data(sim, mon_name, mon_data)
                monitor_data_dict[mon_name] = monitor_data_instance

        return cls(simulation=sim, monitor_data=monitor_data_dict)
