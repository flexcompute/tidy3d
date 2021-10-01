""" Classes for Storing Monitor and Simulation Data """

from abc import ABC, abstractmethod
from typing import Dict, List

import xarray as xr
import numpy as np
import holoviews as hv
import h5py

from .simulation import Simulation
from .geometry import Box
from .monitor import FluxMonitor, FieldMonitor, ModeMonitor, PermittivityMonitor
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
    sampler_label: str
    sampler_values: List
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

    @abstractmethod
    def visualize(self) -> None:
        """make interactive plot (impement in subclasses)"""

    @abstractmethod
    def _make_xarray(self) -> xr.DataArray:
        """returns an xarray representation of data"""

    def sel(self, *args, **kwargs):
        """http://xarray.pydata.org/en/stable/generated/xarray.DataArray.sel.html"""
        return self.data.sel(*args, **kwargs)

    def isel(self, *args, **kwargs):
        """http://xarray.pydata.org/en/stable/generated/xarray.DataArray.isel.html"""
        return self.data.sel(*args, **kwargs)

    def squeeze(self, *args, **kwargs):
        """http://xarray.pydata.org/en/stable/generated/xarray.DataArray.squeeze.html"""
        return self.data.squeeze(*args, **kwargs)

    def interp(self, *args, **kwargs):
        """http://xarray.pydata.org/en/stable/generated/xarray.DataArray.interp.html"""
        return self.data.interp(*args, **kwargs)

    def query(self, *args, **kwargs):
        """http://xarray.pydata.org/en/stable/generated/xarray.DataArray.query.html"""
        return self.data.query(*args, **kwargs)

    def isin(self, *args, **kwargs):
        """http://xarray.pydata.org/en/stable/generated/xarray.DataArray.isin.html"""
        return self.data.isin(*args, **kwargs)

    def where(self, *args):
        """http://xarray.pydata.org/en/stable/generated/xarray.DataArray.where.html"""
        return self.data.where(*args)

    def export(self, fname: str) -> None:
        """Export MonitorData's xarray to hdf5 file"""
        self.data.to_netcdf(fname, engine="h5netcdf", invalid_netcdf=True)

    @classmethod
    def load(cls, fname: str):
        """Load MonitorData from .hdf5 file containing xarray"""

        # open from file
        data_array = xr.open_dataarray(fname, engine="h5netcdf")

        # strip out sampler info and data values
        sampler_label = "f" if "f" in data_array.coords else "t"
        sampler_values = list(data_array.coords[sampler_label])
        values = data_array.values

        # kwargs that all MonitorData instances have
        kwargs = {
            "sampler_label": sampler_label,
            "sampler_values": sampler_values,
            "values": values,
            "monitor_name": data_array.name,
        }

        # get other kwargs from the data array, allow extras
        for name, val in data_array.coords.items():
            if name not in kwargs:
                kwargs[name] = np.array(val)

        return cls(**kwargs)


class FieldData(MonitorData):
    """Stores Electric and Magnetic fields from a FieldMonitor"""

    x: Numpy  # (Nx,)
    y: Numpy  # (Ny,)
    z: Numpy  # (Nz,)
    values: Numpy  # (2, 3, Nx, Ny, Nz, Ns)

    def _make_xarray(self):
        """returns an xarray representation of data"""
        coords = {
            "field": ["E", "H"],
            "component": ["x", "y", "z"],
            "x": self.x,
            "y": self.y,
            "z": self.z,
            self.sampler_label: self.sampler_values,
        }
        return xr.DataArray(self.values, coords=coords, name=self.monitor_name)

    @add_ax_if_none
    def plot(
        self,
        field_component: str,
        sampler_value: float,
        position: float,
        axis: Axis,
        ax: AxesSubplot = None,
    ) -> AxesSubplot:
        """make plot of field data along plane"""
        field, component = field_component
        field_data = self.data.sel(field=field, component=component)
        interp_kwargs = {"xyz"[axis]: position, self.sampler_label: sampler_value}
        data_plane = field_data.interp(**interp_kwargs)
        data_plane.real.plot.imshow(ax=ax)
        ax = self.geometry._add_ax_labels_lims(axis=axis, ax=ax, buffer=0.0)
        return ax

    def visualize(self):
        """make interactive plot"""
        hv_ds = hv.Dataset(np.abs(self.data.copy()))
        image = hv_ds.to(hv.Image, kdims=["x", "y"], dynamic=True)
        return image.options(cmap="magma", colorbar=True, aspect="equal")

    @property
    def geometry(self):
        """return Box representation of self"""
        size_x = np.ptp(self.x)
        size_y = np.ptp(self.y)
        size_z = np.ptp(self.z)
        center_x = np.min(self.x) + Lx / 2.0
        center_y = np.min(self.y) + Ly / 2.0
        center_z = np.min(self.z) + Lz / 2.0
        return Box(center=(center_x, center_y, center_z), size=(size_x, size_y, size_z))


class PermittivityData(MonitorData):
    """Stores Electric and Magnetic fields from a FieldMonitor"""

    x: Numpy  # (Nx,)
    y: Numpy  # (Ny,)
    z: Numpy  # (Nz,)
    values: Numpy  # (3, Nx, Ny, Nz, Ns)

    def _make_xarray(self):
        """returns an xarray representation of data"""
        coords = {
            "component": ["x", "y", "z"],
            "x": self.x,
            "y": self.y,
            "z": self.z,
            self.sampler_label: self.sampler_values,
        }
        return xr.DataArray(self.values, coords=coords, name=self.monitor_name)

    def visualize(self):
        """make interactive plot"""
        hv_ds = hv.Dataset(self.data.real.copy())
        image = hv_ds.to(hv.Image, kdims=["x", "y"], dynamic=True)
        return image.options(cmap="RdBu", colorbar=True, aspect="equal")

    @property
    def geometry(self):
        """return Box representation of self"""
        size_x = np.ptp(self.x)
        size_y = np.ptp(self.y)
        size_z = np.ptp(self.z)
        center_x = np.min(self.x) + Lx / 2.0
        center_y = np.min(self.y) + Ly / 2.0
        center_z = np.min(self.z) + Lz / 2.0
        return Box(center=(center_x, center_y, center_z), size=(size_x, size_y, size_z))


class FluxData(MonitorData):
    """Stores power flux data through a planar FluxMonitor"""

    values: Numpy  # (Ns,)

    def _make_xarray(self):
        """returns an xarray representation of data"""
        coords = {self.sampler_label: self.sampler_values}
        return xr.DataArray(self.values, coords=coords, name=self.monitor_name)

    def visualize(self):
        """make interactive plot"""
        hv.extension("bokeh")
        hv_ds = hv.Dataset(self.data.copy())
        image = hv_ds.to(hv.Curve, self.sampler_label)
        return image


class ModeData(MonitorData):
    """Stores modal amplitdudes from a ModeMonitor"""

    mode_index: Numpy  # (Nm,)
    values: Numpy  # (Nm, Ns)

    def _make_xarray(self):
        """returns an xarray representation of data"""
        coords = {
            "direction": ["+", "-"],
            "mode_index": self.mode_index,
            self.sampler_label: self.sampler_values,
        }
        return xr.DataArray(self.values, coords=coords, name=self.monitor_name)

    def visualize(self):
        """make interactive plot"""
        hv_ds = hv.Dataset(self.data.real.copy())
        image = hv_ds.to(hv.Curve, self.sampler_label, dynamic=True)
        return image


# maps monitor type to corresponding data type
monitor_data_map = {
    FieldMonitor: FieldData,
    PermittivityMonitor: PermittivityData,
    FluxMonitor: FluxData,
    ModeMonitor: ModeData,
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

    def plot_fields(
        self,
        field_mon_name: str,
        position: float,
        axis: Axis,
        field_component: str,
        sampler_value: float,
        ax: AxesSubplot = None,
        **plot_params
    ) -> AxesSubplot:
        """plot the monitor with simulation object overlay"""
        monitor_data = self.monitor_data[field_mon_name]
        assert isinstance(monitor_data, FieldData), "must be data for field monitor"
        plot_params_new = SimDataGeoParams().update_params(**plot_params)
        ax = self.simulation.plot_structures_eps(
            position=position, axis=axis, ax=ax, cbar=False, **plot_params_new
        )
        ax = monitor_data.plot(
            position=position,
            axis=axis,
            field_component=field_component,
            sampler_value=sampler_value,
            ax=ax,
        )
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

        # these fields are specific types, not np.array()
        kwargs["sampler_values"] = list(kwargs["sampler_values"])
        kwargs["sampler_label"] = str(kwargs["sampler_label"])
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
