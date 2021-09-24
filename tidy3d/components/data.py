""" Classes for Storing Monitor and Simulation Data """

from abc import ABC, abstractmethod
from typing import Dict, List

import xarray as xr
import numpy as np
import holoviews as hv
import h5py

from .simulation import Simulation
from .monitor import FluxMonitor, FieldMonitor, ModeMonitor
from .base import Tidy3dBaseModel


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

    # data: xr.DataArray = None
    sampler_label: str
    sampler_values: List
    values: np.ndarray
    monitor_name: str
    data: xr.DataArray = None

    def __init__(self, **kwargs):
        """compute xarray and add to monitor after init"""
        super().__init__(**kwargs)
        self.data = self.load_xarray()

    def __eq__(self, other):
        """check equality against another MonitorData instance"""
        assert isinstance(other, MonitorData), "can only check eqality on two monitor data objects"
        return np.all(self.values == self.values)

    @abstractmethod
    def visualize(self) -> None:
        """make interactive plot (impement in subclasses)"""

    @abstractmethod
    def load_xarray(self) -> xr.DataArray:
        """create an xarray for the dataset and set it to self.data"""

    def export(self, fname: str) -> None:
        """Export MonitorData's xarray to hdf5 file"""
        self.data.to_netcdf(fname, engine="h5netcdf", invalid_netcdf=True)

    @classmethod
    def load(cls, fname: str):
        """Load MonitorData from .hdf5 file containing xarray"""

        # open from file
        data_array = xr.open_dataarray(fname, engine="h5netcdf")

        # strip out sampler info and data values
        sampler_label = "freqs" if "freqs" in data_array.coords else "times"
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

    xs: np.ndarray  # (Nx,)
    ys: np.ndarray  # (Ny,)
    zs: np.ndarray  # (Nz,)
    values: np.ndarray  # (2, 3, Nx, Ny, Nz, Ns)

    def load_xarray(self):
        """returns an xarray representation of data"""
        coords = {
            "field": ["E", "H"],
            "component": ["x", "y", "z"],
            "xs": self.xs,
            "ys": self.ys,
            "zs": self.zs,
            self.sampler_label: self.sampler_values,
        }
        return xr.DataArray(self.values, coords=coords, name=self.monitor_name)

    def visualize(self):
        """make interactive plot"""
        hv_ds = hv.Dataset(self.data.copy())
        image = hv_ds.to(hv.Image, kdims=["xs", "ys"], dynamic=True)
        return image.options(cmap="RdBu", colorbar=True, aspect="equal")


class FluxData(MonitorData):
    """Stores power flux data through a planar FluxMonitor"""

    values: np.ndarray  # (Ns,)

    def load_xarray(self):
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

    mode_index: np.ndarray  # (Nm,)
    values: np.ndarray  # (Nm, Ns)

    def load_xarray(self):
        """returns an xarray representation of data"""
        coords = {
            "direction": ["+", "-"],
            "mode_index": self.mode_index,
            self.sampler_label: self.sampler_values,
        }
        return xr.DataArray(self.values, coords=coords, name=self.monitor_name)

    def visualize(self):
        """make interactive plot"""
        hv_ds = hv.Dataset(self.data.copy())
        image = hv_ds.to(hv.Curve, self.sampler_label, dynamic=True)
        return image


# maps monitor type to corresponding data type
monitor_data_map = {FieldMonitor: FieldData, FluxMonitor: FluxData, ModeMonitor: ModeData}


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
    def _load_monitor_data(sim: Simulation, mon_name: str, mon_data: np.ndarray) -> MonitorData:
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

        # these fields are specific types, not np.arrays()
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
