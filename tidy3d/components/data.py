""" Classes for Storing Monitor and Simulation Data """

from abc import ABC, abstractmethod
from typing import Dict
import json

import xarray as xr
import numpy as np
import holoviews as hv
import h5py
import matplotlib.pylab as plt

from .simulation import Simulation
from .geometry import Box
from .monitor import FluxMonitor, FluxTimeMonitor, FieldMonitor, FieldTimeMonitor, ModeMonitor
from .monitor import (
    PermittivityMonitor,
    Monitor,
    AbstractFluxMonitor,
    AbstractFieldMonitor,
    FreqMonitor,
    TimeMonitor,
)
from .monitor import monitor_type_map
from .base import Tidy3dBaseModel
from .types import AxesSubplot, Axis, Numpy, Literal
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
    monitor: Monitor
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

    @property
    def geometry(self):
        """return Box representation of field data"""
        return self.monitor.geometry

    @abstractmethod
    def _get_xarray_coords(self) -> dict:
        """returns dictionary of coords for xarray creation"""

    def _make_xarray(self) -> xr.DataArray:
        """returns an xarray representation of data"""
        coords = self._get_xarray_coords()
        return xr.DataArray(self.values, coords=coords, name=self.monitor_name)

    def export(self, fname: str) -> None:
        """Export MonitorData to hdf5 file"""

        with h5py.File(fname, "a") as f_handle:

            # save json string as an attribute
            mon_json = self.monitor.json()
            f_handle.attrs["mon_json"] = mon_json

            mon_data_grp = f_handle.create_group("monitor_data")

            for name, value in self.dict().items():

                ignore = ("data", "monitor")
                if name not in ignore:
                    mon_data_grp.create_dataset(name, data=value)

    @classmethod
    def load(cls, fname: str):
        """Load MonitorData from .hdf5 file"""

        with h5py.File(fname, "r") as f_handle:

            # construct the original monitor from the json string
            mon_json = f_handle.attrs["mon_json"]
            monitor_type_str = json.loads(mon_json)["type"]
            monitor_type = monitor_type_map[monitor_type_str]
            monitor = monitor_type.parse_raw(mon_json)

            # load the raw monitor data into a MonitorData instance
            monitor_data = f_handle["monitor_data"]
            return cls._load_from_data(monitor, monitor_data)

    @staticmethod
    def _load_from_data(monitor: Monitor, monitor_data: dict):
        """load the solver data for a monitor into a MonitorData instance"""

        # get info about the original monitor
        mon_type = type(monitor)
        mon_data_type = monitor_data_map[mon_type]

        # construct kwarg dict from hdf5 data group for monitor
        kwargs = {}
        for data_name, data_value in monitor_data.items():
            kwargs[data_name] = np.array(data_value)

        # convert name to string and add monitor to kwargs
        kwargs["monitor_name"] = str(kwargs["monitor_name"])
        kwargs["monitor"] = monitor

        # construct MonitorData and return
        monitor_data_instance = mon_data_type(**kwargs)
        return monitor_data_instance


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


class AbstractFluxData(MonitorData, ABC):
    """stores flux data through a surface"""


""" usable monitors """


class FieldData(AbstractFieldData, FreqData):
    """Stores Electric and Magnetic fields from a FieldMonitor"""

    # values.shape = (2, 3, Nx, Ny, Nz, Nf)

    def _get_xarray_coords(self):
        """returns dictionary of coords for xarray creation"""
        return {
            "field": ["E", "H"],
            "component": ["x", "y", "z"],
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "f": self.f,
        }

    @add_ax_if_none
    def plot(
        self,
        field_component: str,
        freq: float,
        position: float,
        axis: Axis,
        ax: AxesSubplot = None,
        **pcolormesh_params: dict,
    ) -> AxesSubplot:
        """make plot of field data along plane"""
        field, component = field_component
        z_label, (x_label, y_label) = self.geometry._pop_axis("xyz", axis=axis)
        x_coords = self.data.coords[x_label]
        y_coords = self.data.coords[y_label]
        field_data = self.data.sel(field=field, component=component, f=freq)
        data_plane = field_data.interp(**{z_label: position})
        image = ax.pcolormesh(
            x_coords,
            y_coords,
            np.real(data_plane.values),
            cmap="RdBu",
            shading="auto",
            **pcolormesh_params,
        )
        plt.colorbar(image, ax=ax)
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

    @add_ax_if_none
    def plot(
        self,
        field_component: str,
        time: int,
        position: float,
        axis: Axis,
        ax: AxesSubplot = None,
        **pcolormesh_params: dict,
    ) -> AxesSubplot:
        """make plot of field data along plane"""
        field, component = field_component
        z_label, (x_label, y_label) = self.geometry._pop_axis("xyz", axis=axis)
        x_coords = self.data.coords[x_label]
        y_coords = self.data.coords[y_label]
        field_data = self.data.sel(field=field, component=component, t=time)
        data_plane = field_data.interp(**{z_label: position})
        im = ax.pcolormesh(
            x_coords,
            y_coords,
            np.real(data_plane.values),
            cmap="RdBu",
            shading="auto",
            **pcolormesh_params,
        )
        plt.colorbar(im, ax=ax)
        ax = self.geometry._add_ax_labels_lims(axis=axis, ax=ax, buffer=0.0)
        return ax

    def _get_xarray_coords(self):
        """returns dictionary of coords for xarray creation"""
        return {
            "field": ["E", "H"],
            "component": ["x", "y", "z"],
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "t": self.t,
        }


class PermittivityData(AbstractFieldData, FreqData):
    """Stores Reltive Permittivity from a FieldMonitor"""

    # values.shape = (3, Nx, Ny, Nz, Nf)

    @add_ax_if_none
    def plot(
        self,
        freq: float,
        component: Literal["xx", "yy", "zz"],
        position: float,
        axis: Axis,
        ax: AxesSubplot = None,
        **pcolormesh_params: dict,
    ) -> AxesSubplot:
        """make plot of field data along plane"""
        z_label, (x_label, y_label) = self.geometry._pop_axis("xyz", axis=axis)
        x_coords = self.data.coords[x_label]
        y_coords = self.data.coords[y_label]
        field_data = self.data.sel(component=component, f=freq)
        data_plane = field_data.interp(**{z_label: position})
        im = ax.pcolormesh(
            x_coords,
            y_coords,
            np.real(data_plane.values),
            vmin=1,
            vmax=np.max(np.real(data_plane)),
            cmap="gist_yarg",
            shading="auto",
            **pcolormesh_params,
        )
        plt.colorbar(im, ax=ax)
        ax = self.geometry._add_ax_labels_lims(axis=axis, ax=ax, buffer=0.0)
        return ax

    def _get_xarray_coords(self):
        """returns dictionary of coords for xarray creation"""
        return {
            "component": ["xx", "yy", "zz"],
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "f": self.f,
        }

    def visualize(self):
        """make interactive plot"""
        hv_ds = hv.Dataset(self.data.real.copy())
        image = hv_ds.to(hv.Image, kdims=["x", "y"], dynamic=True)
        return image.options(cmap="RdBu", colorbar=True, aspect="equal")


class FluxData(AbstractFluxData, FreqData):
    """Stores power flux data through a planar FluxMonitor"""

    # values.shape = (Nt,)

    @add_ax_if_none
    def plot(self, ax: AxesSubplot = None, **plot_params) -> AxesSubplot:
        """make static plot"""
        ax.plot(self.f, self.values, **plot_params)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("flux")
        return ax

    def _get_xarray_coords(self):
        """returns dictionary of coords for xarray creation"""
        return {"f": self.f}

    def visualize(self):
        """make interactive plot"""
        hv.extension("bokeh")
        hv_ds = hv.Dataset(self.data.copy())
        image = hv_ds.to(hv.Curve, "f")
        return image


class FluxTimeData(AbstractFluxData, TimeData):
    """Stores power flux data through a planar FluxMonitor"""

    # values.shape = (Nt,)

    @add_ax_if_none
    def plot(self, ax: AxesSubplot = None, **plot_params: dict) -> AxesSubplot:
        """make static plot"""
        ax.plot(self.t, self.values, **plot_params)
        ax.set_xlabel("time steps")
        ax.set_ylabel("flux")
        return ax

    def _get_xarray_coords(self):
        """returns dictionary of coords for xarray creation"""
        return {"t": self.t}

    def visualize(self):
        """make interactive plot"""
        hv.extension("bokeh")
        hv_ds = hv.Dataset(self.data.copy())
        image = hv_ds.to(hv.Curve, "t")
        return image


class ModeData(FreqData):
    """Stores modal amplitdudes from a ModeMonitor"""

    mode_index: Numpy  # (Nm,)
    # values.shape = (Nm, Nf)

    @add_ax_if_none
    def plot(
        self, direction: Literal["+", "-"], ax: AxesSubplot = None, **plot_params: dict
    ) -> AxesSubplot:
        """make static plot"""
        values_dir = self.data.sel(direction="+").values
        for mode_index, mode_spectrum in enumerate(values_dir):
            ax.plot(self.f, np.abs(mode_spectrum.real), label=f"mode {mode_index}", **plot_params)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("Re{mode amplitude}")
        ax.legend()
        return ax

    def _get_xarray_coords(self):
        """returns dictionary of coords for xarray creation"""
        return {
            "direction": ["+", "-"],
            "mode_index": self.mode_index,
            "f": self.f,
        }

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

    """ add __getitem__ or __index__ for monitor """

    @add_ax_if_none
    def plot(self, monitor_name: str, ax: AxesSubplot = None, **plot_params: dict) -> AxesSubplot:
        """plot the monitor with simulation object overlay"""

        monitor_data = self.monitor_data[monitor_name]
        ax = monitor_data.plot(ax=ax, **plot_params)
        return ax

    @add_ax_if_none
    def plot_fields(
        self,
        monitor_name: str,
        position: float,
        axis: Axis,
        ax: AxesSubplot = None,
        **plot_params: dict,
    ) -> AxesSubplot:
        """make field plot with structure permittivity overlayed with transparency"""
        monitor_data = self.monitor_data[monitor_name]
        assert isinstance(
            monitor_data, (FieldData, FieldTimeData)
        ), f"must be FieldData or FieldTimeData, given {type(monitor_data)}"
        plot_params_structures = SimDataGeoParams().update_params(**{})
        ax = monitor_data.plot(position=position, axis=axis, ax=ax, **plot_params)
        ax = self.simulation.plot_structures_eps(
            position=position, axis=axis, ax=ax, cbar=False, **plot_params_structures
        )
        ax = monitor_data.geometry._add_ax_labels_lims(axis=axis, ax=ax, buffer=0.0)
        return ax

    def export(self, fname: str) -> None:
        """Export all data to an hdf5"""

        with h5py.File(fname, "a") as f_handle:

            # save json string as an attribute
            sim_json = self.simulation.json()
            f_handle.attrs["sim_json"] = sim_json

            # make a group for monitor_data
            mon_data_grp = f_handle.create_group("monitor_data")
            for mon_name, mon_data in self.monitor_data.items():

                # for each monitor, make new group with the same name
                mon_grp = mon_data_grp.create_group(mon_name)

                # for each attribute in MonitorData
                for name, value in mon_data.dict().items():

                    ignore = ("data", "monitor")
                    if name in ignore:
                        continue

                    # add dataset to hdf5
                    mon_grp.create_dataset(name, data=value)

    @classmethod
    def load(cls, fname: str):
        """Load SimulationData from files"""

        # read from file at fname
        with h5py.File(fname, "r") as f_handle:

            # construct the original simulation from the json string
            sim_json = f_handle.attrs["sim_json"]
            sim = Simulation.parse_raw(sim_json)

            # loop through monitor dataset and create all MonitorData instances
            monitor_data = f_handle["monitor_data"]
            monitor_data_dict = {}
            for monitor_name, monitor_data in monitor_data.items():

                # load this monitor data, add to dict
                monitor = sim.monitors.get(monitor_name)
                monitor_data_instance = MonitorData._load_from_data(monitor, monitor_data)
                monitor_data_dict[monitor_name] = monitor_data_instance

        return cls(simulation=sim, monitor_data=monitor_data_dict)

    def __getitem__(self, monitor_name: str) -> MonitorData:
        """get the monitor directly by name"""
        return self.monitor_data[monitor_name]

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
