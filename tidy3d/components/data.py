""" Classes for Storing Monitor and Simulation Data """

from abc import ABC
from typing import Dict, List, Union
import json

import xarray as xr
import numpy as np
import h5py

from .simulation import Simulation
from .monitor import FluxMonitor, FluxTimeMonitor, FieldMonitor, FieldTimeMonitor, ModeMonitor
from .monitor import PermittivityMonitor, Monitor, AbstractFluxMonitor, AbstractFieldMonitor
from .monitor import FreqMonitor, TimeMonitor

from .monitor import monitor_type_map
from .base import Tidy3dBaseModel
from .types import Numpy, EMField, Component, Direction


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

    # dims stores the keys (strings) of the coordinates of each MonitorData subclass
    # they are in order corresponding to their index into `values`.
    # underscore is used so _dims() is a class variable (static, not stored in json)
    # dims are used to construct xrrays.
    _dims = ()

    def __init__(self, **kwargs):
        """compute xarray and add to monitor after init"""
        super().__init__(**kwargs)
        self.data = self._make_xarray()

    def _make_xarray(self) -> Union[xr.DataArray, xr.Dataset]:
        """make xarray representation of data, either DataArray or Dataset (fields)"""
        data_dict = self.dict()
        coords = {dim: data_dict[dim] for dim in self._dims}
        return xr.DataArray(self.values, coords=coords, name=self.monitor_name)

    def __eq__(self, other):
        """check equality against another MonitorData instance"""
        assert isinstance(other, MonitorData), "can only check eqality on two monitor data objects"
        return np.all(self.values == self.values)

    @property
    def geometry(self):
        """return Box representation of monitor's geometry."""
        return self.monitor.geometry

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
            return cls.load_from_data(monitor, monitor_data)

    @staticmethod
    def load_from_data(monitor: Monitor, monitor_data: Dict[str, Numpy]):
        """load the solver data for a monitor into a MonitorData instance"""

        # kwargs that gets passed to MonitorData.__init__() to make new MonitorData
        kwargs = {}

        # construct kwarg dict from hdf5 data group for monitor
        for data_name, data_value in monitor_data.items():
            kwargs[data_name] = np.array(data_value)

        def _process_string_kwarg(array_of_bytes: Numpy) -> List[str]:
            """convert numpy array containing bytes to list of strings"""
            list_of_bytes = array_of_bytes.tolist()
            list_of_str = [v.decode("utf-8") for v in list_of_bytes]
            return list_of_str

        # handle data stored as np.array() of bytes instead of strings
        for str_kwarg in ("component", "field", "direction"):
            if kwargs.get(str_kwarg) is not None:
                kwargs[str_kwarg] = _process_string_kwarg(kwargs[str_kwarg])

        # convert name to string and add monitor to kwargs
        kwargs["monitor_name"] = str(kwargs["monitor_name"])
        kwargs["monitor"] = monitor

        # get MontiorData type and initialize using kwargs
        mon_type = type(monitor)
        mon_data_type = monitor_data_map[mon_type]
        monitor_data_instance = mon_data_type(**kwargs)
        return monitor_data_instance


""" Differentiates between frequency and time domain data """


class FreqData(MonitorData, ABC):
    """stores data in frequency domain"""

    f: Numpy


class TimeData(MonitorData, ABC):
    """stores data in time domain"""

    t: Numpy


""" Differentiates between types of field data """


class VectorFieldData(MonitorData, ABC):
    """stores general vector field data as a function of {component, x, y, z}"""

    component: List[Component] = ["x", "y", "z"]
    x: Numpy
    y: Numpy
    z: Numpy


class AbstractEMFieldData(VectorFieldData, ABC):
    """stores collections of electromagnetic fields"""

    field: List[EMField] = ["E", "H"]

    def _make_xarray(self):
        """reutrn dataset"""
        data_dict = self.dict()
        data_arrays = {}
        for field_index, field_name in enumerate(self.field):
            for component_index, component in enumerate(self.component):
                name = field_name + component  # Ex, Hy, etc.
                coords = {dim: data_dict[dim] for dim in self._dims}
                values = self.values[field_index, component_index]
                coords.pop("component")
                coords.pop("field")
                for dimension in "xyz":
                    coords[dimension] = coords[dimension][field_index, component_index]
                data_array = xr.DataArray(values, coords=coords, name=self.monitor_name)
                data_arrays[name] = data_array
        return xr.Dataset(data_arrays)


class AbstractFluxData(MonitorData, ABC):
    """stores flux data through a surface"""


""" usable monitors """


class FieldData(AbstractEMFieldData, FreqData):
    """Stores Electric and Magnetic fields from a FieldMonitor"""

    _dims = ("field", "component", "x", "y", "z", "f")


class FieldTimeData(AbstractEMFieldData, TimeData):
    """Stores Electric and Magnetic fields from a FieldTimeMonitor"""

    _dims = ("field", "component", "x", "y", "z", "t")


class PermittivityData(VectorFieldData, FreqData):
    """Stores Reltive Permittivity from a FieldMonitor"""

    _dims = ("component", "x", "y", "z", "f")

    def _make_xarray(self):
        """reutrn dataset"""
        data_dict = self.dict()
        data_arrays = {}
        for component_index, component in enumerate(self.component):
            name = component + component  # xx, yy, zz
            coords = {dim: data_dict[dim] for dim in self._dims}
            values = self.values[component_index]
            coords.pop("component")
            for dimension in "xyz":
                coords[dimension] = coords[dimension][component_index]
            data_array = xr.DataArray(values, coords=coords, name=self.monitor_name)
            data_arrays[name] = data_array
        return xr.Dataset(data_arrays)


class FluxData(AbstractFluxData, FreqData):
    """Stores power flux data through a planar FluxMonitor"""

    _dims = ("f",)


class FluxTimeData(AbstractFluxData, TimeData):
    """Stores power flux data through a planar FluxMonitor"""

    _dims = ("t",)


class ModeData(FreqData):
    """Stores modal amplitdudes from a ModeMonitor"""

    direction: List[Direction] = ["+", "-"]
    mode_index: Numpy

    _dims = ("direction", "mode_index", "f")


# maps monitor type to corresponding data type
monitor_data_map = {
    FieldMonitor: FieldData,
    FieldTimeMonitor: FieldTimeData,
    PermittivityMonitor: PermittivityData,
    FluxMonitor: FluxData,
    FluxTimeMonitor: FluxTimeData,
    ModeMonitor: ModeData,
    AbstractFieldMonitor: VectorFieldData,
    AbstractFluxMonitor: AbstractFluxData,
    FreqMonitor: FreqData,
    TimeMonitor: TimeData,
}


class SimulationData(Tidy3dData):
    """holds simulation and its monitors' data."""

    simulation: Simulation
    monitor_data: Dict[str, MonitorData]

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
                monitor_data_instance = MonitorData.load_from_data(monitor, monitor_data)
                monitor_data_dict[monitor_name] = monitor_data_instance

        return cls(simulation=sim, monitor_data=monitor_data_dict)

    def __getitem__(self, monitor_name: str) -> MonitorData:
        """get the monitor xarray directly by name"""
        return self.monitor_data[monitor_name].data

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
