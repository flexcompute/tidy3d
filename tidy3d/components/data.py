"""Classes for Storing Monitor and Simulation Data."""

from abc import ABC
from typing import Dict, List, Union
import json

import xarray as xr
import numpy as np
import h5py

from .types import Numpy, EMField, FieldType, Direction, Array, numpy_encoding
from .base import Tidy3dBaseModel
from .monitor import FluxMonitor, FluxTimeMonitor, FieldMonitor, FieldTimeMonitor, ModeMonitor
from .monitor import Monitor, PlanarMonitor, AbstractFluxMonitor, ScalarFieldMonitor
from .monitor import FreqMonitor, TimeMonitor, monitor_type_map
from .simulation import Simulation


class Tidy3dData(Tidy3dBaseModel):
    """base class for data associated with a simulation."""

    class Config:  # pylint: disable=too-few-public-methods
        """sets config for all Tidy3dBaseModel objects"""

        validate_all = True  # validate default values too
        extra = "allow"  # allow extra kwargs not specified in model (like dir=['+', '-'])
        validate_assignment = True  # validate when attributes are set after initialization
        arbitrary_types_allowed = True  # allow types like `Array[float]`
        json_encoders = {  # how to write certain types to json files
            np.ndarray: numpy_encoding,  # use custom encoding defined in .types
            np.int64: lambda x: int(x),  # pylint: disable=unnecessary-lambda
            xr.Dataset: lambda x: None,  # dont write
            xr.DataArray: lambda x: None,  # dont write
        }


class MonitorData(Tidy3dData, ABC):
    """Stores data for a Monitor.

    Attributes
    ----------
    data : ``Union[xr.DataArray, xr.Dataset]``
    ``xarray`` representation of the underlying data.
    """

    monitor_name: str
    monitor: Monitor

    """ explanation of values
        ``values`` is a numpy array that stores the raw data associated with each ``MonitorData``.
        It can be complex-valued or real valued, depending on whether data is in the frequency or
        time domain, respectively.
        Each axis in ``values`` corresponds to a specific dimension in the ``MonitorData``, which
        are supplied as arguments to the ``MonitorData`` subclasses.
        The order of the dimensions is specified in the ``_dims`` attribute of each ``MonitorData``
        subclass
    """

    values: Union[Array[float], Array[complex]]

    """ explanation of _dims
        _dims is an attribute of all `MonitorData` objects.
        It is a tuple of strings that stores the keys of the coordinates corresponding to `values`.
        Note: they must be in order corresponding to their index into `values`.
        The underscore is used so _dims() is a class variable and not stored in .json.
        The dims are used to construct xarray objects as it tells the _make_xarray method what
        attribute to use for the keys in the `coords` coordinate dictionary.
    """
    _dims = ()

    def __init__(self, **kwargs):
        """compute xarray and add to monitor after init"""
        super().__init__(**kwargs)
        self.data = self._make_xarray()

    def _make_xarray(self) -> Union[xr.DataArray, xr.Dataset]:
        """make xarray representation of data

        Returns
        -------
        Union[xr.DataArray, xr.Dataset]
            ``xarray`` representation of the underlying data.
        """
        data_dict = self.dict()
        coords = {dim: data_dict[dim] for dim in self._dims}
        return xr.DataArray(self.values, coords=coords, name=self.monitor_name)

    def __eq__(self, other) -> bool:
        """check equality against another MonitorData instance

        Parameters
        ----------
        other : ``MonitorData``
            Other ``MonitorData`` to equate to.

        Returns
        -------
        bool
            Whether the other ``MonitorData`` instance has the same data.
        """
        assert isinstance(other, MonitorData), "can only check eqality on two monitor data objects"
        return np.all(self.values == self.values)

    @property
    def geometry(self):
        """Return ``Box`` representation of monitor's geometry.

        Returns
        -------
        ``Box``
            ``Box`` represention of shape of originl monitor.
        """
        return self.monitor.geometry

    def export(self, fname: str) -> None:
        """Export MonitorData to hdf5 file.

        Parameters
        ----------
        fname : str
            Path to data file (including filename).
        """

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
        """Load MonitorData from .hdf5 file

        Parameters
        ----------
        fname : str
            Path to data file (including filename).

        Returns
        -------
        ``MonitorData``
            A ``MonitorData`` instance.
        """

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
        """load the solver data dict for a specific monitor into a MonitorData instance

        Parameters
        ----------
        monitor : ``Monitor``
            Original monitor that specified how data was stored.
        monitor_data : Dict[str, Numpy]
            Mapping from data value name to numpy array holding data.

        Returns
        -------
        ``MonitorData``
            A ``MonitorData`` instance.
        """

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
        for str_kwarg in ("field", "direction"):
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


""" The following are just abstract classes that separate the MonitorData instances into different
    types depending on what they store. """


class FreqData(MonitorData, ABC):
    """Frequency-domain data stores an `f` attribute for frequency (Hz)."""

    monitor: FreqMonitor
    f: Array[float]


class TimeData(MonitorData, ABC):
    """Time-domain data stores an `t` attribute for time (sec)."""

    monitor: TimeMonitor
    t: Array[float]


class ScalarFieldData(MonitorData, ABC):
    """ScalarFieldData stores some `field` quantities as a function of x, y, and z."""

    monitor: ScalarFieldMonitor
    field: List[EMField] = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    x: Array[float]
    y: Array[float]
    z: Array[float]

    def _make_xarray(self):
        """For field quantities, store a single xarray `xr.DataArray` for each `field`.  These all
        go in a single `xr.Dataset`, which keeps track of the shared coords.

        Returns
        -------
        xr.Dataset
            ``xarray`` representation of the underlying data.
        """

        data_dict = self.dict()

        # for each `field`, create `xr.DataArray` and add to dictionary.
        data_arrays = {}
        for field_index, field_name in enumerate(self.field):

            # get the coordinates from `self._dims` and strip out the 'xyz' coords for this field
            coords = {dim: data_dict[dim] for dim in self._dims}
            coords.pop("field")
            for dimension in "xyz":
                coords[dimension] = coords[dimension][field_index]

            # get the values for this field, use to construct field's DataArray and add to dict.
            values = self.values[field_index]
            data_array = xr.DataArray(values, coords=coords, name=self.monitor_name)
            data_arrays[field_name] = data_array

        # make a `xr.Dataset` out of all of the field components, this is stored as .data attribute.
        return xr.Dataset(data_arrays)


class PlanarData(MonitorData, ABC):
    """stores data that is constrained to the plane."""

    monitor: PlanarMonitor


class AbstractFluxData(PlanarData, ABC):
    """Stores electromagnetic flux through a planar Monitor"""

    monitor: AbstractFluxMonitor


""" usable monitors """


class FieldData(FreqData, ScalarFieldData):
    """Stores Electric and Magnetic fields from a ``FieldMonitor``.

    Parameters
    ----------
    monitor : ``FieldMonitor``
        Original monitor object corresponding to data.
    monitor_name : str
        Name of original monitor in its Simulation object.
    field: List[str], optional
        Electromagnetic fields (E, H) in dtaset defaults to ``['Ex', 'Ey', 'Ez', 'Hx', 'Hy',
        'Hz']``, may also store diagonal components of permittivity tensor as ``'eps_xx', 'eps_yy',
        'eps_zz'``.
    x : np.ndarray
        x locations of each field and component. ``x.shape=(len(fields), num_x)``.
    y : np.ndarray
        y locations of each field and component. ``y.shape=(len(fields), num_y)``.
    z : np.ndarray
        z locations of each field and component. ``z.shape=(len(fields), num_z)``.
    f : np.ndarray
        Frequencies of the data (Hz).
    values : np.ndarray
        Complex-valued array of data values. ``values.shape=(len(field), num_x, num_y, num_z,
        len(f))``
    """

    monitor: FieldMonitor
    field: List[FieldType] = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    values: Array[complex]

    _dims = ("field", "x", "y", "z", "f")


class FieldTimeData(ScalarFieldData, TimeData):
    """Stores Electric and Magnetic fields from a FieldTimeMonitor.

    Parameters
    ----------
    monitor : ``FieldTimeMonitor``
        Original monitor object corresponding to data.
    monitor_name : str
        Name of original monitor in its Simulation object.
    field : List[str], optional
        Electromagnetic fields (E, H) in dtaset defaults to ``['Ex', 'Ey', 'Ez', 'Hx', 'Hy',
        'Hz']``.
    x : np.ndarray
        x locations of each field. ``x.shape=(len(fields), num_x)``.
    y : np.ndarray
        y locations of each field. ``y.shape=(len(fields), num_y)``.
    z : np.ndarray
        z locations of each field. ``z.shape=(len(fields), num_z)``.
    t : np.ndarray
        Time of the data (sec).
    values : np.ndarray
        Real-valued array of data values. ``values.shape=(len(field), num_x, num_y, num_z, len(t))``
    """

    monitor: FieldTimeMonitor
    values: Array[float]

    _dims = ("field", "x", "y", "z", "t")


class FluxData(AbstractFluxData, FreqData):
    """Stores power flux data through a planar ``FluxMonitor``

    Parameters
    ----------
    monitor : ``FluxMonitor``
        Original monitor object corresponding to data.
    monitor_name : str
        Name of original monitor in its Simulation object.
    f : np.ndarray
        Frequencies of the data (Hz).
    values : np.ndarray
        Complex-valued array of data values. ``values.shape=(len(f),)``
    """

    monitor: FluxMonitor
    values: Array[float]

    _dims = ("f",)


class FluxTimeData(AbstractFluxData, TimeData):
    """Stores power flux data through a planar ``FluxTimeMonitor``

    Parameters
    ----------
    monitor : ``FluxTimeMonitor``
        Original monitor object corresponding to data.
    monitor_name : str
        Name of original monitor in its Simulation object.
    t : np.ndarray
        Times of the data (sec).
    values : np.ndarray
        Complex-valued array of data values. ``values.shape=(len(t),)``
    """

    monitor: FluxTimeMonitor
    values: Array[float]

    _dims = ("t",)


class ModeData(PlanarData, FreqData):
    """Stores modal amplitdudes from a ModeMonitor

    Parameters
    ----------
    monitor : ``ModeMonitor``
        Original monitor object corresponding to data.
    monitor_name : str
        Name of original monitor in its Simulation object.
    direction : List[Literal["+", "-"]]
        Direction in which the modes are propagating (normal to monitor plane).
    mode_index : np.ndarray
        Array of integers into ``ModeMonitor.modes`` specifying the mode corresponding to this
        index.
    f : np.ndarray
        Frequencies of the data (Hz).
    values : np.ndarray
        Complex-valued array of data values. ``values.shape=(len(direction), len(mode_index),
        len(f))``
    """

    monitor: ModeMonitor
    direction: List[Direction] = ["+", "-"]
    mode_index: Array[int]
    values: Array[complex]

    _dims = ("direction", "mode_index", "f")


""" monitor_data_map explanation:
This dictionary maps monitor type to its corresponding data type
It is used to figure out what kind of MonitorData to load given a Monitor + raw data.
"""
monitor_data_map = {
    FieldMonitor: FieldData,
    FieldTimeMonitor: FieldTimeData,
    FluxMonitor: FluxData,
    FluxTimeMonitor: FluxTimeData,
    ModeMonitor: ModeData,
    ScalarFieldMonitor: ScalarFieldData,
    PlanarMonitor: PlanarData,
    AbstractFluxMonitor: AbstractFluxData,
    FreqMonitor: FreqData,
    TimeMonitor: TimeData,
}


class SimulationData(Tidy3dData):
    """holds simulation and its monitors' data.

    Parameters
    ----------
    simulation : ``Simulation``
        Original Simulation.
    monitor_data : Dict[str, ``MonitorData``]
        Mapping of monitor name to ``MonitorData`` intance.
    """

    simulation: Simulation
    monitor_data: Dict[str, MonitorData]

    def export(self, fname: str) -> None:
        """Export ``SimulationData`` to single hdf5 file including monitor data.

        Parameters
        ----------
        fname : str
            Path to data file (including filename).
        """

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

                    # add non-ignored names/values as hdf5 dataset
                    ignore = ("data", "monitor")
                    if name not in ignore:
                        mon_grp.create_dataset(name, data=value)

    @classmethod
    def load(cls, fname: str):
        """Load ``SimulationData`` from .hdf5 file

        Parameters
        ----------
        fname : str
            Path to data file (including filename).

        Returns
        -------
        ``SimulationData``
            A ``SimulationData`` instance.
        """

        # read from file at fname
        with h5py.File(fname, "r") as f_handle:

            # construct the original simulation from the json string
            sim_json = f_handle.attrs["sim_json"]
            sim = Simulation.parse_raw(sim_json)

            # loop through monitor dataset and create all MonitorData instances
            monitor_data = f_handle["monitor_data"]
            monitor_data_dict = {}
            for monitor_name, monitor_data in monitor_data.items():

                # load this MonitorData instance, add to monitor_data dict
                monitor = sim.monitors.get(monitor_name)
                monitor_data_instance = MonitorData.load_from_data(monitor, monitor_data)
                monitor_data_dict[monitor_name] = monitor_data_instance

        return cls(simulation=sim, monitor_data=monitor_data_dict)

    def __getitem__(self, monitor_name: str) -> MonitorData:
        """get the ``MonitorData`` xarray representation by name (``sim_data[monitor_name]``).

        Parameters
        ----------
        monitor_name : str
            Name of monitor to get data for.

        Returns
        -------
        Union[``xarray.DataArray``, ``xarray.Dataset``]
            The ``xarray`` representation of the data.
        """
        return self.monitor_data[monitor_name].data

    def __eq__(self, other):
        """check equality against another SimulationData instance

        Parameters
        ----------
        other : ``SimulationData``
            Another ``SimulationData`` instance to equate with self.

        Returns
        -------
        bool
            Whether the other ``SimulationData`` instance had the same data.
        """

        if self.simulation != other.simulation:
            return False
        for mon_name, mon_data in self.monitor_data.items():
            other_data = other.monitor_data.get(mon_name)
            if other_data is None:
                return False
            if mon_data != other.monitor_data[mon_name]:
                return False
        return True
