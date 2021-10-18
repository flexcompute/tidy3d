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
from .mode import Mode  # pylint: disable=unused-import


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
            xr.DataArray: lambda x: None,  # dont write
            xr.Dataset: lambda x: None,  # dont write
        }


class MonitorData(Tidy3dData, ABC):
    """Abstract base class.  Stores data corresponding to a :class:`Monitor`.

    Attributes
    ----------
    data : ``Union[xarray.DataArray xarray.Dataset]``
        Representation of the data as an xarray object.
    """

    monitor_name: str
    monitor: Monitor

    """ explanation of values
        `values` is a numpy array that stores the raw data associated with each
        :class:`MonitorData`.
        It can be complex-valued or real valued, depending on whether data is in the frequency or
        time domain, respectively.
        Each axis in ``values`` corresponds to a specific dimension in the :class:`MonitorData`,
        which are supplied as arguments to the :class:`MonitorData` subclasses.
        The order of the dimensions is specified in the ``_dims`` attribute of each
        :class:`MonitorData` subclass
    """

    values: Union[Array[float], Array[complex]]

    """ explanation of``_dims``
        `_dims` is an attribute of all `MonitorData` objects.
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
        ``Union[xarray.DataArray xarray.Dataset]``
            Representation of the underlying data using xarray.
        """
        data_dict = self.dict()
        coords = {dim: data_dict[dim] for dim in self._dims}
        return xr.DataArray(self.values, coords=coords, name=self.monitor_name)

    def __eq__(self, other) -> bool:
        """check equality against another MonitorData instance

        Parameters
        ----------
        other : :class:`MonitorData`
            Other :class:`MonitorData` to equate to.

        Returns
        -------
        ``bool``
            Whether the other :class:`MonitorData` instance has the same data.
        """
        assert isinstance(other, MonitorData), "can only check eqality on two monitor data objects"
        return np.all(self.values == self.values)

    @property
    def geometry(self):
        """Return :class:`Box` representation of monitor's geometry.

        Returns
        -------
        :class:`Box`
            :class:`Box` represention of shape of originl monitor.
        """
        return self.monitor.geometry

    def export(self, fname: str) -> None:
        """Export :class:`MonitorData` to .hdf5 file.

        Parameters
        ----------
        fname : ``str``
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
        """Load :class:`MonitorData` from .hdf5 file

        Parameters
        ----------
        fname : ``str``
            Path to data file (including filename).

        Returns
        -------
        :class:`MonitorData`
            A :class:`MonitorData` instance.
        """

        with h5py.File(fname, "r") as f_handle:

            # construct the original :class:`Monitor`from the json string
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
        monitor : :class:`Monitor`
            original :class:`Monitor`that specified how data was stored.
        monitor_data : ``Dict[str, Numpy]``
            Mapping from data value name to numpy array holding data.

        Returns
        -------
        :class:`MonitorData`
            A :class:`MonitorData` instance.
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

        # handle data stored as np.array() of bytes instead of strings
        for str_kwarg in ("x", "y", "z"):
            if kwargs.get(str_kwarg) is not None:
                kwargs[str_kwarg] = kwargs[str_kwarg].tolist()

        # convert name to string and add monitor to kwargs
        kwargs["monitor_name"] = str(kwargs["monitor_name"])
        kwargs["monitor"] = monitor

        # get MontiorData type and initialize using kwargs
        mon_type = type(monitor)
        mon_data_type = monitor_data_map[mon_type]

        # convert values to list of arrays, one per field
        if mon_data_type in (FieldData, FieldTimeData):
            kwargs["values"] = list(kwargs["values"])

        monitor_data_instance = mon_data_type(**kwargs)
        return monitor_data_instance


""" The following are abstract classes that separate the :class:`MonitorData` instances into
    different types depending on what they store. 
    They can be useful for keeping argument types and validations separated.
    For example, monitors that should always be defined on planar geometries can have an 
    ``_assert_plane()`` validation in the abstract base class ``PlanarData``.
    This way, ``_assert_plane()`` will always be used if we add more ``PlanarData`` objects in
    the future.
    This organization is also useful when doing conditions based on monitor / data type.
    For example, instead of 
    ``if isinstance(mon_data, (FieldData, FieldTimeData)):`` we can simply do 
    ``if isinstance(mon_data, ScalarFieldData)`` and this will generalize if we add more
    ``ScalarFieldData`` objects in the future.
"""


class FreqData(MonitorData, ABC):
    """Stores frequency-domain data using an ``f`` attribute for frequency (Hz)."""

    monitor: FreqMonitor
    f: Array[float]


class TimeData(MonitorData, ABC):
    """Stores time-domain data using a ``t`` attribute for time (sec)."""

    monitor: TimeMonitor
    t: Array[float]


class ScalarFieldData(MonitorData, ABC):
    """Stores `field` quantities as a function of x, y, and z."""

    monitor: ScalarFieldMonitor
    field: List[EMField] = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    x: List[Array[float]]
    y: List[Array[float]]
    z: List[Array[float]]
    values: List[Union[Array[complex], Array[float]]]

    def _make_xarray(self):
        """For field quantities, store a single xarray DataArray for each ``field``.
        These all go in a single xarray Dataset, which keeps track of the shared coords.

        Returns
        -------
        ```xarray.Dataset <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`__``
            Representation of the underlying data using xarray.
        """

        data_dict = self.dict()

        # for each `field`, create xrray DataArray and add to dictionary.
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

        # make an xarray dataset
        return xr.Dataset(data_arrays)


class PlanarData(MonitorData, ABC):
    """Stores data that is constrained to the plane."""

    monitor: PlanarMonitor


class AbstractFluxData(PlanarData, ABC):
    """Stores electromagnetic flux through a planar :class:`Monitor`"""

    monitor: AbstractFluxMonitor


""" usable monitors """


class FieldData(FreqData, ScalarFieldData):
    """Stores Electric and Magnetic fields from a :class:`FieldMonitor`.

    Parameters
    ----------
    monitor : :class:`FieldMonitor`
        original :class:`Monitor` object corresponding to data.
    monitor_name : str
        Name of original :class:`Monitor` in the original :attr:`Simulation.monitors` dictionary..
    field: List[str], optional
        Electromagnetic fields (E, H) in dtaset defaults to ``['Ex', 'Ey', 'Ez', 'Hx', 'Hy',
        'Hz']``, may also store diagonal components of permittivity tensor as ``'eps_xx', 'eps_yy',
        'eps_zz'``.
    x : ``List[numpy.ndarray]``
        x locations of each field. ``len(x) == len(fields)``, ``len(x[i]) == num_x``.
    y : ``List[numpy.ndarray]``
        y locations of each field. ``len(y) == len(fields)``, ``len(y[i]) == num_y``.
    z : ``List[numpy.ndarray]``
        z locations of each field. ``len(z) == len(fields)``, ``len(z[i]) == num_z``.
    f : ``numpy.ndarray``
        Frequencies of the data (Hz).
    values : ``List[numpy.ndarray]``
        List of Complex-valued array of data values for each field.
        ``len(values) == len(fields)``.
        ``values[i].shape == (x[i], y[i], z[i], len(t))``

    Example
    -------
    >>> f = np.linspace(2e14, 3e14, 1001)
    >>> monitor = FieldMonitor(fields=['Ex'], size=(2, 4, 0), freqs=list(f))
    >>> x = [np.linspace(-1, 1, 10)]
    >>> y = [np.linspace(-2, 2, 20)]
    >>> z = [np.linspace(0, 0, 1)]
    >>> values = [np.random.random((10, 20, 1, len(f)))]
    >>> data = FieldData(
    ...     monitor=monitor,
    ...     monitor_name='ex',
    ...     values=values,
    ...     field=['Ex'],
    ...     x=x,
    ...     y=y,
    ...     z=z,
    ...     f=f)

    """

    monitor: FieldMonitor
    field: List[FieldType] = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    values: List[Array[complex]]

    _dims = ("field", "x", "y", "z", "f")


class FieldTimeData(ScalarFieldData, TimeData):
    """Stores Electric and Magnetic fields from a :class:`FieldTimeMonitor`.

    Parameters
    ----------
    monitor : :class:`FieldTimeMonitor`
        original :class:`Monitor` object corresponding to data.
    monitor_name : str
        Name of original :class:`Monitor` in the original :attr:`Simulation.monitors` dictionary.
    field : List[str], optional
        Electromagnetic fields (E, H) in dtaset defaults to ``['Ex', 'Ey', 'Ez', 'Hx', 'Hy',
        'Hz']``.
    x : ``List[numpy.ndarray]``
        x locations of each field. ``len(x) == len(fields)``, ``len(x[i]) == num_x``.
    y : ``List[numpy.ndarray]``
        y locations of each field. ``len(y) == len(fields)``, ``len(y[i]) == num_y``.
    z : ``List[numpy.ndarray]``
        z locations of each field. ``len(z) == len(fields)``, ``len(z[i]) == num_z``.
    t : ``numpy.ndarray``
        Time of the data (sec).
    values : ``List[numpy.ndarray]``
        List of Real-valued array of data values for each field.
        ``len(values) == len(fields)``.
        ``values[i].shape == (x[i], y[i], z[i], len(t))``

    Example
    -------

    >>> times = np.arange(0, 1000, 101)
    >>> monitor = FieldTimeMonitor(fields=['Hy'], size=(2, 4, 0), times=list(times))
    >>> x = [np.linspace(-1, 1, 10)]
    >>> y = [np.linspace(-2, 2, 20)]
    >>> z = [np.linspace(0, 0, 1)]
    >>> dt = 1e-13
    >>> t = times * dt
    >>> values = [np.random.random((10, 20, 1, len(t)))]
    >>> data = FieldTimeData(
    ...     monitor=monitor,
    ...     monitor_name='hy',
    ...     values=values,
    ...     field=['Hy'],
    ...     x=x,
    ...     y=y,
    ...     z=z,
    ...     t=t)
    """

    monitor: FieldTimeMonitor
    values: List[Array[float]]

    _dims = ("field", "x", "y", "z", "t")


class FluxData(AbstractFluxData, FreqData):
    """Stores power flux data through a planar :class:`FluxMonitor`.

    Parameters
    ----------
    monitor : :class:`FluxMonitor`
        original :class:`Monitor` object corresponding to data.
    monitor_name : str
        Name of original :class:`Monitor` in the original :attr:`Simulation.monitors` dictionary..
    f : ``numpy.ndarray``
        Frequencies of the data (Hz).
    values : ``numpy.ndarray``
        Complex-valued array of data values. ``values.shape=(len(f),)``

    Example
    -------

    >>> f = np.linspace(2e14, 3e14, 1001)
    >>> monitor = FluxMonitor(size=(2, 4, 0), freqs=list(f))
    >>> values = np.random.random((1001,))
    >>> data = FluxData(monitor=monitor, monitor_name='flux', values=values, f=f)
    """

    monitor: FluxMonitor
    values: Array[float]

    _dims = ("f",)


class FluxTimeData(AbstractFluxData, TimeData):
    """Stores power flux data through a planar :class:`FluxTimeMonitor`

    Parameters
    ----------
    monitor : :class:`FluxTimeMonitor`
        Original :class:`Monitor` object corresponding to data.
    monitor_name : ``str``
        Name of original :class:`Monitor` in the original :attr:`Simulation.monitors` dictionary.
    t : ``numpy.ndarray``
        Times of the data (sec).
    values : ``numpy.ndarray``
        Real-valued array of data values. ``values.shape=(len(t),)``

    Example
    -------

    >>> times = np.arange(0, 1000, 51)
    >>> monitor = FluxTimeMonitor(size=(2, 4, 0), times=list(times))
    >>> dt = 1e-13
    >>> t = times * dt
    >>> values = np.random.random(times.shape)
    >>> data = FluxTimeData(monitor=monitor, monitor_name='flux', values=values, t=t)
    """

    monitor: FluxTimeMonitor
    values: Array[float]

    _dims = ("t",)


class ModeData(PlanarData, FreqData):
    """Stores modal amplitdudes from a :class:`ModeMonitor`.

    Parameters
    ----------
    monitor : :class:`ModeMonitor`
        original :class:`Monitor` object corresponding to data.
    monitor_name : ``str``
        Name of original :class:`Monitor` in the original :attr:`Simulation.monitors` dictionary.
    direction : ``List[Literal["+", "-"]]``
        Direction in which the modes are propagating (normal to monitor plane).
    mode_index : ``numpy.ndarray``
        Array of integers into :attr:`ModeMonitor.modes` specifying the mode corresponding to this
        index.
    f : ``numpy.ndarray``
        Frequencies of the data (Hz).
    values : ``numpy.ndarray``
        Complex-valued array of data values. ``values.shape=(len(direction), len(mode_index),
        len(f))``

    Example
    -------

    >>> f = np.linspace(2e14, 3e14, 1001)
    >>> modes = [Mode(mode_index=0), Mode(mode_index=1)]
    >>> monitor = ModeMonitor(direction=['+'], size=(2, 4, 0), modes=modes, freqs=list(f))
    >>> values = (1+1j) * np.random.random((1, 2, 1001))
    >>> data = ModeData(
    ...     monitor=monitor,
    ...     monitor_name='mode',
    ...     values=values,
    ...     direction=['+'],
    ...     mode_index=np.arange(1, 3),
    ...     f=f)
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
    """Holds simulation and its :class:`Monitor` data.

    Parameters
    ----------
    simulation : :class:`Simulation`
        Original :class:`Simulation`.
    monitor_data : ``Dict[str, :class:`MonitorData`]``
        Mapping of monitor name to :class:`MonitorData` intance. The dictionary keys must
        exist in ``simulation.monitors.keys()``.
    task_info : ``dict``, optional
        metadata about task that generated this SimulationData at the time it was downloaded.
    log_string : ``str``, optional
        string containing the log from server.

    Example
    -------

    >>> f = np.linspace(2e14, 3e14, 1001)
    >>> mnt_name = 'flux'
    >>> flux_monitor = FluxMonitor(size=(2, 4, 0), freqs=list(f))
    >>> simulation = Simulation(
    ...     size=(4,4,4),
    ...     grid_size=(0.1, 0.1, 0.1),
    ...     monitors={mnt_name: flux_monitor})
    >>> values = np.random.random((1001,))
    >>> flux_data = FluxData(monitor=flux_monitor, monitor_name=mnt_name, values=values, f=f)
    >>> sim_data = SimulationData(simulation=simulation, monitor_data={mnt_name: flux_data})
    """

    simulation: Simulation
    monitor_data: Dict[str, MonitorData]
    task_info: dict = None
    log_string: str = None

    @property
    def log(self):
        """prints the server-side log
        TODO: store log metadata inside of SimulationData.info or something (credits billed, time)
        """
        if self.log_string:
            print(self.log_string)
        else:
            print("no log stored.")

    def export(self, fname: str) -> None:
        """Export :class:`SimulationData` to single hdf5 file including monitor data.

        Parameters
        ----------
        fname : ``str``
            Path to data file (including filename).
        """

        """ TODO: Provide optional args to only export the MonitorData of selected monitors. """

        with h5py.File(fname, "a") as f_handle:

            # save json string as an attribute
            sim_json = self.simulation.json()

            # TODO: wrap this in a small helper function?
            str_type = h5py.special_dtype(vlen=str)

            f_handle.create_dataset("sim_json", (1,), dtype=str_type)
            f_handle["sim_json"][0] = sim_json

            if self.log_string:
                f_handle.create_dataset("log_string", (1,), dtype=str_type)
                f_handle["log_string"][0] = self.log_string

            if self.task_info:
                f_handle.create_dataset("task_info", (1,), dtype=str_type)
                f_handle["task_info"][0] = json.dumps(self.task_info)

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
        """Load :class:`SimulationData` from .hdf5 file

        Parameters
        ----------
        fname : ``str``
            Path to data file (including filename).

        Returns
        -------
        :class:`SimulationData`
            A :class:`SimulationData` instance.
        """

        # read from file at fname
        with h5py.File(fname, "r") as f_handle:

            # construct the original simulation from the json string
            sim_json = f_handle["sim_json"][0]
            simulation = Simulation.parse_raw(sim_json)

            # get the log if exists
            log_string = f_handle.get("log_string")
            if log_string:
                log_string = log_string[0]

            # get the task_info if exists
            task_info = f_handle.get("task_info")
            if task_info:
                task_info = json.loads(task_info[0])

            # loop through monitor dataset and create all MonitorData instances
            monitor_data = f_handle["monitor_data"]
            monitor_data_dict = {}
            for monitor_name, monitor_data in monitor_data.items():

                # load this MonitorData instance, add to monitor_data dict
                monitor = simulation.monitors.get(monitor_name)
                monitor_data_instance = MonitorData.load_from_data(monitor, monitor_data)
                monitor_data_dict[monitor_name] = monitor_data_instance

        return cls(
            simulation=simulation,
            monitor_data=monitor_data_dict,
            task_info=task_info,
            log_string=log_string,
        )

    def __getitem__(self, monitor_name: str) -> MonitorData:
        """get the :class:`MonitorData` xarray representation by name (``sim_data[monitor_name]``).

        Parameters
        ----------
        monitor_name : str
            Name of :class:`Monitor` to get data for.

        Returns
        -------
        ``Union[xarray.DataArray``, xarray.Dataset]``
            The ``xarray`` representation of the data.
        """
        return self.monitor_data[monitor_name].data

    def __eq__(self, other):
        """check equality against another SimulationData instance

        Parameters
        ----------
        other : :class:`SimulationData`
            Another :class:`SimulationData` instance to equate with self.

        Returns
        -------
        bool
            Whether the other :class:`SimulationData` instance had the same data.
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


""" TODO:
 - assert all SimulationData.monitor_data.keys() are in SimulationData.simulation.monitor.keys()
 - remove MonitorData.monitor and MonitorData.monitor_name attributes - actually, can keep
 - remove MonitorData.load() and MonitorData.export() - can keep but should not be unused, so
   maybe remove
 - provide optional args to SimulationData.export() and SimulationData.load() to only export/load
   a subset of all possible MonitorData objects
 - change data dictionary structure to ['monitor_name']['Ex']['values'], etc.
"""
