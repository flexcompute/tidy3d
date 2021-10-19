"""Classes for Storing Monitor and Simulation Data."""

from abc import ABC, abstractmethod
from typing import Dict, List, Union

import xarray as xr
import numpy as np
import h5py

from .types import Numpy, Direction, Array, numpy_encoding, Literal
from .base import Tidy3dBaseModel
from .simulation import Simulation
from .mode import Mode  # pylint: disable=unused-import
from ..log import log

""" Helper functions """


def save_string(hdf5_grp, string_key: str, string_value: str) -> None:
    """save a string to an hdf5 group"""
    str_type = h5py.special_dtype(vlen=str)
    hdf5_grp.create_dataset(string_key, (1,), dtype=str_type)
    hdf5_grp[string_key][0] = string_value


def decode_bytes(bytes_dataset) -> str:
    """decode an hdf5 dataset containing bytes to a string"""
    return bytes_dataset[0].decode("utf-8")


def load_string(hdf5_grp, string_key: str) -> str:
    """load a string from an hdf5 group"""
    string_value_bytes = hdf5_grp.get(string_key)
    if not string_value_bytes:
        return None
    return decode_bytes(string_value_bytes)


def decode_bytes_array(array_of_bytes: Numpy) -> List[str]:
    """convert numpy array containing bytes to list of strings"""
    list_of_bytes = array_of_bytes.tolist()
    list_of_str = [v.decode("utf-8") for v in list_of_bytes]
    return list_of_str


""" Base Classes """


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

    type: str = None

    def __init__(self, **kwargs):
        """compute xarray representation and add to ``self.data`` after init"""
        super().__init__(**kwargs)
        self.data = self._make_xarray()

    @abstractmethod
    def _make_xarray(self):
        """make xarray representation of stored data."""

    @abstractmethod
    def add_to_group(self, hdf5_grp):
        """add data contents to an hdf5 group"""

    @classmethod
    @abstractmethod
    def load_from_group(cls, hdf5_grp):
        """add data contents to an hdf5 group"""


class MonitorData(Tidy3dData, ABC):
    """Abstract base class.  Stores data.

    Attributes
    ----------
    data : ``Union[xarray.DataArray xarray.Dataset]``
        Representation of the data as an xarray object.
    """

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

    def _make_xarray(self) -> Union[xr.DataArray, xr.Dataset]:
        """make xarray representation of data

        Returns
        -------
        ``Union[xarray.DataArray xarray.Dataset]``
            Representation of the underlying data using xarray.
        """
        data_dict = self.dict()
        coords = {dim: data_dict[dim] for dim in self._dims}
        return xr.DataArray(self.values, coords=coords)

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

    def add_to_group(self, hdf5_grp) -> None:
        """add data contents to an hdf5 group"""

        # save the type information of MonitorData to the group
        save_string(hdf5_grp, "type", self.type)
        for data_name, data_value in self.dict().items():

            # for each data member in self._dims (+ values), add to group.
            if data_name in self._dims or data_name in ("values",):
                hdf5_grp.create_dataset(data_name, data=data_value)

    @classmethod
    def load_from_group(cls, hdf5_grp):
        """load the solver data dict for a specific monitor into a MonitorData instance"""

        # kwargs that gets passed to MonitorData.__init__() to make new MonitorData
        kwargs = {}

        # construct kwarg dict from hdf5 data group for monitor
        for data_name, data_value in hdf5_grp.items():
            kwargs[data_name] = np.array(data_value)

        # handle data stored as np.array() of bytes instead of strings
        # for str_kwarg in ("field", "direction"):
        #     if kwargs.get(str_kwarg) is not None:
        #         kwargs[str_kwarg] = decode_bytes_array(kwargs[str_kwarg])

        # handle data stored as np.array() of bytes instead of strings
        # for str_kwarg in ("x", "y", "z"):
        #     if kwargs.get(str_kwarg) is not None:
        #         kwargs[str_kwarg] = kwargs[str_kwarg].tolist()

        # ignore the "type" dataset as it's used for finding type for loading
        kwargs.pop("type")

        return cls(**kwargs)


class CollectionData(Tidy3dData):
    """Abstract base class.  Stores collection of data with similar dimensions.

    Parameters
    ----------
    data_dict : ``{str : :class:`MonitorData`}
        mapping of field name to corresponding :class:`MonitorData`.
    """

    data_dict: Dict[str, MonitorData]

    def _make_xarray(self):
        """For field quantities, store a single xarray DataArray for each ``field``.
        These all go in a single xarray Dataset, which keeps track of the shared coords.

        Returns
        -------
        ```xarray.Dataset <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`__``
            Representation of the underlying data using xarray.
        """
        data_arrays = {name: arr.data for name, arr in self.data_dict.items()}

        # make an xarray dataset
        return xr.Dataset(data_arrays)

    def __eq__(self, other):
        """check for equality against other :class:`CollectionData` object."""

        # same keys?
        if not all(k in other.data_dict.keys() for k in self.data_dict.keys()):
            return False
        if not all(k in self.data_dict.keys() for k in other.data_dict.keys()):
            return False
        # same data?
        for data_name, data_value in self.data_dict.items():
            if data_value != other.data_dict[data_name]:
                return False
        return True

    def add_to_group(self, hdf5_grp) -> None:
        """add data from a :class:`CollectionData` to an hdf5 group ."""

        # put collection's type information into the group
        save_string(hdf5_grp, "type", self.type)
        for data_name, data_value in self.data_dict.items():

            # create a new group for each member of collection and add its data
            data_grp = hdf5_grp.create_group(data_name)
            data_value.add_to_group(data_grp)

    @classmethod
    def load_from_group(cls, hdf5_grp):
        """load a :class:`CollectionData` from hdf5 group containing data."""
        data_dict = {}
        for data_name, data_value in hdf5_grp.items():

            # hdf5 group contains `type` dataset, ignore it.
            if data_name == "type":
                continue

            # get the type from MonitorData.type and add instance to dict
            data_type = data_type_map[load_string(data_value, "type")]
            data_dict[data_name] = data_type.load_from_group(data_value)

        return cls(data_dict=data_dict)


""" The following
are abstract classes that separate the :class:`MonitorData` instances into
    different types depending on what they store. 
    They can be useful for keeping argument types and validations separated.
    For example, monitors that should always be defined on planar geometries can have an 
    ``_assert_plane()`` validation in the abstract base class ``PlanarData``.
    This way, ``_assert_plane()`` will always be used if we add more ``PlanarData`` objects in
    the future.
    This organization is also useful when doing conditions based on monitor / data type.
    For example, instead of 
    ``if isinstance(mon_data, (FieldData, FieldTimeData)):`` we can simply do 
    ``if isinstance(mon_data, AbstractFieldData)`` and this will generalize if we add more
    ``AbstractFieldData`` objects in the future.
"""


class FreqData(MonitorData, ABC):
    """Stores frequency-domain data using an ``f`` attribute for frequency (Hz)."""

    f: Array[float]


class TimeData(MonitorData, ABC):
    """Stores time-domain data using a ``t`` attribute for time (sec)."""

    t: Array[float]


class AbstractScalarFieldData(MonitorData, ABC):
    """Stores a single field as a functio of x,y,z and sampler"""

    x: Array[float]
    y: Array[float]
    z: Array[float]
    values: Union[Array[complex], Array[float]]


class PlanarData(MonitorData, ABC):
    """Stores data that is constrained to the plane."""


class AbstractFluxData(PlanarData, ABC):
    """Stores electromagnetic flux through a planar :class:`Monitor`"""


""" usable monitors """


class ScalarFieldData(AbstractScalarFieldData, FreqData):
    """stores a single scalar field in frequency domain

    Parameters
    ----------
    data_dict : ``{str : :class:`ScalarFieldData`}
        mapping of field name to corresponding :class:`ScalarFieldData`.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 1001)
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-2, 2, 20)
    >>> z = np.linspace(0, 0, 1)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(z), len(f)))
    >>> data = ScalarFieldData(values=values, x=x, y=y, z=z, f=f)
    """

    values: Array[complex]
    type: Literal["ScalarFieldData"] = "ScalarFieldData"

    _dims = ("x", "y", "z", "f")


class ScalarFieldTimeData(AbstractScalarFieldData, TimeData):
    """stores a single scalar field in time domain

    Parameters
    ----------
    data_dict : ``{str : :class:`ScalarFieldTimeData`}
        mapping of field name to corresponding :class:`ScalarFieldTimeData`.

    Example
    -------
    >>> t = np.linspace(0, 1e-12, 1001)
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-2, 2, 20)
    >>> z = np.linspace(0, 0, 1)
    >>> values = np.random.random((len(x), len(y), len(z), len(t)))
    >>> data = ScalarFieldTimeData(values=values, x=x, y=y, z=z, t=t)
    """

    values: Array[float]
    type: Literal["ScalarFieldTimeData"] = "ScalarFieldTimeData"

    _dims = ("x", "y", "z", "t")


class FieldData(CollectionData):
    """Stores a collectio of scalar field quantities as a function of x, y, and z.

    Parameters
    ----------
    data_dict : ``{str : :class:`ScalarFieldTimeData`}
        mapping of field name to corresponding :class:`ScalarFieldTimeData`.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 1001)
    >>> t = np.linspace(0, 1e-12, 1001)
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-2, 2, 20)
    >>> z = np.linspace(0, 0, 1)
    >>> values_f = np.random.random((len(x), len(y), len(z), len(f)))
    >>> values_t = np.random.random((len(x), len(y), len(z), len(t)))
    >>> field_f = ScalarFieldData(values=values_f, x=x, y=y, z=z, f=f)
    >>> field_t = ScalarFieldTimeData(values=values_t, x=x, y=y, z=z, t=t)
    >>> data_f = FieldData(data_dict={'Ex': field_f, 'Ey': field_f})
    >>> data_t = FieldData(data_dict={'Ex': field_t, 'Ey': field_t})
    """

    data_dict: Dict[str, Union[ScalarFieldData, ScalarFieldTimeData]]
    type: Literal["FieldData"] = "FieldData"


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
    >>> values = np.random.random((len(f),))
    >>> data = FluxData(values=values, f=f)
    """

    values: Array[float]
    type: Literal["FluxData"] = "FluxData"

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

    >>> t = np.linspace(0, 1e-12, 1001)
    >>> values = np.random.random((len(t),))
    >>> data = FluxTimeData(values=values, t=t)
    """

    values: Array[float]
    type: Literal["FluxTimeData"] = "FluxTimeData"

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
    >>> values = (1+1j) * np.random.random((1, 2, len(f)))
    >>> data = ModeData(values=values, direction=['+'], mode_index=np.arange(1, 3), f=f)
    """

    direction: List[Direction] = ["+", "-"]
    mode_index: Array[int]
    values: Array[complex]
    type: Literal["ModeData"] = "ModeData"

    _dims = ("direction", "mode_index", "f")


# maps MonitorData.type string to the actual type, for MonitorData.load()
data_type_map = {
    "ScalarFieldData": ScalarFieldData,
    "ScalarFieldTimeData": ScalarFieldTimeData,
    "FieldData": FieldData,
    "FluxData": FluxData,
    "FluxTimeData": FluxTimeData,
    "ModeData": ModeData,
}


class SimulationData(Tidy3dBaseModel):
    """Holds :class:`Monitor` data associated with :class:`Simulation`.

    Parameters
    ----------
    simulation : :class:`Simulation`
        Original :class:`Simulation`.
    monitor_data : ``Dict[str, :class:`Tidy3dData`]``
        Mapping of monitor name to :class:`Tidy3dData` intance. The dictionary keys must
        exist in ``simulation.monitors.keys()``.
    log_string : ``str``, optional
        string containing the log from server.
    """

    simulation: Simulation
    monitor_data: Dict[str, Union[MonitorData, FieldData]]
    log_string: str = None

    @property
    def log(self):
        """prints the server-side log
        TODO: store log metadata inside of SimulationData.info or something (credits billed, time)
        """
        print(self.log_string if self.log_string else "no log stored")

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
            save_string(f_handle, "sim_json", self.simulation.json())

            if self.log_string:
                save_string(f_handle, "log_string", self.log_string)

            # make a group for monitor_data
            mon_data_grp = f_handle.create_group("monitor_data")
            for mon_name, mon_data in self.monitor_data.items():

                # for each monitor, make new group with the same name
                mon_grp = mon_data_grp.create_group(mon_name)
                mon_data.add_to_group(mon_grp)

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
            sim_json = load_string(f_handle, "sim_json")
            simulation = Simulation.parse_raw(sim_json)

            # get the log if exists
            log_string = load_string(f_handle, "log_string")

            # loop through monitor dataset and create all MonitorData instances
            monitor_data_dict = {}
            for monitor_name, monitor_data in f_handle["monitor_data"].items():

                # load this MonitorData instance, add to monitor_data dict
                data_type = data_type_map[load_string(monitor_data, "type")]
                monitor_data_instance = data_type.load_from_group(monitor_data)
                monitor_data_dict[monitor_name] = monitor_data_instance

        return cls(
            simulation=simulation,
            monitor_data=monitor_data_dict,
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
        monitor_data = self.monitor_data.get(monitor_name)
        if not monitor_data:
            log.error(f"monitor {monitor_name} not found")
        return monitor_data.data

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

        # check if they have the same simulation
        if self.simulation != other.simulation:
            return False

        # check if each monitor data are equal
        for mon_name, mon_data in self.monitor_data.items():
            other_data = other.monitor_data.get(mon_name)
            if other_data is None:
                return False
            if mon_data != other.monitor_data[mon_name]:
                return False

        # if never returned False, they are equal
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
