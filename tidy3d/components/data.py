# pylint: disable=unused-import
"""Classes for Storing Monitor and Simulation Data."""

from abc import ABC, abstractmethod
from typing import Dict, List, Union

import xarray as xr
import numpy as np
import h5py

from .types import Numpy, Direction, Array, numpy_encoding, Literal, Ax
from .base import Tidy3dBaseModel
from .simulation import Simulation
from .mode import Mode
from .viz import add_ax_if_none
from ..log import DataError


""" Helper functions """


def save_string(hdf5_grp, string_key: str, string_value: str) -> None:
    """Save a string to an hdf5 group."""
    str_type = h5py.special_dtype(vlen=str)
    hdf5_grp.create_dataset(string_key, (1,), dtype=str_type)
    hdf5_grp[string_key][0] = string_value


def decode_bytes(bytes_dataset) -> str:
    """Decode an hdf5 dataset containing bytes to a string."""
    return bytes_dataset[0].decode("utf-8")


def load_string(hdf5_grp, string_key: str) -> str:
    """Load a string from an hdf5 group."""
    string_value_bytes = hdf5_grp.get(string_key)
    if not string_value_bytes:
        return None
    return decode_bytes(string_value_bytes)


def decode_bytes_array(array_of_bytes: Numpy) -> List[str]:
    """Convert numpy array containing bytes to list of strings."""
    list_of_bytes = array_of_bytes.tolist()
    list_of_str = [v.decode("utf-8") for v in list_of_bytes]
    return list_of_str


""" Base Classes """


class Tidy3dData(Tidy3dBaseModel):
    """Base class for data associated with a simulation."""

    class Config:  # pylint: disable=too-few-public-methods
        """Configuration for all Tidy3dData objects."""

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

    @abstractmethod
    def add_to_group(self, hdf5_grp):
        """Add data contents to an hdf5 group."""

    @classmethod
    @abstractmethod
    def load_from_group(cls, hdf5_grp):
        """Load data contents from an hdf5 group."""


class MonitorData(Tidy3dData, ABC):
    """Abstract base class for objects storing individual data from simulation."""

    values: Union[Array[float], Array[complex]]
    type: str = None

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

    _dims = ()

    """ explanation of``_dims``
        `_dims` is an attribute of all `MonitorData` objects.
        It is a tuple of strings that stores the keys of the coordinates corresponding to `values`.
        Note: they must be in order corresponding to their index into `values`.
        The underscore is used so _dims() is a class variable and not stored in .json.
        The dims are used to construct xarray objects as it tells the _make_xarray method what
        attribute to use for the keys in the `coords` coordinate dictionary.
    """

    @property
    def data(self) -> xr.DataArray:
        # pylint:disable=line-too-long
        """Returns an xarray representation of the montitor data.

        Returns
        -------
        xarray.DataArray
            Representation of the monitor data using xarray.
            For more details refer to `xarray's Documentaton <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html>`_.
        """
        # pylint:enable=line-too-long

        data_dict = self.dict()
        coords = {dim: data_dict[dim] for dim in self._dims}
        return xr.DataArray(self.values, coords=coords)

    def __eq__(self, other) -> bool:
        """Check equality against another MonitorData instance.

        Parameters
        ----------
        other : :class:`MonitorData`
            Other :class:`MonitorData` to equate to.

        Returns
        -------
        bool
            Whether the other :class:`MonitorData` instance has the same data.
        """
        assert isinstance(other, MonitorData), "can only check eqality on two monitor data objects"
        return np.all(self.values == self.values)

    def add_to_group(self, hdf5_grp) -> None:
        """Add data contents to an hdf5 group."""

        # save the type information of MonitorData to the group
        save_string(hdf5_grp, "type", self.type)
        for data_name, data_value in self.dict().items():

            # for each data member in self._dims (+ values), add to group.
            if data_name in self._dims or data_name in ("values",):
                hdf5_grp.create_dataset(data_name, data=data_value)

    @classmethod
    def load_from_group(cls, hdf5_grp):
        """Load Monitor data instance from an hdf5 group."""

        # kwargs that gets passed to MonitorData.__init__() to make new MonitorData
        kwargs = {}

        # construct kwarg dict from hdf5 data group for monitor
        for data_name, data_value in hdf5_grp.items():
            kwargs[data_name] = np.array(data_value)

        # handle data stored as np.array() of bytes instead of strings
        for str_kwarg in ("direction",):
            if kwargs.get(str_kwarg) is not None:
                kwargs[str_kwarg] = decode_bytes_array(kwargs[str_kwarg])

        # handle data stored as np.array() of bytes instead of strings
        # for str_kwarg in ("x", "y", "z"):
        #     if kwargs.get(str_kwarg) is not None:
        #         kwargs[str_kwarg] = kwargs[str_kwarg].tolist()

        # ignore the "type" dataset as it's used for finding type for loading
        kwargs.pop("type")

        return cls(**kwargs)


class CollectionData(Tidy3dData):
    """Abstract base class.  Stores a collection of data with same dimension types (such as field).

    Parameters
    ----------
    data_dict : Dict[str, :class:`MonitorData`]
        Mapping of collection member name to corresponding :class:`MonitorData`.
    """

    data_dict: Dict[str, MonitorData]
    type: str = None

    @property
    def data(self) -> xr.Dataset:
        # pylint:disable=line-too-long
        """For field quantities, store a single xarray DataArray for each ``field``.
        These all go in a single xarray Dataset, which keeps track of the shared coords.

        Returns
        -------
        xarray.Dataset
            Representation of the underlying data using xarray.
            For more details refer to `xarray's Documentaton <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`_.
        """
        # pylint:enable=line-too-long
        data_arrays = {name: arr.data for name, arr in self.data_dict.items()}

        # make an xarray dataset
        return xr.Dataset(data_arrays)

    def __eq__(self, other):
        """Check for equality against other :class:`CollectionData` object."""

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
        """Add data from a :class:`CollectionData` to an hdf5 group ."""

        # put collection's type information into the group
        save_string(hdf5_grp, "type", self.type)
        for data_name, data_value in self.data_dict.items():

            # create a new group for each member of collection and add its data
            data_grp = hdf5_grp.create_group(data_name)
            data_value.add_to_group(data_grp)

    @classmethod
    def load_from_group(cls, hdf5_grp):
        """Load a :class:`CollectionData` from hdf5 group containing data."""
        data_dict = {}
        for data_name, data_value in hdf5_grp.items():

            # hdf5 group contains `type` dataset, ignore it.
            if data_name == "type":
                continue

            # get the type from MonitorData.type and add instance to dict
            data_type = data_type_map[load_string(data_value, "type")]
            data_dict[data_name] = data_type.load_from_group(data_value)

        return cls(data_dict=data_dict)


""" Classes of Monitor Data """


class FreqData(MonitorData, ABC):
    """Stores frequency-domain data using an ``f`` dimension for frequency in Hz."""

    f: Array[float]


class TimeData(MonitorData, ABC):
    """Stores time-domain data using a ``t`` attribute for time in seconds."""

    t: Array[float]


class AbstractScalarFieldData(MonitorData, ABC):
    """Stores a single, scalar field as a function of spatial coordinates x,y,z."""

    x: Array[float]
    y: Array[float]
    z: Array[float]
    # values: Union[Array[complex], Array[float]]


class PlanarData(MonitorData, ABC):
    """Stores data that must be found via a planar monitor."""


class AbstractFluxData(PlanarData, ABC):
    """Stores electromagnetic flux through a plane."""


""" usable monitors """


class ScalarFieldData(AbstractScalarFieldData, FreqData):
    """Stores a single scalar field in frequency-domain.

    Parameters
    ----------
    x : numpy.ndarray
        Data coordinates in x direction (um).
    y : numpy.ndarray
        Data coordinates in y direction (um).
    z : numpy.ndarray
        Data coordinates in z direction (um).
    f : numpy.ndarray
        Frequency coordinates (Hz).
    values : numpy.ndarray
        Complex-valued array of shape ``(len(x), len(y), len(z), len(f))`` storing field values.

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
    x : numpy.ndarray
        Data coordinates in x direction (um).
    y : numpy.ndarray
        Data coordinates in y direction (um).
    z : numpy.ndarray
        Data coordinates in z direction (um).
    t : numpy.ndarray
        Time coordinates (sec).
    values : numpy.ndarray
        Real-valued array of shape ``(len(x), len(y), len(z), len(t))`` storing field values.

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
    """Stores a collection of scalar fields
    from a :class:`FieldMonitor` or :class:`FieldTimeMonitor`.

    Parameters
    ----------
    data_dict : Dict[str, :class:`ScalarFieldData`] or Dict[str, :class:`ScalarFieldTimeData`]
        Mapping of field name to its scalar field data.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 1001)
    >>> t = np.linspace(0, 1e-12, 1001)
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-2, 2, 20)
    >>> z = np.linspace(0, 0, 1)
    >>> values_f = (1+1j) * np.random.random((len(x), len(y), len(z), len(f)))
    >>> values_t = np.random.random((len(x), len(y), len(z), len(t)))
    >>> field_f = ScalarFieldData(values=values_f, x=x, y=y, z=z, f=f)
    >>> field_t = ScalarFieldTimeData(values=values_t, x=x, y=y, z=z, t=t)
    >>> data_f = FieldData(data_dict={'Ex': field_f, 'Ey': field_f})
    >>> data_t = FieldData(data_dict={'Ex': field_t, 'Ey': field_t})
    """

    data_dict: Union[Dict[str, ScalarFieldData], Dict[str, ScalarFieldTimeData]]
    type: Literal["FieldData"] = "FieldData"


class FluxData(AbstractFluxData, FreqData):
    """Stores frequency-domain power flux data from a :class:`FluxMonitor`.

    Parameters
    ----------
    f : numpy.ndarray
        Frequency coordinates (Hz).
    values : numpy.ndarray
        Complex-valued array of shape ``(len(f),)`` storing field values.

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
    """Stores time-domain power flux data from a :class:`FluxTimeMonitor`.

    Parameters
    ----------
    t : numpy.ndarray
        Time coordinates (sec).
    values : numpy.ndarray
        Real-valued array of shape ``(len(t),)`` storing field values.

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
    direction : List[str]
        List of strings corresponding to the mode propagation direction.
        Allowed elements are ``'+'`` and ``'-'``.
    mode_index : numpy.ndarray
        Array of integer indices into the original monitor's :attr:`ModeMonitor.modes`.
    f : numpy.ndarray
        Frequency coordinates (Hz).
    values : numpy.ndarray
        Complex-valued array of mode amplitude values
        with shape ``values.shape=(len(direction), len(mode_index), len(f))``.

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
        Original :class:`Simulation` that was run to create data.
    monitor_data : Dict[str, :class:`Tidy3dData`]
        Mapping of monitor name to :class:`Tidy3dData` intance.
    log_string : str = None
        A string containing the log information from the simulation run.
    """

    simulation: Simulation
    monitor_data: Dict[str, Union[MonitorData, FieldData]]
    log_string: str = None

    @property
    def log(self):
        """Prints the server-side log."""
        print(self.log_string if self.log_string else "no log stored")

    def __getitem__(self, monitor_name: str) -> Union[xr.DataArray, xr.Dataset]:
        """Get the :class:`MonitorData` xarray representation by name (``sim_data[monitor_name]``).

        Parameters
        ----------
        monitor_name : ``str``
            Name of the :class:`Monitor` to return data for.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The xarray representation of the data.
        """
        monitor_data = self.monitor_data.get(monitor_name)
        if not monitor_data:
            raise DataError(f"monitor {monitor_name} not found")
        return monitor_data.data

    # @add_ax_if_none
    # def plot_field(
    #     self,
    #     field_monitor_name: str,
    #     field_name: str,
    #     x: float = None,
    #     y: float = None,
    #     z: float = None,
    #     freq: float = None,
    #     time: float = None,
    #     eps_alpha: pydantic.confloat(ge=0.0, le=1.0) = 0.5,
    #     ax: Ax = None,
    #     **kwargs,
    # ) -> Ax:
    #     """Plot the field data for a monitor with simulation plot overlayed.

    #     Parameters
    #     ----------
    #     field_monitor_name : ``str``
    #         Name of :class:`FieldMonitor` or :class:`FieldTimeData` to plot.
    #     field_name : ``str``
    #         Name of `field` in monitor to plot (eg. 'Ex').
    #     x : ``float``, optional
    #         Position of plane in x direction.
    #     y : ``float``, optional
    #         Position of plane in y direction.
    #     z : ``float``, optional
    #         Position of plane in z direction.
    #     freq: ``float``, optional
    #         if monitor is a :class:`FieldMonitor`, specifies the frequency (Hz) to plot the field.
    #     time: ``float``, optional
    #         if monitor is a :class:`FieldTimeMonitor`, specifies the time (sec) to plot the field.
    #     cbar: `bool``, optional
    #         if True (default), will include colorbar
    #     ax : ``matplotlib.axes._subplots.Axes``, optional
    #         matplotlib axes to plot on, if not specified, one is created.
    #     **patch_kwargs
    #         Optional keyword arguments passed to ``add_artist(patch, **patch_kwargs)``.

    #     Returns
    #     -------
    #     ``matplotlib.axes._subplots.Axes``
    #         The supplied or created matplotlib axes.

    #     TODO: fully test and finalize arguments.
    #     """

    #     if field_monitor_name not in self.monitor_data:
    #     raise DataError(f"field_monitor_name {field_monitor_name} not found in SimulationData.")

    #     monitor_data = self.monitor_data.get(field_monitor_name)

    #     if not isinstance(monitor_data, FieldData):
    #         raise DataError(f"field_monitor_name {field_monitor_name} not a FieldData instance.")

    #     if field_name not in monitor_data.data_dict:
    #         raise DataError(f"field_name {field_name} not found in {field_monitor_name}.")

    #     xr_data = monitor_data.data_dict.get(field_name)
    #     if isinstance(monitor_data, FieldData):
    #         field_data = xr_data.sel(f=freq)
    #     else:
    #         field_data = xr_data.sel(t=time)

    #     ax = field_data.sel(x=x, y=y, z=z).real.plot.pcolormesh(ax=ax)
    #     ax = self.simulation.plot_structures_eps(
    #         freq=freq, cbar=False, x=x, y=y, z=z, alpha=eps_alpha, ax=ax
    #     )
    #     return ax

    def export(self, fname: str) -> None:
        """Export :class:`SimulationData` to single hdf5 file including monitor data.

        Parameters
        ----------
        fname : str
            Path to .hdf5 data file (including filename).
        """

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
        """Load :class:`SimulationData` from .hdf5 file.

        Parameters
        ----------
        fname : str
            Path to .hdf5 data file (including filename).

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

    def __eq__(self, other):
        """Check equality against another :class:`SimulationData` instance.

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
