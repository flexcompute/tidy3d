# pylint: disable=unused-import, too-many-lines
"""Classes for Storing Monitor and Simulation Data."""

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional
import logging

import xarray as xr
import numpy as np
import h5py

from .types import Numpy, Direction, Array, numpy_encoding, Literal, Ax
from .base import Tidy3dBaseModel
from .simulation import Simulation
from .mode import ModeSpec
from .viz import add_ax_if_none, equal_aspect
from ..log import log, DataError

# TODO: add warning if fields didnt fully decay


# mapping of data coordinates to units for assigning .attrs to the xarray objects
DIM_ATTRS = {
    "x": {"units": "um", "long_name": "x position"},
    "y": {"units": "um", "long_name": "y position"},
    "z": {"units": "um", "long_name": "z position"},
    "f": {"units": "Hz", "long_name": "frequency"},
    "t": {"units": "sec", "long_name": "time"},
    "direction": {"units": None, "long_name": "propagation direction"},
    "mode_index": {"units": None, "long_name": "mode index"},
}


""" xarray subclasses """

# TODO: make this work for Dataset items, which get converted to xr.DataArray


class Tidy3dDataArray(xr.DataArray):
    """Subclass of xarray's DataArray that implements some custom functions."""

    __slots__ = ()

    @property
    def abs(self):
        """Absolute value of complex-valued data."""
        return abs(self)


""" Base data classes """


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
            Tidy3dDataArray: lambda x: None,  # dont write
            xr.Dataset: lambda x: None,  # dont write
        }

    @abstractmethod
    def add_to_group(self, hdf5_grp):
        """Add data contents to an hdf5 group."""

    @classmethod
    @abstractmethod
    def load_from_group(cls, hdf5_grp):
        """Load data contents from an hdf5 group."""

    @staticmethod
    def save_string(hdf5_grp, string_key: str, string_value: str) -> None:
        """Save a string to an hdf5 group."""
        str_type = h5py.special_dtype(vlen=str)
        hdf5_grp.create_dataset(string_key, (1,), dtype=str_type)
        hdf5_grp[string_key][0] = string_value

    @staticmethod
    def decode_bytes(bytes_dataset) -> str:
        """Decode an hdf5 dataset containing bytes to a string."""
        return bytes_dataset[0].decode("utf-8")

    @staticmethod
    def load_string(hdf5_grp, string_key: str) -> str:
        """Load a string from an hdf5 group."""
        string_value_bytes = hdf5_grp.get(string_key)
        if not string_value_bytes:
            return None
        return Tidy3dData.decode_bytes(string_value_bytes)

    @staticmethod
    def decode_bytes_array(array_of_bytes: Numpy) -> List[str]:
        """Convert numpy array containing bytes to list of strings."""
        list_of_bytes = array_of_bytes.tolist()
        list_of_str = [v.decode("utf-8") for v in list_of_bytes]
        return list_of_str


class MonitorData(Tidy3dData, ABC):
    """Abstract base class for objects storing individual data from simulation."""

    values: Union[Array[float], Array[complex]]
    data_attrs: Dict[str, str] = None
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
    def data(self) -> Tidy3dDataArray:
        """Returns an xarray representation of the montitor data.

        Returns
        -------
        xarray.DataArray
            Representation of the monitor data using xarray.
            For more details refer to `xarray's Documentaton <https://tinyurl.com/2zrzsp7b>`_.
        """

        # make DataArray
        data_dict = self.dict()
        coords = {dim: data_dict[dim] for dim in self._dims}
        data_array = Tidy3dDataArray(self.values, coords=coords, dims=self._dims)

        # assign attrs for xarray
        if self.data_attrs:
            data_array.attrs = self.data_attrs
        for name, coord in data_array.coords.items():
            coord[name].attrs = DIM_ATTRS.get(name)

        return data_array

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
        Tidy3dData.save_string(hdf5_grp, "type", self.type)
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
                kwargs[str_kwarg] = Tidy3dData.decode_bytes_array(kwargs[str_kwarg])

        # ignore the "type" dataset as it's used for finding type for loading
        kwargs.pop("type")

        return cls(**kwargs)


class CollectionData(Tidy3dData):
    """Abstract base class.
    Stores a collection of data with same dimension types (such as a field with many components).

    Parameters
    ----------
    data_dict : Dict[str, :class:`MonitorData`]
        Mapping of collection member name to corresponding :class:`MonitorData`.
    """

    data_dict: Dict[str, MonitorData]
    type: str = None

    @property
    def data(self) -> Dict[str, xr.DataArray]:
        """For field quantities, store a single xarray DataArray for each ``field``.
        These all go in a single xarray Dataset, which keeps track of the shared coords.

        Returns
        -------
        Dict[str, xarray.DataArray]
            Mapping of data dict keys to corresponding DataArray from .data property.
            For more details refer to `xarray's Documentaton <https://tinyurl.com/2zrzsp7b>`_.
        """

        data_arrays = {name: arr.data for name, arr in self.data_dict.items()}
        return data_arrays

    def __eq__(self, other):
        """Check for equality against other :class:`AbstractFieldData` object."""

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
        """Add data from a :class:`AbstractFieldData` to an hdf5 group ."""

        # put collection's type information into the group
        Tidy3dData.save_string(hdf5_grp, "type", self.type)
        for data_name, data_value in self.data_dict.items():

            # create a new group for each member of collection and add its data
            data_grp = hdf5_grp.create_group(data_name)
            data_value.add_to_group(data_grp)

    @classmethod
    def load_from_group(cls, hdf5_grp):
        """Load a :class:`AbstractFieldData` from hdf5 group containing data."""
        data_dict = {}
        for data_name, data_value in hdf5_grp.items():

            # hdf5 group contains `type` dataset, ignore it.
            if data_name == "type":
                continue

            # get the type from MonitorData.type and add instance to dict
            _data_type = DATA_TYPE_MAP[Tidy3dData.load_string(data_value, "type")]
            data_dict[data_name] = _data_type.load_from_group(data_value)

        return cls(data_dict=data_dict)

    def ensure_member_exists(self, member_name: str):
        """make sure a member of collection is present in data"""
        if member_name not in self.data_dict:
            raise DataError(f"member_name '{member_name}' not found.")


""" Subclasses of MonitorData and CollectionData """


class AbstractFieldData(CollectionData, ABC):
    """Sores a collection of EM fields either in freq or time domain."""

    """ Get the standard EM components from the dict using convenient "dot" syntax."""

    @property
    def Ex(self):
        """Get Ex component of field using '.Ex' syntax."""
        scalar_data = self.data_dict.get("Ex")
        if scalar_data:
            return scalar_data.data
        return None

    @property
    def Ey(self):
        """Get Ey component of field using '.Ey' syntax."""
        scalar_data = self.data_dict.get("Ey")
        if scalar_data:
            return scalar_data.data
        return None

    @property
    def Ez(self):
        """Get Ez component of field using '.Ez' syntax."""
        scalar_data = self.data_dict.get("Ez")
        if scalar_data:
            return scalar_data.data
        return None

    @property
    def Hx(self):
        """Get Hx component of field using '.Hx' syntax."""
        scalar_data = self.data_dict.get("Hx")
        if scalar_data:
            return scalar_data.data
        return None

    @property
    def Hy(self):
        """Get Hy component of field using '.Hy' syntax."""
        scalar_data = self.data_dict.get("Hy")
        if scalar_data:
            return scalar_data.data
        return None

    @property
    def Hz(self):
        """Get Hz component of field using '.Hz' syntax."""
        scalar_data = self.data_dict.get("Hz")
        if scalar_data:
            return scalar_data.data
        return None

    def colocate(self, x, y, z) -> xr.Dataset:
        """colocate all of the data at a set of x, y, z coordinates.

        Parameters
        ----------
        x : np.array
            x coordinates of locations.
        y : np.array
            y coordinates of locations.
        z : np.array
            z coordinates of locations.

        Returns
        -------
        xr.Dataset
            Dataset containing all fields at the same spatial locations.
            For more details refer to `xarray's Documentaton <https://tinyurl.com/cyca3krz>`_.

        Note
        ----
        For many operations (such as flux calculations and plotting),
        it is important that the fields are colocated at the same spatial locations.
        Be sure to apply this method to your field data in those cases.
        """
        coord_val_map = {"x": x, "y": y, "z": z}
        centered_data_dict = {}
        for field_name, field_data in self.data_dict.items():
            centered_data_array = field_data.data
            for coord_name in "xyz":
                if len(centered_data_array.coords[coord_name]) <= 1:
                    # centered_data_array = centered_data_array.isel(**{coord_name:0})
                    coord_val = coord_val_map[coord_name]
                    coord_kwargs = {coord_name: coord_val}
                    centered_data_array = centered_data_array.assign_coords(**coord_kwargs)
                    centered_data_array = centered_data_array.isel(**{coord_name: 0})
                else:
                    coord_vals = coord_val_map[coord_name]
                    centered_data_array = centered_data_array.interp(**{coord_name: coord_vals})
            centered_data_dict[field_name] = centered_data_array
        # import pdb; pdb.set_trace()
        return xr.Dataset(centered_data_dict)


class FreqData(MonitorData, ABC):
    """Stores frequency-domain data using an ``f`` dimension for frequency in Hz."""

    f: Array[float]

    @abstractmethod
    def normalize(self, source_freq_amps: Array[complex]) -> None:
        """Normalize values of frequency-domain data by source amplitude spectrum."""


class TimeData(MonitorData, ABC):
    """Stores time-domain data using a ``t`` attribute for time in seconds."""

    t: Array[float]


class AbstractScalarFieldData(MonitorData, ABC):
    """Stores a single, scalar field as a function of spatial coordinates x,y,z."""

    x: Array[float]
    y: Array[float]
    z: Array[float]


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
    data_attrs: Dict[str, str] = None  # {'units': '[E] = V/um, [H] = A/um'}
    type: Literal["ScalarFieldData"] = "ScalarFieldData"

    _dims = ("x", "y", "z", "f")

    def normalize(self, source_freq_amps: Array[complex]) -> None:
        """normalize the values by the amplitude of the source."""
        self.values /= 1j * source_freq_amps  # pylint: disable=no-member


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
    data_attrs: Dict[str, str] = None  # {'units': '[E] = V/m, [H] = A/m'}
    type: Literal["ScalarFieldTimeData"] = "ScalarFieldTimeData"

    _dims = ("x", "y", "z", "t")


class FieldData(AbstractFieldData):
    """Stores a collection of scalar fields in the frequency domain from a :class:`FieldMonitor`.

    Parameters
    ----------
    data_dict : Dict[str, :class:`ScalarFieldData`]
        Mapping of field name (eg. 'Ex') to its scalar field data.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 1001)
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-2, 2, 20)
    >>> z = np.linspace(0, 0, 1)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(z), len(f)))
    >>> field = ScalarFieldData(values=values, x=x, y=y, z=z, f=f)
    >>> data = FieldData(data_dict={'Ex': field, 'Ey': field})
    """

    data_dict: Dict[str, ScalarFieldData]
    type: Literal["FieldData"] = "FieldData"


class FieldTimeData(AbstractFieldData):
    """Stores a collection of scalar fields in the time domain from a :class:`FieldTimeMonitor`.

    Parameters
    ----------
    data_dict : Dict[str, :class:`ScalarFieldTimeData`]
        Mapping of field name to its scalar field data.

    Example
    -------
    >>> t = np.linspace(0, 1e-12, 1001)
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-2, 2, 20)
    >>> z = np.linspace(0, 0, 1)
    >>> values = np.random.random((len(x), len(y), len(z), len(t)))
    >>> field = ScalarFieldTimeData(values=values, x=x, y=y, z=z, t=t)
    >>> data = FieldTimeData(data_dict={'Ex': field, 'Ey': field})
    """

    data_dict: Dict[str, ScalarFieldTimeData]
    type: Literal["FieldTimeData"] = "FieldTimeData"


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
    data_attrs: Dict[str, str] = {"units": "W", "long_name": "flux"}
    type: Literal["FluxData"] = "FluxData"

    _dims = ("f",)

    def normalize(self, source_freq_amps: Array[complex]) -> None:
        """normalize the values by the amplitude of the source."""
        self.values /= abs(source_freq_amps) ** 2  # pylint: disable=no-member


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
    data_attrs: Dict[str, str] = {"units": "W", "long_name": "flux"}
    type: Literal["FluxTimeData"] = "FluxTimeData"

    _dims = ("t",)


class AbstractModeData(PlanarData, FreqData, ABC):
    """Abstract class for mode data as a function of frequency and mode index."""

    mode_index: Array[int]


class ModeAmpsData(AbstractModeData):
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
    >>> values = (1+1j) * np.random.random((1, 2, len(f)))
    >>> data = ModeAmpsData(values=values, direction=['+'], mode_index=np.arange(1, 3), f=f)
    """

    direction: List[Direction] = ["+", "-"]
    values: Array[complex]
    data_attrs: Dict[str, str] = {"units": "sqrt(W)", "long_name": "mode amplitudes"}
    type: Literal["ModeAmpsData"] = "ModeAmpsData"

    _dims = ("direction", "mode_index", "f")

    def normalize(self, source_freq_amps: Array[complex]) -> None:
        """normalize the values by the amplitude of the source."""
        self.values /= 1j * source_freq_amps  # pylint: disable=no-member


class ModeIndexData(AbstractModeData):
    """Stores modal amplitdudes from a :class:`ModeMonitor`.

    Parameters
    ----------
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
    >>> values = (1+1j) * np.random.random((2, len(f)))
    >>> data = ModeIndexData(values=values, mode_index=np.arange(1, 3), f=f)
    """

    values: Array[complex]
    data_attrs: Dict[str, str] = {"units": "None", "long_name": "complex effective index"}
    type: Literal["ModeIndexData"] = "ModeIndexData"

    _dims = ("mode_index", "f")

    def normalize(self, source_freq_amps: Array[complex]) -> None:
        """normalize the values by the amplitude of the source."""
        return


class ModeData(CollectionData):
    """Stores a collection of mode decomposition amplitudes and mode effective indexes for all
    modes in a :class:`.ModeMonitor`.

    Parameters
    ----------
    data_dict : Dict[str, :class:`ScalarFieldData`]
        Mapping of field name (eg. 'Ex') to its scalar field data.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 1001)
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-2, 2, 20)
    >>> z = np.linspace(0, 0, 1)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(z), len(f)))
    >>> field = ScalarFieldData(values=values, x=x, y=y, z=z, f=f)
    >>> data = FieldData(data_dict={'Ex': field, 'Ey': field})
    """

    data_dict: Dict[str, AbstractModeData]
    type: Literal["ModeData"] = "ModeData"

    @property
    def amps(self):
        """Get mode amplitudes."""
        scalar_data = self.data_dict.get("amps")
        if scalar_data:
            return scalar_data.data
        return None

    @property
    def n_complex(self):
        """Get complex effective indexes."""
        scalar_data = self.data_dict.get("n_complex")
        if scalar_data:
            return scalar_data.data
        return None

    @property
    def n_eff(self):
        """Get real part of effective index."""
        scalar_data = self.data_dict.get("n_complex")
        if scalar_data:
            return scalar_data.data.real
        return None

    @property
    def k_eff(self):
        """Get imaginary part of effective index."""
        scalar_data = self.data_dict.get("n_complex")
        if scalar_data:
            return scalar_data.data.imag
        return None


# maps MonitorData.type string to the actual type, for MonitorData.from_file()
DATA_TYPE_MAP = {
    "ScalarFieldData": ScalarFieldData,
    "ScalarFieldTimeData": ScalarFieldTimeData,
    "FieldData": FieldData,
    "FieldTimeData": FieldTimeData,
    "FluxData": FluxData,
    "FluxTimeData": FluxTimeData,
    "ModeAmpsData": ModeAmpsData,
    "ModeIndexData": ModeIndexData,
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
    diverged : bool = False
        A boolean flag denoting if the simulation run diverged.
    normalized : bool = Fale
        A boolean flag denoting whether the data has been normalized by the spectrum of a source.
    """

    simulation: Simulation
    monitor_data: Dict[str, Tidy3dData]
    log_string: str = None
    diverged: bool = False
    normalized: bool = False

    @property
    def log(self) -> str:
        """Returns the server-side log as a string."""
        if not self.log_string:
            raise DataError("No log stored in SimulationData.")
        return self.log_string

    @property
    def final_decay_value(self) -> float:
        """Returns value of the field decay at the final time step."""
        log_str = self.log
        lines = log_str.split("\n")
        decay_lines = [l for l in lines if "field decay" in l]
        final_decay_line = decay_lines[-1]
        final_decay = float(final_decay_line.split("field decay: ")[-1])
        return final_decay

    def __getitem__(self, monitor_name: str) -> Union[Tidy3dDataArray, xr.Dataset]:
        """Get the :class:`MonitorData` xarray representation by name (``sim_data[monitor_name]``).

        Parameters
        ----------
        monitor_name : ``str``
            Name of the :class:`Monitor` to return data for.

        Returns
        -------
        xarray.DataArray or CollectionData
            Data from the supplied monitor.
            If the monitor corresponds to collection-like data (such as fields),
            a collection data instance is returned.
            Otherwise, if it is a MonitorData instance, the xarray representation is returned.
        """
        monitor_data = self.monitor_data.get(monitor_name)
        if not monitor_data:
            raise DataError(f"monitor '{monitor_name}' not found")
        if isinstance(monitor_data, MonitorData):
            return monitor_data.data
        return monitor_data

    def ensure_monitor_exists(self, monitor_name: str) -> None:
        """Raise exception if monitor isn't in the simulation data"""
        if monitor_name not in self.monitor_data:
            raise DataError(f"Data for monitor '{monitor_name}' not found in simulation data.")

    def ensure_field_monitor(self, data_obj: Tidy3dData) -> None:
        """Raise exception if monitor isn't a field monitor."""
        if not isinstance(data_obj, (FieldData, FieldTimeData)):
            raise DataError(f"data_obj '{data_obj}' " "not a FieldData or FieldTimeData instance.")

    def at_centers(self, field_monitor_name: str) -> xr.Dataset:
        """return xarray.Dataset representation of field monitor data
        co-located at Yee cell centers.

        Parameters
        ----------
        field_monitor_name : str
            Name of field monitor used in the original :class:`Simulation`.

        Returns
        -------
        xarray.Dataset
            Dataset containing all of the fields in the data
            interpolated to center locations on Yee grid.
        """

        # get the data
        self.ensure_monitor_exists(field_monitor_name)
        field_monitor_data = self.monitor_data.get(field_monitor_name)
        self.ensure_field_monitor(field_monitor_data)

        # get the monitor, discretize, and get center locations
        monitor = self.simulation.get_monitor_by_name(field_monitor_name)
        sub_grid = self.simulation.discretize(monitor)
        centers = sub_grid.centers

        # colocate each of the field components at centers
        field_dataset = field_monitor_data.colocate(x=centers.x, y=centers.y, z=centers.z)
        return field_dataset

    @equal_aspect
    @add_ax_if_none
    def plot_field(  # pylint:disable=too-many-arguments, too-many-locals
        self,
        field_monitor_name: str,
        field_name: str,
        x: float = None,
        y: float = None,
        z: float = None,
        val: Literal["real", "imag", "abs"] = "real",
        freq: float = None,
        time: float = None,
        eps_alpha: float = 0.2,
        robust: bool = True,
        ax: Ax = None,
        **kwargs,
    ) -> Ax:
        """Plot the field data for a monitor with simulation plot overlayed.

        Parameters
        ----------
        field_monitor_name : str
            Name of :class:`FieldMonitor` or :class:`FieldTimeData` to plot.
        field_name : str
            Name of `field` in monitor to plot (eg. 'Ex').
        x : float = None
            Position of plane in x direction.
        y : float = None
            Position of plane in y direction.
        z : float = None
            Position of plane in z direction.
        val : Literal['real', 'imag', 'abs'] = 'real'
            What part of the field to plot (in )
        freq: float = None
            If monitor is a :class:`FieldMonitor`, specifies the frequency (Hz) to plot the field.
            Also sets the frequency at which the permittivity is evaluated at (if dispersive).
            By default, chooses permittivity as frequency goes to infinity.
        time: float = None
            if monitor is a :class:`FieldTimeMonitor`, specifies the time (sec) to plot the field.
        eps_alpha : float = 0.2
            Opacity of the structure permittivity.
            Must be between 0 and 1 (inclusive).
        robust : bool = True
            If specified, uses the 2nd and 98th percentiles of the data to compute the color limits.
            This helps in visualizing the field patterns especially in the presence of a source.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to ``add_artist(patch, **patch_kwargs)``.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        # get the monitor data
        self.ensure_monitor_exists(field_monitor_name)
        monitor_data = self.monitor_data.get(field_monitor_name)
        self.ensure_field_monitor(monitor_data)

        # get the field data component
        monitor_data.ensure_member_exists(field_name)
        xr_data = monitor_data.data_dict.get(field_name).data

        # select the frequency or time value
        if "f" in xr_data.coords:
            if freq is None:
                raise DataError("'freq' must be supplied to plot a FieldMonitor.")
            field_data = xr_data.sel(f=freq, method="nearest")
        elif "t" in xr_data.coords:
            if time is None:
                raise DataError("'time' must be supplied to plot a FieldMonitor.")
            field_data = xr_data.sel(t=time, method="nearest")
        else:
            raise DataError("Field data has neither time nor frequency data, something went wrong.")

        # select the cross section data
        axis, pos = self.simulation.parse_xyz_kwargs(x=x, y=y, z=z)
        axis_label = "xyz"[axis]
        interp_kwarg = {axis_label: pos}
        try:
            field_data = field_data.interp(**interp_kwarg)

        except Exception as e:
            raise DataError(f"Could not interpolate data at {axis_label}={pos}.") from e

        # select the field value
        if val not in ("real", "imag", "abs"):
            raise DataError(f"'val' must be one of ``{'real', 'imag', 'abs'}``, given {val}")
        if val == "real":
            field_data = field_data.real
        elif val == "imag":
            field_data = field_data.imag
        elif val == "abs":
            field_data = abs(field_data)

        # plot the field
        xy_coord_labels = list("xyz")
        xy_coord_labels.pop(axis)
        x_coord_label, y_coord_label = xy_coord_labels  # pylint:disable=unbalanced-tuple-unpacking
        field_data.plot(ax=ax, x=x_coord_label, y=y_coord_label, robust=robust)

        # plot the simulation epsilon
        ax = self.simulation.plot_structures_eps(
            freq=freq, cbar=False, x=x, y=y, z=z, alpha=eps_alpha, ax=ax, **kwargs
        )

        # set the limits based on the xarray coordinates min and max
        x_coord_values = field_data.coords[x_coord_label]
        y_coord_values = field_data.coords[y_coord_label]
        ax.set_xlim(min(x_coord_values), max(x_coord_values))
        ax.set_ylim(min(y_coord_values), max(y_coord_values))

        return ax

    def normalize(self, normalize_index: int = 0):
        """Return a copy of the :class:`.SimulationData` object with data normalized by source.

        Parameters
        ----------
        normalize_index : int = 0
            If specified, normalizes the frequency-domain data by the amplitude spectrum of the
            source corresponding to ``simulation.sources[normalize_index]``.
            This occurs when the data is loaded into a :class:`SimulationData` object.

        Returns
        -------
        :class:`.SimulationData`
            A copy of the :class:`.SimulationData` with the data normalized by source spectrum.
        """

        if self.normalized:
            raise DataError(
                "This SimulationData object has already been normalized"
                "and can't be normalized again."
            )

        try:
            source = self.simulation.sources[normalize_index]
            source_time = source.source_time
        except Exception:  # pylint:disable=broad-except
            logging.warning(f"Could not locate source at normalize_index={normalize_index}.")
            return self

        source_time = source.source_time
        sim_data_norm = self.copy(deep=True)
        times = self.simulation.tmesh
        dt = self.simulation.dt

        def normalize_data(monitor_data):
            """normalize a monitor data instance using the source time parameters."""
            freqs = monitor_data.f
            source_freq_amps = source_time.spectrum(times, freqs, dt)
            monitor_data.normalize(source_freq_amps)

        for monitor_data in sim_data_norm.monitor_data.values():

            if isinstance(monitor_data, (FieldData, FluxData, ModeData)):

                if isinstance(monitor_data, CollectionData):
                    for attr_data in monitor_data.data_dict.values():
                        normalize_data(attr_data)

                else:
                    normalize_data(monitor_data)

        sim_data_norm.normalized = True
        return sim_data_norm

    def to_file(self, fname: str) -> None:
        """Export :class:`SimulationData` to single hdf5 file including monitor data.

        Parameters
        ----------
        fname : str
            Path to .hdf5 data file (including filename).
        """

        with h5py.File(fname, "a") as f_handle:

            # save json string as a dataset
            Tidy3dData.save_string(f_handle, "sim_json", self.simulation.json())

            # save log string as a dataset
            if self.log_string:
                Tidy3dData.save_string(f_handle, "log_string", self.log_string)

            # save diverged flag as an attribute
            f_handle.attrs["diverged"] = self.diverged

            # make a group for monitor_data
            mon_data_grp = f_handle.create_group("monitor_data")
            for mon_name, mon_data in self.monitor_data.items():

                # for each monitor, make new group with the same name
                mon_grp = mon_data_grp.create_group(mon_name)
                mon_data.add_to_group(mon_grp)

    @classmethod
    def from_file(cls, fname: str):
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
            sim_json = Tidy3dData.load_string(f_handle, "sim_json")
            simulation = Simulation.parse_raw(sim_json)

            # get the log if exists
            log_string = Tidy3dData.load_string(f_handle, "log_string")

            # set the diverged flag
            # TODO: add link to documentation discussing divergence
            diverged = f_handle.attrs["diverged"]
            if diverged:
                logging.warning("Simulation run has diverged!")

            # loop through monitor dataset and create all MonitorData instances
            monitor_data_dict = {}
            for monitor_name, monitor_data in f_handle["monitor_data"].items():

                # load this MonitorData instance, add to monitor_data dict
                _data_type = DATA_TYPE_MAP[Tidy3dData.load_string(monitor_data, "type")]
                monitor_data_instance = _data_type.load_from_group(monitor_data)
                monitor_data_dict[monitor_name] = monitor_data_instance

        sim_data = cls(
            simulation=simulation,
            monitor_data=monitor_data_dict,
            log_string=log_string,
            diverged=diverged,
        )

        return sim_data

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
