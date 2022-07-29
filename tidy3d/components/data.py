# pylint: disable=unused-import, too-many-lines
"""Classes for Storing Monitor and Simulation Data."""

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Tuple
import logging

import xarray as xr
import numpy as np
import scipy
import h5py
import pydantic as pd

from rich.progress import track

from .types import Numpy, Direction, Array, ArrayLike, Literal, Ax, Coordinate, Axis
from .types import TYPE_TAG_STR, annotate_type
from .base import Tidy3dBaseModel
from .simulation import Simulation
from .geometry import Geometry
from .boundary import Symmetry, BlochBoundary
from .medium import Medium
from .monitor import Monitor, FieldMonitor
from .monitor import Near2FarAngleMonitor, Near2FarKSpaceMonitor, Near2FarCartesianMonitor
from .grid import Grid, Coords
from .viz import add_ax_if_none, equal_aspect
from ..log import DataError, log, SetupError, ValidationError
from ..constants import HERTZ, SECOND, MICROMETER, RADIAN, C_0, ETA_0
from ..updater import Updater


# mapping of data coordinates to units for assigning .attrs to the xarray objects
DIM_ATTRS = {
    "x": {"units": "um", "long_name": "x position"},
    "y": {"units": "um", "long_name": "y position"},
    "z": {"units": "um", "long_name": "z position"},
    "f": {"units": "Hz", "long_name": "frequency"},
    "t": {"units": "sec", "long_name": "time"},
    "direction": {"units": None, "long_name": "propagation direction"},
    "mode_index": {"units": None, "long_name": "mode index"},
    "theta": {"units": "rad", "long_name": "elevation angle"},
    "phi": {"units": "rad", "long_name": "azimuth angle"},
}


""" xarray subclasses """


class Tidy3dDataArray(xr.DataArray):
    """Subclass of xarray's DataArray that implements some custom functions."""

    __slots__ = ()

    @property
    def abs(self):
        """Absolute value of complex-valued data."""
        return abs(self)


""" Base data classes """


class Tidy3dBaseDataModel(Tidy3dBaseModel):
    """Tidy3dBaseModel, but with yaml and json IO methods disabled."""

    def _json_string(self, include_unset: bool = True) -> str:
        """Disable exporting to json string."""
        raise DataError("Can't export json string of Tidy3dData.")

    def to_json(self, fname: str) -> None:
        """Disable exporting to json file."""
        raise DataError("Can't export json file of Tidy3dData, use `.to_file(fname.hdf5)` instead.")

    def to_yaml(self, fname: str) -> None:
        """Disable exporting to yaml file."""
        raise DataError("Can't export yaml file of Tidy3dData, use `.to_file(fname.hdf5)` instead.")

    @classmethod
    def from_json(cls, fname: str, **parse_file_kwargs):
        """Disable loading from json file."""
        raise DataError(
            "Can't load Tidy3dData from .json file, use `.from_file(fname.hdf5)` instead."
        )

    @classmethod
    def from_yaml(cls, fname: str, **parse_raw_kwargs):
        """Disable loading from yaml file."""
        raise DataError(
            "Can't load Tidy3dData from .yaml file, use `.from_file(fname.hdf5)` instead."
        )


class Tidy3dData(Tidy3dBaseDataModel):
    """Base class for data associated with a simulation."""

    class Config:  # pylint: disable=too-few-public-methods
        """Configuration for all Tidy3dData objects."""

        validate_all = True  # validate default values too
        extra = "allow"  # allow extra kwargs not specified in model (like dir=['+', '-'])
        validate_assignment = True  # validate when attributes are set after initialization
        arbitrary_types_allowed = True  # allow types like `Array[float]`
        json_encoders = {  # how to write certain types to json files
            # np.ndarray: numpy_encoding,  # use custom encoding defined in .types
            np.int64: lambda x: int(x),  # pylint: disable=unnecessary-lambda
            Tidy3dDataArray: lambda x: None,  # dont write
            xr.Dataset: lambda x: None,  # dont write
        }
        frozen = False
        allow_mutation = True

    @abstractmethod
    def add_to_group(self, hdf5_grp):
        """Add data contents to an hdf5 group."""

    @property
    @abstractmethod
    def sim_data_getitem(self):
        """What gets returned by sim_data['monitor_data_name']"""

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
        if isinstance(string_value_bytes, str):
            return string_value_bytes
        return Tidy3dData.decode_bytes(string_value_bytes)

    @staticmethod
    def decode_bytes_array(array_of_bytes: Numpy) -> List[str]:
        """Convert numpy array containing bytes to list of strings."""
        list_of_bytes = array_of_bytes.tolist()
        return [v.decode("utf-8") for v in list_of_bytes]


class MonitorData(Tidy3dData, ABC):
    """Abstract base class for objects storing individual data from simulation."""

    values: Union[Array[float], Array[complex]] = pd.Field(
        ..., title="Values", description="Values of the raw data being stored."
    )

    data_attrs: Dict[str, str] = pd.Field(
        None,
        title="Data Attributes",
        description="Dictionary storing extra attributes associated with the monitor data.",
    )

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
        for name, coord in data_array.coords.items():  # pylint:disable=no-member
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
        return np.all(self.values == other.values)

    def add_to_group(self, hdf5_grp) -> None:
        """Add data contents to an hdf5 group."""

        # save the type information of MonitorData to the group
        Tidy3dData.save_string(hdf5_grp, TYPE_TAG_STR, self.type)  # pylint:disable=no-member
        for data_name, data_value in self.dict().items():

            # for each data member in self._dims (+ values), add to group.
            if data_name in self._dims or data_name in ("values",):
                hdf5_grp.create_dataset(data_name, data=data_value)

    @classmethod
    def load_from_group(cls, hdf5_grp):
        """Load Monitor data instance from an hdf5 group."""

        # kwargs that gets passed to MonitorData.__init__() to make new MonitorData
        kwargs = {data_name: np.array(data_value) for data_name, data_value in hdf5_grp.items()}

        # handle data stored as np.array() of bytes instead of strings
        for str_kwarg in ("direction",):
            if kwargs.get(str_kwarg) is not None:
                kwargs[str_kwarg] = Tidy3dData.decode_bytes_array(kwargs[str_kwarg])

        # ignore the TYPE_TAG_STR dataset as it's used for finding type for loading
        kwargs.pop(TYPE_TAG_STR)

        return cls(**kwargs)

    @property
    def sim_data_getitem(self) -> Tidy3dDataArray:
        """What gets returned by sim_data['monitor_data_name']"""
        return self.data


class CollectionData(Tidy3dData):
    """Abstract base class.
    Stores a collection of data with same dimension types (such as a field with many components).
    """

    data_dict: Dict[str, MonitorData] = pd.Field(
        ...,
        title="Data Dictionary",
        description="Mapping of name to each :class:`.MonitorData` in the collection.",
    )

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

        return {name: arr.data for name, arr in self.data_dict.items()}

    def __eq__(self, other):
        """Check for equality against other :class:`AbstractFieldData` object."""

        # same keys?
        if any(k not in other.data_dict.keys() for k in self.data_dict.keys()):
            return False
        if any(k not in self.data_dict.keys() for k in other.data_dict.keys()):
            return False
        return all(
            data_value == other.data_dict[data_name]
            for data_name, data_value in self.data_dict.items()
        )

    def __getitem__(self, field_name: str) -> xr.DataArray:
        """Get the :class:`MonitorData` xarray representation by name (``col_data[field_name]``).

        Parameters
        ----------
        field_name : ``str``
            Name of the colletion's field, eg. "Ey" for FieldData.

        Returns
        -------
        xarray.DataArray
            Data corresponding to the supplied field name.
        """
        monitor_data = self.data_dict.get(field_name)
        if not monitor_data:
            raise DataError(f"field_name '{field_name}' not found")
        return monitor_data.data

    def add_to_group(self, hdf5_grp) -> None:
        """Add data from a :class:`AbstractFieldData` to an hdf5 group ."""

        # put collection's type information into the group
        Tidy3dData.save_string(hdf5_grp, TYPE_TAG_STR, self.type)  # pylint:disable=no-member
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
            if data_name == TYPE_TAG_STR:
                continue

            # get the type from MonitorData.type and add instance to dict
            _data_type = DATA_TYPE_MAP[Tidy3dData.load_string(data_value, TYPE_TAG_STR)]
            data_dict[data_name] = _data_type.load_from_group(data_value)

        return cls(data_dict=data_dict)

    def ensure_member_exists(self, member_name: str):
        """make sure a member of collection is present in data"""
        if member_name not in self.data_dict:
            raise DataError(f"member_name '{member_name}' not found.")

    @property
    def sim_data_getitem(self) -> Tidy3dData:
        """What gets returned by sim_data['monitor_data_name']"""
        return self


""" Abstract subclasses of MonitorData and CollectionData """


class SpatialCollectionData(CollectionData, ABC):
    """Sores a collection of scalar data defined over x, y, z (among other) coords."""

    """ Attributes storing details about any symmetries that can be used to expand the data. """

    symmetry: Tuple[Symmetry, Symmetry, Symmetry] = pd.Field(
        (0, 0, 0),
        title="Symmetry Eigenvalues",
        description="Eigenvalues of the symmetry under reflection in x, y, and z.",
    )

    symmetry_center: Coordinate = pd.Field(
        None, title="Symmetry Center", description="Position of the symmetry planes in x, y, and z."
    )

    expanded_grid: Dict[str, Coords] = pd.Field(
        {},
        title="Expanded Grid",
        description="Grid after the symmetries (if any) are expanded. "
        "The dictionary keys must correspond to the data keys in the ``data_dict`` "
        "for the expanded grid to be invoked.",
    )

    _sym_dict: Dict[str, Symmetry] = pd.PrivateAttr({})
    """
        title="Symmetry Dict",
        description="Dictionary of the form ``{data_key: Symmetry}``, "
        "defining how data components are affected by a positive symmetry along each of the axes. "
        "If the name of a given data in the ``data_dict`` is not in this dictionary, "
        "then in the presence of symmetry the data is just unwrapped "
        "with a positive symmetry value in each direction. "
        "If the data name is in the dictionary, for each axis, "
        "the corresponding ``_sym_dict`` value times the ``self.symmetry`` eigenvalue is used.",
    """

    @pd.validator("symmetry_center", always=True)
    def _defined_if_sym_present(cls, val, values):
        """If symmetry required, must have symmetry_center."""
        if any(sym != 0 for sym in values.get("symmetry")):
            assert val is not None, "symmetry_center must be supplied."
        return val

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
            interp_dict = {}
            for coord_name in "xyz":
                if len(centered_data_array.coords[coord_name]) <= 1:
                    # centered_data_array = centered_data_array.isel(**{coord_name:0})
                    coord_val = coord_val_map[coord_name]
                    coord_kwargs = {coord_name: coord_val}
                    centered_data_array = centered_data_array.assign_coords(**coord_kwargs)
                    centered_data_array = centered_data_array.isel(**{coord_name: 0})
                else:
                    interp_dict[coord_name] = coord_val_map[coord_name]
            centered_data_dict[field_name] = centered_data_array.interp(
                **interp_dict, kwargs={"bounds_error": True}
            )
        return xr.Dataset(centered_data_dict)

    @property
    def expand_syms(self) -> Tidy3dData:
        """Create a new :class:`SpatialCollectionData` subclass by interpolating on the
        stored ``expanded_grid` using the stored symmetry information.

        Returns
        -------
        :class:`SpatialCollectionData`
            A new data object with the expanded fields. The data is only modified for data keys
            found in the ``self.expanded_grid`` dict, and along dimensions where ``self.symmetry``
            is non-zero.
        """

        new_data_dict = {}

        for data_key, scalar_data in self.data_dict.items():
            new_data = scalar_data.data

            # Apply symmetries
            zipped = zip("xyz", self.symmetry_center, self.symmetry)
            for dim, (dim_name, center, sym) in enumerate(zipped):
                # Continue if no symmetry or the data key is not in the expanded grid
                if sym == 0 or self.expanded_grid.get(data_key) is None:
                    continue

                # Get new grid locations
                coords = self.expanded_grid[data_key].to_list[dim]

                # Get indexes of coords that lie on the left of the symmetry center
                flip_inds = np.where(coords < center)[0]

                # Get the symmetric coordinates on the right
                coords_interp = np.copy(coords)
                coords_interp[flip_inds] = 2 * center - coords[flip_inds]

                # Interpolate. There generally shouldn't be values out of bounds except potentially
                # when handling modes, in which case they should be at the boundary and close to 0.
                new_data = new_data.sel({dim_name: coords_interp}, method="nearest")
                new_data = new_data.assign_coords({dim_name: coords})

                sym_eval = self._sym_dict.get(data_key)
                if sym_eval is not None:
                    # Apply the correct +/-1 for the data_key component
                    new_data[{dim_name: flip_inds}] *= sym * sym_eval[dim]

            new_data_dict[data_key] = type(scalar_data)(values=new_data.values, **new_data.coords)

        return type(self)(data_dict=new_data_dict)

    @property
    def sim_data_getitem(self) -> Tidy3dData:
        """What gets returned by sim_data['monitor_data_name']"""
        return self.expand_syms

    def set_symmetry_attrs(self, simulation: Simulation, monitor_name: str):
        """Set the collection data attributes related to symmetries."""
        monitor = simulation.get_monitor_by_name(monitor_name)
        mnt_grid = simulation.discretize(monitor, extend=True)
        self.expanded_grid = mnt_grid.yee.grid_dict
        self.symmetry = simulation.symmetry
        self.symmetry_center = simulation.center


class AbstractFieldData(SpatialCollectionData, ABC):
    """Sores a collection of EM fields either in freq or time domain."""

    _sym_dict: Dict[str, Symmetry] = pd.PrivateAttr(
        {
            "Ex": (-1, 1, 1),
            "Ey": (1, -1, 1),
            "Ez": (1, 1, -1),
            "Hx": (1, -1, -1),
            "Hy": (-1, 1, -1),
            "Hz": (-1, -1, 1),
        }
    )

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


class FreqData(MonitorData, ABC):
    """Stores frequency-domain data using an ``f`` dimension for frequency in Hz."""

    f: Array[float] = pd.Field(
        ...,
        title="Frequencies",
        description="Array of frequency values to use as coordintes.",
        units=HERTZ,
    )

    @abstractmethod
    def normalize(self, source_freq_amps: Array[complex]) -> None:
        """Normalize values of frequency-domain data by source amplitude spectrum."""


class TimeData(MonitorData, ABC):
    """Stores time-domain data using a ``t`` attribute for time in seconds."""

    t: Array[float] = pd.Field(
        ...,
        title="Times",
        description="Array of time values to use as coordintes.",
        units=SECOND,
    )


class ScalarSpatialData(MonitorData, ABC):
    """Stores a single, scalar variable as a function of spatial coordinates x, y, z."""

    x: Array[float] = pd.Field(
        ...,
        title="X Locations",
        description="Array of x location values to use as coordintes.",
        units=MICROMETER,
    )

    y: Array[float] = pd.Field(
        ...,
        title="Y Locations",
        description="Array of y location values to use as coordintes.",
        units=MICROMETER,
    )

    z: Array[float] = pd.Field(
        ...,
        title="Z Locations",
        description="Array of z location values to use as coordintes.",
        units=MICROMETER,
    )


class PlanarData(MonitorData, ABC):
    """Stores data that must be found via a planar monitor."""


class AbstractModeData(PlanarData, FreqData, ABC):
    """Abstract class for mode data as a function of frequency and mode index."""

    mode_index: Array[int] = pd.Field(
        ..., title="Mode Indices", description="Array of mode index values to use as coordintes."
    )


class AbstractFluxData(PlanarData, ABC):
    """Stores electromagnetic flux through a plane."""


""" Usable individual data containers for CollectionData monitors """


class ScalarFieldData(ScalarSpatialData, FreqData):
    """Stores a single scalar field in frequency-domain.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 1001)
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-2, 2, 20)
    >>> z = np.linspace(0, 0, 1)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(z), len(f)))
    >>> data = ScalarFieldData(values=values, x=x, y=y, z=z, f=f)
    """

    values: Array[complex] = pd.Field(
        ...,
        title="Scalar Field Values",
        description="Multi-dimensional array storing the raw scalar field values in freq. domain.",
    )

    _dims = ("x", "y", "z", "f")

    def normalize(self, source_freq_amps: Array[complex]) -> None:
        """normalize the values by the amplitude of the source."""
        self.values /= source_freq_amps  # pylint: disable=no-member


class ScalarFieldTimeData(ScalarSpatialData, TimeData):
    """stores a single scalar field in time domain

    Example
    -------
    >>> t = np.linspace(0, 1e-12, 1001)
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-2, 2, 20)
    >>> z = np.linspace(0, 0, 1)
    >>> values = np.random.random((len(x), len(y), len(z), len(t)))
    >>> data = ScalarFieldTimeData(values=values, x=x, y=y, z=z, t=t)
    """

    values: Array[float] = pd.Field(
        ...,
        title="Scalar Field Values",
        description="Multi-dimensional array storing the raw scalar field values in time domain.",
    )

    _dims = ("x", "y", "z", "t")


class ScalarPermittivityData(ScalarSpatialData, FreqData):
    """Stores a single scalar permittivity distribution in frequency-domain.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 1001)
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-2, 2, 20)
    >>> z = np.linspace(0, 0, 1)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(z), len(f)))
    >>> data = ScalarPermittivityData(values=values, x=x, y=y, z=z, f=f)
    """

    values: Array[complex] = pd.Field(
        ...,
        title="Scalar Permittivity Values",
        description="Multi-dimensional array storing the raw permittivity values in freq. domain.",
    )

    _dims = ("x", "y", "z", "f")

    def normalize(self, source_freq_amps: Array[complex]) -> None:
        pass


class ScalarModeFieldData(ScalarFieldData, AbstractModeData):
    """Like :class:`.ScalarFieldData`, but with extra dimension ``mode_index``."""

    _dims = ("x", "y", "z", "f", "mode_index")


class ModeAmpsData(AbstractModeData):
    """Stores modal amplitdudes from a :class:`.ModeMonitor`.

    Example
    -------
    >>> f = np.linspace(2e14, 3e14, 1001)
    >>> mode_index = np.arange(1, 3)
    >>> values = (1+1j) * np.random.random((2, len(f), len(mode_index)))
    >>> data = ModeAmpsData(values=values, direction=['+', '-'], mode_index=mode_index, f=f)
    """

    direction: List[Direction] = pd.Field(
        ["+", "-"],
        title="Direction Coordinates",
        description="List of directions contained in the mode amplitude data.",
    )

    values: Array[complex] = pd.Field(
        ...,
        title="Mode Amplitude Values",
        description="Multi-dimensional array storing the raw, complex mode amplitude values.",
    )

    data_attrs: Dict[str, str] = pd.Field(
        {"units": "sqrt(W)", "long_name": "mode amplitudes"},
        title="Data Attributes",
        description="Dictionary storing extra attributes associated with the monitor data.",
    )

    _dims = ("direction", "f", "mode_index")

    def normalize(self, source_freq_amps: Array[complex]) -> None:
        """normalize the values by the amplitude of the source."""
        self.values /= source_freq_amps[None, :, None]  # pylint: disable=no-member


class ModeIndexData(AbstractModeData):
    """Stores effective propagation index from a :class:`.ModeMonitor`.

    Example
    -------
    >>> f = np.linspace(2e14, 3e14, 1001)
    >>> values = (1+1j) * np.random.random((len(f), 2))
    >>> data = ModeIndexData(values=values, mode_index=np.arange(1, 3), f=f)
    """

    values: Array[complex] = pd.Field(
        ..., title="Values", description="Values of the mode's complex effective refractive index."
    )

    data_attrs: Dict[str, str] = pd.Field(
        {"units": "", "long_name": "effective index"},
        title="Data Attributes",
        description="Dictionary storing extra attributes associated with the monitor data.",
    )

    _dims = ("f", "mode_index")

    def normalize(self, source_freq_amps: Array[complex]) -> None:
        """normalize the values by the amplitude of the source."""
        return

    @property
    def n_complex(self):
        """Complex effective index."""
        return self.data

    @property
    def n_eff(self):
        """Get real part of effective index."""
        _n_eff = self.data.real
        _n_eff.attrs["long_name"] = "effective n"
        return _n_eff

    @property
    def k_eff(self):
        """Get imaginary part of effective index."""
        _k_eff = self.data.imag
        _k_eff.attrs["long_name"] = "effective k"
        return _k_eff


""" Usable monitor/collection data """


class FieldData(AbstractFieldData):
    """Stores a collection of scalar fields in the frequency domain from a :class:`.FieldMonitor`.

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

    data_dict: Dict[str, ScalarFieldData] = pd.Field(
        ...,
        title="Data Dictionary",
        description="Mapping of the field names to their corresponding :class:`.ScalarFieldData`.",
    )


class FieldTimeData(AbstractFieldData):
    """Stores a collection of scalar fields in the time domain from a :class:`.FieldTimeMonitor`.

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

    data_dict: Dict[str, ScalarFieldTimeData] = pd.Field(
        ...,
        title="Data Dictionary",
        description="Mapping of the field names to their corresponding "
        ":class:`.ScalarFieldTimeData`.",
    )


class PermittivityData(SpatialCollectionData):
    """Sores a collection of permittivity components over spatial coordinates and frequency
    from a :class:`.PermittivityMonitor`.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 1001)
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-2, 2, 20)
    >>> z = np.linspace(0, 0, 1)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(z), len(f)))
    >>> eps = ScalarPermittivityData(values=values, x=x, y=y, z=z, f=f)
    >>> data = PermittivityData(data_dict={'eps_xx': eps, 'eps_yy': eps, 'eps_zz': eps})
    """

    data_dict: Dict[str, ScalarPermittivityData] = pd.Field(
        ...,
        title="Data Dictionary",
        description="Mapping of the permittivity tensor names to their corresponding "
        ":class:`.ScalarPermittivityData`.",
    )

    """ Get the permittivity components from the dict using convenient "dot" syntax."""

    @property
    def eps_xx(self):
        """Get eps_xx component."""
        scalar_data = self.data_dict.get("eps_xx")
        if scalar_data:
            return scalar_data.data
        return None

    @property
    def eps_yy(self):
        """Get eps_yy component."""
        scalar_data = self.data_dict.get("eps_yy")
        if scalar_data:
            return scalar_data.data
        return None

    @property
    def eps_zz(self):
        """Get eps_zz component."""
        scalar_data = self.data_dict.get("eps_zz")
        if scalar_data:
            return scalar_data.data
        return None

    def set_symmetry_attrs(self, simulation: Simulation, monitor_name: str):
        """Set the collection data attributes related to symmetries."""
        super().set_symmetry_attrs(simulation, monitor_name)
        # Redefine the expanded grid for epsilon rather than for fields.
        self.expanded_grid = {
            "eps_xx": self.expanded_grid["Ex"],
            "eps_yy": self.expanded_grid["Ey"],
            "eps_zz": self.expanded_grid["Ez"],
        }


class FluxData(AbstractFluxData, FreqData):
    """Stores frequency-domain power flux data from a :class:`.FluxMonitor`.

    Example
    -------
    >>> f = np.linspace(2e14, 3e14, 1001)
    >>> values = np.random.random((len(f),))
    >>> data = FluxData(values=values, f=f)
    """

    values: Array[float] = pd.Field(
        ..., title="Values", description="Values of the raw flux data in the frequency domain."
    )
    data_attrs: Dict[str, str] = pd.Field(
        {"units": "W", "long_name": "flux"},
        title="Data Attributes",
        description="Dictionary storing extra attributes associated with the monitor data.",
    )

    _dims = ("f",)

    def normalize(self, source_freq_amps: Array[complex]) -> None:
        """normalize the values by the amplitude of the source."""
        self.values /= abs(source_freq_amps) ** 2  # pylint: disable=no-member


class FluxTimeData(AbstractFluxData, TimeData):
    """Stores time-domain power flux data from a :class:`.FluxTimeMonitor`.

    Example
    -------
    >>> t = np.linspace(0, 1e-12, 1001)
    >>> values = np.random.random((len(t),))
    >>> data = FluxTimeData(values=values, t=t)
    """

    values: Array[float] = pd.Field(
        ..., title="Values", description="Values of the raw flux data in the time domain."
    )
    data_attrs: Dict[str, str] = pd.Field(
        {"units": "W", "long_name": "flux"},
        title="Data Attributes",
        description="Dictionary storing extra attributes associated with the monitor data.",
    )

    _dims = ("t",)


class ModeData(CollectionData):
    """Stores a collection of mode decomposition amplitudes and mode effective indexes for all
    modes in a :class:`.ModeMonitor`.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 1001)
    >>> mode_index = np.arange(2)
    >>> amps = (1+1j) * np.random.random((2, len(f), len(mode_index)))
    >>> amps_data = ModeAmpsData(values=amps, f=f, mode_index=mode_index)
    >>> n_complex = (1+1j) * np.random.random((len(f), len(mode_index)))
    >>> index_data = ModeIndexData(values=n_complex, f=f, mode_index=mode_index)
    >>> data = ModeData(data_dict={'n_complex': index_data, 'amps': amps_data})
    """

    data_dict: Dict[str, Union[ModeAmpsData, ModeIndexData]] = pd.Field(
        ...,
        title="Data Dictionary",
        description="Mapping of 'amps' to :class:`.ModeAmpsData` "
        "and 'n_complex' to :class:`.ModeIndexData` for the :class:`.ModeMonitor`.",
    )

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
            return scalar_data.n_eff
        return None

    @property
    def k_eff(self):
        """Get imaginary part of effective index."""
        scalar_data = self.data_dict.get("n_complex")
        if scalar_data:
            return scalar_data.k_eff
        return None


class ModeFieldData(AbstractFieldData):
    """Like FieldData but with extra dimension ``mode_index``.


    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 1001)
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-2, 2, 20)
    >>> z = np.linspace(0, 0, 1)
    >>> mode_index = np.arange(0, 4)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(z), len(f), len(mode_index)))
    >>> field = ScalarModeFieldData(values=values, x=x, y=y, z=z, f=f, mode_index=mode_index)
    >>> data = ModeFieldData(data_dict={'Ex': field, 'Ey': field})
    """

    data_dict: Dict[str, ScalarModeFieldData] = pd.Field(
        ...,
        title="Data Dictionary",
        description="Mapping of field name to the corresponding :class:`.ScalarModeFieldData`.",
    )

    def sel_mode_index(self, mode_index):
        """Return a FieldData at the selected mode index."""
        if mode_index not in self.Ex.mode_index:
            raise DataError("Requested 'mode_index' not stored in ModeFieldData.")

        data_dict = {}
        for field_name, scalar_data in self.data_dict.items():
            scalar_dict = scalar_data.dict()
            scalar_dict.pop("mode_index")
            scalar_dict.pop(TYPE_TAG_STR)
            scalar_dict["values"] = scalar_data.data.sel(mode_index=mode_index).values
            data_dict[field_name] = ScalarFieldData(**scalar_dict)

        return FieldData(data_dict=data_dict)


""" Near-to-far transformation """

class AbstractRadiationVector(FreqData):
    """Stores a single scalar radiation vector in frequency domain.
    """

    values: Array[complex] = pd.Field(
        ...,
        title="Scalar Field Values",
        description="Multi-dimensional array storing the raw radiation vector "
        "values in freq. domain.",
    )

    def normalize(self, source_freq_amps: Array[complex]) -> None:
        """normalize the values by the amplitude of the source."""
        self.values /= source_freq_amps  # pylint: disable=no-member

class RadiationVectorAngular(AbstractRadiationVector):
    """Stores a single scalar radiation vector in frequency domain
       as a function of angles theta and phi.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> values = (1+1j) * np.random.random((len(theta), len(phi), len(f)))
    >>> data = RadiationVectorAngular(values=values, theta=theta, phi=phi, f=f)
    """

    theta: Array[float] = pd.Field(
        ...,
        title="Elevation angles",
        description="Array of theta observation angles.",
        units=RADIAN,
    )

    phi: Array[float] = pd.Field(
        ...,
        title="Azimuth angles",
        description="Array of phi observation angles.",
        units=RADIAN,
    )

    _dims = ("theta", "phi", "f")

class RadiationVectorCartesian(AbstractRadiationVector):
    """Stores a single scalar radiation vector in frequency domain
       as a function of x, y, and z coordinates.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> x = np.linspace(0, 5, 10)
    >>> y = np.linspace(0, 10, 20)
    >>> z = np.atleast_1d(50)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(z), len(f)))
    >>> data = RadiationVectorCartesian(values=values, x=x, y=y, z=z, f=f)
    """

    x: Array[float] = pd.Field(
        ...,
        title="x coordinates",
        description="Array of observation x coordinates.",
        units=MICROMETER,
    )

    y: Array[float] = pd.Field(
        ...,
        title="y coordinates",
        description="Array of observation y coordinates.",
        units=MICROMETER,
    )

    z: Array[float] = pd.Field(
        ...,
        title="z coordinates",
        description="Array of observation z coordinates.",
        units=MICROMETER,
    )

    _dims = ('x', 'y', 'z', 'f')

class RadiationVectorKSpace(AbstractRadiationVector):
    """Stores a single scalar radiation vector in frequency domain
       as a function of normalized kx and ky vectors on the observation plane.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> ux = np.linspace(0, 5, 10)
    >>> uy = np.linspace(0, 10, 20)
    >>> values = (1+1j) * np.random.random((len(ux), len(uy), len(f)))
    >>> data = RadiationVectorKSpace(values=values, ux=ux, uy=uy, f=f)
    """

    ux: Array[float] = pd.Field(
        ...,
        title="Normalized kx",
        description="Array of observation kx values normalized by wave number.",
    )

    uy: Array[float] = pd.Field(
        ...,
        title="Normalized ky",
        description="Array of observation ky values normalized by wave number.",
    )

    _dims = ('ux', 'uy', 'f')


class AbstractNear2FarData(CollectionData, ABC):
    """Stores a collection of radiation vectors in the frequency domain.
    """

    @property
    def Ntheta(self):
        """Get Ntheta component of field using '.Ntheta' syntax."""
        scalar_data = self.data_dict.get("Ntheta")
        if scalar_data:
            return scalar_data.data
        return None

    @property
    def Nphi(self):
        """Get Nphi component of field using '.Nphi' syntax."""
        scalar_data = self.data_dict.get("Nphi")
        if scalar_data:
            return scalar_data.data
        return None

    @property
    def Ltheta(self):
        """Get Ltheta component of field using '.Ltheta' syntax."""
        scalar_data = self.data_dict.get("Ltheta")
        if scalar_data:
            return scalar_data.data
        return None

    @property
    def Lphi(self):
        """Get Lphi component of field using '.Lphi' syntax."""
        scalar_data = self.data_dict.get("Lphi")
        if scalar_data:
            return scalar_data.data
        return None

    def nk(self, frequency, medium) -> Tuple[float, float]:
        """Returns the real and imaginary parts of the background medium's refractive index."""
        eps_complex = medium.eps_model(frequency)
        return medium.eps_complex_to_nk(eps_complex)

    def k(self, frequency, medium) -> complex:
        """Returns the complex wave number associated with the background medium."""
        index_n, index_k = self.nk(frequency, medium)
        return (2 * np.pi * frequency / C_0) * (index_n + 1j * index_k)

    def eta(self, frequency, medium) -> complex:
        """Returns the complex wave impedance associated with the background medium."""
        eps_complex = medium.eps_model(frequency)
        return ETA_0 / np.sqrt(eps_complex)

    @staticmethod
    def car_2_sph(x: float, y: float, z: float):
        """Convert Cartesian to spherical coordinates.

        Parameters
        ----------
        x : float
            x coordinate relative to ``local_origin``.
        y : float
            y coordinate relative to ``local_origin``.
        z : float
            z coordinate relative to ``local_origin``.

        Returns
        -------
        r : float
            r coordinate relative to ``local_origin``.
        theta : float
            theta coordinate relative to ``local_origin``.
        phi : float
            phi coordinate relative to ``local_origin``.
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi

    @staticmethod
    def sph_2_car(r, theta, phi):
        """Convert spherical to Cartesian coordinates.

        Parameters
        ----------
        r : float
            radius.
        theta : float
            polar angle (rad) downward from x=y=0 line.
        phi : float
            azimuthal (rad) angle from y=z=0 line.

        Returns
        -------
        x : float
            x coordinate relative to ``local_origin``.
        y : float
            y coordinate relative to ``local_origin``.
        z : float
            z coordinate relative to ``local_origin``.
        """
        r_sin_theta = r * np.sin(theta)
        x = r_sin_theta * np.cos(phi)
        y = r_sin_theta * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    @staticmethod
    def sph_2_car_field(f_r, f_theta, f_phi, theta, phi):
        """Convert vector field components in spherical coordinates to cartesian.

        Parameters
        ----------
        f_r : float
            radial component of the vector field.
        f_theta : float
            polar angle component of the vector fielf.
        f_phi : float
            azimuthal angle component of the vector field.
        theta : float
            polar angle (rad) of location of the vector field.
        phi : float
            azimuthal angle (rad) of location of the vector field.

        Returns
        -------
        tuple
            x, y, and z components of the vector field in cartesian coordinates.
        """
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        f_x = f_r * sin_theta * cos_phi + f_theta * cos_theta * cos_phi - f_phi * sin_phi
        f_y = f_r * sin_theta * sin_phi + f_theta * cos_theta * sin_phi + f_phi * cos_phi
        f_z = f_r * cos_theta - f_theta * sin_theta
        return f_x, f_y, f_z


class Near2FarAngleData(AbstractNear2FarData):
    """Stores a collection of radiation vectors in the frequency domain on an angle-based grid
       from a :class:`.Near2FarAngleMonitor`.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> values = (1+1j) * np.random.random((len(theta), len(phi), len(f)))
    >>> fld = RadiationVectorAngular(values=values, theta=theta, phi=phi, f=f)
    >>> data = Near2FarAngleData(
    ...     data_dict={'Ntheta': fld, 'Nphi': fld, 'Ltheta': fld, 'Lphi': fld})
    """

    data_dict: Dict[str, RadiationVectorAngular] = pd.Field(
        ...,
        title="Data Dictionary",
        description="Mapping of the field names to their corresponding "
        ":class:`.RadiationVectorAngular`.",
    )

    # Ntheta: RadiationVectorAngular = pd.Field(
    #     None,
    #     title="Ntheta",
    #     description="Theta component of the radiation vector N.",
    #     )

    # Nphi: RadiationVectorAngular = pd.Field(
    #     None,
    #     title="Nphi",
    #     description="Phi component of the radiation vector N.",
    #     )

    # Ltheta: RadiationVectorAngular = pd.Field(
    #     None,
    #     title="Ltheta",
    #     description="Theta component of the radiation vector L.",
    #     )

    # Lphi: RadiationVectorAngular = pd.Field(
    #     None,
    #     title="Lphi",
    #     description="Phi component of the radiation vector L.",
    #     )

    # pylint:disable=too-many-locals
    def fields_spherical(
        self, r: float = None, medium: Medium = Medium(permittivity=1)
    ) -> xr.Dataset:
        """Get fields in spherical coordinates relative to the monitor's local origin
        for all angles and frequencies specified in the associated :class:`Near2FarMonitor`.
        If the radial distance ``r`` is provided, a corresponding phase factor is applied
        to the returned fields.

        Parameters
        ----------
        r : float = None
            (micron) radial distance relative to the monitor's local origin.
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: free space.

        Returns
        -------
        xarray.Dataset
            xarray dataset containing (Er, Etheta, Ephi), (Hr, Htheta, Hphi)
            in polar coordinates.
        """

        # Assumes that frequencies and angles are the same for all radiation vectors
        theta = self.Ntheta.theta
        phi = self.Ntheta.phi
        frequencies = self.Ntheta.f

        k = np.array([self.k(frequency, medium) for frequency in frequencies])
        eta = np.array([self.eta(frequency, medium) for frequency in frequencies])

        # assemble E felds
        if r is not None:
            scalar_proj_r = -1j * k * np.exp(1j * k * r) / (4 * np.pi / r)

            eta = eta[None, None, None, :]
            scalar_proj_r = scalar_proj_r[None, None, None, :]

            Et_array = -scalar_proj_r * (
                self.Lphi.values[None, ...] + eta * self.Ntheta.values[None, ...]
            )
            Ep_array = scalar_proj_r * (
                self.Ltheta.values[None, ...] - eta * self.Nphi.values[None, ...]
            )
            Er_array = np.zeros_like(Ep_array)

            dims = ("r", "theta", "phi", "f")
            coords = {"r": np.atleast_1d(r), "theta": theta, "phi": phi, "f": frequencies}

        else:
            eta = eta[None, None, ...]
            Et_array = -(self.Lphi.values + eta * self.Ntheta.values)
            Ep_array = self.Ltheta.values - eta * self.Nphi.values
            Er_array = np.zeros_like(Ep_array)

            dims = ("theta", "phi", "f")
            coords = {"theta": theta, "phi": phi, "f": frequencies}

        # assemble H fields
        Ht_array = -Ep_array / eta
        Hp_array = Et_array / eta
        Hr_array = np.zeros_like(Hp_array)

        Er = xr.DataArray(data=Er_array, coords=coords, dims=dims)
        Et = xr.DataArray(data=Et_array, coords=coords, dims=dims)
        Ep = xr.DataArray(data=Ep_array, coords=coords, dims=dims)

        Hr = xr.DataArray(data=Hr_array, coords=coords, dims=dims)
        Ht = xr.DataArray(data=Ht_array, coords=coords, dims=dims)
        Hp = xr.DataArray(data=Hp_array, coords=coords, dims=dims)

        field_data = xr.Dataset(
            {"E_r": Er, "E_theta": Et, "E_phi": Ep, "H_r": Hr, "H_theta": Ht, "H_phi": Hp}
        )

        return field_data

    def radar_cross_section(self, medium: Medium = Medium(permittivity=1)) -> xr.DataArray:
        """Get radar cross section at a point relative to the local origin in
        units of incident power.

        Parameters
        ----------
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: free space.

        Returns
        -------
        RCS : xarray.DataArray
            Radar cross section at angles relative to the local origin.
        """

        # Assumes that frequencies and angles are the same for all radiation vectors
        frequencies = self.Ntheta.f
        theta = self.Ntheta.theta
        phi = self.Ntheta.phi

        for frequency in frequencies:
            _, index_k = self.nk(frequency, medium)
            if index_k != 0.0:
                raise SetupError("Can't compute RCS for a lossy background medium.")

        k = np.array([self.k(frequency, medium) for frequency in frequencies])
        eta = np.array([self.eta(frequency, medium) for frequency in frequencies])

        k = k[None, None, ...]
        eta = eta[None, None, ...]

        constant = k**2 / (8 * np.pi * eta)
        term1 = np.abs(self.Lphi.values + eta * self.Ntheta.values) ** 2
        term2 = np.abs(self.Ltheta.values - eta * self.Nphi.values) ** 2
        rcs_data = constant * (term1 + term2)

        dims = ("theta", "phi", "f")
        coords = {"theta": theta, "phi": phi, "f": frequencies}

        return xr.DataArray(data=rcs_data, coords=coords, dims=dims)

    def power_spherical(self, r: float) -> xr.DataArray:
        """Get power scattered to a point relative to the local origin in spherical coordinates.

        Parameters
        ----------
        r : float
            (micron) radial distance relative to the local origin.

        Returns
        -------
        power : xarray.DataArray
            Power at points relative to the local origin.
        """

        field_data = self.fields_spherical(r)
        Et, Ep = [field_data[comp].values for comp in ["E_theta", "E_phi"]]
        Ht, Hp = [field_data[comp].values for comp in ["H_theta", "H_phi"]]
        power_theta = 0.5 * np.real(Et * np.conj(Hp))
        power_phi = 0.5 * np.real(-Ep * np.conj(Ht))
        power_data = power_theta + power_phi

        dims = ("r", "theta", "phi", "f")
        # Assumes that frequencies and angles are the same for all radiation vectors
        coords = {"r": [r], "theta": self.Ntheta.theta, "phi": self.Ntheta.phi, "f": self.Ntheta.f}

        return xr.DataArray(data=power_data, coords=coords, dims=dims)


class Near2FarCartesianData(AbstractNear2FarData):
    """Stores a collection of radiation vectors in the frequency domain on a Cartesian grid
       from a :class:`.Near2FarCartesianMonitor`.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> x = np.linspace(1, 10, 10)
    >>> y = np.linspace(1, 20, 20)
    >>> z = 5
    >>> values = (1+1j) * np.random.random((len(x), len(y), 1, len(f)))
    >>> fld = RadiationVectorCartesian(values=values, x=x, y=y, z=z f=f)
    >>> data = Near2FarCartesianData(
    ...     data_dict={'Ntheta': fld, 'Nphi': fld, 'Ltheta': fld, 'Lphi': fld})
    """

    data_dict: Dict[str, RadiationVectorCartesian] = pd.Field(
        ...,
        title="Data Dictionary",
        description="Mapping of the field names to their corresponding "
        ":class:`.RadiationVectorCartesian`.",
    )

    # Ntheta: RadiationVectorCartesian = pd.Field(
    #     None,
    #     title="Ntheta",
    #     description="Theta component of the radiation vector N.",
    #     )

    # Nphi: RadiationVectorCartesian = pd.Field(
    #     None,
    #     title="Nphi",
    #     description="Phi component of the radiation vector N.",
    #     )

    # Ltheta: RadiationVectorCartesian = pd.Field(
    #     None,
    #     title="Ltheta",
    #     description="Theta component of the radiation vector L.",
    #     )

    # Lphi: RadiationVectorCartesian = pd.Field(
    #     None,
    #     title="Lphi",
    #     description="Phi component of the radiation vector L.",
    #     )

    # pylint:disable=too-many-arguments, too-many-locals
    def fields_cartesian(
        self, medium: Medium = Medium(permittivity=1)
    ) -> xr.Dataset:
        """Get fields on a cartesian plane at a distance relative to monitor center
        along a given axis.

        Parameters
        ----------
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: free space.

        Returns
        -------
        xarray.Dataset
            xarray dataset containing (Ex, Ey, Ez), (Hx, Hy, Hz) in cartesian coordinates.
        """

        # Assumes that frequencies and coordinates are the same for all radiation vectors
        frequencies = self.Ntheta.f
        x = self.Ntheta.x
        y = self.Ntheta.y
        z = self.Ntheta.z
        x, y, z = [np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)]

        Ex_data = np.zeros((len(x), len(y), len(z), len(frequencies)), dtype=complex)
        Ey_data = np.zeros_like(Ex_data)
        Ez_data = np.zeros_like(Ex_data)

        Hx_data = np.zeros_like(Ex_data)
        Hy_data = np.zeros_like(Ex_data)
        Hz_data = np.zeros_like(Ex_data)

        for f, freq in enumerate(frequencies):
            for i, _x in enumerate(x):
                for j, _y in enumerate(y):
                    for k, _z in enumerate(z):
                        r, theta, phi = self.car_2_sph(_x, _y, _z)

                        wave_number = self.k(freq, medium)
                        eta = self.eta(freq, medium)
                        scalar_proj_r = \
                            -1j * wave_number * np.exp(1j * wave_number * r) / (4 * np.pi / r)

                        e_theta = -(self.Lphi.values[i,j,k,f] + eta * self.Ntheta.values[i,j,k,f])
                        e_phi = (self.Ltheta.values[i,j,k,f] - eta * self.Nphi.values[i,j,k,f])

                        Et = -scalar_proj_r * e_theta
                        Ep = scalar_proj_r * e_phi
                        Er = np.zeros_like(Et)

                        Ht = -Ep / eta
                        Hp = Et / eta
                        Hr = np.zeros_like(Hp)

                        e_fields = self.sph_2_car_field(Er, Et, Ep, theta, phi)
                        h_fields = self.sph_2_car_field(Hr, Ht, Hp, theta, phi)

                        Ex_data[i,j,k,f] = e_fields[0]
                        Ey_data[i,j,k,f] = e_fields[1]
                        Ez_data[i,j,k,f] = e_fields[2]

                        Hx_data[i,j,k,f] = h_fields[0]
                        Hy_data[i,j,k,f] = h_fields[1]
                        Hz_data[i,j,k,f] = h_fields[2]

        dims = ("x", "y", "z", "f")
        coords = {"x": x, "y": y, "z": z, "f": frequencies}

        Ex = xr.DataArray(data=Ex_data, coords=coords, dims=dims)
        Ey = xr.DataArray(data=Ey_data, coords=coords, dims=dims)
        Ez = xr.DataArray(data=Ez_data, coords=coords, dims=dims)

        Hx = xr.DataArray(data=Hx_data, coords=coords, dims=dims)
        Hy = xr.DataArray(data=Hy_data, coords=coords, dims=dims)
        Hz = xr.DataArray(data=Hz_data, coords=coords, dims=dims)

        field_data = xr.Dataset({"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz})

        return field_data


    # def power_cartesian(self, x: ArrayLikeN2F, y: ArrayLikeN2F, z: ArrayLikeN2F) -> xr.DataArray:
    #     """Get power scattered to a point relative to the local origin in cartesian coordinates.

    #     Parameters
    #     ----------
    #     x : Union[float, Tuple[float, ...], np.ndarray]
    #         (micron) x distances relative to the local origin.
    #     y : Union[float, Tuple[float, ...], np.ndarray]
    #         (micron) y distances relative to the local origin.
    #     z : Union[float, Tuple[float, ...], np.ndarray]
    #         (micron) z distances relative to the local origin.

    #     Returns
    #     -------
    #     power : xarray.DataArray
    #         Power at points relative to the local origin.
    #     """

    #     x, y, z = [np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)]

    #     power_data = np.zeros((len(x), len(y), len(z)))

    #     for i in track(np.arange(len(x)), description="Computing far field power"):
    #         _x = x[i]
    #         for j in np.arange(len(y)):
    #             _y = y[j]
    #             for k in np.arange(len(z)):
    #                 _z = z[k]

    #                 r, theta, phi = self._car_2_sph(_x, _y, _z)
    #                 power_data[i, j, k] = self.power_spherical(r, theta, phi).values

    #     dims = ("x", "y", "z")
    #     coords = {"x": x, "y": y, "z": z}

    #     return xr.DataArray(data=power_data, coords=coords, dims=dims)


# Default number of points per wavelength in the background medium to use for resampling fields.
PTS_PER_WVL = 10

# Numpy float array and related array types
ArrayLikeN2F = Union[float, Tuple[float, ...], ArrayLike[float, 4]]


class Near2FarSurface(Tidy3dBaseModel):
    """Data structure to store surface monitor data with associated surface current densities."""

    monitor: FieldMonitor = pd.Field(
        ...,
        title="Near field monitor",
        description=":class:`.FieldMonitor` on which near fields will be sampled and integrated.",
    )

    normal_dir: Direction = pd.Field(
        ...,
        title="Normal vector orientation",
        description=":class:`.Direction` of the surface monitor's normal vector w.r.t. "
        "the positive x, y or z unit vectors. Must be one of '+' or '-'.",
    )

    @property
    def axis(self) -> Axis:
        """Returns the :class:`.Axis` normal to this surface."""
        # assume that the monitor's axis is in the direction where the monitor is thinnest
        return self.monitor.size.index(0.0)

    @pd.validator("monitor", always=True)
    def is_plane(cls, val):
        """Ensures that the monitor is a plane, i.e., its `size` attribute has exactly 1 zero"""
        size = val.size
        if size.count(0.0) != 1 and isinstance(val, FieldMonitor):
            raise ValidationError(f"Monitor '{val.name}' must be planar, given size={size}")
        return val


""" Simulation data """


# maps MonitorData.type string to the actual type, for MonitorData.from_file()
DATA_TYPE_MAP = {
    "ScalarFieldData": ScalarFieldData,
    "ScalarFieldTimeData": ScalarFieldTimeData,
    "ScalarPermittivityData": ScalarPermittivityData,
    "ScalarModeFieldData": ScalarModeFieldData,
    "FieldData": FieldData,
    "FieldTimeData": FieldTimeData,
    "PermittivityData": PermittivityData,
    "FluxData": FluxData,
    "FluxTimeData": FluxTimeData,
    "ModeAmpsData": ModeAmpsData,
    "ModeIndexData": ModeIndexData,
    "ModeData": ModeData,
    "ModeFieldData": ModeFieldData,
    "RadiationVectorAngular": RadiationVectorAngular,
    "RadiationVectorCartesian": RadiationVectorCartesian,
    "RadiationVectorKSpace": RadiationVectorKSpace,
    "Near2FarAngleData": Near2FarAngleData,
    "Near2FarCartesianData": Near2FarCartesianData,
}


class AbstractSimulationData(Tidy3dBaseDataModel, ABC):
    """Abstract class to store a simulation and some data associated with it."""

    simulation: Simulation = pd.Field(
        ...,
        title="Simulation",
        description="Original :class:`.Simulation` associated with the data.",
    )

    @equal_aspect
    @add_ax_if_none
    # pylint:disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def plot_field_array(
        self,
        field_data: xr.DataArray,
        axis: Axis,
        position: float,
        val: Literal["real", "imag", "abs"] = "real",
        freq: float = None,
        eps_alpha: float = 0.2,
        robust: bool = True,
        vmin: float = None,
        vmax: float = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot the field data for a monitor with simulation plot overlayed.

        Parameters
        ----------
        field_data: xr.DataArray
            DataArray with the field data to plot.
        axis: Axis
            Axis normal to the plotting plane.
        position: float
            Position along the axis.
        val : Literal['real', 'imag', 'abs'] = 'real'
            Which part of the field to plot.
        freq: float = None
            Frequency at which the permittivity is evaluated at (if dispersive).
            By default, chooses permittivity as frequency goes to infinity.
        eps_alpha : float = 0.2
            Opacity of the structure permittivity.
            Must be between 0 and 1 (inclusive).
        robust : bool = True
            If True and vmin or vmax are absent, uses the 2nd and 98th percentiles of the data
            to compute the color limits. This helps in visualizing the field patterns especially
            in the presence of a source.
        vmin : float = None
            The lower bound of data range that the colormap covers. If `None`, they are
            inferred from the data and other keyword arguments.
        vmax : float = None
            The upper bound of data range that the colormap covers. If `None`, they are
            inferred from the data and other keyword arguments.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        # select the cross section data
        axis_label = "xyz"[axis]
        interp_kwarg = {axis_label: position}

        if len(field_data.coords[axis_label]) > 1:
            try:
                field_data = field_data.interp(**interp_kwarg)

            except Exception as e:
                raise DataError(f"Could not interpolate data at {axis_label}={position}.") from e

        # select the field value
        if val not in ("real", "imag", "abs"):
            raise DataError(f"'val' must be one of ``{'real', 'imag', 'abs'}``, given {val}")

        if val == "real":
            field_data = field_data.real
        elif val == "imag":
            field_data = field_data.imag
        elif val == "abs":
            field_data = abs(field_data)

        if val == "abs":
            cmap = "magma"
            eps_reverse = True
        else:
            cmap = "RdBu"
            eps_reverse = False

        # plot the field
        xy_coord_labels = list("xyz")
        xy_coord_labels.pop(axis)
        x_coord_label, y_coord_label = xy_coord_labels  # pylint:disable=unbalanced-tuple-unpacking
        field_data.plot(
            ax=ax, x=x_coord_label, y=y_coord_label, robust=robust, cmap=cmap, vmin=vmin, vmax=vmax
        )

        # plot the simulation epsilon
        ax = self.simulation.plot_structures_eps(
            freq=freq,
            cbar=False,
            alpha=eps_alpha,
            reverse=eps_reverse,
            ax=ax,
            **{axis_label: position},
        )

        # set the limits based on the xarray coordinates min and max
        x_coord_values = field_data.coords[x_coord_label]
        y_coord_values = field_data.coords[y_coord_label]
        ax.set_xlim(min(x_coord_values), max(x_coord_values))
        ax.set_ylim(min(y_coord_values), max(y_coord_values))

        return ax


class SimulationData(AbstractSimulationData):
    """Holds :class:`Monitor` data associated with :class:`Simulation`."""

    monitor_data: Dict[str, Tidy3dData] = pd.Field(
        ...,
        title="Monitor Data",
        description="Mapping of monitor name to :class:`Tidy3dData` instance.",
    )

    log_string: str = pd.Field(
        None,
        title="Log String",
        description="A string containing the log information from the simulation run.",
    )

    diverged: bool = pd.Field(
        False,
        title="Diverged Flag",
        description="A boolean flag denoting if the simulation run diverged.",
    )

    # set internally by the normalize function
    _normalize_index: pd.NonNegativeInt = pd.PrivateAttr(None)
    """
        title="Normalization Index",
        description="Index into the ``Simulation.sources`` "
        "indicating which source normalized the data.",
    """

    @property
    def normalized(self) -> bool:
        """Is this data normalized?"""
        return self._normalize_index is not None

    @property
    def normalize_index(self) -> pd.NonNegativeInt:
        """What is the index of the source that normalized this data. If ``None``, unnormalized."""
        return self._normalize_index

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
        final_decay = 1.0
        if decay_lines:
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
        self.ensure_monitor_exists(monitor_name)
        monitor_data = self.monitor_data.get(monitor_name)
        if isinstance(monitor_data, SpatialCollectionData):
            monitor_data.set_symmetry_attrs(self.simulation, monitor_name)
        return monitor_data.sim_data_getitem

    def ensure_monitor_exists(self, monitor_name: str) -> None:
        """Raise exception if monitor isn't in the simulation data"""
        if monitor_name not in self.monitor_data:
            raise DataError(f"Data for monitor '{monitor_name}' not found in simulation data.")

    def ensure_field_monitor(self, data_obj: Tidy3dData) -> None:
        """Raise exception if monitor isn't a field monitor."""
        if not isinstance(data_obj, (FieldData, FieldTimeData, ModeFieldData)):
            raise DataError(f"data_obj '{data_obj}' " "not an AbstractFieldData instance.")

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
        field_monitor_data = self[field_monitor_name]
        self.ensure_field_monitor(field_monitor_data)

        # get the monitor, discretize, and get center locations
        monitor = self.simulation.get_monitor_by_name(field_monitor_name)
        sub_grid = self.simulation.discretize(monitor)
        centers = sub_grid.centers

        xs = np.array(centers.x)
        ys = np.array(centers.y)
        zs = np.array(centers.z)

        return field_monitor_data.colocate(x=xs, y=ys, z=zs)

    # pylint:disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def plot_field(
        self,
        field_monitor_name: str,
        field_name: str,
        x: float = None,
        y: float = None,
        z: float = None,
        val: Literal["real", "imag", "abs"] = "real",
        freq: float = None,
        time: float = None,
        mode_index: int = None,
        eps_alpha: float = 0.2,
        robust: bool = True,
        vmin: float = None,
        vmax: float = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot the field data for a monitor with simulation plot overlayed.

        Parameters
        ----------
        field_monitor_name : str
            Name of :class:`FieldMonitor` or :class:`FieldTimeData` to plot.
        field_name : str
            Name of `field` in monitor to plot (eg. 'Ex').
            Also accepts `'int'` to plot intensity.
        x : float = None
            Position of plane in x direction.
        y : float = None
            Position of plane in y direction.
        z : float = None
            Position of plane in z direction.
        val : Literal['real', 'imag', 'abs'] = 'real'
            Which part of the field to plot.
            If ``field_name='int'``, this has no effect.
        freq: float = None
            If monitor is a :class:`FieldMonitor`, specifies the frequency (Hz) to plot the field.
            Also sets the frequency at which the permittivity is evaluated at (if dispersive).
            By default, chooses permittivity as frequency goes to infinity.
        time: float = None
            if monitor is a :class:`FieldTimeMonitor`, specifies the time (sec) to plot the field.
        mode_index: int = None
            if monitor is a :class:`ModeFieldMonitor`, specifies which mode index to plot.
        eps_alpha : float = 0.2
            Opacity of the structure permittivity.
            Must be between 0 and 1 (inclusive).
        robust : bool = True
            If True and vmin or vmax are absent, uses the 2nd and 98th percentiles of the data
            to compute the color limits. This helps in visualizing the field patterns especially
            in the presence of a source.
        vmin : float = None
            The lower bound of data range that the colormap covers. If `None`, they are
            inferred from the data and other keyword arguments.
        vmax : float = None
            The upper bound of data range that the colormap covers. If `None`, they are
            inferred from the data and other keyword arguments.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        # get the monitor data
        monitor_data = self[field_monitor_name]
        self.ensure_field_monitor(monitor_data)
        if isinstance(monitor_data, ModeFieldData):
            if mode_index is None:
                raise DataError("'mode_index' must be supplied to plot a ModeFieldMonitor.")
            monitor_data = monitor_data.sel_mode_index(mode_index=mode_index)

        # get the field data component
        if field_name == "int":
            monitor_data = self.at_centers(field_monitor_name)
            xr_data = 0.0
            for field in ("Ex", "Ey", "Ez"):
                field_data = monitor_data[field]
                xr_data += abs(field_data) ** 2
                xr_data.name = "Intensity"
            val = "abs"
        else:
            monitor_data.ensure_member_exists(field_name)
            xr_data = monitor_data.data_dict.get(field_name).data

        # select the frequency or time value
        if "f" in xr_data.coords:
            if freq is None:
                raise DataError("'freq' must be supplied to plot a FieldMonitor.")
            field_data = xr_data.sel(f=freq, method="nearest")
        elif "t" in xr_data.coords:
            if time is None:
                raise DataError("'time' must be supplied to plot a FieldTimeMonitor.")
            field_data = xr_data.sel(t=time, method="nearest")
        else:
            raise DataError("Field data has neither time nor frequency data, something went wrong.")

        if x is None and y is None and z is None:
            """If a planar monitor, infer x/y/z based on the plane position and normal."""
            monitor = self.simulation.get_monitor_by_name(field_monitor_name)
            try:
                axis = monitor.geometry.size.index(0.0)
                position = monitor.geometry.center[axis]
            except Exception as e:
                raise ValueError(
                    "If none of 'x', 'y' or 'z' is specified, monitor must have a "
                    "zero-sized dimension"
                ) from e
        else:
            axis, position = self.simulation.parse_xyz_kwargs(x=x, y=y, z=z)

        return self.plot_field_array(
            field_data=field_data,
            axis=axis,
            position=position,
            val=val,
            freq=freq,
            eps_alpha=eps_alpha,
            robust=robust,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )

    def normalize(self, normalize_index: Optional[int] = 0):
        """Return a copy of the :class:`.SimulationData` object with data normalized by source.

        Parameters
        ----------
        normalize_index : int = 0
            If specified, normalizes the frequency-domain data by the amplitude spectrum of the
            source corresponding to ``simulation.sources[normalize_index]``.
            This occurs when the data is loaded into a :class:`.SimulationData` object.

        Returns
        -------
        :class:`.SimulationData`
            A copy of the :class:`.SimulationData` with the data normalized by source spectrum.
        """

        sim_data_norm = self.copy(validate=False)

        # if no normalize index, just return the new copy right away.
        if normalize_index is None:
            return sim_data_norm

        # if data already normalized
        if self.normalized:

            # if with a different normalize index, raise an error
            if self._normalize_index != normalize_index:
                raise DataError(
                    "This SimulationData object has already been normalized "
                    f"with `normalize_index` of {self._normalize_index} "
                    f"and can't be normalized again with `normalize_index` of {normalize_index}."
                )

            # otherwise, just return the data
            return sim_data_norm

        # from here on, normalze_index is not None and the data has not been normalized

        # no sources, just warn and return
        if len(self.simulation.sources) == 0:
            log.warning(
                f"normalize_index={normalize_index} supplied but no sources found, "
                "not normalizing."
            )
            return sim_data_norm

        # try to get the source info
        try:
            source = self.simulation.sources[normalize_index]
            source_time = source.source_time
        except IndexError as e:
            raise DataError(f"Could not locate source at normalize_index={normalize_index}.") from e

        times = self.simulation.tmesh
        dt = self.simulation.dt
        boundaries = self.simulation.boundary_spec.to_list
        boundaries = [item for boundary in boundaries for item in boundary]
        complex_fields = any(isinstance(item, BlochBoundary) for item in boundaries)

        def normalize_data(monitor_data):
            """normalize a monitor data instance using the source time parameters."""
            freqs = monitor_data.f
            source_freq_amps = source_time.spectrum(times, freqs, dt, complex_fields)
            # We remove the user-defined phase from the normalization. Otherwise, with a single
            # source, we would get the exact same fields regardless of the source_time phase.
            # Instead we would like the field phase to be determined by the source_time phase.
            source_freq_amps *= np.exp(-1j * source_time.phase)
            monitor_data.normalize(source_freq_amps)

        for monitor_data in sim_data_norm.monitor_data.values():

            if isinstance(monitor_data, (FieldData, FluxData, ModeData, AbstractNear2FarData)):

                if isinstance(monitor_data, CollectionData):
                    for attr_data in monitor_data.data_dict.values():
                        normalize_data(attr_data)

                else:
                    normalize_data(monitor_data)

        sim_data_norm._normalize_index = normalize_index  # pylint:disable=protected-access
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

            # save diverged and normalized flags as attributes
            f_handle.attrs["diverged"] = self.diverged
            if self._normalize_index:
                f_handle.attrs["normalize_index"] = self._normalize_index

            # make a group for monitor_data
            mon_data_grp = f_handle.create_group("monitor_data")
            for mon_name, mon_data in self.monitor_data.items():

                # for each monitor, make new group with the same name
                mon_grp = mon_data_grp.create_group(mon_name)
                mon_data.add_to_group(mon_grp)

    @classmethod
    def from_file(
        cls, fname: str, normalize_index: Optional[int] = 0, **kwargs
    ):  # pylint:disable=arguments-differ
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
            sim_json_str = Tidy3dData.load_string(f_handle, "sim_json")
            updater = Updater.from_string(sim_json_str)
            sim_dict = updater.update_to_current()
            simulation = Simulation.parse_obj(sim_dict)

            # get the log if exists
            log_string = Tidy3dData.load_string(f_handle, "log_string")

            # set the diverged flag
            # TODO: add link to documentation discussing divergence
            diverged = f_handle.attrs["diverged"]
            if diverged:
                logging.warning("Simulation run has diverged!")

            # get whether this data has been normalized
            normalize_index_file = f_handle.attrs.get("normalize_index")

            # loop through monitor dataset and create all MonitorData instances
            monitor_data_dict = {}
            for monitor_name, monitor_data in f_handle["monitor_data"].items():

                # load this MonitorData instance, add to monitor_data dict
                _data_type = DATA_TYPE_MAP[Tidy3dData.load_string(monitor_data, TYPE_TAG_STR)]
                monitor_data_instance = _data_type.load_from_group(monitor_data)
                monitor_data_dict[monitor_name] = monitor_data_instance

        # create a SimulationData object
        sim_data = cls(
            simulation=simulation,
            monitor_data=monitor_data_dict,
            log_string=log_string,
            diverged=diverged,
            **kwargs,
        )

        # make sure to tag the SimulationData with the normalize_index stored from file
        sim_data._normalize_index = normalize_index_file

        # if normalize_index supplied as None, just return the sim_data right away (norm or not)
        if normalize_index is None:
            return sim_data

        # if the data in the file has not been normalized, normalize with supplied index
        if normalize_index_file is None:
            return sim_data.normalize(normalize_index=normalize_index)

        # from here on, normalze_index and normalize_index_file are present

        # if they are the same, just return as normal
        if normalize_index == normalize_index_file:
            return sim_data

        # if they aren't the same, throw an error
        raise DataError(
            "Data from this file is already normalized with "
            f"normalize_index={normalize_index_file}, can't normalize with supplied "
            f"normalize_index={normalize_index} unless they are the same "
            "or supplied normalize index is `None`."
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


class Near2Far(Tidy3dBaseModel):
    """Near field to far field transformation tool."""

    sim_data: SimulationData = pd.Field(
        ...,
        title="Simulation data",
        description="Container for simulation data containing the near field monitors.",
    )

    surfaces: Tuple[Near2FarSurface, ...] = pd.Field(
        None,
        title="Surface monitor with direction",
        description="Tuple of each :class:`.Near2FarSurface` to use as source of near field.",
    )

    resample: bool = pd.Field(
        True,
        title="Resample surface currents",
        description="Pick whether to resample surface currents based on ``pts_per_wavelength``. "
        "If ``False``, the field ``pts_per_wavelength`` has no effect.",
    )

    pts_per_wavelength: int = pd.Field(
        PTS_PER_WVL,
        title="Points per wavelength",
        description="Number of points per wavelength in the background medium with which "
        "to discretize the surface monitors for the projection.",
    )

    medium: Medium = pd.Field(
        None,
        title="Background medium",
        description="Background medium in which to radiate near fields to far fields. "
        "If ``None``, uses the :class:.Simulation background medium.",
    )

    origin: Coordinate = pd.Field(
        None,
        title="Local origin",
        description="Local origin used for defining observation points. If ``None``, uses the "
        "average of the centers of all surface monitors.",
        units=MICROMETER,
    )

    currents: Dict[str, xr.Dataset] = pd.Field(
        None,
        title="Surface current densities",
        description="Dictionary mapping monitor name to an ``xarray.Dataset`` storing the "
        "surface current densities.",
    )

    @pd.validator("origin", always=True)
    def set_origin(cls, val, values):
        """Sets .origin as the average of centers of all surface monitors if not provided."""
        if val is None:
            surfaces = values.get("surfaces")
            val = np.array([surface.monitor.center for surface in surfaces])
            return tuple(np.mean(val, axis=0))
        return val

    @pd.validator("medium", always=True)
    def set_medium(cls, val, values):
        """Sets the .medium field using the simulation default if no medium was provided."""
        if val is None:
            val = values.get("sim_data").simulation.medium
        return val

    @property
    def frequencies(self) -> Tuple[float, ...]:
        """Return the tuple of frequencies associated with the field monitors."""
        return self.surfaces[0].monitor.freqs

    def nk(self, frequency) -> Tuple[float, float]:
        """Returns the real and imaginary parts of the background medium's refractive index."""
        eps_complex = self.medium.eps_model(frequency)
        return self.medium.eps_complex_to_nk(eps_complex)

    def k(self, frequency) -> complex:
        """Returns the complex wave number associated with the background medium."""
        index_n, index_k = self.nk(frequency)
        return (2 * np.pi * frequency / C_0) * (index_n + 1j * index_k)

    def eta(self, frequency) -> complex:
        """Returns the complex wave impedance associated with the background medium."""
        eps_complex = self.medium.eps_model(frequency)
        return ETA_0 / np.sqrt(eps_complex)

    @classmethod
    # pylint:disable=too-many-arguments
    def from_near_field_monitors(
        cls,
        sim_data: SimulationData,
        monitors: Tuple[FieldMonitor, ...],
        normal_dirs: Tuple[Direction, ...],
        resample: bool = True,
        pts_per_wavelength: int = PTS_PER_WVL,
        medium: Medium = None,
        origin: Coordinate = None,
    ):
        """Constructs :class:`Near2Far` from a tuple of near field monitors and their directions.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitors.
        monitors : Tuple[:class:`.FieldMonitor`, ...]
            Tuple of :class:`.FieldMonitor` objects on which near fields will be sampled.
        normal_dirs : Tuple[:class:`.Direction`, ...]
            Tuple containing the :class:`.Direction` of the normal to each surface monitor
            w.r.t. to the positive x, y or z unit vectors. Must have the same length as monitors.
        resample : bool = True
            Pick whether to resample surface currents based on ``pts_per_wavelength``.
            "If ``False``, the argument ``pts_per_wavelength`` has no effect.
        pts_per_wavelength : int = 10
            Number of points per wavelength with which to discretize the
            surface monitors for the projection.
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: same as the :class:`.Simulation` background medium.
        origin : :class:`.Coordinate`
            Local origin used for defining observation points. If ``None``, uses the
            average of the centers of all surface monitors.
        """

        if len(monitors) != len(normal_dirs):
            raise SetupError(
                f"Number of monitors ({len(monitors)}) does not equal "
                "the number of directions ({len(normal_dirs)})."
            )

        surfaces = []
        for monitor, normal_dir in zip(monitors, normal_dirs):
            surfaces.append(Near2FarSurface(monitor=monitor, normal_dir=normal_dir))

        return cls(
            sim_data=sim_data,
            surfaces=surfaces,
            resample=resample,
            pts_per_wavelength=pts_per_wavelength,
            medium=medium,
            origin=origin,
        )

    @pd.validator("currents", always=True)
    def set_currents(cls, val, values):
        """Sets the surface currents."""
        sim_data = values.get("sim_data")
        surfaces = values.get("surfaces")
        resample = values.get("resample")
        pts_per_wavelength = values.get("pts_per_wavelength")
        medium = values.get("medium")

        if surfaces is None:
            return None

        val = {}
        for surface in surfaces:
            current_data = cls.compute_surface_currents(
                sim_data, surface, medium, resample, pts_per_wavelength
            )
            val[surface.monitor.name] = current_data

        return val

    @staticmethod
    # pylint:disable=too-many-arguments
    def compute_surface_currents(
        sim_data: SimulationData,
        surface: Near2FarSurface,
        medium: Medium,
        resample: bool = True,
        pts_per_wavelength: int = PTS_PER_WVL,
    ) -> xr.Dataset:
        """Returns resampled surface current densities associated with the surface monitor.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitors.
        surface: :class:`.Near2FarSurface`
            :class:`.Near2FarSurface` to use as source of near field.
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: same as the :class:`.Simulation` background medium.
        resample : bool = True
            Pick whether to resample surface currents based on ``pts_per_wavelength``.
            "If ``False``, the argument ``pts_per_wavelength`` has no effect.
        pts_per_wavelength : int = 10
            Number of points per wavelength with which to discretize the
            surface monitors for the projection.

        Returns
        -------
        xarray.Dataset
            Colocated surface current densities for the given surface.
        """

        try:
            field_data = sim_data[surface.monitor.name]
        except Exception as e:
            raise SetupError(
                f"No data for monitor named '{surface.monitor.name}' found in sim_data."
            ) from e

        currents = Near2Far._fields_to_currents(field_data, surface)
        currents = Near2Far._resample_surface_currents(
            currents, sim_data, surface, medium, resample, pts_per_wavelength
        )

        return currents

    @staticmethod
    def _fields_to_currents(field_data: FieldData, surface: Near2FarSurface) -> FieldData:
        """Returns surface current densities associated with a given :class:`.FieldData` object.

        Parameters
        ----------
        field_data : :class:`.FieldData`
            Container for field data associated with the given near field surface.
        surface: :class:`.Near2FarSurface`
            :class:`.Near2FarSurface` to use as source of near field.

        Returns
        -------
        :class:`.FieldData`
            Surface current densities for the given surface.
        """

        # figure out which field components are tangential or normal to the monitor
        normal_field, tangent_fields = surface.monitor.pop_axis(("x", "y", "z"), axis=surface.axis)

        signs = np.array([-1, 1])
        if surface.axis % 2 != 0:
            signs *= -1
        if surface.normal_dir == "-":
            signs *= -1

        # compute surface current densities and delete unneeded field components
        currents = field_data.copy(deep=True)
        cmp_1, cmp_2 = tangent_fields

        currents.data_dict["J" + cmp_2] = currents.data_dict.pop("H" + cmp_1)
        currents.data_dict["J" + cmp_1] = currents.data_dict.pop("H" + cmp_2)
        del currents.data_dict["H" + normal_field]

        currents.data_dict["M" + cmp_2] = currents.data_dict.pop("E" + cmp_1)
        currents.data_dict["M" + cmp_1] = currents.data_dict.pop("E" + cmp_2)
        del currents.data_dict["E" + normal_field]

        currents.data_dict["J" + cmp_1].values *= signs[0]
        currents.data_dict["J" + cmp_2].values *= signs[1]

        currents.data_dict["M" + cmp_1].values *= signs[1]
        currents.data_dict["M" + cmp_2].values *= signs[0]

        return currents

    @staticmethod
    # pylint:disable=too-many-locals, too-many-arguments
    def _resample_surface_currents(
        currents: xr.Dataset,
        sim_data: SimulationData,
        surface: Near2FarSurface,
        medium: Medium,
        resample: bool = True,
        pts_per_wavelength: int = PTS_PER_WVL,
    ) -> xr.Dataset:
        """Returns the surface current densities associated with the surface monitor.

        Parameters
        ----------
        currents : xarray.Dataset
            Surface currents defined on the original Yee grid.
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitors.
        surface: :class:`.Near2FarSurface`
            :class:`.Near2FarSurface` to use as source of near field.
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: same as the :class:`.Simulation` background medium.
        resample : bool = True
            Pick whether to resample surface currents based on ``pts_per_wavelength``.
            "If ``False``, the argument ``pts_per_wavelength`` has no effect.
        pts_per_wavelength : int = 10
            Number of points per wavelength with which to discretize the
            surface monitors for the projection.

        Returns
        -------
        xarray.Dataset
            Colocated surface current densities for the given surface.
        """

        # colocate surface currents on a regular grid of points on the monitor based on wavelength
        colocation_points = [None] * 3
        colocation_points[surface.axis] = surface.monitor.center[surface.axis]

        # use the highest frequency associated with the monitor to resample the surface currents
        frequency = max(surface.monitor.freqs)
        eps_complex = medium.eps_model(frequency)
        index_n, _ = medium.eps_complex_to_nk(eps_complex)
        wavelength = C_0 / frequency / index_n

        _, idx_uv = surface.monitor.pop_axis((0, 1, 2), axis=surface.axis)

        for idx in idx_uv:

            if not resample:
                comp = ["x", "y", "z"][idx]
                colocation_points[idx] = sim_data.at_centers(surface.monitor.name)[comp].values
                continue

            # pick sample points on the monitor and handle the possibility of an "infinite" monitor
            start = np.maximum(
                surface.monitor.center[idx] - surface.monitor.size[idx] / 2.0,
                sim_data.simulation.center[idx] - sim_data.simulation.size[idx] / 2.0,
            )
            stop = np.minimum(
                surface.monitor.center[idx] + surface.monitor.size[idx] / 2.0,
                sim_data.simulation.center[idx] + sim_data.simulation.size[idx] / 2.0,
            )
            size = stop - start

            num_pts = int(np.ceil(pts_per_wavelength * size / wavelength))
            points = np.linspace(start, stop, num_pts)
            colocation_points[idx] = points

        currents = currents.colocate(*colocation_points)
        return currents

    # pylint:disable=too-many-locals, too-many-arguments
    def _radiation_vectors_for_surface(
        self,
        frequency: float,
        theta: ArrayLikeN2F,
        phi: ArrayLikeN2F,
        surface: Near2FarSurface,
        currents: xr.Dataset,
    ):
        """Compute radiation vectors at an angle in spherical coordinates
        for a given set of surface currents and observation angles.

        Parameters
        ----------
        frequency : float
            Frequency to select from each :class:`.FieldMonitor` to use for projection.
            Must be a frequency stored in each :class:`FieldMonitor`.
        theta : Union[float, Tuple[float, ...], np.ndarray]
            Polar angles (rad) downward from x=y=0 line relative to the local origin.
        phi : Union[float, Tuple[float, ...], np.ndarray]
            Azimuthal (rad) angles from y=z=0 line relative to the local origin.
        surface: :class:`Near2FarSurface`
            :class:`Near2FarSurface` object to use as source of near field.
        currents : xarray.Dataset
            xarray Dataset containing surface currents associated with the surface monitor.

        Returns
        -------
        Tuple[numpy.ndarray[float], ...]
            ``N_theta``, ``N_phi``, ``L_theta``, ``L_phi`` radiation vectors for the given surface.
        """

        # make sure that observation points are interpreted w.r.t. the local origin
        pts = [currents[name].values - origin for name, origin in zip(["x", "y", "z"], self.origin)]

        try:
            currents_f = currents.sel(f=frequency)
        except Exception as e:
            raise SetupError(
                f"Frequency {frequency} not found in fields for monitor '{surface.monitor.name}'."
            ) from e

        idx_w, idx_uv = surface.monitor.pop_axis((0, 1, 2), axis=surface.axis)
        _, source_names = surface.monitor.pop_axis(("x", "y", "z"), axis=surface.axis)

        idx_u, idx_v = idx_uv
        cmp_1, cmp_2 = source_names

        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        J = np.zeros((3, len(theta), len(phi)), dtype=complex)
        M = np.zeros_like(J)

        def integrate_2d(function, pts_u, pts_v):
            """Trapezoidal integration in two dimensions."""
            return np.trapz(np.trapz(function, pts_u, axis=0), pts_v, axis=0)
            # return np.sum(np.sum(function, axis=0), axis=0)

        phase = [None] * 3
        propagation_factor = -1j * self.k(frequency)

        def integrate_for_one_theta(i_th: int):
            """Perform integration for a given theta angle index"""

            for j_ph in np.arange(len(phi)):

                phase[0] = np.exp(propagation_factor * pts[0] * sin_theta[i_th] * cos_phi[j_ph])
                phase[1] = np.exp(propagation_factor * pts[1] * sin_theta[i_th] * sin_phi[j_ph])
                phase[2] = np.exp(propagation_factor * pts[2] * cos_theta[i_th])

                phase_ij = phase[idx_u][:, None] * phase[idx_v][None, :] * phase[idx_w]

                J[idx_u, i_th, j_ph] = integrate_2d(
                    currents_f["J" + cmp_1].values * phase_ij, pts[idx_u], pts[idx_v]
                )
                J[idx_v, i_th, j_ph] = integrate_2d(
                    currents_f["J" + cmp_2].values * phase_ij, pts[idx_u], pts[idx_v]
                )

                M[idx_u, i_th, j_ph] = integrate_2d(
                    currents_f["M" + cmp_1].values * phase_ij, pts[idx_u], pts[idx_v]
                )
                M[idx_v, i_th, j_ph] = integrate_2d(
                    currents_f["M" + cmp_2].values * phase_ij, pts[idx_u], pts[idx_v]
                )

        if len(theta) < 2:
            integrate_for_one_theta(0)
        else:
            for i_th in track(
                np.arange(len(theta)),
                description=f"Processing surface monitor '{surface.monitor.name}'...",
            ):
                integrate_for_one_theta(i_th)

        cos_th_cos_phi = cos_theta[:, None] * cos_phi[None, :]
        cos_th_sin_phi = cos_theta[:, None] * sin_phi[None, :]

        # N_theta (8.33a)
        N_theta = J[0] * cos_th_cos_phi + J[1] * cos_th_sin_phi - J[2] * sin_theta[:, None]

        # N_phi (8.33b)
        N_phi = -J[0] * sin_phi[None, :] + J[1] * cos_phi[None, :]

        # L_theta  (8.34a)
        L_theta = M[0] * cos_th_cos_phi + M[1] * cos_th_sin_phi - M[2] * sin_theta[:, None]

        # L_phi  (8.34b)
        L_phi = -M[0] * sin_phi[None, :] + M[1] * cos_phi[None, :]

        return N_theta, N_phi, L_theta, L_phi

    def radiation_vectors(self, theta: ArrayLikeN2F, phi: ArrayLikeN2F) -> Near2FarAngleData:
        """Compute radiation vectors at given angles in spherical coordinates.

        Parameters
        ----------
        theta : Union[float, Tuple[float, ...], np.ndarray]
            Polar angles (rad) downward from x=y=0 line relative to the local origin.
        phi : Union[float, Tuple[float, ...], np.ndarray]
            Azimuthal (rad) angles from y=z=0 line relative to the local origin.

        Returns
        -------
        :class:.`Near2FarAngleData`
            Data structure with ``N_theta``, ``N_phi``, ``L_theta``, ``L_phi`` radiation vectors.
        """

        freqs = self.frequencies

        # compute radiation vectors for the dataset associated with each monitor
        N_theta = np.zeros((len(theta), len(phi), len(freqs)), dtype=complex)
        N_phi = np.zeros_like(N_theta)
        L_theta = np.zeros_like(N_theta)
        L_phi = np.zeros_like(N_theta)

        for idx_f, frequency in enumerate(freqs):
            for surface in self.surfaces:
                _N_th, _N_ph, _L_th, _L_ph = self._radiation_vectors_for_surface(
                    frequency, theta, phi, surface, self.currents[surface.monitor.name]
                )
                N_theta[..., idx_f] += _N_th
                N_phi[..., idx_f] += _N_ph
                L_theta[..., idx_f] += _L_th
                L_phi[..., idx_f] += _L_ph

        nth = RadiationVectorAngular(values=N_theta, theta=theta, phi=phi, f=freqs)
        nph = RadiationVectorAngular(values=N_phi, theta=theta, phi=phi, f=freqs)
        lth = RadiationVectorAngular(values=L_theta, theta=theta, phi=phi, f=freqs)
        lph = RadiationVectorAngular(values=L_phi, theta=theta, phi=phi, f=freqs)

        return Near2FarAngleData(
            data_dict={"Ntheta": nth, "Nphi": nph, "Ltheta": lth, "Lphi": lph})
