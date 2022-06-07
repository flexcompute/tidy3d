# pylint: disable=unused-import, too-many-lines
"""Classes for Storing Monitor and Simulation Data."""

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Tuple
import logging

import xarray as xr
import numpy as np
import h5py
import pydantic as pd

from .types import Numpy, Direction, Array, numpy_encoding, Literal, Ax, Coordinate, Axis
from .types import ArrayLike
from .base import Tidy3dBaseModel, TYPE_TAG_STR
from .simulation import Simulation
from .boundary import Symmetry, BlochBoundary
from .monitor import Monitor
from .grid import Grid, Coords
from .viz import add_ax_if_none, equal_aspect
from ..log import DataError, log
from ..constants import HERTZ, SECOND, MICROMETER
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
            np.ndarray: numpy_encoding,  # use custom encoding defined in .types
            np.int64: lambda x: int(x),  # pylint: disable=unnecessary-lambda
            Tidy3dDataArray: lambda x: None,  # dont write
            xr.Dataset: lambda x: None,  # dont write
        }

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
        list_of_str = [v.decode("utf-8") for v in list_of_bytes]
        return list_of_str


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
        kwargs = {}

        # construct kwarg dict from hdf5 data group for monitor
        for data_name, data_value in hdf5_grp.items():
            kwargs[data_name] = np.array(data_value)

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
        span_inds = simulation.grid.discretize_inds(monitor.geometry, extend=True)
        boundary_dict = {}
        for idim, dim in enumerate(["x", "y", "z"]):
            ind_beg, ind_end = span_inds[idim]
            boundary_dict[dim] = simulation.grid.periodic_subspace(idim, ind_beg, ind_end + 1)
        mnt_grid = Grid(boundaries=Coords(**boundary_dict))
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

    values: ArrayLike = pd.Field(
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
        if len(decay_lines) > 0:
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

        # colocate each of the field components at centers
        field_dataset = field_monitor_data.colocate(x=centers.x, y=centers.y, z=centers.z)
        return field_dataset

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

        sim_data_norm = self.copy(deep=True)

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

            if isinstance(monitor_data, (FieldData, FluxData, ModeData)):

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
