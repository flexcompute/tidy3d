from typing import Dict, List, Union
import json

import pydantic as pd
import h5py
import xarray as xr

from .base import Tidy3dBaseModel
from .types import ArrayLike
from .simulation import Simulation
from .monitor import FieldMonitor, FieldTimeMonitor, FluxMonitor, FluxTimeMonitor, ModeMonitor
from .monitor import PermittivityMonitor, ModeFieldMonitor

# every type that can be used as a component
ComponentType = Union[
    Simulation,
    FieldMonitor,
    FieldTimeMonitor,
    FluxMonitor,
    FluxTimeMonitor,
    ModeMonitor,
    ModeFieldMonitor,
]


class Tidy3dData(Tidy3dBaseModel):

    component: ComponentType = pd.Field(
        None,
        title="Original component",
        description="Copy of the tidy3d component that this data is associated with.",
    )

    data: xr.DataArray = pd.Field(
        None,
        title="Data",
        description="An xarray DataArray corresponding to the data stord by this element.",
    )

    data_dict: Dict[str, "Tidy3dData"] = pd.Field(
        {},
        title="Data dictionary",
        description="Collection of other tidy3d data stored by this element",
    )

    @property
    def coords(self) -> Dict[str, ArrayLike]:
        """Labelled coordinates corresponding to the axes of ``self.values``."""
        return self.data.coords

    @property
    def values(self) -> ArrayLike:
        """The raw data associated with the data, represented as a multi-dimensional array."""
        return self.data.values

    @property
    def dims(self) -> Tuple[str]:
        """Tuple of the ordered coordinate labels corresponding to each axis of ``self.values.``"""
        return self.data.dims

    def __getitem__(self, name) -> "Tidy3dData":
        """Return the ``Tidy3dData`` associated with the collection member using square brackets."""
        item = self.data_dict.get(name)
        if item is None:
            raise Tidy3dKeyError(f'item "{item}" not found in data_dict.')
        return item

    @property
    def json_string(self) -> str:
        """A string representation of everything execpt for the ``.data`` and ``.data_dict``."""
        return self.json(exclude={"data", "data_dict"})

    """ Saving to an hdf5 file."""

    def save_component(self, f_handle, hdf5_grp):
        """Convert everything but the ``.data`` and ``.data_dict`` to a string and save to group."""
        hdf5_grp.create_dataset("json", data=self.json_string)

    def save_data(self, f_handle, hdf5_grp):
        """Save the ``xarray.DataArray`` to an hdf5 group."""
        hdf5_grp.create_dataset("values", data=self.values)
        for dim in self.dims:
            coord = self.coords[dim]
            hdf5_grp.create_dataset(dim, data=coord)

    def save_data_dict(self, f_handle, hdf5_grp):
        """Save the ``data_dict`` to an hdf5 group recursively."""
        for name, tidy3d_data in self.data_dict.items():
            new_group = f_handle.create_group(name)
            tidy3d_data.add_to_hdf5_group(f_handle, new_group)

    def add_to_hdf5_group(self, f_handle, hdf5_grp) -> "hdf5_grp":
        """Save the entire object to an hdf5 group."""
        self.save_component(f_handle, hdf5_grp)
        self.save_data(f_handle, hdf5_grp)
        self.save_data_dict(f_handle, hdf5_grp)

    def to_file(self, fname: str) -> None:
        """Save the entire object to an hdf5 file."""
        with h5py.File(fname, "a") as f_handle:
            root_grp = f_handle.create_group("root")
            self.add_to_hdf5_group(f_handle, root_grp)

    """ Loading from an hdf5 file."""

    @classmethod
    def load_component(cls, f_handle, hdf5_grp):
        """Strip the component json and extra kwargs from an hdf5 group."""
        json_string = hdf5_grp["json"]
        json_dict = json.loads(json_string)
        component = json_dict.pop("component")
        return component, extra_kwargs

    @classmethod
    def load_data(cls, f_handle, hdf5_grp) -> xr.DataArray:
        """Construct the ``xarray.DataArray`` from an hdf5 file and group."""
        return xr.DataArray(**dict(hdf5_grp.items()))

    @classmethod
    def load_data_dict(cls, f_handle, hdf5_grp):
        """Construct the ``data_dict`` recursively from an hdf5 file and group."""
        return {name: cls.load_from_hdf5_group(f_handle, grp) for name, grp in hdf5_grp.items()}

    @classmethod
    def load_from_hdf5_group(cls, f_handle, hdf5_grp):
        """Construct the object from an hdf5 file and group."""
        component, extra_kwargs = cls.load_component(f_handle, hdf5_grp)
        data = cls.load_data(f_handle, hdf5_grp)
        data_dict = cls.load_data_dict(f_handle, hdf5_grp)
        return cls(component=component, data=data, data_dict=data_dict, **extra_kwargs)

    @classmethod
    def from_file(cls, fname) -> "Tidy3dData":
        """Load the entire object from an hdf5 file."""
        with h5py.File(fname, "r") as f_handle:
            root_grp = f_handle["root"]
            return cls.load_from_hdf5_group(f_handle, root_grp)
