""" Base class of all Data Objects"""

from abc import ABC, abstractmethod

import pydantic as pd
import h5py

from ..base import Tidy3dBaseModel


class Tidy3dData(Tidy3dBaseModel, ABC):
    """All data stuff inherits from this."""

    normalized: bool = pd.Field(
        None, title="Normalized", description="Whether the data object has been normalized."
    )

    def to_hdf5(self, fname):
        with open(fname) as f:
            self.to_hdf5_group(f)

    def to_hdf5_group(self, grp):
        group["json"] = self.json(exclude={"data_dict"})
        self.data_to_hdf5_group

    # @abstractmethod
    def data_to_hdf5_group(self, grp):
        """Write just the data to the hdf5 group."""

    @classmethod
    def from_hdf5(cls, fname):
        with open(fname) as f:
            cls.from_hdf5_group(f)

    @classmethod
    def from_hdf5_group(cls, grp):
        json_component = group["json"]
        json["data_dict"] = cls.data_from_hdf5_group(grp)  # note, how to do this.
        return cls.parse_raw(json)

    @classmethod
    # @abstractmethod
    def data_from_hdf5_group(self, grp):
        """Write just the data to the hdf5 group."""
