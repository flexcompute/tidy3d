"""Storing tidy3d data at it's most fundamental level as xr.DataArray objects"""
from abc import ABC, abstractmethod
from typing import Dict

import xarray as xr
import numpy as np

from ...constants import HERTZ, SECOND, MICROMETER

# TODO: docstring examples?
# TODO: constrain xarray creation by specifying type of the data?
# TODO: saving and loading from hdf5 group or json file.

# maps the dimension names to their attributes
DIM_ATTRS = {
    "x": {"units": MICROMETER, "long_name": "x position"},
    "y": {"units": MICROMETER, "long_name": "y position"},
    "z": {"units": MICROMETER, "long_name": "z position"},
    "f": {"units": HERTZ, "long_name": "frequency"},
    "t": {"units": SECOND, "long_name": "time"},
    "direction": {"units": None, "long_name": "propagation direction"},
    "mode_index": {"units": None, "long_name": "mode index"},
}


class DataArray(xr.DataArray):
    """Subclass of ``xr.DataArray`` that requires __slots__ to match the keys of the coords."""

    # stores an ordered tuple of strings corresponding to the data dimensions
    __slots__ = ()

    # stores a dictionary of attributes corresponding to the data values
    _data_attrs: Dict[str, str] = {}

    def __init__(self, *args, **kwargs):

        # if dimensions are supplied, make sure they are valid
        if "dims" in kwargs:
            assert kwargs["dims"] == self.__slots__, "supplied dims different from hardcoded slots."

        # if dimensions not supplied (and not fastpath, which rejects coords), use __slots__ as dims
        if ("dims" not in kwargs) and (not kwargs.get("fastpath")):
            kwargs["dims"] = self.__slots__

        # set up coords to use attributes by supplying coords as sequence of (dim, data, attrs)
        if "coords" in kwargs:
            coords = [(dim, kwargs["coords"][dim], DIM_ATTRS.get(dim)) for dim in self.__slots__]
            kwargs["coords"] = coords

        # add class level attributes to data values, if not supplied
        if (not kwargs.get("fastpath")) and ("attrs" not in kwargs) and (self._data_attrs):
            kwargs["attrs"] = self._data_attrs

        # call xarray's initializer to make the data array
        super().__init__(*args, **kwargs)


class ScalarFieldDataArray(DataArray):
    """Spatial distribution in the frequency-domain."""

    __slots__ = ("x", "y", "z", "f")
    _data_attrs = {"units": None, "long_name": "field value"}


class ScalarFieldTimeDataArray(DataArray):
    """Spatial distribution in the time-domain."""

    __slots__ = ("x", "y", "z", "t")
    _data_attrs = {"units": None, "long_name": "field value"}


class ScalarModeFieldDataArray(DataArray):
    """Spatial distribution of a mode in frequency-domain as a function of mode index."""

    __slots__ = ("x", "y", "z", "f", "mode_index")
    _data_attrs = {"units": None, "long_name": "field value"}


class FluxDataArray(DataArray):
    """Flux through a surface in the frequency-domain."""

    __slots__ = ("f",)
    _data_attrs = {"units": "W", "long_name": "flux"}


class FluxTimeDataArray(DataArray):
    """Flux through a surface in the time-domain."""

    __slots__ = ("t",)
    _data_attrs = {"units": "W", "long_name": "flux"}


class ModeAmpsDataArray(DataArray):
    """Forward and backward propagating complex-valued mode amplitudes."""

    __slots__ = ("direction", "f", "mode_index")
    _data_attrs = {"units": "sqrt(W)", "long_name": "mode amplitudes"}


class ModeIndexDataArray(DataArray):
    """Complex-valued effective propagation index of a mode."""

    __slots__ = ("f", "mode_index")
    _data_attrs = {"units": None, "long_name": "Propagation index"}
