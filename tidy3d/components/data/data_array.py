"""Storing tidy3d data at it's most fundamental level as xr.DataArray objects"""
from __future__ import annotations

from typing import Dict, Tuple
from abc import ABC

import xarray as xr
import numpy as np
import h5py
import pydantic as pd

from ..base import Tidy3dBaseModel
from ...constants import HERTZ, SECOND, MICROMETER, RADIAN
from ...log import DataError, ValidationError

# maps the dimension names to their attributes
DIM_ATTRS = {
    "x": {"units": MICROMETER, "long_name": "x position"},
    "y": {"units": MICROMETER, "long_name": "y position"},
    "z": {"units": MICROMETER, "long_name": "z position"},
    "f": {"units": HERTZ, "long_name": "frequency"},
    "t": {"units": SECOND, "long_name": "time"},
    "direction": {"long_name": "propagation direction"},
    "mode_index": {"long_name": "mode index"},
    "theta": {"units": RADIAN, "long_name": "elevation angle"},
    "phi": {"units": RADIAN, "long_name": "azimuth angle"},
    "ux": {"long_name": "normalized kx"},
    "uy": {"long_name": "normalized ky"},
}


class DataArray(Tidy3dBaseModel, ABC):
    """Holds a single xr.DataArray."""

    _dims: Tuple[str, str] = ()
    _data_attrs: Dict[str, str] = {}

    data: xr.DataArray = pd.Field(
        ...,
        title="Data Array",
        description="An ``xarray.DataArray`` object storing a multi-dimensional array "
        "with labelled coordinates.",
    )

    @pd.validator("data", always=True)
    def _check_dims(cls, val):
        """Make sure class dims match the data values."""
        if cls._dims != val.dims:
            raise ValidationError(f"Supplied dims '{val.dims}' dont match hardcoded '{cls._dims}'")
        return val

    @pd.validator("data", always=True)
    def _assign_data_attrs(cls, val):
        """Assign coordinate and value attributes to the data array."""
        return val.assign_attrs(cls._data_attrs)

    @pd.validator("data", always=True)
    def _assign_coord_attrs(cls, val):
        """Assign coordinate and value attributes to the data array."""
        for coord_name, coord_val in val.coords.items():
            attrs = DIM_ATTRS.get(coord_name)
            val.coords[coord_name].attrs = attrs
        return val

    def __eq__(self, other) -> bool:
        """Whether two data array objects are equal."""
        if not np.all(self.data.data == other.data.data):
            return False
        for key, val in self.data.coords.items():
            if not np.all(np.array(val) == np.array(other.data.coords[key])):
                return False
        return True


class ScalarFieldDataArray(DataArray):
    """Spatial distribution in the frequency-domain.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> data_array = xr.DataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> data = ScalarFieldDataArray(data=data_array)
    """

    _dims = ("x", "y", "z", "f")
    _data_attrs = {"long_name": "field value"}


class ScalarFieldTimeDataArray(DataArray):
    """Spatial distribution in the time-domain.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(x=x, y=y, z=z, t=t)
    >>> data_array = xr.DataArray(np.random.random((2,3,4,3)), coords=coords)
    >>> data = ScalarFieldTimeDataArray(data=data_array)
    """

    _dims = ("x", "y", "z", "t")
    _data_attrs = {"long_name": "field value"}


class ScalarModeFieldDataArray(DataArray):
    """Spatial distribution of a mode in frequency-domain as a function of mode index.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> coords = dict(x=x, y=y, z=z, f=f, mode_index=mode_index)
    >>> data_array = xr.DataArray((1+1j) * np.random.random((2,3,4,2,5)), coords=coords)
    >>> data = ScalarModeFieldDataArray(data=data_array)
    """

    _dims = ("x", "y", "z", "f", "mode_index")
    _data_attrs = {"long_name": "field value"}


class FluxDataArray(DataArray):
    """Flux through a surface in the frequency-domain.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> coords = dict(f=f)
    >>> data_array = xr.DataArray(np.random.random(2), coords=coords)
    >>> data = FluxDataArray(data=data_array)
    """

    _dims = ("f",)
    _data_attrs = {"units": "W", "long_name": "flux"}


class FluxTimeDataArray(DataArray):
    """Flux through a surface in the time-domain.

    Example
    -------
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(t=t)
    >>> data_array = xr.DataArray(np.random.random(3), coords=coords)
    >>> data = FluxTimeDataArray(data=data_array)
    """

    _dims = ("t",)
    _data_attrs = {"units": "W", "long_name": "flux"}


class ModeAmpsDataArray(DataArray):
    """Forward and backward propagating complex-valued mode amplitudes.

    Example
    -------
    >>> direction = ["+", "-"]
    >>> f = [1e14, 2e14, 3e14]
    >>> mode_index = np.arange(4)
    >>> coords = dict(direction=direction, f=f, mode_index=mode_index)
    >>> data_array = xr.DataArray((1+1j) * np.random.random((2, 3, 4)), coords=coords)
    >>> data = ModeAmpsDataArray(data=data_array)
    """

    _dims = ("direction", "f", "mode_index")
    _data_attrs = {"units": "sqrt(W)", "long_name": "mode amplitudes"}


class ModeIndexDataArray(DataArray):
    """Complex-valued effective propagation index of a mode.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(4)
    >>> coords = dict(f=f, mode_index=mode_index)
    >>> data_array = xr.DataArray((1+1j) * np.random.random((2,4)), coords=coords)
    >>> data = ModeIndexDataArray(data=data_array)
    """

    _dims = ("f", "mode_index")
    _data_attrs = {"long_name": "Propagation index"}


class Near2FarAngleDataArray(DataArray):
    """Radiation vectors in frequency domain as a function of angles theta and phi.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> coords = dict(theta=theta, phi=phi, f=f)
    >>> values = (1+1j) * np.random.random((len(theta), len(phi), len(f)))
    >>> data_array = xr.DataArray(values, coords=coords)
    >>> data = Near2FarAngleDataArray(data=data_array)
    """

    _dims = ("theta", "phi", "f")
    _data_attrs = {"long_name": "radiation vectors"}


class Near2FarCartesianDataArray(DataArray):
    """Radiation vectors in frequency domain as a function of local x and y coordinates.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> x = np.linspace(0, 5, 10)
    >>> y = np.linspace(0, 10, 20)
    >>> coords = dict(x=x, y=y, f=f)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(f)))
    >>> data_array = xr.DataArray(values, coords=coords)
    >>> data = Near2FarCartesianDataArray(data=data_array)
    """

    _dims = ("x", "y", "f")
    _data_attrs = {"long_name": "radiation vectors"}


class Near2FarKSpaceDataArray(DataArray):
    """Radiation vector in frequency domain as a function of normalized
    kx and ky vectors on the observation plane.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> ux = np.linspace(0, 5, 10)
    >>> uy = np.linspace(0, 10, 20)
    >>> coords = dict(ux=ux, uy=uy, f=f)
    >>> values = (1+1j) * np.random.random((len(ux), len(uy), len(f)))
    >>> data_array = xr.DataArray(values, coords=coords)
    >>> data = Near2FarKSpaceDataArray(data=data_array)
    """

    _dims = ("ux", "uy", "f")
    _data_attrs = {"long_name": "radiation vectors"}
