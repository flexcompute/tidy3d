"""Electromagnetic monitor data arrays: flux, structure loss, mode, field projection, dipole emission,
and diffraction."""

from __future__ import annotations

import numpy as np  # noqa: F401 - used by doctest examples

from tidy3d.constants import (
    PICOSECOND_PER_NANOMETER_PER_KILOMETER,
    WATT,
)

from ._constants import DIPOLE_EMISSION_INTENSITY_UNITS
from .base import DataArray


class FluxDataArray(DataArray):
    """Flux through a surface in the frequency-domain.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> coords = dict(f=f)
    >>> fd = FluxDataArray(np.random.random(2), coords=coords)
    """

    __slots__ = ()
    _dims = ("f",)
    _data_attrs = {"units": WATT, "long_name": "flux"}


class StructureLossDataArray(DataArray):
    """Total electromagnetic power loss per structure in the frequency domain.

    ``structure_index`` indexes ``Simulation.structures``; ``-1`` is the background medium, and
    ``-2`` collects the lumped elements and automatic PEC frames.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> coords = dict(f=f, structure_index=[-1, 0, 1])
    >>> loss = StructureLossDataArray(np.random.random((2, 3)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "structure_index")
    _data_attrs = {"units": WATT, "long_name": "power loss"}


class StructureLossTimeDataArray(DataArray):
    """Instantaneous electromagnetic power loss per structure in the time domain.

    ``structure_index`` indexes ``Simulation.structures``; ``-1`` is the background medium.

    Example
    -------
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(t=t, structure_index=[-1, 0, 1])
    >>> loss = StructureLossTimeDataArray(np.random.random((3, 3)), coords=coords)
    """

    __slots__ = ()
    _dims = ("t", "structure_index")
    _data_attrs = {"units": WATT, "long_name": "power loss"}


class FluxTimeDataArray(DataArray):
    """Flux through a surface in the time-domain.

    Example
    -------
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(t=t)
    >>> data = FluxTimeDataArray(np.random.random(3), coords=coords)
    """

    __slots__ = ()
    _dims = ("t",)
    _data_attrs = {"units": WATT, "long_name": "flux"}


class ModeAmpsDataArray(DataArray):
    """Forward and backward propagating complex-valued mode amplitudes.

    Example
    -------
    >>> direction = ["+", "-"]
    >>> f = [1e14, 2e14, 3e14]
    >>> mode_index = np.arange(4)
    >>> coords = dict(direction=direction, f=f, mode_index=mode_index)
    >>> data = ModeAmpsDataArray((1+1j) * np.random.random((2, 3, 4)), coords=coords)
    """

    __slots__ = ()
    _dims = ("direction", "f", "mode_index")
    _data_attrs = {"units": "sqrt(W)", "long_name": "mode amplitudes"}


class ModeAmpsTimeDataArray(DataArray):
    """Forward and backward propagating complex-valued mode amplitudes in the time domain.

    Example
    -------
    >>> direction = ["+", "-"]
    >>> t = [0, 1e-12, 2e-12]
    >>> mode_index = np.arange(4)
    >>> coords = dict(direction=direction, t=t, mode_index=mode_index)
    >>> data = ModeAmpsTimeDataArray((1+1j) * np.random.random((2, 3, 4)), coords=coords)
    """

    __slots__ = ()
    _dims = ("direction", "t", "mode_index")
    _data_attrs = {"units": "sqrt(W)", "long_name": "mode amplitudes"}


class ModeIndexDataArray(DataArray):
    """Complex-valued effective propagation index of a mode.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(4)
    >>> coords = dict(f=f, mode_index=mode_index)
    >>> data = ModeIndexDataArray((1+1j) * np.random.random((2,4)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "mode_index")
    _data_attrs = {"long_name": "Propagation index"}


class GroupIndexDataArray(DataArray):
    """Group index of a mode.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(4)
    >>> coords = dict(f=f, mode_index=mode_index)
    >>> data = GroupIndexDataArray((1+1j) * np.random.random((2,4)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "mode_index")
    _data_attrs = {"long_name": "Group index"}


class ModeDispersionDataArray(DataArray):
    """Dispersion parameter of a mode.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(4)
    >>> coords = dict(f=f, mode_index=mode_index)
    >>> data = ModeDispersionDataArray((1+1j) * np.random.random((2,4)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "mode_index")
    _data_attrs = {
        "long_name": "Dispersion parameter",
        "units": PICOSECOND_PER_NANOMETER_PER_KILOMETER,
    }


class FieldProjectionAngleDataArray(DataArray):
    """Far fields in frequency domain as a function of angles theta and phi.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> r = np.atleast_1d(5)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> coords = dict(r=r, theta=theta, phi=phi, f=f)
    >>> values = (1+1j) * np.random.random((len(r), len(theta), len(phi), len(f)))
    >>> data = FieldProjectionAngleDataArray(values, coords=coords)
    """

    __slots__ = ()
    _dims = ("r", "theta", "phi", "f")
    _data_attrs = {"long_name": "radiation vectors"}


class FieldProjectionCartesianDataArray(DataArray):
    """Far fields in frequency domain as a function of local x and y coordinates.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> x = np.linspace(0, 5, 10)
    >>> y = np.linspace(0, 10, 20)
    >>> z = np.atleast_1d(5)
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(z), len(f)))
    >>> data = FieldProjectionCartesianDataArray(values, coords=coords)
    """

    __slots__ = ()
    _dims = ("x", "y", "z", "f")
    _data_attrs = {"long_name": "radiation vectors"}


class FieldProjectionKSpaceDataArray(DataArray):
    """Far fields in frequency domain as a function of normalized
    kx and ky vectors on the observation plane.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> r = np.atleast_1d(5)
    >>> ux = np.linspace(0, 5, 10)
    >>> uy = np.linspace(0, 10, 20)
    >>> coords = dict(ux=ux, uy=uy, r=r, f=f)
    >>> values = (1+1j) * np.random.random((len(ux), len(uy), len(r), len(f)))
    >>> data = FieldProjectionKSpaceDataArray(values, coords=coords)
    """

    __slots__ = ()
    _dims = ("ux", "uy", "r", "f")
    _data_attrs = {"long_name": "radiation vectors"}


class DipoleEmissionDataArray(DataArray):
    """Radiation intensity stored by a single dipole-emission monitor.

    Example
    -------
    >>> dipole_axis = ["x", "y", "z"]
    >>> f = np.linspace(1e14, 2e14, 3)
    >>> coords = dict(dipole_axis=dipole_axis, f=f)
    >>> values = np.random.random((len(dipole_axis), len(f)))
    >>> data = DipoleEmissionDataArray(values, coords=coords)
    """

    __slots__ = ()
    _dims = ("dipole_axis", "f")
    _data_attrs = {
        "long_name": "angular radiation intensity per dipole moment squared",
        "units": DIPOLE_EMISSION_INTENSITY_UNITS,
    }


class DipoleEmissionPositionDataArray(DataArray):
    """Position samples stored by a single dipole-emission monitor.

    Example
    -------
    >>> index = np.arange(2)
    >>> dipole_axis = ["x", "y", "z"]
    >>> f = np.linspace(1e14, 2e14, 3)
    >>> coords = dict(index=index, dipole_axis=dipole_axis, f=f)
    >>> values = np.random.random((len(index), len(dipole_axis), len(f)))
    >>> data = DipoleEmissionPositionDataArray(values, coords=coords)
    """

    __slots__ = ()
    _dims = ("index", "dipole_axis", "f")
    _data_attrs = {
        "long_name": "position-resolved angular radiation intensity per dipole moment squared",
        "units": DIPOLE_EMISSION_INTENSITY_UNITS,
    }


class DiffractionDataArray(DataArray):
    """Diffraction power amplitudes as a function of diffraction orders and frequency.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> orders_x = np.linspace(-1, 1, 3)
    >>> orders_y = np.linspace(-2, 2, 5)
    >>> coords = dict(orders_x=orders_x, orders_y=orders_y, f=f)
    >>> values = (1+1j) * np.random.random((len(orders_x), len(orders_y), len(f)))
    >>> data = DiffractionDataArray(values, coords=coords)
    """

    __slots__ = ()
    _dims = ("orders_x", "orders_y", "f")
    _data_attrs = {"long_name": "diffraction amplitude"}
