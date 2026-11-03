"""Eigenmode-expansion data arrays."""

from __future__ import annotations

import numpy as np  # noqa: F401 - used by doctest examples
from flexcompute.data.arrays import eme as _core

from .abstract_spatial import AbstractSpatialDataArray
from .base import DataArray


class EMEScalarModeFieldDataArray(AbstractSpatialDataArray, _core.EMEScalarModeFieldDataArray):
    """Spatial distribution of a mode in frequency-domain as a function of mode index
    and EME cell index.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3]
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> eme_cell_index = np.arange(5)
    >>> coords = dict(x=x, y=y, z=z, f=f, mode_index=mode_index, eme_cell_index=eme_cell_index)
    >>> fd = EMEScalarModeFieldDataArray((1+1j) * np.random.random((2,3,1,2,5,5)), coords=coords)
    """

    __slots__ = ()


class EMEFreqModeDataArray(DataArray, _core.EMEFreqModeDataArray):
    """Array over frequency, mode index, and EME cell index.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> eme_cell_index = np.arange(5)
    >>> coords = dict(f=f, mode_index=mode_index, eme_cell_index=eme_cell_index)
    >>> fd = EMEFreqModeDataArray((1+1j) * np.random.random((2, 5, 5)), coords=coords)
    """

    __slots__ = ()


class EMEScalarFieldDataArray(AbstractSpatialDataArray, _core.EMEScalarFieldDataArray):
    """Spatial distribution of a field excited from an EME port in frequency-domain as a
    function of mode index at the EME port and the EME port index.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> eme_port_index = [0, 1]
    >>> coords = dict(x=x, y=y, z=z, f=f, mode_index=mode_index, eme_port_index=eme_port_index)
    >>> fd = EMEScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2,5,2)), coords=coords)
    """

    __slots__ = ()


class EMECoefficientDataArray(DataArray, _core.EMECoefficientDataArray):
    """EME expansion coefficient of the mode `mode_index_out` in the EME cell
    `eme_cell_index`, when excited from mode `mode_index_in` of EME port `eme_port_index`.

    Example
    -------
    >>> mode_index_in = [0, 1]
    >>> mode_index_out = [0, 1]
    >>> eme_cell_index = np.arange(5)
    >>> eme_port_index = [0, 1]
    >>> f = [2e14]
    >>> coords = dict(
    ...     f=f,
    ...     mode_index_out=mode_index_out,
    ...     mode_index_in=mode_index_in,
    ...     eme_cell_index=eme_cell_index,
    ...     eme_port_index=eme_port_index
    ... )
    >>> fd = EMECoefficientDataArray((1 + 1j) * np.random.random((1, 2, 2, 5, 2)), coords=coords)
    """

    __slots__ = ()


class EMESMatrixDataArray(DataArray, _core.EMESMatrixDataArray):
    """Scattering matrix elements for a fixed pair of ports, possibly with an extra
    sweep index.

    Example
    -------
    >>> mode_index_in = [0, 1]
    >>> mode_index_out = [0, 1, 2]
    >>> f = [2e14]
    >>> sweep_index = np.arange(10)
    >>> coords = dict(
    ...     f=f,
    ...     mode_index_out=mode_index_out,
    ...     mode_index_in=mode_index_in,
    ...     sweep_index=sweep_index
    ... )
    >>> fd = EMESMatrixDataArray((1 + 1j) * np.random.random((1, 3, 2, 10)), coords=coords)
    """

    __slots__ = ()


class EMEInterfaceSMatrixDataArray(DataArray, _core.EMEInterfaceSMatrixDataArray):
    """Scattering matrix elements at a single cell interface for a fixed pair of ports,
    possibly with an extra sweep index.
    Example
    -------
    >>> mode_index_in = [0, 1]
    >>> mode_index_out = [0, 1, 2]
    >>> eme_cell_index = [2, 4]
    >>> f = [2e14]
    >>> sweep_index = np.arange(10)
    >>> coords = dict(
    ...     f=f,
    ...     sweep_index=sweep_index,
    ...     eme_cell_index=eme_cell_index,
    ...     mode_index_out=mode_index_out,
    ...     mode_index_in=mode_index_in,
    ... )
    >>> fd = EMEInterfaceSMatrixDataArray((1 + 1j) * np.random.random((1, 10, 2, 3, 2)), coords=coords)
    """

    __slots__ = ()


class EMEOverlapDataArray(DataArray, _core.EMEOverlapDataArray):
    """Overlap matrix between EME modes in one cell or across one cell interface."""

    __slots__ = ()
    # Intentionally no ``_legacy_dims``: released overlap blocks retain the historical
    # EMEInterfaceSMatrixDataArray type and normalize through EMEOverlapDataset. No
    # released HDF5 file dispatches a legacy block directly to this replacement class.


class EMEModeIndexDataArray(DataArray, _core.EMEModeIndexDataArray):
    """Complex-valued effective propagation index of an EME mode,
    also indexed by EME cell.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(4)
    >>> eme_cell_index = np.arange(5)
    >>> coords = dict(f=f, mode_index=mode_index, eme_cell_index=eme_cell_index)
    >>> data = EMEModeIndexDataArray((1+1j) * np.random.random((2,4,5)), coords=coords)
    """

    __slots__ = ()


class EMEFluxDataArray(DataArray, _core.EMEFluxDataArray):
    """Power flux of an EME mode, also indexed by EME cell.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> eme_cell_index = np.arange(5)
    >>> mode_index = np.arange(4)
    >>> coords = dict(f=f, eme_cell_index=eme_cell_index, mode_index=mode_index)
    >>> data = EMEFluxDataArray(np.random.random((2,5,4)), coords=coords)
    """

    __slots__ = ()


class EMETraceMetricDataArray(DataArray, _core.EMETraceMetricDataArray):
    """Field-metric matrix used by advanced EME interface diagnostics."""

    __slots__ = ()


class EMEInterfaceCellIndexDataArray(DataArray, _core.EMEInterfaceCellIndexDataArray):
    """Left or right EME cell index for each aggregated EME interface."""

    __slots__ = ()


class EMEInterfaceDiagnosticDataArray(DataArray, _core.EMEInterfaceDiagnosticDataArray):
    """Scalar EME interface diagnostic value.

    Indexed by interface, incident port, and incident mode.
    """

    __slots__ = ()
