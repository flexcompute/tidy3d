"""Compatibility shim for :mod:`tidy3d._common.components.data.data_array`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as partially migrated to _common
from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

from tidy3d._common.components.data.data_array import (
    DATA_ARRAY_MAP,
    DATA_ARRAY_SCHEMA_MAP,
    DATA_ARRAY_SPEC_MAP,
    DATA_ARRAY_TYPES,
    LEGACY_SHIM_WARNINGS,
    AbstractSpatialDataArray,
    DataArray,
    DataArraySpec,
    FreqDataArray,
    ScalarFieldDataArray,
    TimeDataArray,
    TriangleMeshDataArray,
    data_array_annotated_type,
    data_array_spec_for_type,
    data_array_spec_from_name,
    data_array_type_from_name,
    install_legacy_shims,
    is_data_array_name,
    iter_data_array_names,
    td_abs,
    td_angle,
    td_does_cover,
    td_reflect,
    td_sel_inside,
    td_validate,
    td_with_updated_data,
)
from tidy3d._common.constants import (
    AMP,
    OHM,
    PICOSECOND_PER_NANOMETER_PER_KILOMETER,
    VOLT,
    WATT,
)
from tidy3d._common.exceptions import DataError, FileError

if TYPE_CHECKING:
    from xarray.core.types import Self

    from tidy3d._common.components.types.base import Axis, Bound


class FreqVoltageDataArray(DataArray):
    """Frequency-domain array.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> v = [0.1, 0.2, 0.3]
    >>> coords = dict(f=f, v=v)
    >>> fd = FreqVoltageDataArray((1+1j) * np.random.random((2, 3)), coords=coords)
    """

    __slots__ = ()
    _dims = (
        "f",
        "v",
    )


class FreqModeDataArray(DataArray):
    """Array over frequency and mode index.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> coords = dict(f=f, mode_index=mode_index)
    >>> fd = FreqModeDataArray((1+1j) * np.random.random((2, 5)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "mode_index")


class MixedModeDataArray(DataArray):
    """Scalar property associated with mode pairs

    Example
    -------
    >>> f = [1e14, 2e14, 3e14]
    >>> mode_index_0 = np.arange(4)
    >>> mode_index_1 = np.arange(2)
    >>> coords = dict(f=f, mode_index_0=mode_index_0, mode_index_1=mode_index_1)
    >>> data = MixedModeDataArray((1+1j) * np.random.random((3, 4, 2)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "mode_index_0", "mode_index_1")


class SpatialDataArray(AbstractSpatialDataArray):
    """Spatial distribution.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> coords = dict(x=x, y=y, z=z)
    >>> fd = SpatialDataArray((1+1j) * np.random.random((2,3,4)), coords=coords)
    """

    __slots__ = ()

    def reflect(self, axis: Axis, center: float, reflection_only: bool = False) -> Self:
        """Reflect data across the plane define by parameters ``axis`` and ``center`` from right to
        left. Note that the returned data is sorted with respect to spatial coordinates.

        Parameters
        ----------
        axis : Literal[0, 1, 2]
            Normal direction of the reflection plane.
        center : float
            Location of the reflection plane along its normal direction.
        reflection_only : bool = False
            Return only reflected data.

        Returns
        -------
        SpatialDataArray
            Data after reflection is performed.
        """

        sorted_self = self._spatially_sorted

        coords = [sorted_self.x.values, sorted_self.y.values, sorted_self.z.values]
        data = np.array(sorted_self.data)

        data_left_bound = coords[axis][0]

        if np.isclose(center, data_left_bound):
            num_duplicates = 1
        elif center > data_left_bound:
            raise DataError("Reflection center must be outside and to the left of the data region.")
        else:
            num_duplicates = 0

        if reflection_only:
            coords[axis] = 2 * center - coords[axis]
            coords_dict = dict(zip("xyz", coords))

            tmp_arr = SpatialDataArray(sorted_self.data, coords=coords_dict)

            return tmp_arr.sortby("xyz"[axis])

        shape = np.array(np.shape(data))
        old_len = shape[axis]
        shape[axis] = 2 * old_len - num_duplicates

        ind_left = [slice(shape[0]), slice(shape[1]), slice(shape[2])]
        ind_right = [slice(shape[0]), slice(shape[1]), slice(shape[2])]

        ind_left[axis] = slice(old_len - 1, None, -1)
        ind_right[axis] = slice(old_len - num_duplicates, None)

        new_data = np.zeros(shape)

        new_data[ind_left[0], ind_left[1], ind_left[2]] = data
        new_data[ind_right[0], ind_right[1], ind_right[2]] = data

        new_coords = np.zeros(shape[axis])
        new_coords[old_len - num_duplicates :] = coords[axis]
        new_coords[old_len - 1 :: -1] = 2 * center - coords[axis]

        coords[axis] = new_coords
        coords_dict = dict(zip("xyz", coords))

        return SpatialDataArray(new_data, coords=coords_dict)


class ScalarFieldTimeDataArray(AbstractSpatialDataArray):
    """Spatial distribution in the time-domain.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(x=x, y=y, z=z, t=t)
    >>> fd = ScalarFieldTimeDataArray(np.random.random((2,3,4,3)), coords=coords)
    """

    __slots__ = ()
    _dims = ("x", "y", "z", "t")


class ScalarModeFieldDataArray(AbstractSpatialDataArray):
    """Spatial distribution of a mode in frequency-domain as a function of mode index.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> coords = dict(x=x, y=y, z=z, f=f, mode_index=mode_index)
    >>> fd = ScalarModeFieldDataArray((1+1j) * np.random.random((2,3,4,2,5)), coords=coords)
    """

    __slots__ = ()
    _dims = ("x", "y", "z", "f", "mode_index")


class ScalarModeFieldCylindricalDataArray(AbstractSpatialDataArray):
    """Spatial distribution of a mode in frequency-domain as a function of mode index.

    Example
    -------
    >>> rho = [1,2]
    >>> theta = [2,3,4]
    >>> axial = [3,4,5,6]
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> coords = dict(rho=rho, theta=theta, axial=axial, f=f, mode_index=mode_index)
    >>> fd = ScalarModeFieldCylindricalDataArray((1+1j) * np.random.random((2,3,4,2,5)), coords=coords)
    """

    __slots__ = ()
    _dims = ("rho", "theta", "axial", "f", "mode_index")


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


class HeatDataArray(DataArray):
    """Heat data array.

    Example
    -------
    >>> T = [0, 1e-12, 2e-12]
    >>> td = HeatDataArray((1+1j) * np.random.random((3,)), coords=dict(T=T))
    """

    __slots__ = ()
    _dims = ("T",)


class EMEScalarModeFieldDataArray(AbstractSpatialDataArray):
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
    _dims = ("x", "y", "z", "f", "sweep_index", "eme_cell_index", "mode_index")


class EMEFreqModeDataArray(DataArray):
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
    _dims = ("f", "sweep_index", "eme_cell_index", "mode_index")


class EMEScalarFieldDataArray(AbstractSpatialDataArray):
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
    _dims = ("x", "y", "z", "f", "sweep_index", "eme_port_index", "mode_index")


class EMECoefficientDataArray(DataArray):
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
    _dims = (
        "f",
        "sweep_index",
        "eme_port_index",
        "eme_cell_index",
        "mode_index_out",
        "mode_index_in",
    )
    _data_attrs = {"long_name": "mode expansion coefficient"}


class EMESMatrixDataArray(DataArray):
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
    _dims = ("f", "sweep_index", "mode_index_out", "mode_index_in")
    _data_attrs = {"long_name": "scattering matrix element"}


class EMEInterfaceSMatrixDataArray(DataArray):
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
    _dims = ("f", "sweep_index", "eme_cell_index", "mode_index_out", "mode_index_in")
    _data_attrs = {"long_name": "scattering matrix element"}


class EMEModeIndexDataArray(DataArray):
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
    _dims = ("f", "sweep_index", "eme_cell_index", "mode_index")
    _data_attrs = {"long_name": "Propagation index"}


class EMEFluxDataArray(DataArray):
    """Power flux of an EME mode, also indexed by EME cell.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> sweep_index = np.arange(2)
    >>> eme_cell_index = np.arange(5)
    >>> mode_index = np.arange(4)
    >>> coords = dict(f=f, sweep_index=sweep_index, eme_cell_index=eme_cell_index, mode_index=mode_index)
    >>> data = EMEFluxDataArray(np.random.random((2,2,5,4)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "sweep_index", "eme_cell_index", "mode_index")
    _data_attrs = {"units": WATT, "long_name": "flux"}


class ChargeDataArray(DataArray):
    """Charge data array.

    Example
    -------
    >>> n = [0, 1e-12, 2e-12]
    >>> p = [0, 3e-12, 4e-12]
    >>> td = ChargeDataArray((1+1j) * np.random.random((3,3)), coords=dict(n=n, p=p))
    """

    __slots__ = ()
    _dims = ("n", "p")


class SteadyVoltageDataArray(DataArray):
    """Steady voltage data array. Data array used with steady state
    simulations with voltage as dimension.

    Example
    -------
    >>> import tidy3d as td
    >>> intensities = [0, 1, 4]
    >>> V = [-1, -0.5, 0]
    >>> voltage_dataarray = td.SteadyVoltageDataArray(data=intensities, coords={"v": V})
    """

    __slots__ = ()
    _dims = ("v",)


class PointDataArray(DataArray):
    """A two-dimensional array that stores coordinates/field components for a collection of points.
    Dimension ``index`` denotes the index of a point in the collection, and dimension ``axis``
    denotes the field component (or point coordinate) in that direction.

    Example
    -------
    >>> point_array = PointDataArray(
    ...     (1+1j) * np.random.random((5, 3)), coords=dict(index=np.arange(5), axis=np.arange(3)),
    ... )
    >>> # get coordinates of a point number 3
    >>> point3 = point_array.sel(index=3)
    >>> # get x coordinates of all points
    >>> x_coords = point_array.sel(axis=0)
    >>>
    >>> field_da = PointDataArray(
    ...     np.random.random((120, 3)), coords=dict(index=np.arange(120), axis=np.arange(3)),
    ... )
    >>> # get field of point number 90
    >>> field_point90 = field_da.sel(index=90)
    >>> # get z component of all points
    >>> z_field = field_da.sel(axis=2)
    """

    __slots__ = ()
    _dims = ("index", "axis")


class CellDataArray(DataArray):
    """A two-dimensional array that stores indices of points composing each cell in a collection of
    cells of the same type (for example: triangles, tetrahedra, etc). Dimension ``cell_index``
    denotes the index of a cell in the collection, and dimension ``vertex_index`` denotes placement
    (index) of a point in a cell (for example: 0, 1, or 2 for triangles; 0, 1, 2, or 3 for
    tetrahedra).

    Example
    -------
    >>> cell_array = CellDataArray(
    ...     (1+1j) * np.random.random((4, 3)),
    ...     coords=dict(cell_index=np.arange(4), vertex_index=np.arange(3)),
    ... )
    >>> # get indices of points composing cell number 3
    >>> cell3 = cell_array.sel(cell_index=3)
    >>> # get indices of points that represent the first vertex in each cell
    >>> first_vertices = cell_array.sel(vertex_index=0)
    """

    __slots__ = ()
    _dims = ("cell_index", "vertex_index")


class IndexedDataArray(DataArray):
    """Stores a one-dimensional array enumerated by coordinate ``index``. It is typically used
    in conjuction with a ``PointDataArray`` to store point-associated data or a ``CellDataArray``
    to store cell-associated data.

    Example
    -------
    >>> indexed_array = IndexedDataArray(
    ...     (1+1j) * np.random.random((3,)), coords=dict(index=np.arange(3))
    ... )
    """

    __slots__ = ()
    _dims = ("index",)


class IndexedVoltageDataArray(DataArray):
    """Stores a two-dimensional array with coordinates ``index`` and ``voltage``, where
    ``index`` is usually associated with ``PointDataArray`` and ``voltage`` indicates at what
    bias/DC-voltage the data was obtained with.

    Example
    -------
    >>> indexed_array = IndexedVoltageDataArray(
    ...     (1+1j) * np.random.random((3,2)), coords=dict(index=np.arange(3), voltage=[-1, 1])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "voltage")


class IndexedTimeDataArray(DataArray):
    """Stores a two-dimensional array with coordinates ``index`` and ``t``, where
    ``index`` is usually associated with ``PointDataArray`` and ``t`` indicates at what
    simulated time the data was obtained.

    Example
    -------
    >>> indexed_array = IndexedTimeDataArray(
    ...     (1+1j) * np.random.random((3,2)), coords=dict(index=np.arange(3), t=[0, 1])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "t")


class IndexedFieldVoltageDataArray(DataArray):
    """Stores indexed values of vector fields for different voltages. It is typically used
    in conjuction with a ``PointDataArray`` to store point-associated vector data.
    Example
    -------
    >>> indexed_array = IndexedFieldVoltageDataArray(
    ...     (1+1j) * np.random.random((4,3,2)), coords=dict(index=np.arange(4), axis=np.arange(3), voltage=[-1, 1])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "axis", "voltage")


class SpatialVoltageDataArray(AbstractSpatialDataArray):
    """Spatial distribution with voltage mapping.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> v = [-1, 1]
    >>> coords = dict(x=x, y=y, z=z, voltage=v)
    >>> fd = SpatialVoltageDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    """

    __slots__ = ()
    _dims = ("x", "y", "z", "voltage")


class PerturbationCoefficientDataArray(DataArray):
    __slots__ = ()
    _dims = ("wvl", "coeff")


class VoltageArray(DataArray):
    # Always set __slots__ = () to avoid xarray warnings
    __slots__ = ()
    _data_attrs = {"units": VOLT, "long_name": "voltage"}


class CurrentArray(DataArray):
    # Always set __slots__ = () to avoid xarray warnings
    __slots__ = ()
    _data_attrs = {"units": AMP, "long_name": "current"}


class ImpedanceArray(DataArray):
    # Always set __slots__ = () to avoid xarray warnings
    __slots__ = ()
    _data_attrs = {"units": OHM, "long_name": "impedance"}


# Voltage arrays
class VoltageFreqDataArray(VoltageArray, FreqDataArray):
    """Voltage data array in frequency domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9, 4e9]
    >>> coords = dict(f=f)
    >>> data = np.random.random(3) + 1j * np.random.random(3)
    >>> vfd = VoltageFreqDataArray(data, coords=coords)
    """

    __slots__ = ()


class VoltageTimeDataArray(VoltageArray, TimeDataArray):
    """Voltage data array in time domain.

    Example
    -------
    >>> import numpy as np
    >>> t = [0, 1e-9, 2e-9, 3e-9]
    >>> coords = dict(t=t)
    >>> data = np.sin(2 * np.pi * 1e9 * np.array(t))
    >>> vtd = VoltageTimeDataArray(data, coords=coords)
    """

    __slots__ = ()


class VoltageFreqModeDataArray(VoltageArray, FreqModeDataArray):
    """Voltage data array in frequency-mode domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9]
    >>> mode_index = [0, 1]
    >>> coords = dict(f=f, mode_index=mode_index)
    >>> data = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
    >>> vfmd = VoltageFreqModeDataArray(data, coords=coords)
    """

    __slots__ = ()


# Current arrays
class CurrentFreqDataArray(CurrentArray, FreqDataArray):
    """Current data array in frequency domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9, 4e9]
    >>> coords = dict(f=f)
    >>> data = np.random.random(3) + 1j * np.random.random(3)
    >>> cfd = CurrentFreqDataArray(data, coords=coords)
    """

    __slots__ = ()


class CurrentTimeDataArray(CurrentArray, TimeDataArray):
    """Current data array in time domain.

    Example
    -------
    >>> import numpy as np
    >>> t = [0, 1e-9, 2e-9, 3e-9]
    >>> coords = dict(t=t)
    >>> data = np.cos(2 * np.pi * 1e9 * np.array(t))
    >>> ctd = CurrentTimeDataArray(data, coords=coords)
    """

    __slots__ = ()


class CurrentFreqModeDataArray(CurrentArray, FreqModeDataArray):
    """Current data array in frequency-mode domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9]
    >>> mode_index = [0, 1]
    >>> coords = dict(f=f, mode_index=mode_index)
    >>> data = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
    >>> cfmd = CurrentFreqModeDataArray(data, coords=coords)
    """

    __slots__ = ()


# Impedance arrays
class ImpedanceFreqDataArray(ImpedanceArray, FreqDataArray):
    """Impedance data array in frequency domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9, 4e9]
    >>> coords = dict(f=f)
    >>> data = 50.0 + 1j * np.random.random(3)
    >>> zfd = ImpedanceFreqDataArray(data, coords=coords)
    """

    __slots__ = ()


class ImpedanceTimeDataArray(ImpedanceArray, TimeDataArray):
    """Impedance data array in time domain.

    Example
    -------
    >>> import numpy as np
    >>> t = [0, 1e-9, 2e-9, 3e-9]
    >>> coords = dict(t=t)
    >>> data = 50.0 * np.ones_like(t)
    >>> ztd = ImpedanceTimeDataArray(data, coords=coords)
    """

    __slots__ = ()


class ImpedanceFreqModeDataArray(ImpedanceArray, FreqModeDataArray):
    """Impedance data array in frequency-mode domain.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9]
    >>> mode_index = [0, 1]
    >>> coords = dict(f=f, mode_index=mode_index)
    >>> data = 50.0 + 10.0 * np.random.random((2, 2))
    >>> zfmd = ImpedanceFreqModeDataArray(data, coords=coords)
    """

    __slots__ = ()


def _make_base_result_data_array(result: DataArray) -> IntegralResultType:
    """Helper for creating the proper base result type."""
    cls = FreqDataArray
    if "t" in result.coords:
        cls = TimeDataArray
    if "f" in result.coords and "mode_index" in result.coords:
        cls = FreqModeDataArray
    return cls._assign_data_attrs(cls(data=result.data, coords=result.coords))


def _make_voltage_data_array(result: DataArray) -> VoltageIntegralResultType:
    """Helper for creating the proper voltage array type."""
    cls = VoltageFreqDataArray
    if "t" in result.coords:
        cls = VoltageTimeDataArray
    if "f" in result.coords and "mode_index" in result.coords:
        cls = VoltageFreqModeDataArray
    return cls._assign_data_attrs(cls(data=result.data, coords=result.coords))


def _make_current_data_array(result: DataArray) -> CurrentIntegralResultType:
    """Helper for creating the proper current array type."""
    cls = CurrentFreqDataArray
    if "t" in result.coords:
        cls = CurrentTimeDataArray
    if "f" in result.coords and "mode_index" in result.coords:
        cls = CurrentFreqModeDataArray
    return cls._assign_data_attrs(cls(data=result.data, coords=result.coords))


def _make_impedance_data_array(result: DataArray) -> ImpedanceResultType:
    """Helper for creating the proper impedance array type."""
    cls = ImpedanceFreqDataArray
    if "t" in result.coords:
        cls = ImpedanceTimeDataArray
    if "f" in result.coords and "mode_index" in result.coords:
        cls = ImpedanceFreqModeDataArray
    return cls._assign_data_attrs(cls(data=result.data, coords=result.coords))


IndexedDataArrayTypes = Union[
    IndexedDataArray,
    IndexedVoltageDataArray,
    IndexedTimeDataArray,
    IndexedFieldVoltageDataArray,
    PointDataArray,
]

IntegralResultType = Union[FreqDataArray, FreqModeDataArray, TimeDataArray]
VoltageIntegralResultType = Union[
    VoltageFreqDataArray, VoltageFreqModeDataArray, VoltageTimeDataArray
]
CurrentIntegralResultType = Union[
    CurrentFreqDataArray, CurrentFreqModeDataArray, CurrentTimeDataArray
]
ImpedanceResultType = Union[
    ImpedanceFreqDataArray, ImpedanceFreqModeDataArray, ImpedanceTimeDataArray
]
