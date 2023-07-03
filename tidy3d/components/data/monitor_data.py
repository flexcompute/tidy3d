# pylint:disable=too-many-lines
""" Monitor Level Data, store the DataArrays associated with a single monitor."""
from __future__ import annotations

from abc import ABC
from typing import Union, Tuple, Callable, Dict, List, Any
import warnings

import xarray as xr
import numpy as np
import pydantic as pd

from .data_array import FluxTimeDataArray, FluxDataArray
from .data_array import MixedModeDataArray, ModeIndexDataArray, ModeAmpsDataArray
from .data_array import FieldProjectionAngleDataArray, FieldProjectionCartesianDataArray
from .data_array import FieldProjectionKSpaceDataArray
from .data_array import DataArray, DiffractionDataArray
from .data_array import ScalarFieldDataArray, ScalarFieldTimeDataArray
from .data_array import FreqDataArray, TimeDataArray, FreqModeDataArray
from .dataset import Dataset, AbstractFieldDataset, ElectromagneticFieldDataset
from .dataset import FieldDataset, FieldTimeDataset, ModeSolverDataset, PermittivityDataset
from ..base import TYPE_TAG_STR, cached_property
from ..types import Coordinate, Symmetry, ArrayFloat1D, ArrayFloat2D, Size, Numpy, TrackFreq
from ..types import EpsSpecType
from ..grid.grid import Grid, Coords
from ..validators import enforce_monitor_fields_present, required_if_symmetry_present
from ..monitor import MonitorType, FieldMonitor, FieldTimeMonitor, ModeSolverMonitor
from ..monitor import ModeMonitor, FluxMonitor, FluxTimeMonitor, PermittivityMonitor
from ..monitor import FieldProjectionAngleMonitor, FieldProjectionCartesianMonitor
from ..monitor import FieldProjectionKSpaceMonitor, FieldProjectionSurface
from ..monitor import DiffractionMonitor
from ..source import SourceTimeType, CustomFieldSource
from ..medium import Medium, MediumType
from ...exceptions import SetupError, DataError, Tidy3dNotImplementedError, ValidationError
from ...constants import ETA_0, C_0, MICROMETER
from ...log import log


Coords1D = ArrayFloat1D


class MonitorData(Dataset, ABC):
    """Abstract base class of objects that store data pertaining to a single :class:`.monitor`."""

    monitor: MonitorType = pd.Field(
        ...,
        title="Monitor",
        description="Monitor associated with the data.",
        discriminator=TYPE_TAG_STR,
    )

    @property
    def symmetry_expanded_copy(self) -> MonitorData:
        """Return copy of self with symmetry applied."""
        return self.copy()

    # pylint:disable=unused-argument
    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> Dataset:
        """Return copy of self after normalization is applied using source spectrum function."""
        return self.copy()


class AbstractFieldData(MonitorData, AbstractFieldDataset, ABC):
    """Collection of scalar fields with some symmetry properties."""

    monitor: Union[FieldMonitor, FieldTimeMonitor, PermittivityMonitor, ModeSolverMonitor]

    symmetry: Tuple[Symmetry, Symmetry, Symmetry] = pd.Field(
        (0, 0, 0),
        title="Symmetry",
        description="Symmetry eigenvalues of the original simulation in x, y, and z.",
    )

    symmetry_center: Coordinate = pd.Field(
        None,
        title="Symmetry Center",
        description="Center of the symmetry planes of the original simulation in x, y, and z. "
        "Required only if any of the ``symmetry`` field are non-zero.",
    )
    grid_expanded: Grid = pd.Field(
        None,
        title="Expanded Grid",
        description=":class:`.Grid` on which the symmetry will be expanded. "
        "Required only if any of the ``symmetry`` field are non-zero.",
    )

    _require_sym_center = required_if_symmetry_present("symmetry_center")
    _require_grid_expanded = required_if_symmetry_present("grid_expanded")

    def _expanded_grid_field_coords(self, field_name: str) -> Coords:
        """Coordinates in the expanded grid corresponding to a given field component."""
        return self.grid_expanded[self.grid_locations[field_name]]

    @property
    def symmetry_expanded_copy(self) -> AbstractFieldData:
        """Create a copy of the :class:`.AbstractFieldData` with fields expanded based on symmetry.

        Returns
        -------
        :class:`AbstractFieldData`
            A data object with the symmetry expanded fields.
        """

        if all(sym == 0 for sym in self.symmetry):
            return self.copy()

        update_dict = {}
        for field_name, scalar_data in self.field_components.items():

            eigenval_fn = self.symmetry_eigenvalues[field_name]

            # get grid locations for this field component on the expanded grid
            field_coords = self._expanded_grid_field_coords(field_name)

            for sym_dim, (sym_val, sym_loc) in enumerate(zip(self.symmetry, self.symmetry_center)):

                dim_name = "xyz"[sym_dim]

                # Continue if no symmetry along this dimension
                if sym_val == 0:
                    continue

                # Get coordinates for this field component on the expanded grid
                coords = field_coords.to_list[sym_dim]

                # Get indexes of coords that lie on the left of the symmetry center
                flip_inds = np.where(coords < sym_loc)[0]

                # Get the symmetric coordinates on the right
                coords_interp = np.copy(coords)
                coords_interp[flip_inds] = 2 * sym_loc - coords[flip_inds]

                # Interpolate. There generally shouldn't be values out of bounds except potentially
                # when handling modes, in which case they should be at the boundary and close to 0.
                scalar_data = scalar_data.sel({dim_name: coords_interp}, method="nearest")
                scalar_data = scalar_data.assign_coords({dim_name: coords})

                # apply the symmetry eigenvalue (if defined) to the flipped values
                if eigenval_fn is not None:
                    sym_eigenvalue = eigenval_fn(sym_dim)
                    scalar_data = scalar_data.multiply_at(
                        value=sym_val * sym_eigenvalue, coord_name=dim_name, indices=flip_inds
                    )

            # assign the final scalar data to the update_dict
            update_dict[field_name] = scalar_data

        update_dict.update({"symmetry": (0, 0, 0), "symmetry_center": None})
        return self.copy(update=update_dict)


class ElectromagneticFieldData(AbstractFieldData, ElectromagneticFieldDataset, ABC):
    """Collection of electromagnetic fields."""

    grid_primal_correction: Union[
        float, FreqDataArray, TimeDataArray, FreqModeDataArray
    ] = pd.Field(
        1.0,
        title="Field correction factor",
        description="Correction factor that needs to be applied for data corresponding to a 2D "
        "monitor to take into account the finite grid in the normal direction in the simulation in "
        "which the data was computed. The factor is applied to fields defined on the primal grid "
        "locations along the normal direction.",
    )
    grid_dual_correction: Union[float, FreqDataArray, TimeDataArray, FreqModeDataArray] = pd.Field(
        1.0,
        title="Field correction factor",
        description="Correction factor that needs to be applied for data corresponding to a 2D "
        "monitor to take into account the finite grid in the normal direction in the simulation in "
        "which the data was computed. The factor is applied to fields defined on the dual grid "
        "locations along the normal direction.",
    )

    def _expanded_grid_field_coords(self, field_name: str):
        """Coordinates in the expanded grid corresponding to a given field component."""
        if self.monitor.colocate:
            return self.grid_expanded["centers"]
        return self.grid_expanded[self.grid_locations[field_name]]

    @property
    def _grid_correction_dict(self):
        """Return the primal and dual finite grid correction factors as a dictionary."""
        return {
            "grid_primal_correction": self.grid_primal_correction,
            "grid_dual_correction": self.grid_dual_correction,
        }

    @property
    def _tangential_dims(self) -> List[str]:
        """For a 2D monitor data, return the names of the tangential dimensions. Raise if cannot
        confirm that the associated monitor is 2D."""
        if len(self.monitor.zero_dims) != 1:
            raise DataError("Data must be 2D to get tangential dimensions.")
        tangential_dims = ["x", "y", "z"]
        tangential_dims.pop(self.monitor.zero_dims[0])

        return tangential_dims

    @property
    def _in_plane(self) -> Dict[str, DataArray]:
        """Dictionary of field components with finite grid correction factors applied and symmetry
        expanded."""
        if len(self.monitor.zero_dims) != 1:
            raise DataError("Data must be 2D to apply grid corrections.")

        normal_dim = "xyz"[self.monitor.zero_dims[0]]
        fields = {}
        for field_name, field in self.grid_corrected_copy.field_components.items():
            fields[field_name] = field.squeeze(dim=normal_dim, drop=True)
        return fields

    @property
    def _plane_grid_boundaries(self) -> Tuple[Coords1D, Coords1D]:
        """For a 2D monitor data, return the boundaries of the in-plane grid to be used to compute
        differential area and to colocate fields to grid centers if needed."""
        if np.any(np.array(self.monitor.interval_space) > 1):
            raise Tidy3dNotImplementedError(
                "Cannot determine grid boundaries corresponding to "
                "down-sampled monitor data ('interval_space' > 1 along a direction)."
            )
        dim1, dim2 = self._tangential_dims

        # In-plane grid bounds inferred from the field data
        if self.monitor.colocate:
            # Should be equivalent to sim.discretize(self.monitor, extend=True)
            # Take ``unique`` to handle 0D cases along a given dimension
            plane_bounds1 = np.unique(self.grid_expanded.boundaries.to_dict[dim1])
            plane_bounds2 = np.unique(self.grid_expanded.boundaries.to_dict[dim2])
        else:
            # Last pixel missing compared to sim.discretize(self.monitor, extend=True).
            # This is because we cannot colocate the fields to the center of that pixel.
            # However, the monitor should be completely outside of this last pixel.
            fields = self.symmetry_expanded_copy.field_components
            plane_bounds1 = np.array(fields["E" + dim2].coords[dim1])
            plane_bounds2 = np.array(fields["E" + dim1].coords[dim2])

        return plane_bounds1, plane_bounds2

    @property
    def _plane_grid_centers(self) -> Tuple[Coords1D, Coords1D]:
        """For 2D monitor data, return the centers of the in-plane grid"""
        return [(bs[1:] + bs[:-1]) / 2 for bs in self._plane_grid_boundaries]

    @property
    def _diff_area(self) -> xr.DataArray:
        """For a 2D monitor data, return the area of each cell in the plane, for use in numerical
        integrations."""
        bounds = [bs.copy() for bs in self._plane_grid_boundaries]

        # Fix the grid boundaries to match the analytic monitor boundaries.
        _, plane_inds = self.monitor.pop_axis([0, 1, 2], self.monitor.size.index(0.0))
        mnt_bounds = np.array(self.monitor.bounds)
        mnt_bounds = mnt_bounds[:, plane_inds].T

        """Truncate bounds to monitor boundaries. This implicitly makes extra pixels which may be
        present have size 0 and so won't be included in the integration. For pixels intersected
        by the monitor edge, the size is truncated to the part covered by the monitor. When using
        the differential area sizes defined in this way together with integrand values
        defined at pixel centers, the integration is equivalent to trapezoidal rule with the first
        and last values interpolated to the exact monitor start/end location, if the integrand
        is zero outside of the monitor geometry. This should usually be the case for flux and dot
        computations"""
        bounds[0][np.argwhere(bounds[0] < mnt_bounds[0, 0])] = mnt_bounds[0, 0]
        bounds[0][np.argwhere(bounds[0] > mnt_bounds[0, 1])] = mnt_bounds[0, 1]
        bounds[1][np.argwhere(bounds[1] < mnt_bounds[1, 0])] = mnt_bounds[1, 0]
        bounds[1][np.argwhere(bounds[1] > mnt_bounds[1, 1])] = mnt_bounds[1, 1]

        sizes_dim0 = bounds[0][1:] - bounds[0][:-1] if bounds[0].size > 1 else [1.0]
        sizes_dim1 = bounds[1][1:] - bounds[1][:-1] if bounds[1].size > 1 else [1.0]
        return xr.DataArray(np.outer(sizes_dim0, sizes_dim1), dims=self._tangential_dims)

    def _tangential_corrected(self, fields: Dict[str, DataArray]) -> Dict[str, DataArray]:
        """For a 2D monitor data, extract the tangential components from fields and orient them
        such that the third component would be the normal axis. This just means that the H field
        gets an extra minus sign if the normal axis is ``"y"``. Raise if any of the tangential
        field components is missing.
        """
        # Tangential field components
        tan_dims = self._tangential_dims
        components = [fname + dim for fname in "EH" for dim in tan_dims]

        normal_dim = "xyz"[self.monitor.size.index(0)]

        tan_fields = {}
        for component in components:
            if component not in fields:
                raise DataError(f"Tangential field component '{component}' missing in field data.")

            if normal_dim == "y" and component[0] == "H":
                tan_fields[component] = -fields[component]
            else:
                tan_fields[component] = fields[component]

        return tan_fields

    @property
    def _tangential_fields(self) -> Dict[str, DataArray]:
        """For a 2D monitor data, get the tangential E and H fields in the 2D plane grid.  Fields
        are oriented such that the third component would be the normal axis. This just means that
        the H field gets an extra minus sign if the normal axis is ``"y"``.

        Note
        ----
            The finite grid correction factors are applied and symmetry is expanded.
        """
        return self._tangential_corrected(self._in_plane)

    @property
    def _centered_fields(self) -> Dict[str, DataArray]:
        """For a 2D monitor data, get all E and H fields colocated to the cell centers in the 2D
        plane grid.

        Note
        ----
            The finite grid correction factors are applied and symmetry is expanded.
        """

        field_components = self._in_plane

        if self.monitor.colocate:
            return field_components

        # Interpolate field components to cell centers
        interp_dict = {"assume_sorted": True}
        for dim, cents in zip(self._tangential_dims, self._plane_grid_centers):
            if cents.size > 0:
                interp_dict[dim] = cents

        centered_fields = {key: val.interp(**interp_dict) for key, val in field_components.items()}

        return centered_fields

    @property
    def _centered_tangential_fields(self) -> Dict[str, DataArray]:
        """For a 2D monitor data, get the tangential E and H fields colocated to the cell centers in
        the 2D plane grid.  Fields are oriented such that the third component would be the normal
        axis. This just means that the H field gets an extra minus sign if the normal axis is
        ``"y"``. Raise if any of the tangential field components is missing.

        Note
        ----
            The finite grid correction factors are applied and symmetry is expanded.
        """
        return self._tangential_corrected(self._centered_fields)

    @property
    def grid_corrected_copy(self) -> ElectromagneticFieldData:
        """Return a copy of self with grid correction factors applied (if necessary) and symmetry
        expanded."""
        field_data = self.symmetry_expanded_copy
        if len(self.monitor.zero_dims) != 1:
            return field_data

        normal_dim = "xyz"[self.monitor.zero_dims[0]]
        update = {"grid_primal_correction": 1.0, "grid_dual_correction": 1.0}
        for field_name, field in field_data.field_components.items():
            eig_val = self.symmetry_eigenvalues[field_name](normal_dim)
            if eig_val < 0:
                update[field_name] = field * self.grid_dual_correction
            else:
                update[field_name] = field * self.grid_primal_correction
        return field_data.copy(update=update)

    @property
    def intensity(self) -> ScalarFieldDataArray:
        """Return the sum of the squared absolute electric field components."""
        fields = self._centered_fields
        components = ("Ex", "Ey", "Ez")
        if any(cmp not in fields for cmp in components):
            raise KeyError("Can't compute intensity, all E field components must be present.")
        return sum(fields[cmp].abs ** 2 for cmp in components)

    @property
    def poynting(self) -> ScalarFieldDataArray:
        """Time-averaged Poynting vector for frequency-domain data associated to a 2D monitor,
        projected to the direction normal to the monitor plane."""

        # Tangential fields are ordered as E1, E2, H1, H2
        tan_fields = self._centered_tangential_fields
        dim1, dim2 = self._tangential_dims

        e_x_h_star = tan_fields["E" + dim1] * tan_fields["H" + dim2].conj()
        e_x_h_star -= tan_fields["E" + dim2] * tan_fields["H" + dim1].conj()
        poynting = 0.5 * np.real(e_x_h_star)
        return poynting

    def package_flux_results(self, flux_values: xr.DataArray) -> Any:
        """How to package flux"""
        return FluxDataArray(flux_values)

    @cached_property
    def flux(self) -> FluxDataArray:
        """Flux for data corresponding to a 2D monitor."""

        # Compute flux by integrating Poynting vector in-plane
        d_area = self._diff_area
        poynting = self.poynting

        flux_values = poynting * d_area
        flux_values = flux_values.sum(dim=d_area.dims)

        return self.package_flux_results(flux_values)

    @cached_property
    def mode_area(self) -> FreqModeDataArray:
        """Effective mode area corresponding to a 2D monitor.

        Effective mode area is calculated as: (∫|E|²dA)² / (∫|E|⁴dA)
        """
        intensity = self.intensity
        # integrate over the plane
        d_area = self._diff_area
        num = (intensity * d_area).sum(dim=d_area.dims) ** 2
        den = (intensity**2 * d_area).sum(dim=d_area.dims)

        return FreqModeDataArray(num / den)

    def dot(
        self, field_data: Union[FieldData, ModeSolverData], conjugate: bool = True
    ) -> ModeAmpsDataArray:
        """Dot product (modal overlap) with another :class:`.FieldData` object. Both datasets have
        to be frequency-domain data associated with a 2D monitor. Along the tangential directions,
        the datasets have to have the same discretization. Along the normal direction, the monitor
        position may differ and is ignored. Other coordinates (``frequency``, ``mode_index``) have
        to be either identical or broadcastable. Broadcasting is also supported in the case in
        which the other ``field_data`` has a dimension of size ``1`` whose coordinate is not in the
        list of coordinates in the ``self`` dataset along the corresponding dimension. In that case,
        the coordinates of the ``self`` dataset are used in the output.

        Parameters
        ----------
        field_data : :class:`ElectromagneticFieldData`
            A data instance to compute the dot product with.
        conjugate : bool, optional
            If ``True`` (default), the dot product is defined as ``1 / 4`` times the integral of
            ``E_self* x H_other - H_self* x E_other``, where ``x`` is the cross product and ``*`` is
            complex conjugation. If ``False``, the complex conjugation is skipped.

        Note
        ----
            The dot product with and without conjugation is equivalent (up to a phase) for
            modes in lossless waveguides but differs for modes in lossy materials. In that case,
            the conjugated dot product can be interpreted as the fraction of the power of the first
            mode carried by the second, but modes are not orthogonal with respect to that product
            and the sum of carried power fractions may be different from the total flux.
            In the non-conjugated definition, modes are orthogonal, but the interpretation of the
            dot product power carried by a given mode is no longer valid.
        """

        # Tangential fields for current and other field data
        fields_self = self._centered_tangential_fields
        # pylint:disable=protected-access
        fields_other = field_data._centered_tangential_fields
        if conjugate:
            fields_self = {key: field.conj() for key, field in fields_self.items()}

        # Drop size-1 dimensions in the other data
        fields_other = {key: field.squeeze(drop=True) for key, field in fields_other.items()}

        # Cross products of fields
        dim1, dim2 = self._tangential_dims
        e_self_x_h_other = fields_self["E" + dim1] * fields_other["H" + dim2]
        e_self_x_h_other -= fields_self["E" + dim2] * fields_other["H" + dim1]
        h_self_x_e_other = fields_self["H" + dim1] * fields_other["E" + dim2]
        h_self_x_e_other -= fields_self["H" + dim2] * fields_other["E" + dim1]

        # Integrate over plane
        d_area = self._diff_area
        integrand = (e_self_x_h_other - h_self_x_e_other) * d_area
        return ModeAmpsDataArray(0.25 * integrand.sum(dim=d_area.dims))

    def _interpolated_tangential_fields(self, centers: ArrayFloat2D) -> Dict[str, DataArray]:
        """For 2D monitors, interpolate this fields to given centers in the tangential plane.

        Parameters
        ----------
        centers : ArrayFloat2D
            Interpolation centers in the monitor's tangential plane.

        Return
        ------
            Dictionary with interpolated fields.
        """
        fields = self._tangential_fields

        interp_dict = {"assume_sorted": True}
        for dim, cents in zip(self._tangential_dims, centers):
            if cents.size > 0:
                interp_dict[dim] = cents

        kwargs = {"bounds_error": False, "fill_value": 0.0}
        for component, field in fields.items():
            fields[component] = field.interp(kwargs=kwargs, **interp_dict)

        return fields

    # pylint: disable=too-many-locals
    def outer_dot(
        self, field_data: Union[FieldData, ModeSolverData], conjugate: bool = True
    ) -> MixedModeDataArray:
        """Dot product (modal overlap) with another :class:`.FieldData` object.

        The tangential fields from ``field_data`` are interpolated to this object's grid, so the
        data arrays don't need to have the same discretization.  The calculation is performed for
        all common frequencies between data arrays.  In the output, ``mode_index_0`` and
        ``mode_index_1`` are the mode indices from this object and ``field_data``, respectively, if
        they are instances of ``ModeSolverData``.

        Parameters
        ----------
        field_data : :class:`ElectromagneticFieldData`
            A data instance to compute the dot product with.
        conjugate : bool = True
            If ``True`` (default), the dot product is defined as ``1 / 4`` times the integral of
            ``E_self* x H_other - H_self* x E_other``, where ``x`` is the cross product and ``*`` is
            complex conjugation. If ``False``, the complex conjugation is skipped.

        Returns
        -------
        :class:`xarray.DataArray`
            Data array with the complex-valued modal overlaps between the two mode data.

        See also
        --------
        :member:`dot`
        """

        tan_dims = self._tangential_dims

        # pylint: disable=protected-access
        if not all(a == b for a, b in zip(tan_dims, field_data._tangential_dims)):
            raise DataError("Tangential dimentions must match between the two monitors.")

        # Tangential fields for current
        fields_self = self._centered_tangential_fields
        if conjugate:
            fields_self = {component: field.conj() for component, field in fields_self.items()}

        # Tangential fields for other data
        # pylint:disable=protected-access
        fields_other = field_data._interpolated_tangential_fields(self._plane_grid_centers)

        # Tangential field component names
        dim1, dim2 = tan_dims
        e_1 = "E" + dim1
        e_2 = "E" + dim2
        h_1 = "H" + dim1
        h_2 = "H" + dim2

        # Prepare array with proper dimensions for the dot product data
        arrays = (fields_self[e_1], fields_other[e_1])
        coords = (arrays[0].coords, arrays[1].coords)

        # Common frequencies to both data arrays
        f = np.array(sorted(set(coords[0]["f"].values).intersection(coords[1]["f"].values)))

        # Mode indices, if available
        modes_in_self = "mode_index" in coords[0]
        mode_index_0 = coords[0]["mode_index"].values if modes_in_self else np.zeros(1, dtype=int)
        modes_in_other = "mode_index" in coords[1]
        mode_index_1 = coords[1]["mode_index"].values if modes_in_other else np.zeros(1, dtype=int)

        dtype = np.promote_types(arrays[0].dtype, arrays[1].dtype)
        dot = np.empty((f.size, mode_index_0.size, mode_index_1.size), dtype=dtype)

        # Calculate overlap for each common frequency and each mode pair
        for i, freq in enumerate(f):
            indexer_self = {"f": freq}
            indexer_other = {"f": freq}
            for mi0 in mode_index_0:
                if modes_in_self:
                    indexer_self["mode_index"] = mi0
                e_self_1 = fields_self[e_1].sel(indexer_self, drop=True)
                e_self_2 = fields_self[e_2].sel(indexer_self, drop=True)
                h_self_1 = fields_self[h_1].sel(indexer_self, drop=True)
                h_self_2 = fields_self[h_2].sel(indexer_self, drop=True)

                for mi1 in mode_index_1:
                    if modes_in_other:
                        indexer_other["mode_index"] = mi1
                    e_other_1 = fields_other[e_1].sel(indexer_other, drop=True)
                    e_other_2 = fields_other[e_2].sel(indexer_other, drop=True)
                    h_other_1 = fields_other[h_1].sel(indexer_other, drop=True)
                    h_other_2 = fields_other[h_2].sel(indexer_other, drop=True)

                    # Cross products of fields
                    e_self_x_h_other = e_self_1 * h_other_2 - e_self_2 * h_other_1
                    h_self_x_e_other = h_self_1 * e_other_2 - h_self_2 * e_other_1

                    # Integrate over plane
                    d_area = self._diff_area
                    integrand = (e_self_x_h_other - h_self_x_e_other) * d_area
                    dot[i, mi0, mi1] = 0.25 * integrand.sum(dim=d_area.dims)

        coords = {"f": f, "mode_index_0": mode_index_0, "mode_index_1": mode_index_1}
        result = xr.DataArray(dot, coords=coords)

        # Remove mode index coordinate if the input did not have it
        if not modes_in_self:
            result = result.isel(mode_index_0=0, drop=True)
        if not modes_in_other:
            result = result.isel(mode_index_1=0, drop=True)

        return result

    @property
    def time_reversed_copy(self) -> FieldData:
        """Make a copy of the data with time-reversed fields."""

        # Time reversal for frequency-domain fields; overwritten in :class:`FieldTimeData`
        # and :class:`ModeSolverData`.
        new_data = {}
        for comp, field in self.field_components.items():
            if comp[0] == "H":
                new_data[comp] = -np.conj(field)
            else:
                new_data[comp] = np.conj(field)
        return self.copy(update=new_data)


class FieldData(FieldDataset, ElectromagneticFieldData):
    """Data associated with a :class:`.FieldMonitor`: scalar components of E and H fields.

    Example
    -------
    >>> from tidy3d import ScalarFieldDataArray
    >>> x = [-1,1]
    >>> y = [-2,0,2]
    >>> z = [-3,-1,1,3]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> scalar_field = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> monitor = FieldMonitor(size=(2,4,6), freqs=[2e14, 3e14], name='field', fields=['Ex', 'Hz'])
    >>> data = FieldData(monitor=monitor, Ex=scalar_field, Hz=scalar_field)
    """

    monitor: FieldMonitor = pd.Field(
        ..., title="Monitor", description="Frequency-domain field monitor associated with the data."
    )

    _contains_monitor_fields = enforce_monitor_fields_present()

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> FieldDataset:
        """Return copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        for field_name, field_data in self.field_components.items():
            src_amps = source_spectrum_fn(field_data.f)
            fields_norm[field_name] = (field_data / src_amps).astype(field_data.dtype)

        return self.copy(update=fields_norm)

    def to_source(
        self, source_time: SourceTimeType, center: Coordinate, size: Size = None, **kwargs
    ) -> CustomFieldSource:
        """Create a :class:`.CustomFieldSource` from the fields stored in the :class:`.FieldData`.

        Parameters
        ----------
        source_time: :class:`.SourceTime`
            Specification of the source time-dependence.
        center: Tuple[float, float, float]
            Source center in x, y and z.
        size: Tuple[float, float, float]
            Source size in x, y, and z. If not provided, the size of the monitor associated to the
            data is used.
        **kwargs
            Extra keyword arguments passed to :class:`.CustomFieldSource`.

        Returns
        -------
        :class:`.CustomFieldSource`
            Source injecting the fields stored in the :class:`.FieldData`, with other settings as
            provided in the input arguments.
        """

        if not size:
            size = self.monitor.size

        fields = {}
        for name, field in self.symmetry_expanded_copy.field_components.items():
            fields[name] = field.copy()
            for dim, dim_name in enumerate("xyz"):
                coords_shift = field.coords[dim_name] - self.monitor.center[dim]
                fields[name].coords[dim_name] = coords_shift

        dataset = FieldDataset(**fields)
        return CustomFieldSource(
            field_dataset=dataset, source_time=source_time, center=center, size=size, **kwargs
        )


class FieldTimeData(FieldTimeDataset, ElectromagneticFieldData):
    """Data associated with a :class:`.FieldTimeMonitor`: scalar components of E and H fields.

    Example
    -------
    >>> from tidy3d import ScalarFieldTimeDataArray
    >>> x = [-1,1]
    >>> y = [-2,0,2]
    >>> z = [-3,-1,1,3]
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(x=x, y=y, z=z, t=t)
    >>> scalar_field = ScalarFieldTimeDataArray(np.random.random((2,3,4,3)), coords=coords)
    >>> monitor = FieldTimeMonitor(size=(2,4,6), interval=100, name='field', fields=['Ex', 'Hz'])
    >>> data = FieldTimeData(monitor=monitor, Ex=scalar_field, Hz=scalar_field)
    """

    monitor: FieldTimeMonitor = pd.Field(
        ..., title="Monitor", description="Time-domain field monitor associated with the data."
    )

    _contains_monitor_fields = enforce_monitor_fields_present()

    @property
    def poynting(self) -> ScalarFieldTimeDataArray:
        """Instantaneous Poynting vector for time-domain data associated to a 2D monitor, projected
        to the direction normal to the monitor plane."""

        # Tangential fields are ordered as E1, E2, H1, H2
        tan_fields = self._centered_tangential_fields
        dim1, dim2 = self._tangential_dims
        e_x_h = tan_fields["E" + dim1] * tan_fields["H" + dim2]
        e_x_h -= tan_fields["E" + dim2] * tan_fields["H" + dim1]
        return e_x_h

    @cached_property
    def flux(self) -> FluxTimeDataArray:
        """Flux for data corresponding to a 2D monitor."""

        # Compute flux by integrating Poynting vector in-plane
        d_area = self._diff_area
        return FluxTimeDataArray((self.poynting * d_area).sum(dim=d_area.dims))

    def dot(self, field_data: ElectromagneticFieldData, conjugate: bool = True) -> xr.DataArray:
        """Inner product is not defined for time-domain data."""
        raise DataError("Inner product is not defined for time-domain data.")

    @property
    def time_reversed_copy(self) -> FieldTimeData:
        """Make a copy of the data with time-reversed fields. The sign of the magnetic fields is
        flipped, and the data is reversed along the ``t`` dimension, such that for a given field,
        ``field[t_beg + t] -> field[t_end - t]``, where ``t_beg`` and ``t_end`` are the first and
        last coordinates along the ``t`` dimension.
        """
        new_data = {}
        for comp, field in self.field_components.items():
            if comp[0] == "H":
                new_data[comp] = -field
            else:
                new_data[comp] = field
            # Reverse time coordinates
            new_data[comp] = new_data[comp].assign_coords({"t": field.t[::-1]}).sortby("t")
        return self.copy(update=new_data)


class ModeSolverData(ModeSolverDataset, ElectromagneticFieldData):
    """Data associated with a :class:`.ModeSolverMonitor`: scalar components of E and H fields.

    Example
    -------
    >>> from tidy3d import ModeSpec
    >>> from tidy3d import ScalarModeFieldDataArray, ModeIndexDataArray
    >>> x = [-1,1]
    >>> y = [0]
    >>> z = [-3,-1,1,3]
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> field_coords = dict(x=x, y=y, z=z, f=f, mode_index=mode_index)
    >>> field = ScalarModeFieldDataArray((1+1j)*np.random.random((2,1,4,2,5)), coords=field_coords)
    >>> index_coords = dict(f=f, mode_index=mode_index)
    >>> index_data = ModeIndexDataArray((1+1j) * np.random.random((2,5)), coords=index_coords)
    >>> monitor = ModeSolverMonitor(
    ...    size=(2,0,6),
    ...    freqs=[2e14, 3e14],
    ...    mode_spec=ModeSpec(num_modes=5),
    ...    name='mode_solver',
    ... )
    >>> data = ModeSolverData(
    ...     monitor=monitor,
    ...     Ex=field,
    ...     Ey=field,
    ...     Ez=field,
    ...     Hx=field,
    ...     Hy=field,
    ...     Hz=field,
    ...     n_complex=index_data
    ... )
    """

    monitor: ModeSolverMonitor = pd.Field(
        ..., title="Monitor", description="Mode solver monitor associated with the data."
    )

    eps_spec: List[EpsSpecType] = pd.Field(
        None,
        title="Permettivity Specification",
        description="Characterization of the permittivity profile on the plane where modes are "
        "computed. Possible values are 'diagonal', 'tensorial_real', 'tensorial_complex'.",
    )

    @pd.validator("eps_spec", always=True)
    def eps_spec_match_mode_spec(cls, val, values):
        """Raise validation error if frequencies in eps_spec does not match frequency list"""
        if val:
            mode_data_freqs = values["monitor"].freqs
            if len(val) != len(mode_data_freqs):
                raise ValidationError(
                    "eps_spec must be provided at the same frequencies as mode solver data."
                )
        return val

    # pylint:disable=too-many-locals
    def overlap_sort(
        self,
        track_freq: TrackFreq,
        overlap_thresh: float = 0.9,
    ) -> ModeSolverData:
        """Starting from the base frequency defined by parameter ``track_freq``, sort modes at each
        frequency according to their overlap values with the modes at the previous frequency.
        That is, it attempts to rearrange modes in such a way that a given ``mode_index``
        corresponds to physically the same mode at all frequencies. Modes with overlap values over
        ``overlap_tresh`` are considered matching and not rearranged.

        Parameters
        ----------
        track_freq : Literal["central", "lowest", "highest"]
            Parameter that specifies which frequency will serve as a starting point in
            the reordering process.
        overlap_thresh : float = 0.9
            Modal overlap threshold above which two modes are considered to be the same and are not
            rearranged. If after the sorting procedure the overlap value between two corresponding
            modes is less than this threshold, a warning about a possible discontinuity is
            displayed.
        """
        num_freqs = len(self.monitor.freqs)
        num_modes = self.monitor.mode_spec.num_modes

        if track_freq == "lowest":
            f0_ind = 0
        elif track_freq == "highest":
            f0_ind = num_freqs - 1
        elif track_freq == "central":
            f0_ind = num_freqs // 2

        # Compute sorting order and overlaps with neighboring frequencies
        sorting = -np.ones((num_freqs, num_modes), dtype=int)
        overlap = np.zeros((num_freqs, num_modes))
        phase = np.zeros((num_freqs, num_modes))
        sorting[f0_ind, :] = np.arange(num_modes)  # base frequency won't change
        overlap[f0_ind, :] = np.ones(num_modes)

        # Sort in two directions from the base frequency
        for step, last_ind in zip([-1, 1], [-1, num_freqs]):

            # Start with the base frequency
            data_template = self._isel(f=[f0_ind])

            # March to lower/higher frequencies
            for freq_id in range(f0_ind + step, last_ind, step):

                # Get next frequency to sort
                data_to_sort = self._isel(f=[freq_id])

                # Compute "sorting w.r.t. to neighbor" and overlap values
                # pylint:disable=protected-access
                sorting_one_mode, amps_one_mode = data_template._find_ordering_one_freq(
                    data_to_sort, overlap_thresh
                )

                # Transform "sorting w.r.t. neighbor" to "sorting w.r.t. to f0_ind"
                sorting[freq_id, :] = sorting_one_mode[sorting[freq_id - step, :]]
                overlap[freq_id, :] = np.abs(amps_one_mode[sorting[freq_id - step, :]])
                phase[freq_id, :] = phase[freq_id - step, :] + np.angle(
                    amps_one_mode[sorting[freq_id - step, :]]
                )

                # Check for discontinuities and show warning if any
                for mode_ind in list(np.nonzero(overlap[freq_id, :] < overlap_thresh)[0]):
                    log.warning(
                        f"WARNING: Mode '{mode_ind}' appears to undergo a discontinuous change "
                        f"between frequencies '{self.monitor.freqs[freq_id]}' "
                        f"and '{self.monitor.freqs[freq_id - step]}' "
                        f"(overlap: '{overlap[freq_id, mode_ind]:.2f}')."
                    )

                # Reassign for the next iteration
                data_template = data_to_sort

        # Rearrange modes using computed sorting values
        mode_data_sorted = self._reorder_modes(
            sorting=sorting,
            phase=phase,
            track_freq=track_freq,
        )

        return mode_data_sorted

    def _isel(self, **isel_kwargs):
        """Wraps ``xarray.DataArray.isel`` for all data fields that are defined over frequency and
        mode index. Used in ``overlap_sort`` but not officially supported since for example
        ``self.monitor.mode_spec`` and ``self.monitor.freqs`` will no longer be matching the
        newly created data."""

        update_dict = dict(self._grid_correction_dict, **self.field_components)
        update_dict = {key: field.isel(**isel_kwargs) for key, field in update_dict.items()}
        return self.copy(update=update_dict)

    def _find_ordering_one_freq(
        self,
        data_to_sort: ModeSolverData,
        overlap_thresh: float,
    ) -> Tuple[Numpy, Numpy]:
        """Find new ordering of modes in data_to_sort based on their similarity to own modes."""

        num_modes = self.n_complex.sizes["mode_index"]

        # Current pairs and their overlaps
        pairs = np.arange(num_modes)
        complex_amps = self.dot(data_to_sort).data.ravel()
        if self.monitor.direction == "-":
            complex_amps *= -1

        # Check whether modes already match
        modes_to_sort = np.where(np.abs(complex_amps) < overlap_thresh)[0]
        num_modes_to_sort = len(modes_to_sort)
        if num_modes_to_sort <= 1:
            return pairs, complex_amps

        # Compute an overlap matrix for modes chosen for sorting
        amps_reduced = np.zeros((num_modes_to_sort, num_modes_to_sort), dtype=np.complex128)

        # Extract all modes of interest from template data
        data_template_reduced = self._isel(mode_index=modes_to_sort)

        for i, mode_index in enumerate(modes_to_sort):

            # Get one mode from data_to_sort
            # pylint:disable=protected-access
            one_mode = data_to_sort._isel(mode_index=[mode_index])

            # Project to all modes of interest from data_template
            amps_reduced[:, i] = data_template_reduced.dot(one_mode).data.ravel()
            if self.monitor.direction == "-":
                amps_reduced[:, i] *= -1

        # Find the most similar modes and corresponding overlap values
        pairs_reduced, amps_reduced = self._find_closest_pairs(amps_reduced)

        # Insert new sorting and overlap values into arrays with all data
        complex_amps[modes_to_sort] = amps_reduced
        pairs[modes_to_sort] = modes_to_sort[pairs_reduced]

        return pairs, complex_amps

    @staticmethod
    def _find_closest_pairs(arr: Numpy) -> Tuple[Numpy, Numpy]:
        """Given a complex overlap matrix pair row and column entries."""

        n, k = np.shape(arr)
        if n != k:
            raise DataError("Overlap matrix must be square.")

        arr_abs = np.abs(arr)
        pairs = -np.ones(n, dtype=int)
        values = np.zeros(n, dtype=np.complex128)
        for _ in range(n):
            imax, jmax = np.unravel_index(np.argmax(arr_abs, axis=None), (n, k))
            pairs[imax] = jmax
            values[imax] = arr[imax, jmax]
            arr_abs[imax, :] = -1
            arr_abs[:, jmax] = -1

        return pairs, values

    # pylint:disable=too-many-locals
    def _reorder_modes(
        self,
        sorting: Numpy,
        phase: Numpy,
        track_freq: TrackFreq,
    ) -> ModeSolverData:
        """Rearrange modes for the i-th frequency according to sorting[i, :] and apply phase
        shifts."""

        num_freqs, _ = np.shape(sorting)

        # Create new dict with rearranged field components
        update_dict = {}
        for field_name, field in self.field_components.items():

            field_sorted = field.copy()

            # Rearrange modes
            for freq_id in range(num_freqs):
                field_sorted.data[..., freq_id, :] = field_sorted.data[
                    ..., freq_id, sorting[freq_id, :]
                ]

            # Apply phase shift
            field_sorted.data = field_sorted.data * np.exp(-1j * phase[None, None, None, :, :])

            update_dict[field_name] = field_sorted

        # Rearrange data over f and mode_index
        data_dict = dict(**self._grid_correction_dict, n_complex=self.n_complex)
        for key, data in data_dict.items():
            update_dict[key] = data.copy()
            for freq_id in range(num_freqs):
                update_dict[key].data[freq_id, :] = update_dict[key].data[
                    freq_id, sorting[freq_id, :]
                ]

        # Update mode_spec in the monitor
        mode_spec = self.monitor.mode_spec.copy(update=dict(track_freq=track_freq))
        update_dict["monitor"] = self.monitor.copy(update=dict(mode_spec=mode_spec))

        return self.copy(update=update_dict)

    def _group_index_post_process(self, frequency_step: float) -> ModeSolverData:
        """Calculate group index and remove added frequencies used only for this calculation.

        Parameters
        ----------
        frequency_step: float
            Fractional frequency step used to calculate the group index.

        Returns
        -------
        :class:`.ModeSolverData`
            Filtered data with calulated group index.
        """

        freqs = self.n_complex.coords["f"].values
        num_freqs = freqs.size
        back = slice(0, num_freqs, 3)
        center = slice(1, num_freqs, 3)
        fwd = slice(2, num_freqs, 3)

        # calculate group index
        n_center = self.n_eff.isel(f=center).values
        n_backward = self.n_eff.isel(f=back).values
        n_forward = self.n_eff.isel(f=fwd).values

        n_group_data = n_center + (n_forward - n_backward) / (2 * frequency_step)
        n_group = ModeIndexDataArray(
            n_group_data,
            coords={
                "f": list(freqs[center]),
                "mode_index": list(self.n_complex.coords["mode_index"].values),
            },
            attrs={"long name": "Group index"},
        )

        # remove data corresponding to frequencies used only for group index calculation
        update_dict = {"n_complex": self.n_complex.isel(f=center), "n_group": n_group}

        for key, field in self.field_components.items():
            update_dict[key] = field.isel(f=center)

        # pylint: disable=protected-access
        for key, data in self._grid_correction_dict.items():
            update_dict[key] = data.isel(f=center)

        if self.eps_spec:
            update_dict["eps_spec"] = self.eps_spec[center]

        update_dict["monitor"] = self.monitor.updated_copy(freqs=freqs[center])

        return self.copy(update=update_dict)

    @property
    def time_reversed_copy(self) -> FieldData:
        """Make a copy of the data with direction-reversed fields. In lossy or gyrotropic systems,
        the time-reversed fields will not be the same as the backward-propagating modes."""

        # Time reversal
        new_data = {}
        for comp, field in self.field_components.items():
            if comp[0] == "H":
                new_data[comp] = -np.conj(field)
            else:
                new_data[comp] = np.conj(field)

        # switch direction in the monitor
        mnt = self.monitor
        new_direction = "+" if mnt.direction == "-" else "-"
        new_data["monitor"] = mnt.updated_copy(direction=new_direction)
        return self.copy(update=new_data)


class PermittivityData(PermittivityDataset, AbstractFieldData):
    """Data for a :class:`.PermittivityMonitor`: diagonal components of the permittivity tensor.

    Example
    -------
    >>> from tidy3d import ScalarFieldDataArray
    >>> x = [-1,1]
    >>> y = [-2,0,2]
    >>> z = [-3,-1,1,3]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> sclr_fld = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> monitor = PermittivityMonitor(size=(2,4,6), freqs=[2e14, 3e14], name='eps')
    >>> data = PermittivityData(monitor=monitor, eps_xx=sclr_fld, eps_yy=sclr_fld, eps_zz=sclr_fld)
    """

    monitor: PermittivityMonitor = pd.Field(
        ..., title="Monitor", description="Permittivity monitor associated with the data."
    )


class ModeData(MonitorData):
    """Data associated with a :class:`.ModeMonitor`: modal amplitudes and propagation indices.

    Example
    -------
    >>> from tidy3d import ModeSpec
    >>> from tidy3d import ModeAmpsDataArray, ModeIndexDataArray
    >>> direction = ["+", "-"]
    >>> f = [1e14, 2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> index_coords = dict(f=f, mode_index=mode_index)
    >>> index_data = ModeIndexDataArray((1+1j) * np.random.random((3, 5)), coords=index_coords)
    >>> amp_coords = dict(direction=direction, f=f, mode_index=mode_index)
    >>> amp_data = ModeAmpsDataArray((1+1j) * np.random.random((2, 3, 5)), coords=amp_coords)
    >>> monitor = ModeMonitor(
    ...    size=(2,0,6),
    ...    freqs=[2e14, 3e14],
    ...    mode_spec=ModeSpec(num_modes=5),
    ...    name='mode',
    ... )
    >>> data = ModeData(monitor=monitor, amps=amp_data, n_complex=index_data)
    """

    monitor: ModeMonitor = pd.Field(
        ..., title="Monitor", description="Mode monitor associated with the data."
    )

    amps: ModeAmpsDataArray = pd.Field(
        ..., title="Amplitudes", description="Complex-valued amplitudes associated with the mode."
    )

    n_complex: ModeIndexDataArray = pd.Field(
        ...,
        title="Propagation Index",
        description="Complex-valued effective propagation constants associated with the mode.",
    )

    n_group: ModeIndexDataArray = pd.Field(
        None,
        title="Group Index",
        description="Index associated with group velocity of the mode.",
    )

    @property
    def n_eff(self):
        """Real part of the propagation index."""
        return self.n_complex.real

    @property
    def k_eff(self):
        """Imaginary part of the propagation index."""
        return self.n_complex.imag

    def normalize(self, source_spectrum_fn) -> ModeData:
        """Return copy of self after normalization is applied using source spectrum function."""
        source_freq_amps = source_spectrum_fn(self.amps.f)[None, :, None]
        new_amps = (self.amps / source_freq_amps).astype(self.amps.dtype)
        return self.copy(update=dict(amps=new_amps))


class FluxData(MonitorData):
    """Data associated with a :class:`.FluxMonitor`: flux data in the frequency-domain.

    Example
    -------
    >>> from tidy3d import FluxDataArray
    >>> f = [2e14, 3e14]
    >>> coords = dict(f=f)
    >>> flux_data = FluxDataArray(np.random.random(2), coords=coords)
    >>> monitor = FluxMonitor(size=(2,0,6), freqs=[2e14, 3e14], name='flux')
    >>> data = FluxData(monitor=monitor, flux=flux_data)
    """

    monitor: FluxMonitor = pd.Field(
        ..., title="Monitor", description="Frequency-domain flux monitor associated with the data."
    )

    flux: FluxDataArray = pd.Field(
        ..., title="Flux", description="Flux values in the frequency-domain."
    )

    def normalize(self, source_spectrum_fn) -> FluxData:
        """Return copy of self after normalization is applied using source spectrum function."""
        source_freq_amps = source_spectrum_fn(self.flux.f)
        source_power = abs(source_freq_amps) ** 2
        new_flux = (self.flux / source_power).astype(self.flux.dtype)
        return self.copy(update=dict(flux=new_flux))


class FluxTimeData(MonitorData):
    """Data associated with a :class:`.FluxTimeMonitor`: flux data in the time-domain.

    Example
    -------
    >>> from tidy3d import FluxTimeDataArray
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(t=t)
    >>> flux_data = FluxTimeDataArray(np.random.random(3), coords=coords)
    >>> monitor = FluxTimeMonitor(size=(2,0,6), interval=100, name='flux_time')
    >>> data = FluxTimeData(monitor=monitor, flux=flux_data)
    """

    monitor: FluxTimeMonitor = pd.Field(
        ..., title="Monitor", description="Time-domain flux monitor associated with the data."
    )

    flux: FluxTimeDataArray = pd.Field(
        ..., title="Flux", description="Flux values in the time-domain."
    )


ProjFieldType = Union[
    FieldProjectionAngleDataArray,
    FieldProjectionCartesianDataArray,
    FieldProjectionKSpaceDataArray,
    DiffractionDataArray,
]

ProjMonitorType = Union[
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionKSpaceMonitor,
    DiffractionMonitor,
]


class AbstractFieldProjectionData(MonitorData):
    """Collection of projected fields in spherical coordinates in the frequency domain."""

    monitor: ProjMonitorType = pd.Field(
        ...,
        title="Projection monitor",
        description="Field projection monitor.",
    )

    Er: ProjFieldType = pd.Field(
        ...,
        title="Ephi",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: ProjFieldType = pd.Field(
        ...,
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: ProjFieldType = pd.Field(
        ...,
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: ProjFieldType = pd.Field(
        ...,
        title="Hphi",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: ProjFieldType = pd.Field(
        ...,
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: ProjFieldType = pd.Field(
        ...,
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
    )

    medium: MediumType = pd.Field(
        Medium(),
        title="Background Medium",
        description="Background medium through which to project fields.",
    )

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return dict(
            Er=self.Er,
            Etheta=self.Etheta,
            Ephi=self.Ephi,
            Hr=self.Hr,
            Htheta=self.Htheta,
            Hphi=self.Hphi,
        )

    @property
    def f(self) -> np.ndarray:
        """Frequencies."""
        return np.array(self.Etheta.coords["f"])

    @property
    def coords(self) -> Dict[str, np.ndarray]:
        """Coordinates of the fields contained."""
        return self.Etheta.coords

    @property
    def coords_spherical(self) -> Dict[str, np.ndarray]:
        """Coordinates grid for the fields in the spherical system."""
        if "theta" in self.coords.keys():
            r, theta, phi = np.meshgrid(
                self.coords["r"].values,
                self.coords["theta"].values,
                self.coords["phi"].values,
                indexing="ij",
            )
        elif "z" in self.coords.keys():
            xs, ys, zs = np.meshgrid(
                self.coords["x"].values,
                self.coords["y"].values,
                self.coords["z"].values,
                indexing="ij",
            )
            r, theta, phi = self.monitor.car_2_sph(xs, ys, zs)
        else:
            uxs, uys, r = np.meshgrid(
                self.coords["ux"].values,
                self.coords["uy"].values,
                self.coords["r"].values,
                indexing="ij",
            )
            theta, phi = self.monitor.kspace_2_sph(uxs, uys, self.monitor.proj_axis)
        return {"r": r, "theta": theta, "phi": phi}

    @property
    def dims(self) -> Tuple[str, ...]:
        """Dimensions of the radiation vectors contained."""
        return self.Etheta.dims

    def make_data_array(self, data: np.ndarray) -> xr.DataArray:
        """Make an xr.DataArray with data and same coords and dims as fields of self."""
        return xr.DataArray(data=data, coords=self.coords, dims=self.dims)

    def make_dataset(self, keys: Tuple[str, ...], vals: Tuple[np.ndarray, ...]) -> xr.Dataset:
        """Make an xr.Dataset with keys and data with same coords and dims as fields."""
        data_arrays = tuple(map(self.make_data_array, vals))
        return xr.Dataset(dict(zip(keys, data_arrays)))

    def make_renormalized_data(
        self, phase: np.ndarray, proj_distance: float
    ) -> AbstractFieldProjectionData:
        """Helper to apply the re-projection phase to a copied dataset."""
        new_data = self.copy()
        for field in new_data.field_components.values():
            field.values *= phase
            if "r" in self.coords.keys():
                field["r"] = np.atleast_1d(proj_distance)
        return new_data

    def normalize(
        self, source_spectrum_fn: Callable[[float], complex]
    ) -> AbstractFieldProjectionData:
        """Return copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        for field_name, field_data in self.field_components.items():
            src_amps = source_spectrum_fn(field_data.f)
            fields_norm[field_name] = (field_data / src_amps).astype(field_data.dtype)

        return self.copy(update=fields_norm)

    @staticmethod
    def wavenumber(medium: MediumType, frequency: float) -> complex:
        """Complex valued wavenumber associated with a frequency."""
        index_n, index_k = medium.nk_model(frequency=frequency)
        return (2 * np.pi * frequency / C_0) * (index_n + 1j * index_k)

    @property
    def nk(self) -> Tuple[float, float]:
        """Returns the real and imaginary parts of the background medium's refractive index."""
        return self.medium.nk_model(frequency=self.f)

    @property
    def k(self) -> complex:
        """Returns the complex wave number associated with the background medium."""
        return self.wavenumber(medium=self.medium, frequency=self.f)

    @property
    def eta(self) -> complex:
        """Returns the complex wave impedance associated with the background medium."""
        eps_complex = self.medium.eps_model(frequency=self.f)
        return ETA_0 / np.sqrt(eps_complex)

    @staticmethod
    def propagation_phase(dist: Union[float, None], k: complex) -> complex:
        """Phase associated with propagation of a distance with a given wavenumber."""
        if dist is None:
            return 1.0
        return -1j * k * np.exp(1j * k * dist) / (4 * np.pi * dist)

    @property
    def fields_spherical(self) -> xr.Dataset:
        """Get all field components in spherical coordinates relative to the monitor's
        local origin for all projection grid points and frequencies specified in the
        :class:`AbstractFieldProjectionMonitor`.

        Returns
        -------
        ``xarray.Dataset``
            xarray dataset containing
            (``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``)
            in spherical coordinates.
        """
        return self.make_dataset(
            keys=self.field_components.keys(), vals=self.field_components.values()
        )

    @property
    def fields_cartesian(self) -> xr.Dataset:
        """Get all field components in Cartesian coordinates relative to the monitor's
        local origin for all projection grid points and frequencies specified in the
        :class:`AbstractFieldProjectionMonitor`.

        Returns
        -------
        ``xarray.Dataset``
            xarray dataset containing (``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, ``Hz``)
            in Cartesian coordinates.
        """
        # convert the field components to the Cartesian coordinate system
        coords_sph = self.coords_spherical
        e_data = self.monitor.sph_2_car_field(
            self.Er.values,
            self.Etheta.values,
            self.Ephi.values,
            coords_sph["theta"][..., None],
            coords_sph["phi"][..., None],
        )
        h_data = self.monitor.sph_2_car_field(
            self.Hr.values,
            self.Htheta.values,
            self.Hphi.values,
            coords_sph["theta"][..., None],
            coords_sph["phi"][..., None],
        )

        # package into dataset
        keys = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        field_components = np.concatenate((e_data, h_data), axis=0)
        return self.make_dataset(keys=keys, vals=field_components)

    @property
    def power(self) -> xr.DataArray:
        """Get power measured on the projection grid relative to the monitor's local origin.

        Returns
        -------
        ``xarray.DataArray``
            Power at points relative to the local origin.
        """
        power_theta = 0.5 * np.real(self.Etheta.values * np.conj(self.Hphi.values))
        power_phi = 0.5 * np.real(-self.Ephi.values * np.conj(self.Htheta.values))
        power = power_theta + power_phi

        return self.make_data_array(data=power)

    @property
    def radar_cross_section(self) -> xr.DataArray:
        """Radar cross section in units of incident power."""

        _, index_k = self.nk
        if not np.all(index_k == 0):
            raise SetupError("Can't compute RCS for a lossy background medium.")

        k = self.k[None, None, None, ...]
        eta = self.eta[None, None, None, ...]

        constant = k**2 / (8 * np.pi * eta)

        # normalize fields by the distance-based phase factor
        coords_sph = self.coords_spherical
        if coords_sph["r"] is None:
            phase = 1.0
        else:
            phase = self.propagation_phase(dist=coords_sph["r"][..., None], k=k)
        Etheta = self.Etheta.values / phase
        Ephi = self.Ephi.values / phase
        rcs_data = constant * (np.abs(Etheta) ** 2 + np.abs(Ephi) ** 2)

        return self.make_data_array(data=rcs_data)


class FieldProjectionAngleData(AbstractFieldProjectionData):
    """Data associated with a :class:`.FieldProjectionAngleMonitor`: components of projected fields.

    Example
    -------
    >>> from tidy3d import FieldProjectionAngleDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> r = np.atleast_1d(5)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> coords = dict(r=r, theta=theta, phi=phi, f=f)
    >>> values = (1+1j) * np.random.random((len(r), len(theta), len(phi), len(f)))
    >>> scalar_field = FieldProjectionAngleDataArray(values, coords=coords)
    >>> monitor = FieldProjectionAngleMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', phi=phi, theta=theta
    ...     )
    >>> data = FieldProjectionAngleData(
    ...     monitor=monitor, Er=scalar_field, Etheta=scalar_field, Ephi=scalar_field,
    ...     Hr=scalar_field, Htheta=scalar_field, Hphi=scalar_field,
    ...     projection_surfaces=monitor.projection_surfaces,
    ...     )
    """

    monitor: FieldProjectionAngleMonitor = pd.Field(
        ...,
        title="Projection monitor",
        description="Field projection monitor with an angle-based projection grid.",
    )

    projection_surfaces: Tuple[FieldProjectionSurface, ...] = pd.Field(
        ...,
        title="Projection surfaces",
        description="Surfaces of the monitor where near fields were recorded for projection",
    )

    Er: FieldProjectionAngleDataArray = pd.Field(
        ...,
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: FieldProjectionAngleDataArray = pd.Field(
        ...,
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: FieldProjectionAngleDataArray = pd.Field(
        ...,
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: FieldProjectionAngleDataArray = pd.Field(
        ...,
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: FieldProjectionAngleDataArray = pd.Field(
        ...,
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: FieldProjectionAngleDataArray = pd.Field(
        ...,
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
    )

    @property
    def r(self) -> np.ndarray:
        """Radial distance."""
        return self.Etheta.r.values

    @property
    def theta(self) -> np.ndarray:
        """Polar angles."""
        return self.Etheta.theta.values

    @property
    def phi(self) -> np.ndarray:
        """Azimuthal angles."""
        return self.Etheta.phi.values

    def renormalize_fields(self, proj_distance: float) -> FieldProjectionAngleData:
        """Return a :class:`.FieldProjectionAngleData` with fields re-normalized to a new
        projection distance, by applying a phase factor based on ``proj_distance``.

        Parameters
        ----------
        proj_distance : float = None
            (micron) new radial distance relative to the monitor's local origin.

        Returns
        -------
        :class:`.FieldProjectionAngleData`
            Copy of this :class:`.FieldProjectionAngleData` with fields re-projected
            to ``proj_distance``.
        """
        if self.monitor and not self.monitor.far_field_approx:
            raise DataError(
                "Fields projected without invoking the far field approximation "
                "cannot be re-projected to a new distance."
            )

        # the phase factor associated with the old distance must be removed
        r = self.coords_spherical["r"][..., None]
        old_phase = self.propagation_phase(dist=r, k=self.k[None, None, None, :])

        # the phase factor associated with the new distance must be applied
        new_phase = self.propagation_phase(dist=proj_distance, k=self.k)

        # net phase
        phase = new_phase[None, None, None, :] / old_phase

        # compute updated fields and their coordinates
        return self.make_renormalized_data(phase, proj_distance)


class FieldProjectionCartesianData(AbstractFieldProjectionData):
    """Data associated with a :class:`.FieldProjectionCartesianMonitor`: components of
    projected fields.

    Example
    -------
    >>> from tidy3d import FieldProjectionCartesianDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> x = np.linspace(0, 5, 10)
    >>> y = np.linspace(0, 10, 20)
    >>> z = np.atleast_1d(5)
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(z), len(f)))
    >>> scalar_field = FieldProjectionCartesianDataArray(values, coords=coords)
    >>> monitor = FieldProjectionCartesianMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', x=x, y=y,
    ...     proj_axis=2, proj_distance=50
    ...     )
    >>> data = FieldProjectionCartesianData(
    ...     monitor=monitor, Er=scalar_field, Etheta=scalar_field, Ephi=scalar_field,
    ...     Hr=scalar_field, Htheta=scalar_field, Hphi=scalar_field,
    ...     projection_surfaces=monitor.projection_surfaces,
    ...     )
    """

    monitor: FieldProjectionCartesianMonitor = pd.Field(
        ...,
        title="Projection monitor",
        description="Field projection monitor with a Cartesian projection grid.",
    )

    projection_surfaces: Tuple[FieldProjectionSurface, ...] = pd.Field(
        ...,
        title="Projection surfaces",
        description="Surfaces of the monitor where near fields were recorded for projection",
    )

    Er: FieldProjectionCartesianDataArray = pd.Field(
        ...,
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: FieldProjectionCartesianDataArray = pd.Field(
        ...,
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: FieldProjectionCartesianDataArray = pd.Field(
        ...,
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: FieldProjectionCartesianDataArray = pd.Field(
        ...,
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: FieldProjectionCartesianDataArray = pd.Field(
        ...,
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: FieldProjectionCartesianDataArray = pd.Field(
        ...,
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
    )

    @property
    def x(self) -> np.ndarray:
        """X positions."""
        return self.Etheta.x.values

    @property
    def y(self) -> np.ndarray:
        """Y positions."""
        return self.Etheta.y.values

    @property
    def z(self) -> np.ndarray:
        """Z positions."""
        return self.Etheta.z.values

    def renormalize_fields(self, proj_distance: float) -> FieldProjectionCartesianData:
        """Return a :class:`.FieldProjectionCartesianData` with fields re-normalized to a new
        projection distance, by applying a phase factor based on ``proj_distance``.

        Parameters
        ----------
        proj_distance : float = None
            (micron) new plane distance relative to the monitor's local origin.

        Returns
        -------
        :class:`.FieldProjectionCartesianData`
            Copy of this :class:`.FieldProjectionCartesianData` with fields re-projected
            to ``proj_distance``.
        """
        if not self.monitor.far_field_approx:
            raise DataError(
                "Fields projected without invoking the far field approximation "
                "cannot be re-projected to a new distance."
            )

        # the phase factor associated with the old distance must be removed
        k = self.k[None, None, None, :]
        r = self.coords_spherical["r"][..., None]
        old_phase = self.propagation_phase(dist=r, k=k)

        # update the field components' projection distance
        norm_dir, _ = self.monitor.pop_axis(["x", "y", "z"], axis=self.monitor.proj_axis)
        for field in self.field_components.values():
            field[norm_dir] = np.atleast_1d(proj_distance)

        # the phase factor associated with the new distance must be applied
        r = self.coords_spherical["r"][..., None]
        new_phase = self.propagation_phase(dist=r, k=k)

        # net phase
        phase = new_phase / old_phase

        # compute updated fields and their coordinates
        return self.make_renormalized_data(phase, proj_distance)


class FieldProjectionKSpaceData(AbstractFieldProjectionData):
    """Data associated with a :class:`.FieldProjectionKSpaceMonitor`: components of
    projected fields.

    Example
    -------
    >>> from tidy3d import FieldProjectionKSpaceDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> ux = np.linspace(0, 0.4, 10)
    >>> uy = np.linspace(0, 0.6, 20)
    >>> r = np.atleast_1d(5)
    >>> coords = dict(ux=ux, uy=uy, r=r, f=f)
    >>> values = (1+1j) * np.random.random((len(ux), len(uy), len(r), len(f)))
    >>> scalar_field = FieldProjectionKSpaceDataArray(values, coords=coords)
    >>> monitor = FieldProjectionKSpaceMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', ux=ux, uy=uy, proj_axis=2
    ...     )
    >>> data = FieldProjectionKSpaceData(
    ...     monitor=monitor, Er=scalar_field, Etheta=scalar_field, Ephi=scalar_field,
    ...     Hr=scalar_field, Htheta=scalar_field, Hphi=scalar_field,
    ...     projection_surfaces=monitor.projection_surfaces,
    ...     )
    """

    monitor: FieldProjectionKSpaceMonitor = pd.Field(
        ...,
        title="Projection monitor",
        description="Field projection monitor with a projection grid defined in k-space.",
    )

    projection_surfaces: Tuple[FieldProjectionSurface, ...] = pd.Field(
        ...,
        title="Projection surfaces",
        description="Surfaces of the monitor where near fields were recorded for projection",
    )

    Er: FieldProjectionKSpaceDataArray = pd.Field(
        ...,
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: FieldProjectionKSpaceDataArray = pd.Field(
        ...,
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: FieldProjectionKSpaceDataArray = pd.Field(
        ...,
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: FieldProjectionKSpaceDataArray = pd.Field(
        ...,
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: FieldProjectionKSpaceDataArray = pd.Field(
        ...,
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: FieldProjectionKSpaceDataArray = pd.Field(
        ...,
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
    )

    @property
    def ux(self) -> np.ndarray:
        """Reciprocal X positions."""
        return self.Etheta.ux.values

    @property
    def uy(self) -> np.ndarray:
        """Reciprocal Y positions."""
        return self.Etheta.uy.values

    @property
    def r(self) -> np.ndarray:
        """Radial distance."""
        return self.Etheta.r.values

    def renormalize_fields(self, proj_distance: float) -> FieldProjectionKSpaceData:
        """Return a :class:`.FieldProjectionKSpaceData` with fields re-normalized to a new
        projection distance, by applying a phase factor based on ``proj_distance``.

        Parameters
        ----------
        proj_distance : float = None
            (micron) new radial distance relative to the monitor's local origin.

        Returns
        -------
        :class:`.FieldProjectionKSpaceData`
            Copy of this :class:`.FieldProjectionKSpaceData` with fields re-projected
            to ``proj_distance``.
        """
        if self.monitor and not self.monitor.far_field_approx:
            raise DataError(
                "Fields projected without invoking the far field approximation "
                "cannot be re-projected to a new distance."
            )

        # the phase factor associated with the old distance must be removed
        r = self.coords_spherical["r"][..., None]
        old_phase = self.propagation_phase(dist=r, k=self.k[None, None, None, :])

        # the phase factor associated with the new distance must be applied
        new_phase = self.propagation_phase(dist=proj_distance, k=self.k)

        # net phase
        phase = new_phase[None, None, None, :] / old_phase

        # compute updated fields and their coordinates
        return self.make_renormalized_data(phase, proj_distance)


# pylint: disable=too-many-public-methods
class DiffractionData(AbstractFieldProjectionData):
    """Data for a :class:`.DiffractionMonitor`: complex components of diffracted far fields.

    Example
    -------
    >>> from tidy3d import DiffractionDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> orders_x = list(range(-4, 5))
    >>> orders_y = list(range(-6, 7))
    >>> pol = ["s", "p"]
    >>> coords = dict(orders_x=orders_x, orders_y=orders_y, f=f)
    >>> values = (1+1j) * np.random.random((len(orders_x), len(orders_y), len(f)))
    >>> field = DiffractionDataArray(values, coords=coords)
    >>> monitor = DiffractionMonitor(
    ...     center=(1,2,3), size=(np.inf,np.inf,0), freqs=f, name='diffraction'
    ... )
    >>> data = DiffractionData(
    ...     monitor=monitor, sim_size=[1,1], bloch_vecs=[1,2],
    ...     Etheta=field, Ephi=field, Er=field,
    ...     Htheta=field, Hphi=field, Hr=field,
    ... )
    """

    monitor: DiffractionMonitor = pd.Field(
        ..., title="Monitor", description="Diffraction monitor associated with the data."
    )

    Er: DiffractionDataArray = pd.Field(
        ...,
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: DiffractionDataArray = pd.Field(
        ...,
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: DiffractionDataArray = pd.Field(
        ...,
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: DiffractionDataArray = pd.Field(
        ...,
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: DiffractionDataArray = pd.Field(
        ...,
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: DiffractionDataArray = pd.Field(
        ...,
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
    )

    sim_size: Tuple[float, float] = pd.Field(
        ...,
        title="Domain size",
        description="Size of the near field in the local x and y directions.",
        units=MICROMETER,
    )

    bloch_vecs: Tuple[float, float] = pd.Field(
        ...,
        title="Bloch vectors",
        description="Bloch vectors along the local x and y directions in units of "
        "``2 * pi / (simulation size along the respective dimension)``.",
    )

    @staticmethod
    def shifted_orders(orders: Tuple[int, ...], bloch_vec: float) -> np.ndarray:
        """Diffraction orders shifted by the Bloch vector."""
        return bloch_vec + np.atleast_1d(orders)

    @staticmethod
    def reciprocal_coords(
        orders: np.ndarray, size: float, bloch_vec: float, f: float, medium: MediumType
    ) -> np.ndarray:
        """Get the normalized "u" reciprocal coords for a vector of orders, size, and bloch vec."""
        if size == 0:
            return np.atleast_2d(0)
        epsilon = medium.eps_model(f)
        bloch_array = DiffractionData.shifted_orders(orders, bloch_vec)
        return bloch_array[:, None] / size * C_0 / f / np.real(np.sqrt(epsilon))

    @staticmethod
    def compute_angles(
        reciprocal_vectors: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the polar and azimuth angles associated with the given reciprocal vectors."""
        # some wave number pairs are outside the light cone, leading to warnings from numpy.arcsin
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in arcsin", category=RuntimeWarning
            )
            ux, uy = reciprocal_vectors
            thetas, phis = DiffractionMonitor.kspace_2_sph(ux[:, None, :], uy[None, :, :], axis=2)
        return (thetas, phis)

    @property
    def coords_spherical(self) -> Dict[str, np.ndarray]:
        """Coordinates grid for the fields in the spherical system."""
        theta, phi = self.angles
        return {"r": None, "theta": theta, "phi": phi}

    @property
    def orders_x(self) -> np.ndarray:
        """Allowed orders along x."""
        return np.atleast_1d(np.array(self.Etheta.coords["orders_x"]))

    @property
    def orders_y(self) -> np.ndarray:
        """Allowed orders along y."""
        return np.atleast_1d(np.array(self.Etheta.coords["orders_y"]))

    @property
    def reciprocal_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the normalized "ux" and "uy" reciprocal vectors."""
        return (self.ux, self.uy)

    @property
    def ux(self) -> np.ndarray:
        """Normalized wave vector along x relative to ``local_origin`` and oriented
        with respect to ``monitor.normal_dir``, normalized by the wave number in the
        projection medium."""
        return self.reciprocal_coords(
            orders=self.orders_x,
            size=self.sim_size[0],
            bloch_vec=self.bloch_vecs[0],
            f=self.f,
            medium=self.medium,
        )

    @property
    def uy(self) -> np.ndarray:
        """Normalized wave vector along y relative to ``local_origin`` and oriented
        with respect to ``monitor.normal_dir``, normalized by the wave number in the
        projection medium."""
        return self.reciprocal_coords(
            orders=self.orders_y,
            size=self.sim_size[1],
            bloch_vec=self.bloch_vecs[1],
            f=self.f,
            medium=self.medium,
        )

    @property
    def angles(self) -> Tuple[xr.DataArray]:
        """The (theta, phi) angles corresponding to each allowed pair of diffraction
        orders storeds as data arrays. Disallowed angles are set to ``np.nan``.
        """
        thetas, phis = self.compute_angles(self.reciprocal_vectors)
        theta_data = xr.DataArray(thetas, coords=self.coords)
        phi_data = xr.DataArray(phis, coords=self.coords)
        return theta_data, phi_data

    @property
    def amps(self) -> xr.DataArray:
        """Complex power amplitude in each order for 's' and 'p' polarizations, normalized so that
        the power carried by the wave of that order and polarization equals ``abs(amps)^2``.
        """
        cos_theta = np.cos(np.nan_to_num(self.angles[0]))

        # will set amplitudes to 0 for glancing or negative angles
        cos_theta[cos_theta <= 0] = np.inf

        norm = 1.0 / np.sqrt(2.0 * self.eta) / np.sqrt(cos_theta)
        amp_theta = self.Etheta.values * norm
        amp_phi = self.Ephi.values * norm

        # stack the amplitudes in s- and p-components along a new polarization axis
        coords = {}
        coords["orders_x"] = np.atleast_1d(self.orders_x)
        coords["orders_y"] = np.atleast_1d(self.orders_y)
        coords["f"] = np.atleast_1d(self.f)
        coords["polarization"] = ["s", "p"]
        return xr.DataArray(np.stack([amp_phi, amp_theta], axis=3), coords=coords)

    @property
    def power(self) -> xr.DataArray:
        """Total power in each order, summed over both polarizations."""
        return (np.abs(self.amps) ** 2).sum(dim="polarization")

    @property
    def fields_spherical(self) -> xr.Dataset:
        """Get all field components in spherical coordinates relative to the monitor's
        local origin for all allowed diffraction orders and frequencies specified in the
        :class:`DiffractionMonitor`.

        Returns
        -------
        ``xarray.Dataset``
            xarray dataset containing
            (``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``)
            in spherical coordinates.
        """
        fields = [field.values for field in self.field_components.values()]
        keys = ["Er", "Etheta", "Ephi", "Hr", "Htheta", "Hphi"]
        return self._make_dataset(fields, keys)

    @property
    def fields_cartesian(self) -> xr.Dataset:
        """Get all field components in Cartesian coordinates relative to the monitor's
        local origin for all allowed diffraction orders and frequencies specified in the
        :class:`DiffractionMonitor`.

        Returns
        -------
        ``xarray.Dataset``
            xarray dataset containing (``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, ``Hz``)
            in Cartesian coordinates.
        """
        theta, phi = self.angles
        theta = theta.values
        phi = phi.values

        e_x, e_y, e_z = self.monitor.sph_2_car_field(
            0, self.Etheta.values, self.Ephi.values, theta, phi
        )
        h_x, h_y, h_z = self.monitor.sph_2_car_field(
            0, self.Htheta.values, self.Hphi.values, theta, phi
        )
        e_x, e_y, e_z, h_x, h_y, h_z = [
            np.nan_to_num(fld) for fld in [e_x, e_y, e_z, h_x, h_y, h_z]
        ]

        fields = [e_x, e_y, e_z, h_x, h_y, h_z]
        keys = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
        return self._make_dataset(fields, keys)

    def _make_dataset(self, fields: Tuple[np.ndarray, ...], keys: Tuple[str, ...]) -> xr.Dataset:
        """Make an xr.Dataset for fields with given field names."""
        data_arrays = []
        for field in fields:
            data_arrays.append(xr.DataArray(data=field, coords=self.coords, dims=self.dims))
        return xr.Dataset(dict(zip(keys, data_arrays)))


MonitorDataTypes = (
    FieldData,
    FieldTimeData,
    PermittivityData,
    ModeSolverData,
    ModeData,
    FluxData,
    FluxTimeData,
    FieldProjectionKSpaceData,
    FieldProjectionCartesianData,
    FieldProjectionAngleData,
    DiffractionData,
)

MonitorDataType = Union[MonitorDataTypes]
