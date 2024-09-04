"""Monitor Level Data, store the DataArrays associated with a single monitor."""

from __future__ import annotations

import warnings
from abc import ABC
from typing import Any, Callable, Dict, List, Tuple, Union

import autograd.numpy as np
import pydantic.v1 as pd
import xarray as xr
from pandas import DataFrame
from xarray.core.types import Self

from ...constants import C_0, ETA_0, MICROMETER, MU_0
from ...exceptions import DataError, SetupError, Tidy3dNotImplementedError, ValidationError
from ...log import log
from ..base import TYPE_TAG_STR, cached_property, skip_if_fields_missing
from ..base_sim.data.monitor_data import AbstractMonitorData
from ..grid.grid import Coords, Grid
from ..medium import Medium, MediumType
from ..monitor import (
    DiffractionMonitor,
    FieldMonitor,
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionKSpaceMonitor,
    FieldProjectionSurface,
    FieldTimeMonitor,
    FluxMonitor,
    FluxTimeMonitor,
    ModeMonitor,
    ModeSolverMonitor,
    MonitorType,
    PermittivityMonitor,
)
from ..source import (
    CustomCurrentSource,
    CustomFieldSource,
    GaussianPulse,
    ModeSource,
    PlaneWave,
    PointDipole,
    Source,
    SourceTimeType,
)
from ..types import (
    ArrayFloat1D,
    ArrayFloat2D,
    Coordinate,
    EpsSpecType,
    Literal,
    Numpy,
    Size,
    Symmetry,
    TrackFreq,
)
from ..validators import enforce_monitor_fields_present, required_if_symmetry_present
from .data_array import (
    DataArray,
    DiffractionDataArray,
    EMEFreqModeDataArray,
    FieldProjectionAngleDataArray,
    FieldProjectionCartesianDataArray,
    FieldProjectionKSpaceDataArray,
    FluxDataArray,
    FluxTimeDataArray,
    FreqDataArray,
    FreqModeDataArray,
    GroupIndexDataArray,
    MixedModeDataArray,
    ModeAmpsDataArray,
    ModeDispersionDataArray,
    ScalarFieldDataArray,
    ScalarFieldTimeDataArray,
    TimeDataArray,
)
from .dataset import (
    AbstractFieldDataset,
    Dataset,
    ElectromagneticFieldDataset,
    FieldDataset,
    FieldTimeDataset,
    ModeSolverDataset,
    PermittivityDataset,
)

Coords1D = ArrayFloat1D

# how much to shift the adjoint field source for 0-D axes dimensions
SHIFT_VALUE_ADJ_FLD_SRC = 1e-5


class MonitorData(AbstractMonitorData, ABC):
    """
    Abstract base class of objects that store data pertaining to a single :class:`.monitor`.
    """

    monitor: MonitorType = pd.Field(
        ...,
        title="Monitor",
        description="Monitor associated with the data.",
        discriminator=TYPE_TAG_STR,
    )

    @property
    def symmetry_expanded(self) -> MonitorData:
        """Return self with symmetry applied."""
        return self

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> Dataset:
        """Return copy of self after normalization is applied using source spectrum function."""
        return self.copy()

    def _updated(self, update: Dict) -> MonitorData:
        """Similar to ``updated_copy``, but does not actually copy components, for speed.

        Note
        ----
            This does **not** produce a copy of mutable objects, so e.g. if some of the data arrays
            are not updated, they will point to the values in the original data. This method should
            thus be used carefully.

        """
        data_dict = self.dict()
        data_dict.update(update)
        return type(self).parse_obj(data_dict)

    def make_adjoint_sources(self, dataset_names: list[str], fwidth: float) -> list[Source]:
        """Generate adjoint sources for this ``MonitorData`` instance."""

        # TODO: if there's data in the MonitorData, but no adjoint source, then
        # user is trying to differentiate something that is un-supported by us
        # warn?

        return []

    @staticmethod
    def flip_direction(direction: Union[str, DataArray]) -> str:
        """Flip the direction of a string ``('+', '-') -> ('-', '+')``."""

        if isinstance(direction, DataArray):
            direction = str(direction.values)

        if direction not in ("+", "-"):
            raise ValueError(f"Direction must be in {('+', '-')}, got '{direction}'.")

        return "-" if direction == "+" else "+"

    @staticmethod
    def get_amplitude(x) -> complex:
        """Get the complex amplitude out of some data."""

        if isinstance(x, DataArray):
            x = complex(x.values)

        return 1j * complex(x)


class AbstractFieldData(MonitorData, AbstractFieldDataset, ABC):
    """Collection of scalar fields with some symmetry properties."""

    monitor: Union[FieldMonitor, FieldTimeMonitor, PermittivityMonitor, ModeMonitor]

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
        description=":class:`.Grid` discretization of the associated monitor in the simulation "
        "which created the data. Required if symmetries are present, as "
        "well as in order to use some functionalities like getting Poynting vector and flux.",
    )

    @pd.validator("grid_expanded", always=True)
    def warn_missing_grid_expanded(cls, val, values):
        """If ``grid_expanded`` not provided and fields data is present, warn that some methods
        will break."""
        field_comps = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
        if val is None and any(values.get(comp) is not None for comp in field_comps):
            log.warning(
                "Monitor data requires 'grid_expanded' to be defined to compute values like "
                "flux, Poynting and dot product with other data."
            )
        return val

    _require_sym_center = required_if_symmetry_present("symmetry_center")
    _require_grid_expanded = required_if_symmetry_present("grid_expanded")

    def _expanded_grid_field_coords(self, field_name: str) -> Coords:
        """Coordinates in the expanded grid corresponding to a given field component."""
        return self.grid_expanded[self.grid_locations[field_name]]

    @property
    def symmetry_expanded(self):
        """Return the :class:`.AbstractFieldData` with fields expanded based on symmetry. If
        any symmetry is nonzero (i.e. expanded), the interpolation implicitly creates a copy of the
        data array. However, if symmetry is not expanded, the returned array contains a view of
        the data, not a copy.

        Returns
        -------
        :class:`AbstractFieldData`
            A data object with the symmetry expanded fields.
        """

        if all(sym == 0 for sym in self.symmetry):
            return self

        return self._updated(self._symmetry_update_dict)

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

        return self.copy(update=self._symmetry_update_dict)

    @property
    def _symmetry_update_dict(self) -> Dict:
        """Dictionary of data fields to create data with expanded symmetry."""

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
                coords = self.monitor.downsample(coords, axis=sym_dim)

                # Get indexes of coords that lie on the left of the symmetry center
                flip_inds = np.where(coords < sym_loc)[0]

                # Get the symmetric coordinates on the right
                coords_interp = np.copy(coords)
                coords_interp[flip_inds] = 2 * sym_loc - coords[flip_inds]

                # Interpolate. There generally shouldn't be values out of bounds except potentially
                # when handling modes, in which case they should be at the boundary and close to 0.

                scalar_data = scalar_data.sel(**{dim_name: coords_interp}, method="nearest")
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

        return update_dict

    def at_coords(self, coords: Coords) -> xr.Dataset:
        """Colocate data to some supplied coordinates. This is a convenience method that wraps
        ``colocate``, and skips dimensions for which the data has a single data point only
        (``colocate`` will error in that case.) If the coords are out of bounds for the data
        otherwise, an error will still be produced.

        Parameters
        ----------
        coords : :class:`Coords`
            Coordinates in x, y and z to colocate to.

        Returns
        -------
        xarray.Dataset
            Dataset containing all of the fields in the data interpolated to boundary locations on
            the Yee grid.
        """

        # pass coords if each of the scalar field data have more than one coordinate along a dim
        xyz_kwargs = {}
        for dim, coords_dim in zip("xyz", (coords.x, coords.y, coords.z)):
            scalar_data = list(self.field_components.values())
            coord_lens = [len(data.coords[dim]) for data in scalar_data]
            if all(ncoords > 1 for ncoords in coord_lens):
                xyz_kwargs[dim] = coords_dim

        return self.colocate(**xyz_kwargs)


class ElectromagneticFieldData(AbstractFieldData, ElectromagneticFieldDataset, ABC):
    """Collection of electromagnetic fields."""

    grid_primal_correction: Union[
        float,
        FreqDataArray,
        TimeDataArray,
        FreqModeDataArray,
        EMEFreqModeDataArray,
    ] = pd.Field(
        1.0,
        title="Field correction factor",
        description="Correction factor that needs to be applied for data corresponding to a 2D "
        "monitor to take into account the finite grid in the normal direction in the simulation in "
        "which the data was computed. The factor is applied to fields defined on the primal grid "
        "locations along the normal direction.",
    )
    grid_dual_correction: Union[
        float,
        FreqDataArray,
        TimeDataArray,
        FreqModeDataArray,
        EMEFreqModeDataArray,
    ] = pd.Field(
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
            bounds_dict = self.grid_expanded.boundaries.to_dict
            return Coords(**{key: val[:-1] for key, val in bounds_dict.items()})
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
    def colocation_boundaries(self) -> Coords:
        """Coordinates to be used for colocation of the data to grid boundaries."""

        if not self.grid_expanded:
            raise DataError(
                "Monitor data requires 'grid_expanded' to be defined in order to "
                "compute colocation coordinates."
            )

        # Get boundaries from the expanded grid
        grid_bounds = self.grid_expanded.boundaries.to_dict

        # Non-colocating monitors can only colocate starting from the first boundary
        # (unless there's a single data point, in which case data has already been snapped).
        # Regardless of colocation, we also drop the last boundary.
        colocate_bounds = {}
        for dim, bounds in grid_bounds.items():
            cbs = bounds[:-1]
            if not self.monitor.colocate and cbs.size > 1:
                cbs = cbs[1:]
            colocate_bounds[dim] = cbs

        return Coords(**colocate_bounds)

    @property
    def colocation_centers(self) -> Coords:
        """Coordinates to be used for colocation of the data to grid centers."""
        colocate_centers = {}
        for dim, coords in self.colocation_boundaries.to_dict.items():
            colocate_centers[dim] = (coords[1:] + coords[:-1]) / 2

        return Coords(**colocate_centers)

    @property
    def _plane_grid_boundaries(self) -> Tuple[Coords1D, Coords1D]:
        """For a 2D monitor data, return the boundaries of the in-plane grid to be used to compute
        differential area and to colocate fields if needed."""
        if np.any(np.array(self.monitor.interval_space) > 1):
            raise Tidy3dNotImplementedError(
                "Cannot determine grid boundaries corresponding to "
                "down-sampled monitor data ('interval_space' > 1 along a direction)."
            )
        dim1, dim2 = self._tangential_dims
        bounds_dict = self.colocation_boundaries.to_dict
        return (bounds_dict[dim1], bounds_dict[dim2])

    @property
    def _plane_grid_centers(self) -> Tuple[Coords1D, Coords1D]:
        """For 2D monitor data, return the centers of the in-plane grid"""
        return [(bs[1:] + bs[:-1]) / 2 for bs in self._plane_grid_boundaries]

    @property
    def _diff_area(self) -> DataArray:
        """For a 2D monitor data, return the area of each cell in the plane, for use in numerical
        integrations. This assumes that data is colocated to grid boundaries, and uses the
        difference in the surrounding grid centers to compute the area.
        """

        # Monitor values are interpolated to bounds
        bounds = self._plane_grid_boundaries
        # Coords to compute cell sizes around the interpolation locations
        coords = [bs.copy() for bs in self._plane_grid_centers]

        # Append the first and last boundary
        _, plane_inds = self.monitor.pop_axis([0, 1, 2], self.monitor.size.index(0.0))
        coords[0] = np.array([bounds[0][0]] + coords[0].tolist() + [bounds[0][-1]])
        coords[1] = np.array([bounds[1][0]] + coords[1].tolist() + [bounds[1][-1]])

        """Truncate coords to monitor boundaries. This implicitly makes extra pixels which may be
        present have size 0 and so won't be included in the integration. For pixels intersected
        by the monitor edge, the size is truncated to the part covered by the monitor. When using
        the differential area sizes defined in this way together with integrand values
        defined at cell boundaries, the integration is equivalent to trapezoidal rule with the first
        and last values interpolated to the exact monitor start/end location, if the integrand
        is zero outside of the monitor geometry. This should usually be the case for flux and dot
        computations"""
        mnt_bounds = np.array(self.monitor.bounds)
        mnt_bounds = mnt_bounds[:, plane_inds].T
        coords[0][np.argwhere(coords[0] < mnt_bounds[0, 0])] = mnt_bounds[0, 0]
        coords[0][np.argwhere(coords[0] > mnt_bounds[0, 1])] = mnt_bounds[0, 1]
        coords[1][np.argwhere(coords[1] < mnt_bounds[1, 0])] = mnt_bounds[1, 0]
        coords[1][np.argwhere(coords[1] > mnt_bounds[1, 1])] = mnt_bounds[1, 1]

        # Do not apply the spurious dl along a dimension where the simulation is 2D.
        # Instead, we just set the boundaries such that the cell size along the zero dimension is 1,
        # such that quantities like flux will come out in units of W / um.
        sizes_dim0 = coords[0][1:] - coords[0][:-1] if bounds[0].size > 1 else [1.0]
        sizes_dim1 = coords[1][1:] - coords[1][:-1] if bounds[1].size > 1 else [1.0]

        return DataArray(np.outer(sizes_dim0, sizes_dim1), dims=self._tangential_dims)

    def _tangential_corrected(self, fields: Dict[str, DataArray]) -> Dict[str, DataArray]:
        """For a 2D monitor data, extract the tangential components from fields and orient them
        such that the third component would be the normal axis. This just means that the H field
        gets an extra minus sign if the normal axis is ``"y"``. Raise if any of the tangential
        field components is missing.

        The finite grid correction is also applied, so the intended use of these fields is in
        poynting, flux, and dot-like methods. The normal coordinate is dropped from the field data.
        """

        if len(self.monitor.zero_dims) != 1:
            raise DataError("Data must be 2D to get tangential fields.")

        # Tangential field components
        tan_dims = self._tangential_dims
        components = [fname + dim for fname in "EH" for dim in tan_dims]

        normal_dim = "xyz"[self.monitor.size.index(0)]

        tan_fields = {}
        for component in components:
            if component not in fields:
                raise DataError(f"Tangential field component '{component}' missing in field data.")

            correction = 1

            # sign correction to H
            if normal_dim == "y" and component[0] == "H":
                correction *= -1

            # finite grid correction to all fields
            eig_val = self.symmetry_eigenvalues[component](normal_dim)
            if eig_val < 0:
                correction *= self.grid_dual_correction
            else:
                correction *= self.grid_primal_correction

            field_squeezed = fields[component].squeeze(dim=normal_dim, drop=True)
            tan_fields[component] = field_squeezed * correction

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
        return self._tangential_corrected(self.symmetry_expanded.field_components)

    @property
    def _colocated_fields(self) -> Dict[str, DataArray]:
        """For a 2D monitor data, get all E and H fields colocated to the cell boundaries in the 2D
        plane grid, with symmetries expanded.
        """

        field_components = self.symmetry_expanded.field_components

        if self.monitor.colocate:
            return field_components

        # Interpolate field components to cell boundaries
        interp_dict = {"assume_sorted": True}
        for dim, bounds in zip(self._tangential_dims, self._plane_grid_boundaries):
            if bounds.size > 1:
                interp_dict[dim] = bounds

        colocated_fields = {key: val.interp(**interp_dict) for key, val in field_components.items()}
        return colocated_fields

    @property
    def _colocated_tangential_fields(self) -> Dict[str, DataArray]:
        """For a 2D monitor data, get the tangential E and H fields colocated to the cell boundaries
        in the 2D plane grid.  Fields are oriented such that the third component would be the normal
        axis. This just means that the H field gets an extra minus sign if the normal axis is
        ``"y"``. Raise if any of the tangential field components is missing.

        Note
        ----
            The finite grid correction factors are applied and symmetry is expanded.
        """
        return self._tangential_corrected(self._colocated_fields)

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
        normal_dim = "xyz"[self.monitor.size.index(0)]
        fields = self._colocated_fields
        components = ("Ex", "Ey", "Ez")
        if any(cmp not in fields for cmp in components):
            raise KeyError("Can't compute intensity, all E field components must be present.")
        intensity = sum(fields[cmp].abs ** 2 for cmp in components)
        return intensity.squeeze(dim=normal_dim, drop=True)

    @property
    def poynting(self) -> ScalarFieldDataArray:
        """Time-averaged Poynting vector for frequency-domain data associated to a 2D monitor,
        projected to the direction normal to the monitor plane."""

        # Tangential fields are ordered as E1, E2, H1, H2
        tan_fields = self._colocated_tangential_fields
        dim1, dim2 = self._tangential_dims

        e1 = tan_fields["E" + dim1]
        e2 = tan_fields["E" + dim2]
        h1 = tan_fields["H" + dim1]
        h2 = tan_fields["H" + dim2]

        e1_h2 = e1 * h2.conj()
        e2_h1 = e2 * h1.conj()

        e_x_h_star = e1_h2 - e2_h1
        poynting = 0.5 * np.real(e_x_h_star)

        return poynting

    def package_flux_results(self, flux_values: DataArray) -> Any:
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
        r"""Effective mode area corresponding to a 2D monitor.

        .. math:

           \frac{\left(\int |E|^2 \, {\rm d}S\right)^2}{\int |E|^4 \, {\rm d}S}
        """
        intensity = self.intensity
        # integrate over the plane
        d_area = self._diff_area
        num = (intensity * d_area).sum(dim=d_area.dims) ** 2
        den = (intensity**2 * d_area).sum(dim=d_area.dims)

        area = num / den
        if hasattr(self.monitor, "mode_spec"):
            area *= np.cos(self.monitor.mode_spec.angle_theta)

        return FreqModeDataArray(area)

    def dot(
        self, field_data: Union[FieldData, ModeData, ModeSolverData], conjugate: bool = True
    ) -> ModeAmpsDataArray:
        r"""Dot product (modal overlap) with another :class:`.FieldData` object. Both datasets have
        to be frequency-domain data associated with a 2D monitor. Along the tangential directions,
        the datasets have to have the same discretization. Along the normal direction, the monitor
        position may differ and is ignored. Other coordinates (``frequency``, ``mode_index``) have
        to be either identical or broadcastable. Broadcasting is also supported in the case in
        which the other ``field_data`` has a dimension of size 1 whose coordinate is not in the list
        of coordinates in the ``self`` dataset along the corresponding dimension. In that case, the
        coordinates of the ``self`` dataset are used in the output.

        The dot product is defined as:

        .. math:

           \frac{1}{4} \int \left( E_0 \times H_1^* + H_0^* \times E_1 \) \, {\rm d}S

        Parameters
        ----------
        field_data : :class:`ElectromagneticFieldData`
            A data instance to compute the dot product with.
        conjugate : bool, optional
            If ``True`` (default), the dot product is defined as above. If ``False``, the definition
            is similar, but without the complex conjugation of the $H$ fields.

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
        fields_self = self._colocated_tangential_fields

        fields_other = field_data._colocated_tangential_fields
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

    def _interpolated_tangential_fields(self, coords: ArrayFloat2D) -> Dict[str, DataArray]:
        """For 2D monitors, interpolate this fields to given coords in the tangential plane.

        Parameters
        ----------
        coords : ArrayFloat2D
            Interpolation coords in the monitor's tangential plane.

        Return
        ------
            Dictionary with interpolated fields.
        """
        fields = self._tangential_fields

        interp_dict = {"assume_sorted": True}
        for dim, cents in zip(self._tangential_dims, coords):
            if cents.size > 0:
                interp_dict[dim] = cents

        kwargs = {"bounds_error": False, "fill_value": 0.0}
        for component, field in fields.items():
            fields[component] = field.interp(kwargs=kwargs, **interp_dict)

        return fields

    def outer_dot(
        self, field_data: Union[FieldData, ModeData], conjugate: bool = True
    ) -> MixedModeDataArray:
        r"""Dot product (modal overlap) with another :class:`.FieldData` object.

        The tangential fields from ``field_data`` are interpolated to this object's grid, so the
        data arrays don't need to have the same discretization.  The calculation is performed for
        all common frequencies between data arrays.  In the output, ``mode_index_0`` and
        ``mode_index_1`` are the mode indices from this object and ``field_data``, respectively, if
        they are instances of ``ModeData``.

        The dot product is defined as:

        .. math:

           \frac{1}{4} \int \left( E_0 \times H_1^* + H_0^* \times E_1 \) \, {\rm d}S

        Parameters
        ----------
        field_data : :class:`ElectromagneticFieldData`
            A data instance to compute the dot product with.
        conjugate : bool = True
            If ``True`` (default), the dot product is defined as above. If ``False``, the definition
            is similar, but without the complex conjugation of the $H$ fields.

        Returns
        -------
        :class:`xarray.DataArray`
            Data array with the complex-valued modal overlaps between the two mode data.

        See also
        --------
        :member:`dot`
        """

        tan_dims = self._tangential_dims

        if not all(a == b for a, b in zip(tan_dims, field_data._tangential_dims)):
            raise DataError("Tangential dimensions must match between the two monitors.")

        # Tangential fields for current
        fields_self = self._colocated_tangential_fields
        if conjugate:
            fields_self = {component: field.conj() for component, field in fields_self.items()}

        # Tangential fields for other data

        fields_other = field_data._interpolated_tangential_fields(self._plane_grid_boundaries)

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
        isel1 = [list(coords[0]["f"].values).index(freq) for freq in f]
        isel2 = [list(coords[1]["f"].values).index(freq) for freq in f]

        # Mode indices, if available
        modes_in_self = "mode_index" in coords[0]
        modes_in_other = "mode_index" in coords[1]

        keys = (e_1, e_2, h_1, h_2)
        for key in keys:
            fields_self[key] = fields_self[key].isel(f=isel1)
            if modes_in_self:
                fields_self[key] = fields_self[key].rename(mode_index="mode_index_0")
            else:
                fields_self[key] = fields_self[key].expand_dims(
                    dim={"mode_index_0": [0]}, axis=len(fields_self[key].shape)
                )
            fields_other[key] = fields_other[key].isel(f=isel2)
            if modes_in_other:
                fields_other[key] = fields_other[key].rename(mode_index="mode_index_1")
            else:
                fields_other[key] = fields_other[key].expand_dims(
                    dim={"mode_index_1": [0]}, axis=len(fields_other[key].shape)
                )

        d_area = self._diff_area.expand_dims(dim={"f": f}, axis=2).to_numpy()

        # function to apply at each pair of mode indices before integrating
        def fn(fields_1, fields_2):
            e_self_1 = fields_1[e_1]
            e_self_2 = fields_1[e_2]
            h_self_1 = fields_1[h_1]
            h_self_2 = fields_1[h_2]
            e_other_1 = fields_2[e_1]
            e_other_2 = fields_2[e_2]
            h_other_1 = fields_2[h_1]
            h_other_2 = fields_2[h_2]

            # Cross products of fields
            e_self_x_h_other = e_self_1 * h_other_2 - e_self_2 * h_other_1
            h_self_x_e_other = h_self_1 * e_other_2 - h_self_2 * e_other_1

            summand = 0.25 * (e_self_x_h_other - h_self_x_e_other) * d_area
            return summand

        result = self._outer_fn_summation(
            fields_1=fields_self,
            fields_2=fields_other,
            outer_dim_1="mode_index_0",
            outer_dim_2="mode_index_1",
            sum_dims=tan_dims,
            fn=fn,
        )

        # Remove mode index coordinate if the input did not have it
        if not modes_in_self:
            result = result.isel(mode_index_0=0, drop=True)
        if not modes_in_other:
            result = result.isel(mode_index_1=0, drop=True)

        return result

    @staticmethod
    def _outer_fn_summation(
        fields_1: Dict[str, xr.DataArray],
        fields_2: Dict[str, xr.DataArray],
        outer_dim_1: str,
        outer_dim_2: str,
        sum_dims: List[str],
        fn: Callable,
    ) -> DataArray:
        """
        Loop over ``outer_dim_1`` and ``outer_dim_2``, apply ``fn`` to ``fields_1`` and ``fields_2``, and sum over ``sum_dims``.
        The resulting ``DataArray`` has has dimensions any dimensions in the fields which are not contained in sum_dims.
        This can be more memory efficient than vectorizing over the ``outer_dims``, which can involve broadcasting and reshaping data.
        It also converts to numpy arrays outside the loops to minimize xarray overhead.
        """
        # first, convert to numpy outside the loop to reduce xarray overhead
        fields_1_numpy = {key: val.to_numpy() for key, val in fields_1.items()}
        fields_2_numpy = {key: val.to_numpy() for key, val in fields_2.items()}

        # get one of the data arrays to look at for indexing
        # assuming all data arrays have the same structure
        data_array_temp_1 = list(fields_1.values())[0]
        data_array_temp_2 = list(fields_2.values())[0]
        numpy_temp_1 = data_array_temp_1.to_numpy()
        numpy_temp_2 = data_array_temp_2.to_numpy()

        # find the numpy axes associated with the provided dimensions
        outer_axis_1 = data_array_temp_1.get_axis_num(outer_dim_1)
        outer_axis_2 = data_array_temp_2.get_axis_num(outer_dim_2)
        sum_axes = [data_array_temp_1.get_axis_num(dim) for dim in sum_dims]

        # coords and array for result of calculation
        coords = {key: val.to_numpy() for key, val in data_array_temp_1.coords.items()}
        for dim in sum_dims:
            coords.pop(dim)
        # last two inds are the outer_dims
        coords.pop(outer_dim_1)
        coords[outer_dim_1] = data_array_temp_1.coords[outer_dim_1].to_numpy()
        coords[outer_dim_2] = data_array_temp_2.coords[outer_dim_2].to_numpy()
        # drop scalar non-indexing dimensions
        coords = {key: val for key, val in coords.items() if len(val.shape) != 0}
        shape = [len(val) for val in coords.values()]
        dtype = np.promote_types(numpy_temp_1.dtype, numpy_temp_2.dtype)
        data = np.zeros(shape, dtype=dtype)

        # indexing tuples
        idx_1 = [slice(None)] * numpy_temp_1.ndim
        idx_2 = [slice(None)] * numpy_temp_2.ndim
        idx_data = [slice(None)] * data.ndim

        # calculate the sums of products
        for outer_1 in range(numpy_temp_1.shape[outer_axis_1]):
            for outer_2 in range(numpy_temp_2.shape[outer_axis_2]):
                idx_1[outer_axis_1] = outer_1
                idx_2[outer_axis_2] = outer_2
                idx_data[-2] = outer_1
                idx_data[-1] = outer_2
                fields_1_curr = {key: val[tuple(idx_1)] for key, val in fields_1_numpy.items()}
                fields_2_curr = {key: val[tuple(idx_2)] for key, val in fields_2_numpy.items()}
                summand_curr = fn(fields_1_curr, fields_2_curr)
                data_curr = np.sum(summand_curr, axis=tuple(sum_axes))
                data[tuple(idx_data)] = data_curr

        return DataArray(data, coords=coords)

    @property
    def time_reversed_copy(self) -> FieldData:
        """Make a copy of the data with time-reversed fields."""

        # Time reversal for frequency-domain fields; overwritten in :class:`FieldTimeData`,
        # :class:`ModeData`, and :class:`ModeSolverData`.
        new_data = {}
        for comp, field in self.field_components.items():
            if comp[0] == "H":
                new_data[comp] = -np.conj(field)
            else:
                new_data[comp] = np.conj(field)
        return self.copy(update=new_data)


class FieldData(FieldDataset, ElectromagneticFieldData):
    """
    Data associated with a :class:`.FieldMonitor`: scalar components of E and H fields.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

        This dataset can contain all electric and magnetic field components: ``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``,
        and ``Hz``.

    Example
    -------
    >>> from tidy3d import ScalarFieldDataArray
    >>> x = [-1,1,3]
    >>> y = [-2,0,2,4]
    >>> z = [-3,-1,1,3,5]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], f=f)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> scalar_field = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> monitor = FieldMonitor(
    ...     size=(2,4,6), freqs=[2e14, 3e14], name='field', fields=['Ex', 'Hz'], colocate=True
    ... )
    >>> data = FieldData(monitor=monitor, Ex=scalar_field, Hz=scalar_field, grid_expanded=grid)

    .. TODO sort out standalone data example.

    See Also
    --------

    **Notebooks:**
        * `Quickstart <../../notebooks/StartHere.html>`_: Usage in a basic simulation flow.
        * `Performing visualization of simulation data <../../notebooks/VizData.html>`_
        * `Advanced monitor data manipulation and visualization <../../notebooks/XarrayTutorial.html>`_
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

    def make_adjoint_sources(
        self, dataset_names: list[str], fwidth: float
    ) -> List[Union[CustomCurrentSource, PointDipole]]:
        """Converts a :class:`.FieldData` to a list of adjoint current or point sources."""

        if np.allclose(self.monitor.size, 0):
            return self.to_adjoint_point_sources(fwidth=fwidth)

        return self.to_adjoint_field_sources(fwidth=fwidth)

    def to_adjoint_point_sources(self, fwidth: float) -> List[PointDipole]:
        """Create adjoint point dipole source if this field data contains one item."""

        sources = []

        for polarization, field_component in self.field_components.items():
            if field_component is None:
                continue

            for freq0 in field_component.coords["f"]:
                omega0 = 2 * np.pi * freq0
                scaling_factor = 33 / (MU_0 * omega0)

                forward_amp = self.get_amplitude(field_component.sel(f=freq0))

                if forward_amp == 0.0:
                    continue

                adj_phase = np.pi + np.angle(forward_amp)
                adj_amp = scaling_factor * forward_amp

                src_adj = PointDipole(
                    center=self.monitor.center,
                    polarization=polarization,
                    source_time=GaussianPulse(
                        freq0=freq0,
                        fwidth=fwidth,
                        amplitude=abs(adj_amp),
                        phase=adj_phase,
                    ),
                    interpolate=True,
                )

                sources.append(src_adj)

        return sources

    def to_adjoint_field_sources(self, fwidth: float) -> List[CustomCurrentSource]:
        """Create adjoint custom field sources if this field data has some dimensionality."""

        sources = []
        source_geo = self.monitor.geometry
        freqs = self.monitor.freqs

        for freq0 in freqs:
            src_field_components = {}
            for name, field_component in self.field_components.items():
                # get the VJP values at frequency and apply adjoint phase
                field_component = field_component.sel(f=freq0)
                values = 2 * -1j * field_component.values

                # make source go backwards
                if "H" in name:
                    values *= -1

                # make coords that are shifted relative to geometry (0,0,0) = geometry.center
                coords = dict(field_component.coords.copy())
                for dim, key in enumerate("xyz"):
                    coords[key] = np.array(coords[key]) - source_geo.center[dim]
                coords["f"] = np.array([freq0])
                values = np.expand_dims(values, axis=-1)

                # ignore zero components
                if not np.all(values == 0):
                    src_field_components[name] = ScalarFieldDataArray(values, coords=coords)

            # dont include this source if no data
            if all(fld_cmp is None for fld_cmp in src_field_components.values()):
                continue

            # construct custom Current source
            dataset = FieldDataset(**src_field_components)
            custom_source = CustomCurrentSource(
                center=source_geo.center,
                size=source_geo.size,
                source_time=GaussianPulse(
                    freq0=freq0,
                    fwidth=fwidth,
                ),
                current_dataset=dataset,
                interpolate=True,
            )

            sources.append(custom_source)

        return sources


class FieldTimeData(FieldTimeDataset, ElectromagneticFieldData):
    """
    Data associated with a :class:`.FieldTimeMonitor`: scalar components of E and H fields.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

    Example
    -------
    >>> from tidy3d import ScalarFieldTimeDataArray
    >>> x = [-1,1,3]
    >>> y = [-2,0,2,4]
    >>> z = [-3,-1,1,3,5]
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], t=t)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> scalar_field = ScalarFieldTimeDataArray(np.random.random((2,3,4,3)), coords=coords)
    >>> monitor = FieldTimeMonitor(
    ...     size=(2,4,6), interval=100, name='field', fields=['Ex', 'Hz'], colocate=True
    ... )
    >>> data = FieldTimeData(monitor=monitor, Ex=scalar_field, Hz=scalar_field, grid_expanded=grid)
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
        tan_fields = self._colocated_tangential_fields
        dim1, dim2 = self._tangential_dims
        e_x_h = np.real(tan_fields["E" + dim1]) * np.real(tan_fields["H" + dim2])
        e_x_h -= np.real(tan_fields["E" + dim2]) * np.real(tan_fields["H" + dim1])
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


class PermittivityData(PermittivityDataset, AbstractFieldData):
    """Data for a :class:`.PermittivityMonitor`: diagonal components of the permittivity tensor.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

    Example
    -------
    >>> from tidy3d import ScalarFieldDataArray
    >>> x = [-1,1,3]
    >>> y = [-2,0,2,4]
    >>> z = [-3,-1,1,3,5]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], f=f)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> sclr_fld = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> monitor = PermittivityMonitor(size=(2,4,6), freqs=[2e14, 3e14], name='eps')
    >>> data = PermittivityData(
    ...     monitor=monitor, eps_xx=sclr_fld, eps_yy=sclr_fld, eps_zz=sclr_fld, grid_expanded=grid
    ... )
    """

    monitor: PermittivityMonitor = pd.Field(
        ..., title="Monitor", description="Permittivity monitor associated with the data."
    )


class ModeData(ModeSolverDataset, ElectromagneticFieldData):
    """
    Data associated with a :class:`.ModeMonitor`: modal amplitudes, propagation indices and mode profiles.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

        The mode monitor data contains the complex effective indices and the complex mode amplitudes at the monitor
        position calculated by mode decomposition. The data structure of the complex effective
        indices :attr`n_complex` contains two coordinates: ``f`` and ``mode_index``, both of which are specified when
        defining the :class:``ModeMonitor`` in the simulation.

        Besides the effective index, :class:``ModeMonitor`` is primarily used to calculate the transmission of
        certain modes in certain directions. We can extract the complex amplitude and square it to compute the mode
        transmission power.

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

    eps_spec: List[EpsSpecType] = pd.Field(
        None,
        title="Permettivity Specification",
        description="Characterization of the permittivity profile on the plane where modes are "
        "computed. Possible values are 'diagonal', 'tensorial_real', 'tensorial_complex'.",
    )

    @pd.validator("eps_spec", always=True)
    @skip_if_fields_missing(["monitor"])
    def eps_spec_match_mode_spec(cls, val, values):
        """Raise validation error if frequencies in eps_spec does not match frequency list"""
        if val:
            mode_data_freqs = values["monitor"].freqs
            if len(val) != len(mode_data_freqs):
                raise ValidationError(
                    "eps_spec must be provided at the same frequencies as mode solver data."
                )
        return val

    def normalize(self, source_spectrum_fn) -> ModeData:
        """Return copy of self after normalization is applied using source spectrum function."""
        source_freq_amps = source_spectrum_fn(self.amps.f)[None, :, None]
        new_amps = (self.amps / source_freq_amps).astype(self.amps.dtype)
        return self.copy(update=dict(amps=new_amps))

    def overlap_sort(
        self,
        track_freq: TrackFreq,
        overlap_thresh: float = 0.9,
    ) -> ModeData:
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
        if len(self.field_components) == 0:
            return self.copy()
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
                # Assign to the base frequency so that outer_dot will compare them
                data_to_sort = data_to_sort._assign_coords(f=[self.monitor.freqs[f0_ind]])

                # Compute "sorting w.r.t. to neighbor" and overlap values

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
                        f"Mode '{mode_ind}' appears to undergo a discontinuous change "
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
        return self._updated(update=update_dict)

    def _assign_coords(self, **assign_coords_kwargs):
        """Wraps ``xarray.DataArray.assign_coords`` for all data fields that are defined over frequency and
        mode index. Used in ``overlap_sort`` but not officially supported since for example
        ``self.monitor.mode_spec`` and ``self.monitor.freqs`` will no longer be matching the
        newly created data."""

        update_dict = dict(self._grid_correction_dict, **self.field_components)
        update_dict = {
            key: field.assign_coords(**assign_coords_kwargs) for key, field in update_dict.items()
        }
        return self._updated(update=update_dict)

    def _find_ordering_one_freq(
        self,
        data_to_sort: ModeData,
        overlap_thresh: float,
    ) -> Tuple[Numpy, Numpy]:
        """Find new ordering of modes in data_to_sort based on their similarity to own modes."""

        num_modes = self.n_complex.sizes["mode_index"]

        # Current pairs and their overlaps
        pairs = np.arange(num_modes)
        complex_amps = self.dot(data_to_sort).data.ravel()
        if self.monitor.store_fields_direction == "-":
            complex_amps *= -1

        # Check whether modes already match
        modes_to_sort = np.where(np.abs(complex_amps) < overlap_thresh)[0]
        num_modes_to_sort = len(modes_to_sort)
        if num_modes_to_sort <= 1:
            return pairs, complex_amps

        # Extract all modes of interest from template data
        data_template_reduced = self._isel(mode_index=modes_to_sort)

        amps_reduced = data_template_reduced.outer_dot(
            data_to_sort._isel(mode_index=modes_to_sort)
        ).to_numpy()[0, :, :]

        if self.monitor.store_fields_direction == "-":
            amps_reduced *= -1

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

    def _reorder_modes(
        self,
        sorting: Numpy,
        phase: Numpy,
        track_freq: TrackFreq,
    ) -> ModeData:
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
            phase_fact = np.exp(-1j * phase[None, None, None, :, :]).astype(field_sorted.data.dtype)
            field_sorted.data = field_sorted.data * phase_fact

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

    def _group_index_post_process(self, frequency_step: float) -> ModeData:
        """Calculate group index and remove added frequencies used only for this calculation.

        Parameters
        ----------
        frequency_step: float
            Fractional frequency step used to calculate the group index.

        Returns
        -------
        :class:`.ModeData`
            Filtered data with calculated group index.
        """

        freqs = self.n_complex.coords["f"].values
        num_freqs = freqs.size
        back = slice(0, num_freqs, 3)
        center = slice(1, num_freqs, 3)
        fwd = slice(2, num_freqs, 3)
        freqs = freqs[center]

        # calculate group index
        n_center = self.n_eff.isel(f=center).values
        n_backward = self.n_eff.isel(f=back).values
        n_forward = self.n_eff.isel(f=fwd).values

        inv_step = 1 / frequency_step
        # n_g = n + f * df/dn
        # dn/df = (n+ - n-) / (2 f df)
        n_group_data = n_center + (n_forward - n_backward) * inv_step * 0.5
        # D = -2 * pi * c / lda^2 * d(v_g^-1)/dw = -(f / c)^2 * (2 * dn/df + f * d2n/df2)
        # d2n/df2 = (n+ - 2n + n-) / (f df)^2
        # The '1e18' factor converts from s/um^2 to ps/(nm km)
        dispersion_data = (
            (n_forward * (inv_step + 1) + n_backward * (inv_step - 1) - n_center * inv_step * 2)
            * freqs.reshape((-1, 1))
            * (-1e18 * inv_step / C_0**2)
        )

        mode_index = list(self.n_complex.coords["mode_index"].values)
        f = list(freqs)
        n_group = GroupIndexDataArray(
            n_group_data,
            coords={"f": f, "mode_index": mode_index},
        )

        dispersion = ModeDispersionDataArray(
            dispersion_data,
            coords={"f": f, "mode_index": mode_index},
        )

        # remove data corresponding to frequencies used only for group index calculation
        update_dict = {
            "n_complex": self.n_complex.isel(f=center),
            "n_group_raw": n_group,
            "dispersion_raw": dispersion,
        }

        for key, field in self.field_components.items():
            update_dict[key] = field.isel(f=center)

        for key, data in self._grid_correction_dict.items():
            update_dict[key] = data.isel(f=center)

        if self.eps_spec:
            update_dict["eps_spec"] = self.eps_spec[center]

        update_dict["monitor"] = self.monitor.updated_copy(freqs=freqs)

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
        new_dir = "+" if mnt.store_fields_direction == "-" else "-"
        new_data["monitor"] = mnt.updated_copy(store_fields_direction=new_dir)
        return self.copy(update=new_data)

    def _colocated_propagation_axes_field(self, field_name: Literal["E", "H"]) -> DataArray:
        """Collect a field DataArray containing all 3 field components and rotate from frame
        with normal axis along z to frame with propagation axis along z.
        """
        tan_dims = self._tangential_dims
        normal_dim = "xyz"[self.monitor.zero_dims[0]]
        fields = self._colocated_fields
        fields = {key: val.squeeze(dim=normal_dim, drop=True) for key, val in fields.items()}
        mode_spec = self.monitor.mode_spec

        # fields as a (3, ...) numpy array ordered as [tangential1, tagential2, normal]
        field = [fields[field_name + dim].values for dim in tan_dims]
        field = np.array(field + [fields[field_name + normal_dim].values])

        # rotate axes
        if mode_spec.angle_phi != 0:
            field = self.monitor.rotate_points(field, [0, 0, 1], -mode_spec.angle_phi)
        if mode_spec.angle_theta != 0:
            field = self.monitor.rotate_points(field, [0, 1, 0], -mode_spec.angle_theta)

        # new coords for the (3, ...) array
        coords = {"component": [0, 1, 2]}
        # fields are colocated, so all components should have the same coords
        for dim in fields["Ex"].dims:
            coords.update({dim: fields["Ex"].coords[dim]})

        return DataArray(data=field, coords=coords)

    @cached_property
    def pol_fraction(self) -> xr.Dataset:
        r"""Compute the TE and TM polarization fraction defined as the field intensity along the
        first or the second of the two tangential axes. More precisely, if $E_1$ and $E_2$ are
        the electric field components along the two tangential axes, the TE fraction is defined as:

        .. math::

           \frac{\int |E_1|^2 \, {\rm d}S}{\int \left(|E_1|^2 + |E_2|^2\right) \, {\rm d}S}

        and the TM fraction is equal to one minus the TE fraction. The tangential axes are defined
        by popping the normal axis from the list of ``x, y, z``, so e.g. ``x`` and ``z`` for
        propagation in the ``y`` direction.
        """
        if len(self.field_components) == 0:
            raise DataError(
                "Field data not included in this ModeData onbject. Set "
                "'ModeMonitor.store_fields_direction' to the desired propagation direction to "
                "include the mode field profiles in the corresponding 'ModeData'."
            )

        tan_dims = self._tangential_dims
        e_field = self._colocated_propagation_axes_field("E")
        diff_area = self._diff_area
        tm_int = (diff_area * np.abs(e_field.sel(component=1, drop=True)) ** 2).sum(dim=tan_dims)
        te_int = (diff_area * np.abs(e_field.sel(component=0, drop=True)) ** 2).sum(dim=tan_dims)
        te_frac = te_int / (te_int + tm_int)

        return xr.Dataset(data_vars={"te": te_frac, "tm": 1 - te_frac})

    @cached_property
    def pol_fraction_waveguide(self) -> xr.Dataset:
        r"""Compute the TE and TM polarization fraction using the waveguide definition. If $n$ is
        the propagation direction, the TE fraction is defined as:

        .. math::

           1 - \frac{\int |E \cdot n|^2 \, {\rm d}S}{\int |E|^2 \, {\rm d}S}

        and the TM fraction is defined as

        .. math::

           1 - \frac{\int |H \cdot n|^2 \, {\rm d}S}{\int |H|^2 \, {\rm d}S}

        Note
        ----
            The waveguide TE and TM fractions do not sum to one. For example, TEM modes that
            are completely transverse (zero electric and magnetic field in the propagation
            direction) have TE fraction and TM fraction both equal to one.
        """
        if len(self.field_components) == 0:
            raise DataError(
                "Field data not included in this ModeData onbject. Set "
                "'ModeMonitor.store_fields_direction' to the desired propagation direction to "
                "include the mode field profiles in the corresponding 'ModeData'."
            )

        tan_dims = self._tangential_dims
        e_field = self._colocated_propagation_axes_field("E")
        h_field = self._colocated_propagation_axes_field("H")
        diff_area = self._diff_area

        # te fraction
        field_int = [np.abs(e_field.sel(component=ind, drop=True)) ** 2 for ind in range(3)]
        norm_int = (diff_area * field_int[2]).sum(dim=tan_dims)
        tot_int = norm_int + (diff_area * (field_int[0] + field_int[1])).sum(dim=tan_dims)
        te_frac = 1 - norm_int / tot_int

        # tm fraction
        field_int = [np.abs(h_field.sel(component=ind, drop=True)) ** 2 for ind in range(3)]
        norm_int = (diff_area * field_int[2]).sum(dim=tan_dims)
        tot_int = norm_int + (diff_area * (field_int[0] + field_int[1])).sum(dim=tan_dims)
        tm_frac = 1 - norm_int / tot_int

        return xr.Dataset(data_vars={"te": te_frac, "tm": tm_frac})

    @property
    def modes_info(self) -> xr.Dataset:
        """Dataset collecting various properties of the stored modes."""

        lambda_cm = C_0 / self.k_eff.f / 1e4
        loss_db_cm = 20 * 2 * np.pi * np.log10(np.e) * self.k_eff / lambda_cm

        info = {
            "wavelength": C_0 / self.n_eff.f,
            "n eff": self.n_eff,
            "k eff": self.k_eff,
            "loss (dB/cm)": loss_db_cm,
            f"TE (E{self._tangential_dims[0]}) fraction": None,
            "wg TE fraction": None,
            "wg TM fraction": None,
            "mode area": None,
            "group index": self.n_group_raw,  # Use raw field to avoid issuing a warning
            "dispersion (ps/(nm km))": self.dispersion_raw,  # Use raw field to avoid issuing a warning
        }

        if self.n_group_raw is not None:
            info["group index"] = self.n_group_raw

        if len(self.field_components) == 6:
            info["mode area"] = self.mode_area
            info[f"TE (E{self._tangential_dims[0]}) fraction"] = self.pol_fraction["te"]
            info["wg TE fraction"] = self.pol_fraction_waveguide["te"]
            info["wg TM fraction"] = self.pol_fraction_waveguide["tm"]

        return xr.Dataset(data_vars=info)

    def to_dataframe(self) -> DataFrame:
        """xarray-like method to export the ``modes_info`` into a pandas dataframe which is e.g.
        simple to visualize as a table."""

        dataset = self.modes_info
        drop = []

        if not np.any(dataset["group index"].values):
            drop.append("group index")
        if not np.any(dataset["dispersion (ps/(nm km))"].values):
            drop.append("dispersion (ps/(nm km))")
        if np.all(dataset["loss (dB/cm)"] == 0):
            drop.append("loss (dB/cm)")

        return dataset.drop_vars(drop).to_dataframe()

    def make_adjoint_sources(self, dataset_names: list[str], fwidth: float) -> list[ModeSource]:
        """Get all adjoint sources for the ``ModeMonitorData``."""

        adjoint_sources = []

        for name in dataset_names:
            if name == "amps":
                adjoint_sources += self.make_adjoint_sources_amps(fwidth=fwidth)
            elif not np.all(self.n_complex.values == 0.0):
                log.warning(
                    f"Can't create adjoint source for 'ModeData.{type(self)}.{name}'. "
                    f"for monitor '{self.monitor.name}'. "
                    "It's likely your objective function depends on sim data that is un-traced. "
                    "Double check your post-processing function to confirm. "
                )

        return adjoint_sources

    def make_adjoint_sources_amps(self, fwidth: float) -> list[ModeSource]:
        """Generate adjoint sources for ``ModeMonitorData.amps``."""

        coords = self.amps.coords

        adjoint_sources = []

        # TODO: speed up with ufunc?
        for freq in coords["f"]:
            for direction in coords["direction"]:
                for mode_index in coords["mode_index"]:
                    amp_single = self.amps.sel(f=freq, direction=direction, mode_index=mode_index)

                    if self.get_amplitude(amp_single) == 0.0:
                        continue

                    adjoint_source = self.adjoint_source_amp(amp=amp_single, fwidth=fwidth)
                    adjoint_sources.append(adjoint_source)

        return adjoint_sources

    def adjoint_source_amp(self, amp: DataArray, fwidth: float) -> ModeSource:
        """Generate an adjoint ``ModeSource`` for a single amplitude."""

        monitor = self.monitor

        # grab coordinates
        coords = amp.coords
        freq0 = coords["f"]
        direction = coords["direction"]
        mode_index = coords["mode_index"]

        # determine the complex amplitude
        amp_complex = self.get_amplitude(amp)
        k0 = 2 * np.pi * freq0 / C_0
        grad_const = k0 / 4 / ETA_0
        src_amp = grad_const * amp_complex

        # construct source
        src_adj = ModeSource(
            source_time=GaussianPulse(
                amplitude=abs(src_amp),
                phase=np.angle(src_amp),
                freq0=freq0,
                fwidth=fwidth,
            ),
            mode_spec=monitor.mode_spec,
            size=monitor.size,
            center=monitor.center,
            direction=self.flip_direction(direction),
            mode_index=mode_index,
        )

        return src_adj


class ModeSolverData(ModeData):
    """
    Data associated with a :class:`.ModeSolverMonitor`: scalar components of E and H fields.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

    Example
    -------
    >>> from tidy3d import ModeSpec
    >>> from tidy3d import ScalarModeFieldDataArray, ModeIndexDataArray
    >>> x = [-1,1,3]
    >>> y = [-2,0]
    >>> z = [-3,-1,1,3,5]
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> field_coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], f=f, mode_index=mode_index)
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
    ...     n_complex=index_data,
    ...     grid_expanded=grid
    ... )
    """

    monitor: ModeSolverMonitor = pd.Field(
        ..., title="Monitor", description="Mode solver monitor associated with the data."
    )

    amps: ModeAmpsDataArray = pd.Field(
        None, title="Amplitudes", description="Unused for ModeSolverData."
    )

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> ModeSolverData:
        """Return copy of self after normalization is applied using source spectrum function."""
        return self.copy()

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
        new_dir = "+" if mnt.store_fields_direction == "-" else "-"
        new_data["monitor"] = mnt.updated_copy(direction=new_dir, store_fields_direction=new_dir)
        return self.copy(update=new_data)


class FluxData(MonitorData):
    """
    Data associated with a :class:`.FluxMonitor`: flux data in the frequency-domain.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

        We can access the data for each monitor by indexing into the :class:`SimulationData` with the monitor
        ``.name``. For the flux monitor data, we can access the raw flux data as a function of frequency with
        ``.flux``. As most data are multidimensional, it’s often very helpful to print out the data and directly
        inspect its structure.

    Example
    -------
    >>> from tidy3d import FluxDataArray
    >>> f = [2e14, 3e14]
    >>> coords = dict(f=f)
    >>> flux_data = FluxDataArray(np.random.random(2), coords=coords)
    >>> monitor = FluxMonitor(size=(2,0,6), freqs=[2e14, 3e14], name='flux')
    >>> data = FluxData(monitor=monitor, flux=flux_data)

    See Also
    --------

    **Notebooks:**
        * `Advanced monitor data manipulation and visualization <../../notebooks/XarrayTutorial.html>`_
    """

    monitor: FluxMonitor = pd.Field(
        ..., title="Monitor", description="Frequency-domain flux monitor associated with the data."
    )

    flux: FluxDataArray = pd.Field(
        ..., title="Flux", description="Flux values in the frequency-domain."
    )

    def make_adjoint_sources(
        self, dataset_names: list[str], fwidth: float
    ) -> List[Union[CustomCurrentSource, PointDipole]]:
        """Converts a :class:`.FieldData` to a list of adjoint current or point sources."""

        # avoids error in edge case where there are extraneous flux monitors not used in objective
        if np.all(self.flux.values == 0.0):
            return []

        raise NotImplementedError(
            "Could not formulate adjoint source for 'FluxMonitor' output. To compute derivatives "
            "with respect to flux data, please use a 'FieldMonitor' and call '.flux' on the "
            "resulting 'FieldData' object. Using 'FluxMonitor' directly is not supported as "
            "the full field information is required to construct the adjoint source for this "
            "problem. The 'FluxData' does not contain the information necessary for gradient "
            "computation."
        )

    def normalize(self, source_spectrum_fn) -> FluxData:
        """Return copy of self after normalization is applied using source spectrum function."""
        source_freq_amps = source_spectrum_fn(self.flux.f)
        source_power = abs(source_freq_amps) ** 2
        new_flux = (self.flux / source_power).astype(self.flux.dtype)
        return self.copy(update=dict(flux=new_flux))


class FluxTimeData(MonitorData):
    """
    Data associated with a :class:`.FluxTimeMonitor`: flux data in the time-domain.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

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
        discriminator=TYPE_TAG_STR,
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
        discriminator=TYPE_TAG_STR,
    )

    is_2d_simulation: bool = pd.Field(
        False,
        title="2D Simulation",
        description="Indicates whether the monitor data is for a 2D simulation.",
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

    def make_data_array(self, data: np.ndarray) -> DataArray:
        """Make an DataArray with data and same coords and dims as fields of self."""
        return DataArray(data=data, coords=self.coords, dims=self.dims)

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
    def propagation_factor(dist: Union[float, None], k: complex, is_2d_simulation: bool) -> complex:
        """A normalization factor that includes both phase and amplitude decay associated with propagation over a distance with a given wavenumber."""
        if dist is None:
            return 1.0

        if is_2d_simulation:
            return np.exp(1j * k * dist) * np.sqrt(-1j * k / (8 * np.pi * dist))

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
    def power(self) -> DataArray:
        """Get power measured on the projection grid relative to the monitor's local origin.

        Returns
        -------
        ``xarray.DataArray``
            Power at points relative to the local origin.
        """
        power_theta = 0.5 * np.real(self.Etheta * self.Hphi.conj())
        power_phi = 0.5 * np.real(-self.Ephi * self.Htheta.conj())
        power = power_theta + power_phi

        return self.make_data_array(data=power)

    @property
    def radar_cross_section(self) -> DataArray:
        """Radar cross section in units of incident power."""

        _, index_k = self.nk
        if not np.all(index_k == 0):
            raise SetupError("Can't compute RCS for a lossy background medium.")

        k = self.k[None, None, None, ...]
        eta = self.eta[None, None, None, ...]

        if self.is_2d_simulation:
            constant = k**2 / (16 * np.pi * eta)
        else:
            constant = k**2 / (8 * np.pi * eta)

        # normalize fields by the distance-based phase factor
        coords_sph = self.coords_spherical
        if coords_sph["r"] is None:
            phase = 1.0
        else:
            phase = self.propagation_factor(
                dist=coords_sph["r"][..., None], k=k, is_2d_simulation=self.is_2d_simulation
            )
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
        old_phase = self.propagation_factor(
            dist=r, k=self.k[None, None, None, :], is_2d_simulation=self.is_2d_simulation
        )

        # the phase factor associated with the new distance must be applied
        new_phase = self.propagation_factor(
            dist=proj_distance, k=self.k, is_2d_simulation=self.is_2d_simulation
        )

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

    @property
    def tangential_dims(self):
        tangential_dims = ["x", "y", "z"]
        tangential_dims.pop(self.monitor.proj_axis)
        return tangential_dims

    @property
    def poynting(self) -> DataArray:
        """Time-averaged Poynting vector for field data associated to a Cartesian field projection monitor."""
        fc = self.fields_cartesian
        dim1, dim2 = self.tangential_dims

        e1 = fc["E" + dim1]
        e2 = fc["E" + dim2]
        h1 = fc["H" + dim1]
        h2 = fc["H" + dim2]

        e1_h2 = e1 * h2.conj()
        e2_h1 = e2 * h1.conj()

        e_x_h_star = e1_h2 - e2_h1
        return 0.5 * np.real(e_x_h_star)

    @cached_property
    def flux(self) -> FluxDataArray:
        """Flux for projecteded field data corresponding to a Cartesian field projection monitor."""
        flux = self.poynting.integrate(self.tangential_dims)
        return FluxDataArray(flux)

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
        old_phase = self.propagation_factor(dist=r, k=k, is_2d_simulation=self.is_2d_simulation)

        # update the field components' projection distance
        norm_dir, _ = self.monitor.pop_axis(["x", "y", "z"], axis=self.monitor.proj_axis)
        for field in self.field_components.values():
            field[norm_dir] = np.atleast_1d(proj_distance)

        # the phase factor associated with the new distance must be applied
        r = self.coords_spherical["r"][..., None]
        new_phase = self.propagation_factor(dist=r, k=k, is_2d_simulation=self.is_2d_simulation)

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
        old_phase = self.propagation_factor(
            dist=r, k=self.k[None, None, None, :], is_2d_simulation=self.is_2d_simulation
        )

        # the phase factor associated with the new distance must be applied
        new_phase = self.propagation_factor(
            dist=proj_distance, k=self.k, is_2d_simulation=self.is_2d_simulation
        )

        # net phase
        phase = new_phase[None, None, None, :] / old_phase

        # compute updated fields and their coordinates
        return self.make_renormalized_data(phase, proj_distance)


class DiffractionData(AbstractFieldProjectionData):
    """Data for a :class:`.DiffractionMonitor`: complex components of diffracted far fields.

    Note
    ----

        The diffraction data are separated into S and P polarizations. At normal incidence when
        S and P are undefined, P(S) corresponds to ``Ey``(``Ez``) polarization for monitor normal
        to x, P(S) corresponds to ``Ex``(``Ez``) polarization for monitor normal to y, and P(S)
        corresponds to ``Ex``(``Ey``) polarization for monitor normal to z.


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
        reciprocal_vectors: Tuple[np.ndarray, np.ndarray],
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
    def angles(self) -> Tuple[DataArray]:
        """The (theta, phi) angles corresponding to each allowed pair of diffraction
        orders storeds as data arrays. Disallowed angles are set to ``np.nan``.
        """
        thetas, phis = self.compute_angles(self.reciprocal_vectors)
        theta_data = DataArray(thetas, coords=self.coords)
        phi_data = DataArray(phis, coords=self.coords)
        return theta_data, phi_data

    @property
    def amps(self) -> DataArray:
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
        return DataArray(np.stack([amp_phi, amp_theta], axis=3), coords=coords)

    @property
    def power(self) -> Self:
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
        e_x, e_y, e_z, h_x, h_y, h_z = (
            np.nan_to_num(fld) for fld in [e_x, e_y, e_z, h_x, h_y, h_z]
        )

        fields = [e_x, e_y, e_z, h_x, h_y, h_z]
        keys = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
        return self._make_dataset(fields, keys)

    def _make_dataset(self, fields: Tuple[np.ndarray, ...], keys: Tuple[str, ...]) -> xr.Dataset:
        """Make an xr.Dataset for fields with given field names."""
        data_arrays = []
        for field in fields:
            data_arrays.append(DataArray(data=field, coords=self.coords, dims=self.dims))
        return xr.Dataset(dict(zip(keys, data_arrays)))

    """ Autograd code """

    def make_adjoint_sources(self, dataset_names: list[str], fwidth: float) -> list[PlaneWave]:
        """Get all adjoint sources for the ``DiffractionMonitor.amps``."""

        # NOTE: everything just goes through `.amps`, any post-processing is encoded in E-fields
        return self.make_adjoint_sources_amps(fwidth=fwidth)

    def make_adjoint_sources_amps(self, fwidth: float) -> list[PlaneWave]:
        """Make adjoint sources for outputs that depend on DiffractionData.`amps`."""

        amps = self.amps
        coords = amps.coords

        adjoint_sources = []

        # TODO: speed up with ufunc?
        # loop over all coordinates in the diffraction amplitudes
        for freq in coords["f"]:
            for pol in coords["polarization"]:
                for order_x in coords["orders_x"]:
                    for order_y in coords["orders_y"]:
                        amp_single = amps.sel(
                            f=freq,
                            polarization=pol,
                            orders_x=order_x,
                            orders_y=order_y,
                        )

                        # ignore any amplitudes of 0.0 or nan
                        amp_complex = self.get_amplitude(amp_single)
                        if (amp_complex == 0.0) or np.isnan(amp_complex):
                            continue

                        # compute a plane wave for this amplitude (if propagating / not None)
                        adjoint_source = self.adjoint_source_amp(amp=amp_single, fwidth=fwidth)
                        if adjoint_source is not None:
                            adjoint_sources.append(adjoint_source)

        return adjoint_sources

    def adjoint_source_amp(self, amp: DataArray, fwidth: float) -> PlaneWave:
        """Generate an adjoint ``PlaneWave`` for a single amplitude."""

        monitor = self.monitor

        # grab the coordinates
        coords = amp.coords
        freq0 = coords["f"]
        pol = coords["polarization"]
        order_x = coords["orders_x"]
        order_y = coords["orders_y"]

        # compute the angle corresponding to this amplitude
        theta_data, phi_data = self.angles
        angle_sel_kwargs = dict(orders_x=int(order_x), orders_y=int(order_y), f=float(freq0))
        angle_theta = float(theta_data.sel(**angle_sel_kwargs))
        angle_phi = float(phi_data.sel(**angle_sel_kwargs))

        # if the angle is nan, this amplitude is set to 0 in the fwd pass, so should skip adj
        if np.isnan(angle_theta):
            return None

        # get the polarization angle from the data
        pol_str = str(pol.values)
        if pol_str not in ("p", "s"):
            raise ValueError(f"Something went wrong, given pol='{pol_str}' in adjoint source.")

        pol_angle = 0.0 if pol_str == "p" else np.pi / 2

        # compute the source amplitude
        amp_complex = self.get_amplitude(amp)
        k0 = 2 * np.pi * freq0 / C_0
        bck_eps = self.medium.eps_model(freq0)
        grad_const = 0.5 * k0 / np.sqrt(bck_eps) * np.cos(angle_theta)
        src_amp = grad_const * amp_complex

        # construct plane wave source
        adj_src = PlaneWave(
            size=self.monitor.size,
            center=self.monitor.center,
            source_time=GaussianPulse(
                amplitude=abs(src_amp),
                phase=np.angle(src_amp),
                freq0=freq0,
                fwidth=fwidth,
            ),
            direction=self.flip_direction(monitor.normal_dir),
            angle_theta=angle_theta,
            angle_phi=angle_phi,
            pol_angle=pol_angle,
        )

        return adj_src


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
