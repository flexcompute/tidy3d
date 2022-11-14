# pylint:disable=too-many-lines
""" Monitor Level Data, store the DataArrays associated with a single monitor."""
from __future__ import annotations

from abc import ABC
from typing import Union, Tuple, Callable, Dict, List
import warnings
import xarray as xr
import numpy as np
import pydantic as pd

from .data_array import FluxTimeDataArray, FluxDataArray, ModeIndexDataArray, ModeAmpsDataArray
from .data_array import Near2FarAngleDataArray, Near2FarCartesianDataArray, Near2FarKSpaceDataArray
from .data_array import DataArray, DiffractionDataArray
from .data_array import ScalarFieldDataArray, ScalarFieldTimeDataArray
from .dataset import Dataset, AbstractFieldDataset, ElectromagneticFieldDataset
from .dataset import FieldDataset, FieldTimeDataset, ModeSolverDataset, PermittivityDataset
from ..base import TYPE_TAG_STR
from ..types import Coordinate, Symmetry, ArrayLike, Size
from ..grid.grid import Grid
from ..validators import enforce_monitor_fields_present, required_if_symmetry_present
from ..monitor import MonitorType, FieldMonitor, FieldTimeMonitor, ModeSolverMonitor
from ..monitor import ModeMonitor, FluxMonitor, FluxTimeMonitor, PermittivityMonitor
from ..monitor import Near2FarAngleMonitor, Near2FarCartesianMonitor, Near2FarKSpaceMonitor
from ..monitor import DiffractionMonitor
from ..source import SourceTimeType, CustomFieldSource
from ..medium import Medium, MediumType
from ...log import SetupError, DataError
from ...constants import ETA_0, C_0, MICROMETER


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

            grid_key = self.grid_locations[field_name]
            eigenval_fn = self.symmetry_eigenvalues[field_name]

            # get grid locations for this field component on the expanded grid
            grid_locations = self.grid_expanded[grid_key]

            for sym_dim, (sym_val, sym_loc) in enumerate(zip(self.symmetry, self.symmetry_center)):

                dim_name = "xyz"[sym_dim]

                # Continue if no symmetry along this dimension
                if sym_val == 0:
                    continue

                # Get coordinates for this field component on the expanded grid
                coords = grid_locations.to_list[sym_dim]

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
                    scalar_data[{dim_name: flip_inds}] *= sym_val * sym_eigenvalue

            # assign the final scalar data to the update_dict
            update_dict[field_name] = scalar_data

        update_dict.update({"symmetry": (0, 0, 0), "symmetry_center": None, "grid_expanded": None})
        return self.copy(update=update_dict)


class ElectromagneticFieldData(AbstractFieldData, ElectromagneticFieldDataset, ABC):
    """Collection of electromagnetic fields."""

    @property
    def _tangential_dims(self) -> List[str]:
        """For a 2D monitor data, return the names of the tangential dimensions. Raise if cannot
        confirm that the associated monitor is 2D."""
        zero_dims = np.where(np.array(self.monitor.size) == 0)[0]
        if zero_dims.size != 1:
            raise DataError("Data must be 2D to get tangential dimensions.")
        tangential_dims = ["x", "y", "z"]
        tangential_dims.pop(zero_dims[0])

        return tangential_dims

    @property
    def _tangential_fields(self) -> Dict[str, DataArray]:
        """For a 2D monitor data, return a dictionary with only the tangential field components,
        oriented such that the third component would be the normal axis. This just means that the
        H field gets an extra minus sign if the normal axis is ``"y"``. Raise if any of the
        tangential field components is missing."""

        tan_dims = self._tangential_dims
        normal_dim = "xyz"[self.monitor.size.index(0)]
        field_components = ["E" + dim for dim in tan_dims] + ["H" + dim for dim in tan_dims]
        tan_fields = {}
        for field in field_components:
            if field not in self.field_components:
                raise DataError(f"Tangential field component {field} is missing in data.")
            tan_fields[field] = self.field_components[field].squeeze(dim=normal_dim)
            if normal_dim == "y" and field[0] == "H":
                tan_fields[field] *= -1
        return tan_fields

    @property
    def _plane_grid_boundaries(self) -> Tuple[ArrayLike[float, 1], ArrayLike[float, 1]]:
        """For a 2D monitor data, return the boundaries of the in-plane grid from the stored field
        coordinates."""
        dim1, dim2 = self._tangential_dims
        tan_fields = self._tangential_fields
        plane_bounds1 = tan_fields["E" + dim2].coords[dim1].values
        plane_bounds2 = tan_fields["E" + dim1].coords[dim2].values
        return plane_bounds1, plane_bounds2

    @property
    def _diff_area(self) -> xr.DataArray:
        """For a 2D monitor data, return the area of each cell in the plane, for use in numerical
        integrations."""
        bounds = [bs.copy() for bs in self._plane_grid_boundaries]

        """Fix first and last boundary to match the analytic monitor boundary within that pixel.
        When using the differential area sizes defined in this way together with integrand values
        defined at pixel centers, the integration is equivalent to trapezoidal rule with the first
        and last values interpolated to the exact monitor start/end location."""
        _, plane_inds = self.monitor.pop_axis([0, 1, 2], self.monitor.size.index(0.0))
        mnt_bounds = np.array(self.monitor.bounds)
        mnt_bounds = mnt_bounds[:, plane_inds].T
        bounds[0][0] = max(bounds[0][0], mnt_bounds[0, 0])
        bounds[0][-1] = min(bounds[0][-1], mnt_bounds[0, 1])
        bounds[1][0] = max(bounds[1][0], mnt_bounds[1, 0])
        bounds[1][-1] = min(bounds[1][-1], mnt_bounds[1, 1])

        sizes = [bs[1:] - bs[:-1] for bs in bounds]
        return xr.DataArray(np.outer(sizes[0], sizes[1]), dims=self._tangential_dims)

    @property
    def _centered_tangential_fields(self) -> Dict[str, DataArray]:
        """For a 2D monitor data, get the tangential E and H fields colocated to the cell centers in
        the 2D plane grid."""

        # Tangential directions and fields
        tan_dims = self._tangential_dims
        tan_fields = self._tangential_fields

        # Plane center coordinates
        bounds = self._plane_grid_boundaries
        centers = [(bs[1:] + bs[:-1]) / 2 for bs in bounds]

        # Interpolate tangential field components to cell centers
        interp_dict = dict(zip(tan_dims, centers))
        centered_fields = {key: val.interp(**interp_dict) for key, val in tan_fields.items()}
        return centered_fields

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

    @property
    def flux(self) -> FluxDataArray:
        """Flux for data corresponding to a 2D monitor.

        Note
        ----
            Here, the exact monitor center and size is used in the numerical integration.
            This differs from the on-the-fly computation using a :class:`.FluxMonitor`, where a
            discretization of the monitor plane in terms of an integer number of Yee grid cells is
            used. Thus, the two computations are only expected to match if a :class:`.FieldMonitor`
            is placed exactly at a Yee grid cell center in the normal direction, and spans an
            integer number of cells in both tangential directions.
        """

        # Compute flux by integrating Poynting vector in-plane
        d_area = self._diff_area
        return FluxDataArray((self.poynting * d_area).sum(dim=d_area.dims))

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
            and the sum of carried power fractions exceed 1. In the non-conjugated definition,
            orthogonal modes can be defined, but the interpretation of modal overlap as power
            carried by a given mode is no longer valid.
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
        e_self_x_h_other -= fields_other["E" + dim2] * fields_self["H" + dim1]
        h_self_x_e_other = fields_self["H" + dim1] * fields_other["E" + dim2]
        h_self_x_e_other -= fields_other["H" + dim2] * fields_self["E" + dim1]

        # Integrate over plane
        d_area = self._diff_area
        integrand = (e_self_x_h_other - h_self_x_e_other) * d_area
        return ModeAmpsDataArray(0.25 * integrand.sum(dim=d_area.dims))


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

    monitor: FieldMonitor

    _contains_monitor_fields = enforce_monitor_fields_present()

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> FieldDataset:
        """Return copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        for field_name, field_data in self.field_components.items():
            src_amps = source_spectrum_fn(field_data.f)
            fields_norm[field_name] = (field_data / src_amps).astype(field_data.dtype)

        return self.copy(update=fields_norm)

    def to_source(
        self, source_time: SourceTimeType, center: Coordinate, size: Size = None
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
            field_dataset=dataset, source_time=source_time, center=center, size=size
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

    monitor: FieldTimeMonitor

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

    @property
    def flux(self) -> FluxTimeDataArray:
        """Flux for data corresponding to a 2D monitor.

        Note
        ----
            Here, the exact monitor center and size is used in the numerical integration.
            This differs from the on-the-fly computation using a :class:`.FluxTimeMonitor`, where a
            discretization of the monitor plane in terms of an integer number of Yee grid cells is
            used. Thus, the two computations are only expected to match if a
            :class:`.FieldTimeMonitor` is placed exactly at a Yee grid cell center in the normal
            direction, and spans an integer number of cells in both tangential directions.
        """

        # Compute flux by integrating Poynting vector in-plane
        d_area = self._diff_area
        return FluxTimeDataArray((self.poynting * d_area).sum(dim=d_area.dims))

    def dot(self, field_data: ElectromagneticFieldData, conjugate: bool = True) -> xr.DataArray:
        """Inner product is not defined for time-domain data."""
        raise DataError("Inner product is not defined for time-domain data.")


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

    monitor: ModeSolverMonitor


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

    monitor: PermittivityMonitor


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

    monitor: ModeMonitor

    amps: ModeAmpsDataArray = pd.Field(
        ..., title="Amplitudes", description="Complex-valued amplitudes associated with the mode."
    )

    n_complex: ModeIndexDataArray = pd.Field(
        ...,
        title="Propagation Index",
        description="Complex-valued effective propagation constants associated with the mode.",
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

    monitor: FluxMonitor
    flux: FluxDataArray

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

    monitor: FluxTimeMonitor
    flux: FluxTimeDataArray


PROJFIELDTYPE = Union[
    Near2FarAngleDataArray,
    Near2FarCartesianDataArray,
    Near2FarKSpaceDataArray,
    DiffractionDataArray,
]


class AbstractFieldProjectionData(MonitorData):
    """Collection of projected fields in spherical coordinates in the frequency domain."""

    monitor: Union[
        Near2FarAngleMonitor,
        Near2FarCartesianMonitor,
        Near2FarKSpaceMonitor,
        DiffractionMonitor,
    ] = None

    Er: PROJFIELDTYPE = pd.Field(
        ...,
        title="Ephi",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: PROJFIELDTYPE = pd.Field(
        ...,
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: PROJFIELDTYPE = pd.Field(
        ...,
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: PROJFIELDTYPE = pd.Field(
        ...,
        title="Hphi",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: PROJFIELDTYPE = pd.Field(
        ...,
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: PROJFIELDTYPE = pd.Field(
        ...,
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
    )

    medium: MediumType = pd.Field(
        Medium(),
        title="Background Medium",
        description="Background medium in which to radiate near fields to far fields.",
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
        return self.Etheta.f.values

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
        return self.Ntheta.dims

    def make_data_array(self, data: np.ndarray) -> xr.DataArray:
        """Make an xr.DataArray with data and same coords and dims as fields of self."""
        return xr.DataArray(data=data, coords=self.coords, dims=self.dims)

    def make_dataset(self, keys: Tuple[str, ...], vals: Tuple[np.ndarray, ...]) -> xr.Dataset:
        """Make an xr.Dataset with keys and data with same coords and dims as fields."""
        data_arrays = tuple(map(self.make_data_array, vals))
        return xr.Dataset(dict(zip(keys, data_arrays)))

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
        :class:`AbstractNear2FarMonitor`.

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
        :class:`AbstractNear2FarMonitor`.

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


class Near2FarAngleData(AbstractFieldProjectionData):
    """Data associated with a :class:`.Near2FarAngleMonitor`: components of projected fields.

    Example
    -------
    >>> from tidy3d import Near2FarAngleDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> r = np.atleast_1d(5)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> coords = dict(r=r, theta=theta, phi=phi, f=f)
    >>> values = (1+1j) * np.random.random((len(r), len(theta), len(phi), len(f)))
    >>> scalar_field = Near2FarAngleDataArray(values, coords=coords)
    >>> monitor = Near2FarAngleMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', phi=phi, theta=theta
    ...     )
    >>> data = Near2FarAngleData(
    ...     monitor=monitor, Er=scalar_field, Etheta=scalar_field, Ephi=scalar_field,
    ...     Hr=scalar_field, Htheta=scalar_field, Hphi=scalar_field
    ...     )
    """

    monitor: Near2FarAngleMonitor = None

    Er: Near2FarAngleDataArray = pd.Field(
        ...,
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: Near2FarAngleDataArray = pd.Field(
        ...,
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: Near2FarAngleDataArray = pd.Field(
        ...,
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: Near2FarAngleDataArray = pd.Field(
        ...,
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: Near2FarAngleDataArray = pd.Field(
        ...,
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: Near2FarAngleDataArray = pd.Field(
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

    def renormalize_fields(self, proj_distance: float):
        """Re-normalize stored fields to a new projection distance by applying a phase factor
        based on ``proj_distance``.

        Parameters
        ----------
        proj_distance : float = None
            (micron) new radial distance relative to the monitor's local origin.
        """
        # the phase factor associated with the old distance must be removed
        r = self.coords_spherical["r"][..., None]
        old_phase = self.propagation_phase(dist=r, k=self.k[None, None, None, :])

        # the phase factor associated with the new distance must be applied
        new_phase = self.propagation_phase(dist=proj_distance, k=self.k)

        # net phase
        phase = new_phase[None, None, None, :] / old_phase

        # compute updated fields and their coordinates
        for field in self.field_components.values():
            field.values *= phase
            field["r"] = np.atleast_1d(proj_distance)


class Near2FarCartesianData(AbstractFieldProjectionData):
    """Data associated with a :class:`.Near2FarCartesianMonitor`: components of projected fields.

    Example
    -------
    >>> from tidy3d import Near2FarCartesianDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> x = np.linspace(0, 5, 10)
    >>> y = np.linspace(0, 10, 20)
    >>> z = np.atleast_1d(5)
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(z), len(f)))
    >>> scalar_field = Near2FarCartesianDataArray(values, coords=coords)
    >>> monitor = Near2FarCartesianMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', x=x, y=y,
    ...     proj_axis=2, proj_distance=50
    ...     )
    >>> data = Near2FarCartesianData(
    ...     monitor=monitor, Er=scalar_field, Etheta=scalar_field, Ephi=scalar_field,
    ...     Hr=scalar_field, Htheta=scalar_field, Hphi=scalar_field
    ...     )
    """

    monitor: Near2FarCartesianMonitor

    Er: Near2FarCartesianDataArray = pd.Field(
        ...,
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: Near2FarCartesianDataArray = pd.Field(
        ...,
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: Near2FarCartesianDataArray = pd.Field(
        ...,
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: Near2FarCartesianDataArray = pd.Field(
        ...,
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: Near2FarCartesianDataArray = pd.Field(
        ...,
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: Near2FarCartesianDataArray = pd.Field(
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

    def renormalize_fields(self, proj_distance: float):
        """Re-normalize stored fields to a new projection distance by applying a phase factor
        based on ``proj_distance``.

        Parameters
        ----------
        proj_distance : float = None
            (micron) new plane distance relative to the monitor's local origin.
        """
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
        for field in self.field_components.values():
            field.values *= phase


class Near2FarKSpaceData(AbstractFieldProjectionData):
    """Data associated with a :class:`.Near2FarKSpaceMonitor`: components of projected fields.

    Example
    -------
    >>> from tidy3d import Near2FarKSpaceDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> ux = np.linspace(0, 0.4, 10)
    >>> uy = np.linspace(0, 0.6, 20)
    >>> r = np.atleast_1d(5)
    >>> coords = dict(ux=ux, uy=uy, r=r, f=f)
    >>> values = (1+1j) * np.random.random((len(ux), len(uy), len(r), len(f)))
    >>> scalar_field = Near2FarKSpaceDataArray(values, coords=coords)
    >>> monitor = Near2FarKSpaceMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', ux=ux, uy=uy, proj_axis=2
    ...     )
    >>> data = Near2FarKSpaceData(
    ...     monitor=monitor, Er=scalar_field, Etheta=scalar_field, Ephi=scalar_field,
    ...     Hr=scalar_field, Htheta=scalar_field, Hphi=scalar_field
    ...     )
    """

    monitor: Near2FarKSpaceMonitor = None

    Er: Near2FarKSpaceDataArray = pd.Field(
        ...,
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: Near2FarKSpaceDataArray = pd.Field(
        ...,
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: Near2FarKSpaceDataArray = pd.Field(
        ...,
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: Near2FarKSpaceDataArray = pd.Field(
        ...,
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: Near2FarKSpaceDataArray = pd.Field(
        ...,
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: Near2FarKSpaceDataArray = pd.Field(
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

    def renormalize_fields(self, proj_distance: float):
        """Re-normalize stored fields to a new projection distance by applying a phase factor
        based on ``proj_distance``.

        Parameters
        ----------
        proj_distance : float = None
            (micron) new radial distance relative to the monitor's local origin.
        """
        # the phase factor associated with the old distance must be removed
        r = self.coords_spherical["r"][..., None]
        old_phase = self.propagation_phase(dist=r, k=self.k[None, None, None, :])

        # the phase factor associated with the new distance must be applied
        new_phase = self.propagation_phase(dist=proj_distance, k=self.k)

        # net phase
        phase = new_phase[None, None, None, :] / old_phase

        # compute updated fields and their coordinates
        for field in self.field_components.values():
            field.values *= phase
            field["r"] = np.atleast_1d(proj_distance)


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
    ...     center=(1,2,3), size=(np.inf,np.inf,0), freqs=f, name='diffraction',
    ...     orders_x=orders_x, orders_y=orders_y
    ... )
    >>> data = DiffractionData(
    ...     monitor=monitor, sim_size=[1,1], bloch_vecs=[1,2],
    ...     Etheta=field, Ephi=field, Er=field,
    ...     Htheta=field, Hphi=field, Hr=field,
    ... )
    """

    monitor: DiffractionMonitor

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
    def shifted_orders(orders: Tuple[int], bloch_vec: float) -> np.ndarray:
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
        return np.atleast_1d(self.Etheta.orders_x.values)

    @property
    def orders_y(self) -> np.ndarray:
        """Allowed orders along y."""
        return np.atleast_1d(self.Etheta.orders_y.values)

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
    Near2FarKSpaceData,
    Near2FarCartesianData,
    Near2FarAngleData,
    DiffractionData,
)

MonitorDataType = Union[MonitorDataTypes]
