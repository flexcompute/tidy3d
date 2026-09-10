from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np
from flexcompute.core._migration.em.numerical.raw import (
    source_normalization as source_normalization_numerics,
)
from pydantic import Field

from tidy3d.components.base import TYPE_TAG_STR
from tidy3d.components.data.data_array import DataArray, _TracedDataset
from tidy3d.components.data.monitor_data.base import MonitorData
from tidy3d.components.medium import (
    Medium,
    MediumType,
)
from tidy3d.constants import (
    C_0,
    ETA_0,
)
from tidy3d.exceptions import SetupError

from ._types import ProjFieldType, ProjMonitorType

if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr

    from tidy3d.components.source.current import CustomCurrentSource, PointDipole


class AbstractFieldProjectionData(MonitorData):
    """Collection of projected fields in spherical coordinates in the frequency domain."""

    monitor: ProjMonitorType = Field(
        title="Projection monitor",
        description="Field projection monitor.",
        discriminator=TYPE_TAG_STR,
    )

    Er: ProjFieldType = Field(
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: ProjFieldType = Field(
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: ProjFieldType = Field(
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: ProjFieldType = Field(
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: ProjFieldType = Field(
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: ProjFieldType = Field(
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
    )

    medium: MediumType = Field(
        default_factory=Medium,
        title="Background Medium",
        description="Background medium through which to project fields.",
        discriminator=TYPE_TAG_STR,
    )

    is_2d_simulation: bool = Field(
        default=False,
        title="2D Simulation",
        description="Indicates whether the monitor data is for a 2D simulation.",
    )

    @property
    def field_components(self) -> dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return {
            "Er": self.Er,
            "Etheta": self.Etheta,
            "Ephi": self.Ephi,
            "Hr": self.Hr,
            "Htheta": self.Htheta,
            "Hphi": self.Hphi,
        }

    @property
    def f(self) -> np.ndarray:
        """Frequencies."""
        return np.array(self.Etheta.coords["f"])

    @property
    def coords(self) -> dict[str, np.ndarray]:
        """Coordinates of the fields contained."""
        return self.Etheta.coords

    @property
    def coords_spherical(self) -> dict[str, np.ndarray]:
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
    def dims(self) -> tuple[str, ...]:
        """Dimensions of the radiation vectors contained."""
        return self.Etheta.dims

    def make_data_array(self, data: np.ndarray) -> DataArray:
        """Make an DataArray with data and same coords and dims as fields of self."""
        return DataArray(data=data, coords=self.coords, dims=self.dims)

    def make_dataset(self, keys: tuple[str, ...], vals: tuple[np.ndarray, ...]) -> xr.Dataset:
        """Make a dataset with keys and data with same coords and dims as fields.

        Using _TracedDataset preserves tidy3d's custom DataArray subclass when items are
        accessed later, which is required for autograd-safe .values handling.
        """
        data_arrays = tuple(map(self.make_data_array, vals))
        return _TracedDataset(dict(zip(keys, data_arrays)))

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
            fields_norm[field_name] = source_normalization_numerics.normalize_frequency_component(
                field_data, src_amps
            )

        return self.copy(deep=False, update=fields_norm)

    @staticmethod
    def wavenumber(medium: MediumType, frequency: float) -> complex:
        """Complex valued wavenumber associated with a frequency."""
        index_n, index_k = medium.nk_model(frequency=frequency)
        return (2 * np.pi * frequency / C_0) * (index_n + 1j * index_k)

    @property
    def nk(self) -> tuple[float, float]:
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
    def propagation_factor(dist: float | None, k: complex, is_2d_simulation: bool) -> complex:
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
        :class:`~tidy3d.components.monitor.AbstractFieldProjectionMonitor`.

        Returns
        -------
        xarray.Dataset
            xarray-backed dataset containing
            (``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``)
            in spherical coordinates. Accessing items returns tidy3d DataArrays.
        """
        return self.make_dataset(
            keys=self.field_components.keys(), vals=self.field_components.values()
        )

    @property
    def fields_cartesian(self) -> xr.Dataset:
        """Get all field components in Cartesian coordinates relative to the monitor's
        local origin for all projection grid points and frequencies specified in the
        :class:`~tidy3d.components.monitor.AbstractFieldProjectionMonitor`.

        Returns
        -------
        xarray.Dataset
            xarray-backed dataset containing (``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, ``Hz``)
            in Cartesian coordinates. Accessing items returns tidy3d DataArrays.
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
        # Stack traced tuples before concatenation to avoid creating object arrays.
        e_data = np.stack(e_data, axis=0)
        h_data = np.stack(h_data, axis=0)
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

        n_leading = max(0, len(self.dims) - 1)
        expand_idx = (None,) * n_leading + (Ellipsis,)
        k = self.k[expand_idx]
        eta = self.eta[expand_idx]

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

    def _make_adjoint_sources(
        self, dataset_names: list[str], fwidth: float
    ) -> list[CustomCurrentSource | PointDipole]:
        """Error if server-side field projection is used for autograd"""

        raise NotImplementedError(
            "Adjoint is currently not implemented for server-side field projections. "
            "To compute derivatives with respect to field projection data, please use a 'FieldMonitor' "
            "and use a local projection in your objective function via 'FieldProjector.from_near_field_monitors'. "
            "Using field projection monitors directly is not supported as the full field information is required "
            "to construct the adjoint source for this problem. The field projection data does not contain the "
            "information necessary for gradient computation."
        )
