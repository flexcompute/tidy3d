from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np
import xarray as xr
from flexcompute.core._migration.em.numerical.raw import (
    source_normalization as source_normalization_numerics,
)
from pydantic import Field

from tidy3d.components.data.data_array import (
    FieldProjectionAngleDataArray,
    FluxDataArray,
    FreqDataArray,
    _TracedDataset,
)
from tidy3d.components.data.monitor_data._constants import AXIAL_RATIO_CAP
from tidy3d.components.monitor import (
    DirectivityMonitor,
    FieldProjectionAngleMonitor,
)

from .angle import FieldProjectionAngleData

if TYPE_CHECKING:
    from collections.abc import Callable

    from tidy3d.components.types import PolarizationBasis


class DirectivityData(FieldProjectionAngleData):
    """
    Data associated with a :class:`.DirectivityMonitor`.

    Example
    -------
    >>> from tidy3d import FluxDataArray, FieldProjectionAngleDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> r = np.atleast_1d(1e6)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> coords = dict(r=r, theta=theta, phi=phi, f=f)
    >>> coords_flux = dict(f=f)
    >>> values = (1+1j) * np.random.random((len(r), len(theta), len(phi), len(f)))
    >>> flux_data = FluxDataArray(np.random.random(len(f)), coords=coords_flux)
    >>> scalar_field = FieldProjectionAngleDataArray(values, coords=coords)
    >>> monitor = DirectivityMonitor(center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', phi=phi, theta=theta)
    >>> data = DirectivityData(monitor=monitor, flux=flux_data, Er=scalar_field, Etheta=scalar_field, Ephi=scalar_field,
    ...     Hr=scalar_field, Htheta=scalar_field, Hphi=scalar_field, projection_surfaces=monitor.projection_surfaces)
    """

    monitor: DirectivityMonitor = Field(
        title="Monitor",
        description="Monitor describing the angle-based projection grid on which to measure directivity data.",
    )

    flux: FluxDataArray = Field(
        title="Flux",
        description="Flux values that are either computed from fields recorded on the "
        "projection surfaces or by integrating the projected fields over a spherical surface.",
    )

    @staticmethod
    def from_spherical_field_dataset(
        monitor: DirectivityMonitor,
        field_dataset: xr.Dataset,
    ) -> DirectivityData:
        """Creates a :class:`.DirectivityData` instance from a spherical field dataset.

        Parameters
        ----------
        monitor : :class:`.DirectivityMonitor`
            Monitor defining measurement parameters.
        field_dataset : ``xr.Dataset``
            Dataset containing spherical field components (Er, Etheta, etc.).
            Must sample the entire spherical surface to compute flux correctly.

        Returns
        -------
        :class:`.DirectivityData`
            New :class:`.DirectivityData` instance with computed flux from spherical field integration.
        """
        field_dataset = _TracedDataset.from_dataset(field_dataset)
        f = list(monitor.freqs)
        flux = FluxDataArray(np.zeros(len(f)), coords={"f": f})
        dir_data = DirectivityData(
            monitor=monitor,
            flux=flux,
            Er=field_dataset.Er,
            Etheta=field_dataset.Etheta,
            Ephi=field_dataset.Ephi,
            Hr=field_dataset.Hr,
            Htheta=field_dataset.Htheta,
            Hphi=field_dataset.Hphi,
            projection_surfaces=monitor.projection_surfaces,
        )
        flux = dir_data.flux_from_projected_fields()
        return dir_data.updated_copy(flux=flux, deep=False, validate=False)

    def __add__(self, other: DirectivityData) -> DirectivityData:
        """Form the superposition of two :class:`.DirectivityData`. Flux is recomputed by
        integrating the projected fields over a sphere.

        Note
        ----
        Intended use is for combining fields from different simulations that were recorded
        using the same ``monitor``. The returned :class:`.DirectivityData` takes the ``monitor``
        from ``self``.
        """
        fields_dataset = self.fields_spherical + other.fields_spherical
        combined_data = DirectivityData.from_spherical_field_dataset(self.monitor, fields_dataset)
        return combined_data

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> DirectivityData:
        """
        Return a copy of self after normalization is applied using the source
        spectrum function, for both field components and flux data.
        """

        fields_norm = {}
        for field_name, field_data in self.field_components.items():
            src_amps = source_spectrum_fn(field_data.f)
            fields_norm[field_name] = source_normalization_numerics.normalize_frequency_component(
                field_data, src_amps
            )

        # Normalize flux
        source_freq_amps = source_spectrum_fn(self.flux.f)
        new_flux = source_normalization_numerics.normalize_flux(self.flux, source_freq_amps)

        return self.copy(deep=False, update=dict(fields_norm, flux=new_flux))

    @staticmethod
    def _check_valid_pol_basis(pol_basis: PolarizationBasis, tilt_angle: float) -> None:
        if pol_basis != "linear" and pol_basis != "circular":
            raise ValueError("'pol_basis' must be either 'linear' or 'circular'")
        if tilt_angle is not None and pol_basis == "circular":
            raise ValueError("'tilt_angle' is only defined for linear polarization.")

    def partial_radiation_intensity(
        self, pol_basis: PolarizationBasis = "linear", tilt_angle: float | None = None
    ) -> xr.Dataset:
        """Partial radiation intensity in the frequency domain as a function of angles theta and phi.
        The partial radiation intensities are computed in the ``linear`` or ``circular`` polarization
        bases. If ``tilt_angle`` is not ``None``, the radiation intensity is computed in the linear
        polarization basis rotated by ``tilt_angle`` from the theta-axis. Radiation intensity is
        measured in units of Watts per unit solid angle.

        Parameters
        ----------
        pol_basis : PolarizationBasis
            The desired polarization basis used to express partial radiation intensity, either
            ``linear`` or ``circular``.
        tilt_angle : float
            The angle by which the co-polar vector is rotated from the theta-axis.
            At ``tilt_angle`` = 0, the co-polar vector coincides with the theta-axis and the cross-polar
            vector coincides with the phi-axis; while at ``tilt_angle = pi/2``, the co-polar vector
            coincides with the phi-axis.

        Returns
        -------
        xarray.Dataset
            Dataset containing the partial radiation intensities split into the two polarization states.
        """
        self._check_valid_pol_basis(pol_basis, tilt_angle)
        if pol_basis == "linear":
            if tilt_angle is not None:
                tilt_fields = self.fields_linear_polarization_tilted(tilt_angle)
                E1 = tilt_fields.Eco
                E2 = tilt_fields.Ecross
                H1 = tilt_fields.Hco
                H2 = tilt_fields.Hcross
                keys = ("Uco", "Ucross")
            else:
                E1 = self.Etheta
                E2 = self.Ephi
                H1 = self.Htheta
                H2 = self.Hphi
                keys = ("Utheta", "Uphi")
        else:
            E1 = self.fields_circular_polarization.Eright
            E2 = self.fields_circular_polarization.Eleft
            # needs extra -1 to counteract -1 in cross product below
            H1 = -1.0 * self.fields_circular_polarization.Hleft
            H2 = self.fields_circular_polarization.Hright
            keys = ("Uright", "Uleft")

        U_1 = (self.monitor.proj_distance**2) * 0.5 * np.real(E1 * np.conj(H2))
        U_2 = (self.monitor.proj_distance**2) * 0.5 * np.real(-E2 * np.conj(H1))

        data_arrays = (U_1, U_2)
        return _TracedDataset(dict(zip(keys, data_arrays)))

    @property
    def radiation_intensity(self) -> FieldProjectionAngleDataArray:
        """Radiation intensity in the frequency domain as a function of angles theta and phi.
        Radiation intensity is measured in units of Watts per unit solid angle.
        """
        # Calls partial radiation intensity using default linear polarization basis
        partial_U = self.partial_radiation_intensity()
        return partial_U.Utheta + partial_U.Uphi

    @property
    def radiated_power(self) -> FreqDataArray:
        """Total radiated power in the frequency domain with units of Watts."""
        # If this data was created using FieldProjectionAngleData, the sign
        # will already be correct. Also will be correct if monitor size is all nonzero.
        # TODO fix this sign issue in the backend if possible
        if (
            isinstance(self.monitor, FieldProjectionAngleMonitor)
            or self.monitor.size.count(0.0) == 0
        ):
            return FreqDataArray(self.flux.values, {"f": self.f})
        # The monitor could be planar and directed downward
        sign = 1.0 if self.monitor.normal_dir == "+" else -1.0
        return FreqDataArray(sign * self.flux.values, {"f": self.f})

    def partial_directivity(
        self, pol_basis: PolarizationBasis = "linear", tilt_angle: float | None = None
    ) -> xr.Dataset:
        """Directivity in the frequency domain as a function of angles theta and phi.
        The partial directivities are computed in the ``linear`` or ``circular`` polarization
        bases. If ``tilt_angle`` is not ``None``, the radiation intensity is computed in the linear
        polarization basis rotated by ``tilt_angle`` from the theta-axis. Directivity is a dimensionless
        quantity defined as the ratio of the radiation intensity in a given direction to the average
        radiation intensity over all directions.

        Parameters
        ----------
        pol_basis : PolarizationBasis
            The desired polarization basis used to express partial directivity, either
            ``linear`` or ``circular``.
        tilt_angle : float
            The angle by which the co-polar vector is rotated from the theta-axis.
            At ``tilt_angle`` = 0, the co-polar vector coincides with the theta-axis and the cross-polar
            vector coincides with the phi-axis; while at ``tilt_angle = pi/2``, the co-polar vector
            coincides with the phi-axis.

        Returns
        -------
        ``xarray.Dataset``
            Dataset containing the partial directivities split into the two polarization states.
        """
        self._check_valid_pol_basis(pol_basis, tilt_angle)
        if pol_basis == "linear":
            if tilt_angle is None:
                rename_mapping = {"Utheta": "Dtheta", "Uphi": "Dphi"}
            else:
                rename_mapping = {"Uco": "Dco", "Ucross": "Dcross"}
        else:
            rename_mapping = {"Uright": "Dright", "Uleft": "Dleft"}
        # Average radiation intensity is total radiated power divided by 4 pi
        avg_radiation_intensity = self.radiated_power / (4 * np.pi)
        partial_U = self.partial_radiation_intensity(pol_basis=pol_basis, tilt_angle=tilt_angle)
        partial_D = partial_U / avg_radiation_intensity
        return _TracedDataset.from_dataset(partial_D.rename(rename_mapping))

    @property
    def directivity(self) -> FieldProjectionAngleDataArray:
        """Directivity in the frequency domain as a function of angles theta and phi.
        Directivity is a dimensionless quantity defined as the ratio of the radiation
        intensity in a given direction to the average radiation intensity over all directions.
        """
        # Calls partial directivity using default linear polarization basis
        partial_D = self.partial_directivity()
        return FieldProjectionAngleDataArray(partial_D.Dtheta + partial_D.Dphi)

    def calc_radiation_efficiency(self, power_in: FreqDataArray) -> FreqDataArray:
        """Calculate radiation efficiency as the ratio of radiated power to input power.

        Parameters
        ----------
        power_in : FreqDataArray
            Power supplied to the radiating element in the frequency domain, in units of Watts.

        Returns
        -------
        FreqDataArray
            Radiation efficiency (dimensionless) in the frequency domain, computed as
            radiated_power / power_in.
        """
        return FreqDataArray((self.radiated_power / power_in).values, {"f": self.f})

    def calc_partial_gain(
        self,
        power_in: FreqDataArray,
        pol_basis: PolarizationBasis = "linear",
        tilt_angle: float | None = None,
    ) -> xr.Dataset:
        """The partial gain figures of merit for antennas. The partial gains are computed
        in the ``linear`` or ``circular`` polarization bases. If ``tilt_angle`` is not ``None``,
        the partial directivity is computed in the linear polarization basis rotated by ``tilt_angle``
        from the theta-axis. Gain is dimensionless.

        Parameters
        ----------
        power_in : FreqDataArray
            Power, in units of Watts, supplied to the radiating element in the frequency domain.

        pol_basis : PolarizationBasis
            The desired polarization basis used to express partial gain, either
            ``linear`` or ``circular``.

        tilt_angle : float
            The angle by which the co-polar vector is rotated from the theta-axis.
            At ``tilt_angle`` = 0, the co-polar vector coincides with the theta-axis and the cross-polar
            vector coincides with the phi-axis; while at ``tilt_angle = pi/2``, the co-polar vector
            coincides with the phi-axis.

        Returns
        -------
        ``xarray.Dataset``
            Dataset containing the partial gains split into the two polarization states.
        """
        self._check_valid_pol_basis(pol_basis, tilt_angle)
        radiation_efficiency = self.calc_radiation_efficiency(power_in)
        partial_D = self.partial_directivity(pol_basis=pol_basis, tilt_angle=tilt_angle)
        partial_G = radiation_efficiency * partial_D
        if pol_basis == "linear":
            if tilt_angle is None:
                rename_mapping = {"Dtheta": "Gtheta", "Dphi": "Gphi"}
            else:
                rename_mapping = {"Dco": "Gco", "Dcross": "Gcross"}
        else:
            rename_mapping = {"Dright": "Gright", "Dleft": "Gleft"}
        return partial_G.rename(rename_mapping)

    def calc_gain(self, power_in: FreqDataArray) -> FieldProjectionAngleDataArray:
        """The gain figure of merit for antennas. Gain is dimensionless.

        Parameters
        ----------
        power_in : FreqDataArray
            Power, in units of Watts, supplied to the radiating element in the frequency domain.
        """
        partial_G = self.calc_partial_gain(power_in)
        return FieldProjectionAngleDataArray(partial_G.Gtheta + partial_G.Gphi)

    @property
    def axial_ratio(self) -> FieldProjectionAngleDataArray:
        """Axial Ratio (AR) in the frequency domain as a function of angles theta and phi.
        AR is a dimensionless quantity defined as the ratio of the major axis to the minor
        axis of the polarization ellipse.

        Note
        ----
        The axial ratio computation is based on:

        Balanis, Constantine A., "Antenna Theory: Analysis and Design,"
        John Wiley & Sons, Chapter 2.12 (2016).
        """

        # Axial ratio calculations based on equations (2-65) to (2-67)
        # from Balanis, Constantine A., "Antenna Theory: Analysis and Design,"
        # John Wiley & Sons, 2016.
        #
        # The standard formula computes AR_denominator = (A + B) - |C| where
        # A = |Etheta|², B = |Ephi|², C = Etheta² + Ephi². For near-linear
        # polarization |C| ≈ A + B, causing catastrophic cancellation.
        #
        # Instead, we use the identity (A+B)² - |C|² = 4*(ad - bc)² where
        # a,b = Re,Im(Etheta) and c,d = Re,Im(Ephi). This "cross" term
        # avoids the subtraction entirely.
        cross = self.Etheta.real * self.Ephi.imag - self.Etheta.imag * self.Ephi.real

        AR_numerator = (
            np.abs(self.Etheta) ** 2
            + np.abs(self.Ephi) ** 2
            + np.abs(self.Etheta**2 + self.Ephi**2)
        )

        inds_zero = AR_numerator == 0
        axial_ratio_inverse = xr.zeros_like(AR_numerator)
        axial_ratio_inverse = axial_ratio_inverse.where(inds_zero, 2 * np.abs(cross) / AR_numerator)

        # Safety-net cap for the degenerate case of zero total field
        axial_ratio_inverse = axial_ratio_inverse.where(
            axial_ratio_inverse >= 1 / AXIAL_RATIO_CAP, 1 / AXIAL_RATIO_CAP
        )

        return 1 / axial_ratio_inverse

    @property
    def left_polarization(self) -> FieldProjectionAngleDataArray:
        """Electric far field for left-hand circular polarization
        (counterclockwise component) with an angle-based projection grid.
        """
        return self.fields_circular_polarization.Eleft

    @property
    def right_polarization(self) -> FieldProjectionAngleDataArray:
        """Electric far field for right-hand circular polarization
        (clockwise component) with an angle-based projection grid.
        """
        return self.fields_circular_polarization.Eright

    def fields_linear_polarization_tilted(self, tilt_angle: float) -> xr.Dataset:
        """Electric and magnetic fields in the linear polarization basis that is rotated
        at the pole of the radiation sphere by `tilt_angle`.

        Parameters
        ----------
        tilt_angle : float
            The angle by which the co-polar vector is rotated from the theta-axis.
            At ``tilt_angle`` = 0, the co-polar vector coincides with the theta-axis and the cross-polar
            vector coincides with the phi-axis; while at ``tilt_angle = pi/2``, the co-polar vector
            coincides with the phi-axis.

        Returns
        -------
        ``xarray.Dataset``
            Dataset containing (``Eco``, ``Ecross``, ``Hco``, ``Hcross``)
        """
        Eco = np.cos(tilt_angle) * self.Etheta + np.sin(tilt_angle) * self.Ephi
        Ecross = -np.sin(tilt_angle) * self.Etheta + np.cos(tilt_angle) * self.Ephi
        Hco = np.cos(tilt_angle) * self.Htheta + np.sin(tilt_angle) * self.Hphi
        Hcross = -np.sin(tilt_angle) * self.Htheta + np.cos(tilt_angle) * self.Hphi

        keys = ("Eco", "Ecross", "Hco", "Hcross")
        data_arrays = (Eco, Ecross, Hco, Hcross)
        return _TracedDataset(dict(zip(keys, data_arrays)))

    @property
    def fields_circular_polarization(self) -> xr.Dataset:
        """Electric and magnetic fields in the circular polarization basis.

        Note
        ----
        Uses IEEE handedness convention for polarization state, which means right-handed circularly
        polarization is associated with a clockwise rotation of the electric field vector from the
        point of the view of the source. However, we use the physics convention for time evolution
        of time-harmonic fields, which modifies the computation when compared to engineering references.

        Returns
        -------
        ``xarray.Dataset``
            xarray dataset containing (``Eleft``, ``Eright``, ``Hleft``, ``Hright``)
            in Spherical coordinates.
        """
        Eleft = (self.Etheta + 1j * self.Ephi) / np.sqrt(2.0)
        Eright = (self.Etheta - 1j * self.Ephi) / np.sqrt(2.0)
        Hleft = (self.Hphi - 1j * self.Htheta) / np.sqrt(2.0)
        Hright = (self.Hphi + 1j * self.Htheta) / np.sqrt(2.0)

        keys = ("Eleft", "Eright", "Hleft", "Hright")
        data_arrays = (Eleft, Eright, Hleft, Hright)
        return _TracedDataset(dict(zip(keys, data_arrays)))
