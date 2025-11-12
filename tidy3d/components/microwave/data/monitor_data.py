"""Post-processing data and figures of merit for antennas, including radiation efficiency,
reflection efficiency, gain, and realized gain.
"""

from __future__ import annotations

from typing import Literal, Optional

import pydantic.v1 as pd
import xarray as xr

from tidy3d.components.data.data_array import FieldProjectionAngleDataArray, FreqDataArray
from tidy3d.components.data.monitor_data import DirectivityData, ModeData, ModeSolverData
from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.components.microwave.data.dataset import TransmissionLineDataset
from tidy3d.components.microwave.monitor import MicrowaveModeMonitor, MicrowaveModeSolverMonitor
from tidy3d.components.types import FreqArray, PolarizationBasis


class AntennaMetricsData(DirectivityData, MicrowaveBaseModel):
    """Data representing the main parameters and figures of merit for antennas.

    Example
    -------
    >>> import numpy as np
    >>> from tidy3d.components.data.monitor_data import FluxDataArray, FieldProjectionAngleDataArray
    >>> from tidy3d.components.monitor import DirectivityMonitor
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> r = np.atleast_1d(1e6)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> coords = dict(r=r, theta=theta, phi=phi, f=f)
    >>> coords_flux = dict(f=f)
    >>> field_values = (1+1j) * np.random.random((len(r), len(theta), len(phi), len(f)))
    >>> flux_data = FluxDataArray(np.random.random(len(f)), coords=coords_flux)
    >>> scalar_field = FieldProjectionAngleDataArray(field_values, coords=coords)
    >>> monitor = DirectivityMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     freqs=f,
    ...     name="rad_monitor",
    ...     phi=phi,
    ...     theta=theta
    ... )
    >>> power_data = FreqDataArray(np.random.random(len(f)), coords=coords_flux)
    >>> data = AntennaMetricsData(
    ...     monitor=monitor,
    ...     projection_surfaces=monitor.projection_surfaces,
    ...     flux=flux_data,
    ...     Er=scalar_field,
    ...     Etheta=scalar_field,
    ...     Ephi=scalar_field,
    ...     Hr=scalar_field,
    ...     Htheta=scalar_field,
    ...     Hphi=scalar_field,
    ...     power_incident=power_data,
    ...     power_reflected=power_data
    ... )

    Notes
    -----
    The definitions of radiation efficiency, reflection efficiency, gain, and realized gain
    are based on:

    Balanis, Constantine A., "Antenna Theory: Analysis and Design,"
    John Wiley & Sons, Chapter 2.9 (2016).
    """

    power_incident: FreqDataArray = pd.Field(
        ...,
        title="Power incident",
        description="Array of values representing the incident power to an antenna.",
    )

    power_reflected: FreqDataArray = pd.Field(
        ...,
        title="Power reflected",
        description="Array of values representing power reflected due to an impedance mismatch with the antenna.",
    )

    @staticmethod
    def from_directivity_data(
        dir_data: DirectivityData, power_inc: FreqDataArray, power_refl: FreqDataArray
    ) -> AntennaMetricsData:
        """Create :class:`.AntennaMetricsData` from directivity data and power measurements.

        Parameters
        ----------
        dir_data : :class:`.DirectivityData`
            Directivity data containing field components and flux measurements.
        power_inc : :class:`.FreqDataArray`
            Array of values representing the incident power to an antenna.
        power_refl : :class:`.FreqDataArray`
            Array of values representing power reflected due to impedance mismatch with the antenna.

        Returns
        -------
        :class:`.AntennaMetricsData`
            New instance combining directivity data with incident and reflected power measurements.
        """
        antenna_params_dict = {
            **dir_data.dict(),
            "power_incident": power_inc,
            "power_reflected": power_refl,
        }
        antenna_params_dict.pop("type")
        return AntennaMetricsData(**antenna_params_dict)

    @property
    def supplied_power(self) -> FreqDataArray:
        """The power supplied to the antenna, which takes into account reflections."""
        return self.power_incident - self.power_reflected

    @property
    def radiation_efficiency(self) -> FreqDataArray:
        """The radiation efficiency of the antenna."""
        return self.calc_radiation_efficiency(self.supplied_power)

    @property
    def reflection_efficiency(self) -> FreqDataArray:
        """The reflection efficiency of the antenna, which is due to an impedance mismatch."""
        reflection_efficiency = self.supplied_power / self.power_incident
        return reflection_efficiency

    def partial_gain(
        self, pol_basis: PolarizationBasis = "linear", tilt_angle: Optional[float] = None
    ) -> xr.Dataset:
        """The partial gain figures of merit for antennas. The partial gains are computed
        in the ``linear`` or ``circular`` polarization bases. If ``tilt_angle`` is not ``None``,
        the partial directivity is computed in the linear polarization basis rotated by ``tilt_angle``
        from the theta-axis. Gain is dimensionless.

        Parameters
        ----------
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
        partial_D = self.partial_directivity(pol_basis=pol_basis, tilt_angle=tilt_angle)
        if pol_basis == "linear":
            if tilt_angle is None:
                rename_mapping = {"Dtheta": "Gtheta", "Dphi": "Gphi"}
            else:
                rename_mapping = {"Dco": "Gco", "Dcross": "Gcross"}
        else:
            rename_mapping = {"Dright": "Gright", "Dleft": "Gleft"}
        return self.radiation_efficiency * partial_D.rename(rename_mapping)

    @property
    def gain(self) -> FieldProjectionAngleDataArray:
        """The gain figure of merit for antennas. Gain is dimensionless."""
        partial_G = self.partial_gain()
        return partial_G.Gtheta + partial_G.Gphi

    def partial_realized_gain(
        self, pol_basis: PolarizationBasis = "linear", tilt_angle: Optional[float] = None
    ) -> xr.Dataset:
        """The partial realized gain figures of merit for antennas. The partial gains are computed
        in the ``linear`` or ``circular`` polarization bases. If ``tilt_angle`` is not ``None``,
        the partial directivity is computed in the linear polarization basis rotated by ``tilt_angle``
        from the theta-axis. Gain is dimensionless.

        Parameters
        ----------
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
            Dataset containing the partial realized gains split into the two polarization states.
        """
        self._check_valid_pol_basis(pol_basis, tilt_angle)
        reflection_efficiency = self.reflection_efficiency
        partial_G = self.partial_gain(pol_basis=pol_basis, tilt_angle=tilt_angle)
        return reflection_efficiency * partial_G

    @property
    def realized_gain(self) -> FieldProjectionAngleDataArray:
        """The realized gain figure of merit for antennas. Realized gain is dimensionless."""
        partial_G = self.partial_realized_gain()
        return partial_G.Gtheta + partial_G.Gphi


class MicrowaveModeData(ModeData, MicrowaveBaseModel):
    """
    Data associated with a :class:`.ModeMonitor` for microwave and RF applications: modal amplitudes,
    propagation indices, mode profiles, and transmission line data.

    Notes
    -----

        This class extends :class:`.ModeData` with additional microwave-specific data including
        characteristic impedance, voltage coefficients, and current coefficients. The data is
        stored as `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        objects using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

        The microwave mode data contains all the information from :class:`.ModeData` plus additional
        microwave dataset with impedance calculations performed using voltage and current line integrals
        as specified in the :class:`.MicrowaveModeSpec`.

    Example
    -------
    >>> import tidy3d as td
    >>> import numpy as np
    >>> from tidy3d.components.data.data_array import (
    ...     CurrentFreqModeDataArray,
    ...     ImpedanceFreqModeDataArray,
    ...     ModeAmpsDataArray,
    ...     ModeIndexDataArray,
    ...     VoltageFreqModeDataArray,
    ... )
    >>> from tidy3d.components.microwave.data.dataset import TransmissionLineDataset
    >>> direction = ["+", "-"]
    >>> f = [1e14, 2e14, 3e14]
    >>> mode_index = np.arange(3)
    >>> index_coords = dict(f=f, mode_index=mode_index)
    >>> index_data = ModeIndexDataArray((1+1j) * np.random.random((3, 3)), coords=index_coords)
    >>> amp_coords = dict(direction=direction, f=f, mode_index=mode_index)
    >>> amp_data = ModeAmpsDataArray((1+1j) * np.random.random((2, 3, 3)), coords=amp_coords)
    >>> impedance_data = ImpedanceFreqModeDataArray(50 * np.ones((3, 3)), coords=index_coords)
    >>> voltage_data = VoltageFreqModeDataArray((1+1j) * np.random.random((3, 3)), coords=index_coords)
    >>> current_data = CurrentFreqModeDataArray((0.02+0.01j) * np.random.random((3, 3)), coords=index_coords)
    >>> tl_data = TransmissionLineDataset(
    ...     Z0=impedance_data,
    ...     voltage_coeffs=voltage_data,
    ...     current_coeffs=current_data
    ... )
    >>> monitor = td.MicrowaveModeMonitor(
    ...    center=(0, 0, 0),
    ...    size=(2, 0, 6),
    ...    freqs=[2e14, 3e14],
    ...    mode_spec=td.MicrowaveModeSpec(num_modes=3, impedance_specs=td.AutoImpedanceSpec()),
    ...    name='microwave_mode',
    ... )
    >>> data = MicrowaveModeData(
    ...     monitor=monitor,
    ...     amps=amp_data,
    ...     n_complex=index_data,
    ...     transmission_line_data=tl_data
    ... )
    """

    monitor: MicrowaveModeMonitor = pd.Field(
        ..., title="Monitor", description="Mode monitor associated with the data."
    )

    transmission_line_data: Optional[TransmissionLineDataset] = pd.Field(
        None,
        title="Transmission Line Data",
        description="Additional data relevant to transmission lines in RF and microwave applications, "
        "like characteristic impedance. This field is populated when a :class:`MicrowaveModeSpec` has "
        "been used to set up the monitor or mode solver.",
    )

    @property
    def modes_info(self) -> xr.Dataset:
        """Dataset collecting various properties of the stored modes."""
        super_info = super().modes_info
        if self.transmission_line_data is not None:
            super_info["Re(Z0)"] = self.transmission_line_data.Z0.real
            super_info["Im(Z0)"] = self.transmission_line_data.Z0.imag
        return super_info

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
        super_data = super()._group_index_post_process(frequency_step)
        if self.transmission_line_data is not None:
            _, center_inds, _ = self._group_index_freq_slices()
            update_dict = {
                "Z0": self.transmission_line_data.Z0.isel(f=center_inds),
                "voltage_coeffs": self.transmission_line_data.voltage_coeffs.isel(f=center_inds),
                "current_coeffs": self.transmission_line_data.current_coeffs.isel(f=center_inds),
            }
            super_data = super_data.updated_copy(**update_dict, path="transmission_line_data")
        return super_data

    def _apply_mode_reorder(self, sort_inds_2d):
        """Apply a mode reordering along mode_index for all frequency indices.

        Parameters
        ----------
        sort_inds_2d : np.ndarray
            Array of shape (num_freqs, num_modes) where each row is the
            permutation to apply to the mode_index for that frequency.
        """
        main_data_reordered = super()._apply_mode_reorder(sort_inds_2d)
        if self.transmission_line_data is not None:
            transmission_line_data_reordered = self.transmission_line_data._apply_mode_reorder(
                sort_inds_2d
            )
            main_data_reordered = main_data_reordered.updated_copy(
                transmission_line_data=transmission_line_data_reordered
            )
        return main_data_reordered


class MicrowaveModeSolverData(ModeSolverData, MicrowaveModeData):
    """
    Data associated with a :class:`.ModeSolverMonitor` for microwave and RF applications: scalar components
    of E and H fields plus characteristic impedance data.

    Notes
    -----

        This class extends :class:`.ModeSolverData` with additional microwave-specific data including
        characteristic impedance, voltage coefficients, and current coefficients. The data is
        stored as `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        objects using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

        The microwave mode solver data contains all field components (Ex, Ey, Ez, Hx, Hy, Hz) and
        effective indices from :class:`.ModeSolverData`, plus impedance calculations performed using
        voltage and current line integrals as specified in the :class:`.MicrowaveModeSpec`.

    Example
    -------
    >>> import tidy3d as td
    >>> import numpy as np
    >>> from tidy3d import Grid, Coords
    >>> from tidy3d.components.data.data_array import (
    ...     CurrentFreqModeDataArray,
    ...     ImpedanceFreqModeDataArray,
    ...     ScalarModeFieldDataArray,
    ...     ModeIndexDataArray,
    ...     VoltageFreqModeDataArray,
    ... )
    >>> from tidy3d.components.microwave.data.dataset import TransmissionLineDataset
    >>> x = [-1, 1, 3]
    >>> y = [-2, 0]
    >>> z = [-3, -1, 1, 3, 5]
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(3)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> field_coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], f=f, mode_index=mode_index)
    >>> field = ScalarModeFieldDataArray((1+1j)*np.random.random((2,1,4,2,3)), coords=field_coords)
    >>> index_coords = dict(f=f, mode_index=mode_index)
    >>> index_data = ModeIndexDataArray((1+1j) * np.random.random((2,3)), coords=index_coords)
    >>> impedance_data = ImpedanceFreqModeDataArray(50 * np.ones((2, 3)), coords=index_coords)
    >>> voltage_data = VoltageFreqModeDataArray((1+1j) * np.random.random((2, 3)), coords=index_coords)
    >>> current_data = CurrentFreqModeDataArray((0.02+0.01j) * np.random.random((2, 3)), coords=index_coords)
    >>> tl_data = TransmissionLineDataset(
    ...     Z0=impedance_data,
    ...     voltage_coeffs=voltage_data,
    ...     current_coeffs=current_data
    ... )
    >>> monitor = td.MicrowaveModeSolverMonitor(
    ...    center=(0, 0, 0),
    ...    size=(2, 0, 6),
    ...    freqs=[2e14, 3e14],
    ...    mode_spec=td.MicrowaveModeSpec(num_modes=3, impedance_specs=td.AutoImpedanceSpec()),
    ...    name='microwave_mode_solver',
    ... )
    >>> data = MicrowaveModeSolverData(
    ...     monitor=monitor,
    ...     Ex=field,
    ...     Ey=field,
    ...     Ez=field,
    ...     Hx=field,
    ...     Hy=field,
    ...     Hz=field,
    ...     n_complex=index_data,
    ...     grid_expanded=grid,
    ...     transmission_line_data=tl_data
    ... )
    """

    monitor: MicrowaveModeSolverMonitor = pd.Field(
        ..., title="Monitor", description="Mode monitor associated with the data."
    )

    def interp_in_freq(
        self,
        freqs: FreqArray,
        method: Literal["linear", "cubic", "poly"] = "linear",
        renormalize: bool = True,
        recalculate_grid_correction: bool = True,
        assume_sorted: bool = False,
    ) -> MicrowaveModeData:
        """Interpolate mode data to new frequency points.

        Interpolates all stored mode data (effective indices, field components, group indices,
        and dispersion) from the current frequency grid to a new set of frequencies. This is
        useful for obtaining mode data at many frequencies from computations at fewer frequencies,
        when modes vary smoothly with frequency.

        Parameters
        ----------
        freqs : FreqArray
            New frequency points to interpolate to. Should generally span a similar range
            as the original frequencies to avoid extrapolation.
        method : Literal["linear", "cubic", "cheb"]
            Interpolation method. ``"linear"`` for linear interpolation (requires 2+ source
            frequencies), ``"cubic"`` for cubic spline interpolation (requires 4+ source
            frequencies), ``"cheb"`` for Chebyshev polynomial interpolation using barycentric
            formula (requires 3+ source frequencies at Chebyshev nodes).
            For complex-valued data, real and imaginary parts are interpolated independently.
        renormalize : Optional[bool] = True
            Whether to renormalize the mode profiles to unity power after interpolation.
        recalculate_grid_correction : bool = True
            Whether to recalculate the grid correction factors after interpolation or use interpolated
            grid corrections.
        assume_sorted: bool = False,
            Whether to assume the frequency points are sorted.

        Returns
        -------
        ModeSolverData
            New :class:`ModeSolverData` object with data interpolated to the requested frequencies.

        Raises
        ------
        DataError
            If interpolation parameters are invalid (e.g., too few source frequencies for the
            chosen method, or source frequencies not at Chebyshev nodes for 'cheb' method).

        Note
        ----
            Interpolation assumes modes vary smoothly with frequency. Results may be inaccurate
            near mode crossings or regions of rapid mode variation. Use frequency tracking
            (``mode_spec.sort_spec.track_freq``) to help maintain mode ordering consistency.

            For Chebyshev interpolation, source frequencies must be at Chebyshev nodes of the
            second kind within the frequency range.

        Example
        -------
        >>> # Compute modes at 5 frequencies
        >>> import numpy as np
        >>> freqs_sparse = np.linspace(1e14, 2e14, 5)
        >>> # ... create mode_solver and compute modes ...
        >>> # mode_data = mode_solver.solve()
        >>> # Interpolate to 50 frequencies
        >>> freqs_dense = np.linspace(1e14, 2e14, 50)
        >>> # mode_data_interp = mode_data.interp(freqs=freqs_dense, method='linear')
        """
        main_data_interp = super().interp_in_freq(
            freqs, method, renormalize, recalculate_grid_correction, assume_sorted
        )
        if self.transmission_line_data is not None:
            update_dict = self.transmission_line_data._interp_in_freq_update_dict(
                freqs, method, assume_sorted
            )
            transmission_line_data_interp = self.transmission_line_data.updated_copy(**update_dict)
            main_data_interp = main_data_interp.updated_copy(
                transmission_line_data=transmission_line_data_interp
            )
        return main_data_interp
