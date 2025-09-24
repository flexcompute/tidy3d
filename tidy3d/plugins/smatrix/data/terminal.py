"""Data structures for post-processing terminal component simulations to calculate S-matrices."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.data.data_array import FreqDataArray, ModeAmpsDataArray
from tidy3d.components.data.monitor_data import ModeData, MonitorData
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.microwave.data.monitor_data import AntennaMetricsData
from tidy3d.constants import fp_eps
from tidy3d.exceptions import DataError
from tidy3d.log import log
from tidy3d.plugins.smatrix.component_modelers.terminal import TerminalComponentModeler
from tidy3d.plugins.smatrix.data.base import AbstractComponentModelerData
from tidy3d.plugins.smatrix.data.data_array import PortDataArray, TerminalPortDataArray
from tidy3d.plugins.smatrix.ports.types import TerminalPortType
from tidy3d.plugins.smatrix.ports.wave import WavePort
from tidy3d.plugins.smatrix.types import SParamDef
from tidy3d.plugins.smatrix.utils import (
    ab_to_s,
    check_port_impedance_sign,
    compute_F,
    compute_port_VI,
    compute_power_delivered_by_port,
    compute_power_wave_amplitudes,
    s_to_z,
)

# Minimum number of trusted frequency points required for polynomial fitting
MIN_NUM_TRUSTED_FREQUENCY_POINTS = 5


class MicrowaveSMatrixData(Tidy3dBaseModel):
    """Stores the computed S-matrix and reference impedances for the terminal ports."""

    port_reference_impedances: Optional[PortDataArray] = pd.Field(
        None,
        title="Port Reference Impedances",
        description="Reference impedance for each port used in the S-parameter calculation. This is optional and may not be present if not specified or computed.",
    )

    data: TerminalPortDataArray = pd.Field(
        ...,
        title="S-Matrix Data",
        description="An array containing the computed S-matrix of the device. The data is organized by terminal ports, representing the scattering parameters between them.",
    )

    s_param_def: SParamDef = pd.Field(
        "pseudo",
        title="Scattering Parameter Definition",
        description="Whether scattering parameters are defined using the 'pseudo' or 'power' wave definitions.",
    )


class LowFrequencySmoothingSpec(Tidy3dBaseModel):
    """Specifies the low frequency smoothing parameters for the terminal component simulation.
    The low frequency smoothing is performed by fitting a polynomial to the data in the trusted frequency range,
    defined by the minimum and maximum sampling times, and then using the polynomial to extrapolate
    the data outside of the trusted frequency range into lower frequencies.

    Example
    -------
    >>> low_freq_smoothing = LowFrequencySmoothingSpec(
    ...     min_sampling_time=3,
    ...     max_sampling_time=6,
    ...     order=1,
    ...     max_deviation=0.5,
    ... )
    """

    min_sampling_time: pd.NonNegativeFloat = pd.Field(
        1,
        title="Minimum Sampling Time (periods)",
        description="The minimum simulation time in periods of the corresponding frequency for which frequency domain results will be used to fit the polynomial for the low frequency extrapolation. "
        "Results below this threshold will be completely discarded.",
    )

    max_sampling_time: pd.NonNegativeFloat = pd.Field(
        5,
        title="Maximum Sampling Time (periods)",
        description="The maximum simulation time in periods of the corresponding frequency for which frequency domain results will be used to fit the polynomial for the low frequency extrapolation. "
        "Results above this threshold will be not be modified.",
    )

    order: int = pd.Field(
        1,
        title="Extrapolation Order",
        description="The order of the polynomial to use for the low frequency extrapolation.",
        ge=0,
        le=3,
    )

    max_deviation: Optional[float] = pd.Field(
        0.5,
        title="Maximum Deviation",
        description="The maximum deviation (in fraction of the trusted values) to allow for the low frequency smoothing.",
        ge=0,
    )

    @pd.root_validator
    def _validate_sampling_times(cls, values):
        min_sampling_time = values.get("min_sampling_time")
        max_sampling_time = values.get("max_sampling_time")
        if min_sampling_time is not None and max_sampling_time is not None:
            if min_sampling_time >= max_sampling_time:
                raise ValueError(
                    "The minimum sampling time must be less than the maximum sampling time."
                )
        return values

    def _smoothstep(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        NumPy vectorized smoothstep function using sine. Returns 0 for x <= a, 1 for x >= b, and a smooth sinusoidal transition in between.

        Parameters
        ----------
        x: np.ndarray
            The values to smooth.
        a: float
            The lower bound of the smoothstep.
        b: float
            The upper bound of the smoothstep.

        Returns
        -------
        np.ndarray
            The smoothed values.
        """
        if a == b:
            return 0.5 * np.ones_like(x)
        t = np.clip((x - a) / (b - a), 0, 1)
        return 0.5 * (1 - np.cos(np.pi * t))

    def _arctan_smooth_transition(
        self,
        values: np.ndarray,
        trusted_bound: float,
        constraint_bound: float,
    ) -> np.ndarray:
        """Apply arctan-based smooth transition from trusted_bound to constraint_bound.

        Parameters
        ----------
        values: np.ndarray
            The values to smooth.
        trusted_bound: float
            The trusted boundary value.
        constraint_bound: float
            The constraint boundary value.

        Returns
        -------
        np.ndarray
            The smoothed values using arctan transition.
        """
        # Use arctan for smooth transition from trusted_bound to constraint_bound
        # Scale the arctan to map [trusted_bound, constraint_bound] smoothly
        # The arctan provides smooth asymptotic approach to the limits
        x = (values - trusted_bound) / (constraint_bound - trusted_bound)
        # Arctan maps [0, inf] to [0, π/2], scale to [0, 1]
        smooth_factor = 2 * np.arctan(x * np.pi / 2) / np.pi
        return trusted_bound + smooth_factor * (constraint_bound - trusted_bound)

    def _smooth_constraint(
        self,
        values: np.ndarray,
        trusted_min: float,
        trusted_max: float,
        constraint_min: float,
        constraint_max: float,
    ) -> np.ndarray:
        """Smoothly constrain values using arctan for smooth clipping while preserving convexity.

        Parameters
        ----------
        values: np.ndarray
            The values to smooth.
        trusted_min: float
            The minimum value of the trusted range. Values below this will be smoothly clipped to the constraint_min.
        trusted_max: float
            The maximum value of the trusted range. Values above this will be smoothly clipped to the constraint_max.
        constraint_min: float
            The minimum value of the constraint range. This is the hard lower bound for the values.
        constraint_max: float
            The maximum value of the constraint range. This is the hard upper bound for the values.

        Returns
        -------
        np.ndarray
            The smoothed values.
        """
        if self.max_deviation is None:
            return values

        if constraint_min > trusted_min:
            raise ValueError("The constraint minimum must be less than the trusted minimum.")
        if constraint_max < trusted_max:
            print(constraint_max, trusted_max)
            raise ValueError("The constraint maximum must be greater than the trusted maximum.")
        if trusted_min > trusted_max:
            raise ValueError("The trusted minimum must be less than the trusted maximum.")

        result = values.copy()

        # Handle values below trusted range
        below_mask = values < trusted_min
        if np.any(below_mask):
            below_values = values[below_mask]
            result[below_mask] = self._arctan_smooth_transition(
                below_values, trusted_min, constraint_min
            )

        # Handle values above trusted range
        above_mask = values > trusted_max
        if np.any(above_mask):
            above_values = values[above_mask]
            result[above_mask] = self._arctan_smooth_transition(
                above_values, trusted_max, constraint_max
            )

        return result

    def _smooth_freq_data(
        self, data: ModeAmpsDataArray, run_time_actual: float
    ) -> ModeAmpsDataArray:
        """
        Smooth low frequency results in a ``ModeAmpsDataArray`` object.

        Parameters
        ----------
        data: :class:`.ModeAmpsDataArray`
            The mode amps data to smooth.
        run_time_actual: float
            The actual run time of the simulation.

        Returns
        -------
        :class:`.ModeAmpsDataArray`
            The smoothed mode amps data.
        """

        # get the indices of the data that are within the range from which we extrapolate the data
        selection = self._trusted_selection(data.f, run_time_actual)

        f_sel = data.f[selection]
        amps_sel = data.sel(f=f_sel).data

        # fit the data - handle complex y_data
        # Option 2: Fit magnitude and phase separately (often better for RF data)
        coeffs_mag = np.polyfit(f_sel, np.abs(amps_sel), self.order)

        # Unwrap phase to avoid jumps near ±π
        phase_data = np.angle(amps_sel)
        phase_unwrapped = np.unwrap(phase_data)
        coeffs_phase = np.polyfit(f_sel, phase_unwrapped, self.order)

        # evaluate the fit at all frequency points
        extrapolated_mag = np.polyval(coeffs_mag, data.f)
        extrapolated_phase_unwrapped = np.polyval(coeffs_phase, data.f)
        # Wrap phase back to [-π, π] range
        extrapolated_phase = np.angle(np.exp(1j * extrapolated_phase_unwrapped))

        # Smoothly constrain extrapolated magnitude to not deviate more than max_deviation from trusted region
        if self.max_deviation is not None:
            trusted_mag = np.abs(amps_sel)
            trusted_min = np.min(trusted_mag)
            trusted_max = np.max(trusted_mag)
            mag_min = trusted_min * max(1 - self.max_deviation, 0)
            mag_max = trusted_max * (1 + self.max_deviation)

            extrapolated_mag = self._smooth_constraint(
                extrapolated_mag, trusted_min, trusted_max, mag_min, mag_max
            )

        extrapolated = extrapolated_mag * np.exp(1j * extrapolated_phase)

        # blend the data with the extrapolated data
        # such that the original data is completely discarded for frequencies below min_periods
        # and is kept without any changes for frequencies above max_periods
        blending = self._smoothstep(
            data.f * run_time_actual, self.min_sampling_time, self.max_sampling_time
        )
        blended = data * blending + extrapolated * (1 - blending)

        return blended

    def _smooth_mode_data(self, mode_data: ModeData, run_time_actual: float) -> ModeData:
        """
        Smooth undersampled low frequency results in a ``ModeData`` object.

        Parameters
        ----------
        mode_data: :class:`.ModeData`
            The mode data to smooth.
        run_time_actual: float
            The actual run time of the simulation.

        Returns
        -------
        :class:`.ModeData`
            The smoothed mode data.
        """
        amps = mode_data.amps.copy()

        for direction in ["+", "-"]:
            for mode_index in amps.mode_index:
                amps_one = amps.sel(direction=direction, mode_index=mode_index, drop=True)
                amps_one = self._smooth_freq_data(amps_one, run_time_actual)
                amps.loc[{"direction": direction, "mode_index": mode_index}] = amps_one

        return mode_data.updated_copy(amps=amps)

    def _smooth_sim_data(
        self, sim_data: SimulationData, mode_monitors_to_smooth: list[str]
    ) -> SimulationData:
        """Smooth undersampled low frequency results in a ``SimulationData`` object.

        Parameters
        ----------
        sim_data: :class:`.SimulationData`
            The simulation data to smooth.
        mode_monitors_to_smooth: list[str]
            The names of the mode monitors to smooth.

        Returns
        -------
        :class:`.SimulationData`
            The smoothed simulation data.
        """
        run_time_actual = self._get_actual_run_time(sim_data)
        mnt_data_dict = sim_data.monitor_data
        for mode_monitor_name in mode_monitors_to_smooth:
            mnt_data_dict[mode_monitor_name] = self._smooth_mode_data(
                mnt_data_dict[mode_monitor_name], run_time_actual
            )
        return sim_data.updated_copy(data=list(mnt_data_dict.values()))

    @classmethod
    def _get_actual_run_time(cls, sim_data: SimulationData) -> float:
        """Get the actual run time of the simulation."""
        total_time_steps = sim_data.field_decay.t[-1]
        run_time_actual = sim_data.simulation.tmesh[total_time_steps]
        return run_time_actual

    def _trusted_selection(self, freqs: np.ndarray, run_time_actual: float) -> np.ndarray:
        """Get the indices of the data that are within the trusted sampling time range."""
        num_periods_passed = np.array(freqs) * run_time_actual
        return np.logical_and(
            num_periods_passed >= self.min_sampling_time - fp_eps,
            num_periods_passed <= self.max_sampling_time + fp_eps,
        )

    def _smooth_tcm_data(
        self, tcm_data: TerminalComponentModelerData
    ) -> TerminalComponentModelerData:
        """Smooth undersampled low frequency results in a ``TerminalComponentModelerData`` object.

        Parameters
        ----------
        tcm_data: :class:`.TerminalComponentModelerData`
            The terminal component modeler data to smooth.

        Returns
        -------
        :class:`.TerminalComponentModelerData`
            The smoothed terminal component modeler data.
        """
        new_sim_data = []
        mode_monitors_to_smooth = [
            port._mode_monitor_name for port in tcm_data.modeler.ports if isinstance(port, WavePort)
        ]

        if len(mode_monitors_to_smooth) == 0:
            return tcm_data

        for key, sim_data in tcm_data.data.items():
            skip_smoothing = False
            try:
                run_time_actual = self._get_actual_run_time(sim_data)
            except DataError:
                log.warning(
                    f"Could not get actual run time for simulation data '{key}'. Skipping smoothing."
                )
                skip_smoothing = True

            if not skip_smoothing:
                trusted_freq_indices = self._trusted_selection(
                    tcm_data.modeler.freqs, run_time_actual
                )

                if (
                    sum(trusted_freq_indices) <= MIN_NUM_TRUSTED_FREQUENCY_POINTS
                    and np.min(tcm_data.modeler.freqs) * run_time_actual < self.min_sampling_time
                ):
                    log.warning(
                        "Not enough data to fit a polynomial for low frequency extrapolation. Returning original data."
                    )
                else:
                    sim_data = self._smooth_sim_data(sim_data, mode_monitors_to_smooth)

            new_sim_data.append(sim_data)

        return tcm_data.updated_copy(data=tcm_data.data.updated_copy(values_tuple=new_sim_data))


DEFAULT_LOW_FREQUENCY_SMOOTHING_SPEC = LowFrequencySmoothingSpec()


class TerminalComponentModelerData(AbstractComponentModelerData):
    """
    Data associated with a :class:`.TerminalComponentModeler` simulation run.


    Notes
    -----

    This class serves as a data container for the results of a component modeler simulation,
    with the original simulation definition, and port simulation data, and the solver log.


    **References**

    .. [1]  R. B. Marks and D. F. Williams, "A general waveguide circuit theory,"
            J. Res. Natl. Inst. Stand. Technol., vol. 97, pp. 533, 1992.

    .. [2]  D. M. Pozar, Microwave Engineering, 4th ed. Hoboken, NJ, USA:
            John Wiley & Sons, 2012.
    """

    modeler: TerminalComponentModeler = pd.Field(
        ...,
        title="TerminalComponentModeler",
        description="The original :class:`.TerminalComponentModeler` object that defines the simulation setup "
        "and from which this data was generated.",
    )

    def smatrix(
        self,
        assume_ideal_excitation: Optional[bool] = None,
        s_param_def: Optional[SParamDef] = None,
        low_freq_smoothing: Optional[
            LowFrequencySmoothingSpec
        ] = DEFAULT_LOW_FREQUENCY_SMOOTHING_SPEC,
    ) -> MicrowaveSMatrixData:
        """Computes and returns the S-matrix and port reference impedances.

        Parameters
        ----------
        assume_ideal_excitation: If ``True``, assumes that exciting one port
            does not produce incident waves at other ports. This simplifies the
            S-matrix calculation and is required if not all ports are excited. If not
            provided, ``modeler.assume_ideal_excitation`` is used.
        s_param_def: The definition of S-parameters to use, determining whether
            "pseudo waves" or "power waves" are calculated. If not provided,
            ``modeler.s_param_def`` is used.
        low_freq_smoothing: The specification for low frequency smoothing.

        Returns
        -------
        :class:`.MicrowaveSMatrixData`
            Container with the computed S-parameters and the port reference impedances.
        """
        from tidy3d.plugins.smatrix.analysis.terminal import terminal_construct_smatrix

        if low_freq_smoothing is not None and any(
            isinstance(port, WavePort) for port in self.modeler.ports
        ):
            modeler_data = low_freq_smoothing._smooth_tcm_data(self)
        else:
            modeler_data = self

        terminal_port_data = terminal_construct_smatrix(
            modeler_data=modeler_data,
            assume_ideal_excitation=assume_ideal_excitation
            if (assume_ideal_excitation is not None)
            else self.modeler.assume_ideal_excitation,
            s_param_def=s_param_def if (s_param_def is not None) else self.modeler.s_param_def,
        )
        smatrix_data = MicrowaveSMatrixData(
            data=terminal_port_data,
            port_reference_impedances=modeler_data.port_reference_impedances,
            s_param_def=s_param_def if (s_param_def is not None) else self.modeler.s_param_def,
        )
        return smatrix_data

    @pd.root_validator(pre=False)
    def _warn_rf_license(cls, values):
        log.warning(
            "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
            log_once=True,
        )
        return values

    def _monitor_data_at_port_amplitude(
        self,
        port: TerminalPortType,
        monitor_name: str,
        a_port: Union[FreqDataArray, complex],
    ) -> MonitorData:
        """Normalize the monitor data to a desired complex amplitude of a port,
        represented by ``a_port``, where :math:`\frac{1}{2}|a|^2` is the power
        incident from the port into the system.
        """
        sim_data_port = self.data[self.modeler.get_task_name(port)]
        monitor_data = sim_data_port[monitor_name]
        a_raw, _ = self.compute_power_wave_amplitudes_at_each_port(sim_data=sim_data_port)
        a_raw_port = a_raw.sel(port=self.modeler.network_index(port))
        if not isinstance(a_port, FreqDataArray):
            freqs = list(monitor_data.monitor.freqs)
            array_vals = a_port * np.ones(len(freqs))
            a_port = FreqDataArray(array_vals, coords={"f": freqs})
        scale_array = a_port / a_raw_port
        return monitor_data.scale_fields_by_freq_array(scale_array, method="nearest")

    def get_antenna_metrics_data(
        self,
        port_amplitudes: Optional[dict[str, complex]] = None,
        monitor_name: Optional[str] = None,
    ) -> AntennaMetricsData:
        """Calculate antenna parameters using superposition of fields from multiple port excitations.

        The method computes the radiated far fields and port excitation power wave amplitudes
        for a superposition of port excitations, which can be used to analyze antenna radiation
        characteristics.

        Parameters
        ----------
        port_amplitudes : dict[str, complex]
            Dictionary mapping port names to their desired excitation amplitudes. For each port,
            :math:`\\frac{1}{2}|a|^2` represents the incident power from that port into the system.
            If None, uses only the first port without any scaling of the raw simulation data.  When ``None``
            is passed as a port amplitude, the raw simulation data is used for that port. Note that in this method ``a`` represents
            the incident wave amplitude using the power wave definition in [2].
        monitor_name : str
            Name of the :class:`.DirectivityMonitor` to use for calculating far fields.
            If None, uses the first monitor in `radiation_monitors`.

        Returns
        -------
        :class:`.AntennaMetricsData`
            Container with antenna parameters including directivity, gain, and radiation efficiency,
            computed from the superposition of fields from all excited ports.
        """
        from tidy3d.plugins.smatrix.analysis.antenna import get_antenna_metrics_data

        antenna_metrics_data = get_antenna_metrics_data(
            terminal_component_modeler_data=self,
            port_amplitudes=port_amplitudes,
            monitor_name=monitor_name,
        )
        return antenna_metrics_data

    @cached_property
    def port_reference_impedances(self) -> PortDataArray:
        """Calculates the reference impedance for each port across all frequencies.

        This function determines the characteristic impedance for every port defined
        in the modeler. It handles two types of ports differently: for a
        :class:`.WavePort`, the impedance is frequency-dependent and computed from
        modal properties, while for other types like :class:`.LumpedPort`, the
        impedance is a user-defined constant value.

        Returns:
            A data array containing the complex impedance for each port at each
            frequency.
        """
        from tidy3d.plugins.smatrix.analysis.terminal import port_reference_impedances

        return port_reference_impedances(self)

    def compute_wave_amplitudes_at_each_port(
        self,
        sim_data: SimulationData,
        port_reference_impedances: Optional[PortDataArray] = None,
        s_param_def: SParamDef = "pseudo",
    ) -> tuple[PortDataArray, PortDataArray]:
        """Compute the incident and reflected amplitudes at each port.
        The computed amplitudes have not been normalized.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Results from the simulation.
        port_reference_impedances : :class:`.PortDataArray`, optional
            Reference impedance at each port. If not provided, it is computed from the cached
            property :meth:`.port_reference_impedances`. Defaults to ``None``.
        s_param_def : SParamDef
            The type of waves computed, either pseudo waves defined by Equation 53 and Equation 54 in [1],
            or power waves defined by Equation 4.67 in [2].

        Returns
        -------
        tuple[:class:`.PortDataArray`, :class:`.PortDataArray`]
            Incident (a) and reflected (b) wave amplitudes at each port.
        """
        from tidy3d.plugins.smatrix.analysis.terminal import compute_wave_amplitudes_at_each_port

        port_reference_impedances_i = (
            port_reference_impedances
            if port_reference_impedances is not None
            else self.port_reference_impedances
        )

        return compute_wave_amplitudes_at_each_port(
            modeler=self.modeler,
            port_reference_impedances=port_reference_impedances_i,
            sim_data=sim_data,
            s_param_def=s_param_def,
        )

    def compute_power_wave_amplitudes_at_each_port(
        self,
        sim_data: SimulationData,
        port_reference_impedances: Optional[PortDataArray] = None,
    ) -> tuple[PortDataArray, PortDataArray]:
        """Compute the incident and reflected power wave amplitudes at each port.
        The computed amplitudes have not been normalized.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Results from the simulation.
        port_reference_impedances : :class:`.PortDataArray`, optional
            Reference impedance at each port. If not provided, it is computed from the cached
            property :meth:`.port_reference_impedances`. Defaults to ``None``.

        Returns
        -------
        tuple[:class:`.PortDataArray`, :class:`.PortDataArray`]
            Incident (a) and reflected (b) power wave amplitudes at each port.
        """
        from tidy3d.plugins.smatrix.analysis.terminal import (
            compute_power_wave_amplitudes_at_each_port,
        )

        port_reference_impedances_i = (
            port_reference_impedances
            if port_reference_impedances is not None
            else self.port_reference_impedances
        )

        return compute_power_wave_amplitudes_at_each_port(
            modeler=self.modeler,
            port_reference_impedances=port_reference_impedances_i,
            sim_data=sim_data,
        )

    def s_to_z(
        self,
        reference: Union[complex, PortDataArray],
        assume_ideal_excitation: Optional[bool] = None,
        s_param_def: SParamDef = "pseudo",
    ) -> TerminalPortDataArray:
        """Converts the S-matrix to the Z-matrix using a specified reference impedance.

        This method first computes the S-matrix of the device and then transforms it into the
        corresponding impedance matrix (Z-matrix). The conversion can be performed using either a
        single, uniform reference impedance for all ports or a more general set of per-port,
        frequency-dependent reference impedances.

        This method :meth:`.TerminalComponentModelerData.s_to_z` is called on a
        :class:`.TerminalComponentModelerData` object, which contains the S-matrix and other
        simulation data internally.

        Parameters
        ----------
        reference : Union[complex, :class:`.PortDataArray`]
            The reference impedance(s) to use for the conversion. If a single complex value is
            provided, it is assumed to be the reference impedance for all ports. If a
            :class:`.PortDataArray` is given, it should contain the specific reference
            impedance for each port.
        assume_ideal_excitation: If ``True``, assumes that exciting one port
            does not produce incident waves at other ports. This simplifies the
            S-matrix calculation and is required if not all ports are excited. If not
            provided, ``modeler.assume_ideal_excitation`` is used.
        s_param_def : SParamDef, optional
            The definition of the scattering parameters used in the S-matrix calculation.
            This can be either "pseudo" for pseudo waves (see [1]) or "power" for power
            waves (see [2]). Defaults to "pseudo".

        Returns
        -------
        DataArray
            The computed impedance (Z) matrix, with dimensions corresponding to the ports of
            the device.

        Examples
        --------
        >>> z_matrix = component_modeler_data.s_to_z(reference=50) # doctest: +SKIP
        >>> z_11 = z_matrix.sel(port_out="port_1@0", port_in="port_1@0") # doctest: +SKIP

        See Also
        --------
        smatrix : Computes the scattering matrix.
        """
        s_matrix = self.smatrix(
            assume_ideal_excitation=assume_ideal_excitation, s_param_def=s_param_def
        )
        return s_to_z(s_matrix=s_matrix.data, reference=reference, s_param_def=s_param_def)

    # Mirror Utils
    # So they can be reused elsewhere without a class reimport
    ab_to_s = staticmethod(ab_to_s)
    compute_F = staticmethod(compute_F)
    check_port_impedance_sign = staticmethod(check_port_impedance_sign)
    compute_port_VI = staticmethod(compute_port_VI)
    compute_power_wave_amplitudes = staticmethod(compute_power_wave_amplitudes)
    compute_power_delivered_by_port = staticmethod(compute_power_delivered_by_port)
