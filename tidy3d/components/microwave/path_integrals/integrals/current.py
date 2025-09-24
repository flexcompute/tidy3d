"""Current integral classes"""

from __future__ import annotations

from typing import Union

import numpy as np
import xarray as xr

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import (
    CurrentIntegralResultType,
    FreqDataArray,
    FreqModeDataArray,
    IntegralResultType,
    _make_current_data_array,
)
from tidy3d.components.data.monitor_data import FieldTimeData
from tidy3d.components.microwave.path_integrals.integrals.base import (
    AxisAlignedPathIntegral,
    Custom2DPathIntegral,
    IntegrableMonitorDataType,
)
from tidy3d.components.microwave.path_integrals.specs.current import (
    AxisAlignedCurrentIntegralSpec,
    CompositeCurrentIntegralSpec,
    Custom2DCurrentIntegralSpec,
)
from tidy3d.exceptions import DataError
from tidy3d.log import log


class AxisAlignedCurrentIntegral(AxisAlignedCurrentIntegralSpec):
    """Class for computing conduction current via Ampère's circuital law on an axis-aligned loop.

    Example
    -------
    >>> current = AxisAlignedCurrentIntegral(
    ...     center=(0, 0, 0),
    ...     size=(1, 1, 0),
    ...     sign="+",
    ...     extrapolate_to_endpoints=True,
    ...     snap_contour_to_grid=True,
    ... )
    """

    def compute_current(self, em_field: IntegrableMonitorDataType) -> CurrentIntegralResultType:
        """Compute current flowing in loop defined by the outer edge of a rectangle."""

        AxisAlignedPathIntegral._check_monitor_data_supported(em_field=em_field)
        ax1 = self.remaining_axes[0]
        ax2 = self.remaining_axes[1]
        h_component = "xyz"[ax1]
        v_component = "xyz"[ax2]
        h_field_name = f"H{h_component}"
        v_field_name = f"H{v_component}"
        # Validate that fields are present
        em_field._check_fields_stored([h_field_name, v_field_name])
        h_horizontal = em_field.field_components[h_field_name]
        h_vertical = em_field.field_components[v_field_name]

        # Decompose contour into path integrals
        (bottom, right, top, left) = self._to_path_integrals(h_horizontal, h_vertical)

        current = 0
        # Compute and add contributions from each part of the contour
        current += bottom.compute_integral(h_horizontal)
        current += right.compute_integral(h_vertical)
        current -= top.compute_integral(h_horizontal)
        current -= left.compute_integral(h_vertical)

        if self.sign == "-":
            current *= -1
        return _make_current_data_array(current)

    def _to_path_integrals(
        self, h_horizontal=None, h_vertical=None
    ) -> tuple[AxisAlignedPathIntegral, ...]:
        """Returns four ``AxisAlignedPathIntegral`` instances, which represent a contour
        integral around the surface defined by ``self.size``."""
        path_specs = self._to_path_integral_specs(h_horizontal=h_horizontal, h_vertical=h_vertical)
        path_integrals = tuple(
            AxisAlignedPathIntegral(**path_spec.dict(exclude={"type"})) for path_spec in path_specs
        )
        return path_integrals


class Custom2DCurrentIntegral(Custom2DPathIntegral, Custom2DCurrentIntegralSpec):
    """Class for computing conduction current via Ampère's circuital law on a custom path.
    To compute the current flowing in the positive ``axis`` direction, the vertices should be
    ordered in a counterclockwise direction.

    Example
    -------
    >>> import numpy as np
    >>> vertices = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [0, 0]])
    >>> current = Custom2DCurrentIntegral(
    ...     axis=2,
    ...     position=0.0,
    ...     vertices=vertices,
    ... )
    """

    def compute_current(self, em_field: IntegrableMonitorDataType) -> CurrentIntegralResultType:
        """Compute current flowing in a custom loop.

        Parameters
        ----------
        em_field : :class:`.IntegrableMonitorDataType`
            The electromagnetic field data that will be used for integrating.

        Returns
        -------
        :class:`.CurrentIntegralResultType`
            Result of current computation over remaining dimensions (frequency, time, mode indices).
        """

        AxisAlignedPathIntegral._check_monitor_data_supported(em_field=em_field)
        current = self.compute_integral(field="H", em_field=em_field)
        return _make_current_data_array(current)


class CompositeCurrentIntegral(CompositeCurrentIntegralSpec):
    """Current integral comprising one or more disjoint paths

    Example
    -------
    >>> spec1 = AxisAlignedCurrentIntegralSpec(center=(0, 0, 0), size=(1, 1, 0), sign="+")
    >>> spec2 = AxisAlignedCurrentIntegralSpec(center=(2, 0, 0), size=(1, 1, 0), sign="+")
    >>> composite = CompositeCurrentIntegral(path_specs=(spec1, spec2), sum_spec="sum")
    """

    @cached_property
    def current_integrals(
        self,
    ) -> tuple[Union[AxisAlignedCurrentIntegral, Custom2DCurrentIntegral], ...]:
        """ "Collection of closed current path integrals."""
        from tidy3d.components.microwave.path_integrals.factory import (
            make_current_integral,
        )

        current_integrals = [make_current_integral(path_spec) for path_spec in self.path_specs]
        return current_integrals

    def compute_current(self, em_field: IntegrableMonitorDataType) -> IntegralResultType:
        """Compute current flowing in loop defined by the outer edge of a rectangle."""
        if isinstance(em_field, FieldTimeData) and self.sum_spec == "split":
            raise DataError(
                "Only frequency domain field data is supported when using the 'split' sum_spec. "
                "Either switch the sum_spec to 'sum' or supply frequency domain data."
            )

        current_integrals = self.current_integrals

        # Calculate currents from each path integral and store in dataarray with path index dimension
        path_currents = []
        for path in current_integrals:
            term = path.compute_current(em_field)
            path_currents.append(term)

        # Stack all path currents along a new 'path_index' dimension
        path_currents_array = xr.concat(path_currents, dim="path_index")
        path_currents_array = path_currents_array.assign_coords(
            path_index=range(len(path_currents))
        )

        # Initialize output arrays with zeros
        first_term = path_currents[0]
        current_in_phase = xr.zeros_like(first_term)
        current_out_phase = xr.zeros_like(first_term)

        # Choose phase reference for each frequency using phase from current with largest magnitude
        path_magnitudes = np.abs(path_currents_array)
        max_magnitude_indices = path_magnitudes.argmax(dim="path_index")

        # Get the phase reference for each frequency from the path resulting in the largest magnitude current
        phase_reference = xr.zeros_like(first_term.angle)
        for freq_idx in range(len(first_term.f.values)):
            if hasattr(first_term, "mode_index"):
                max_path_indices = max_magnitude_indices.isel(f=freq_idx).values
                for mode_idx in range(len(first_term.mode_index.values)):
                    max_path_idx = max_path_indices[mode_idx]
                    phase_reference[freq_idx, mode_idx] = path_currents_array.isel(
                        path_index=max_path_idx, f=freq_idx, mode_index=mode_idx
                    ).angle.values
            else:
                max_path_idx = max_magnitude_indices.isel(f=freq_idx).values
                phase_reference[freq_idx] = path_currents_array.isel(
                    path_index=max_path_idx, f=freq_idx
                ).angle.values

        # Perform phase splitting into in and out of phase for each frequency separately
        for term in path_currents:
            if np.all(term.abs == 0):
                continue

            # Compare phase to reference for each frequency
            phase_diff = term.angle - phase_reference
            # Wrap phase difference to [-pi, pi]
            phase_diff.values = np.mod(phase_diff.values + np.pi, 2 * np.pi) - np.pi

            # Add to in-phase or out-of-phase current based on phase difference
            is_in_phase = np.abs(phase_diff) <= np.pi / 2
            current_in_phase += xr.where(is_in_phase, term, 0)
            current_out_phase += xr.where(~is_in_phase, term, 0)

        current_in_phase = _make_current_data_array(current_in_phase)
        current_out_phase = _make_current_data_array(current_out_phase)

        if self.sum_spec == "sum":
            current = current_in_phase + current_out_phase
        else:
            # For split mode, return the larger magnitude current
            current = xr.where(
                abs(current_in_phase) >= abs(current_out_phase), current_in_phase, current_out_phase
            )
            # Choose sign for current when using the split method.
            # We prefer both V and I to be positive
            current = xr.where(current.real >= 0.0, current, -current)
            current = _make_current_data_array(current)

        return current

    def _check_phase_sign_consistency(
        self,
        phase_difference: Union[FreqDataArray, FreqModeDataArray],
    ) -> bool:
        """
        Check that the provided current data has a consistent phase with respect to the reference
        phase. A consistent phase allows for the automatic identification of currents flowing in
        opposite directions. However, when the provided data does not correspond with a transmission
        line mode, this consistent phase condition will likely fail, so we emit a warning here to
        notify the user.
        """

        # Check phase consistency across frequencies
        freq_axis = phase_difference.get_axis_num("f")
        all_in_phase = np.all(abs(phase_difference) <= np.pi / 2, axis=freq_axis)
        all_out_of_phase = np.all(abs(phase_difference) > np.pi / 2, axis=freq_axis)
        consistent_phase = np.logical_or(all_in_phase, all_out_of_phase)

        if not np.all(consistent_phase) and self.sum_spec == "split":
            warning_msg = (
                "Phase alignment of computed current is not consistent across frequencies. "
                "The provided fields are not suitable for the 'split' method of computing current. "
                "Please provide the current path specifications manually."
            )

            if isinstance(phase_difference, FreqModeDataArray):
                inconsistent_modes = []
                mode_indices = phase_difference.mode_index.values
                for mode_idx in range(len(mode_indices)):
                    if not consistent_phase[mode_idx]:
                        inconsistent_modes.append(mode_idx)

                warning_msg += (
                    f" Modes with indices {inconsistent_modes} violated the phase consistency "
                    "requirement."
                )

            log.warning(warning_msg)

            return False
        return True

    def _check_phase_amplitude_consistency(
        self,
        current_in_phase: Union[FreqDataArray, FreqModeDataArray],
        current_out_phase: Union[FreqDataArray, FreqModeDataArray],
    ) -> bool:
        """
        Check that the summed in phase and out of phase components of current have a consistent relative amplitude.
        A consistent amplitude across frequencies allows for the automatic identification of the total conduction
        current flowing in the transmission line. If the amplitudes are not consistent, we emit a warning.
        """

        # For split mode, return the larger magnitude current
        freq_axis = current_in_phase.get_axis_num("f")
        in_all_larger = np.all(abs(current_in_phase) >= abs(current_out_phase), axis=freq_axis)
        in_all_smaller = np.all(abs(current_in_phase) < abs(current_out_phase), axis=freq_axis)
        consistent_max_current = np.logical_or(in_all_larger, in_all_smaller)
        if not np.all(consistent_max_current) and self.sum_spec == "split":
            warning_msg = (
                "There is not a consistently larger current across frequencies between the in-phase "
                "and out-of-phase components. The provided fields are not suitable for the "
                "'split' method of computing current. Please provide the current path "
                "specifications manually."
            )

            if isinstance(current_in_phase, FreqModeDataArray):
                inconsistent_modes = []
                mode_indices = current_in_phase.mode_index.values
                for mode_idx in range(len(mode_indices)):
                    if not consistent_max_current[mode_idx]:
                        inconsistent_modes.append(int(mode_indices[mode_idx]))

                warning_msg += (
                    f" Modes with indices {inconsistent_modes} violated the amplitude consistency "
                    "requirement."
                )

            log.warning(warning_msg)

            return False
        return True
