"""Helper classes for performing custom path integrals with fields on the Yee grid"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import xarray as xr

from tidy3d.components.data.data_array import FreqDataArray, FreqModeDataArray
from tidy3d.components.data.monitor_data import FieldTimeData
from tidy3d.components.geometry.base import Geometry
from tidy3d.components.microwave.path_integrals.base_spec import CustomPathIntegral2DSpec
from tidy3d.components.microwave.path_integrals.current_spec import (
    CompositeCurrentIntegralSpec,
    CustomCurrentIntegral2DSpec,
)
from tidy3d.components.microwave.path_integrals.voltage_spec import CustomVoltageIntegral2DSpec
from tidy3d.components.types import Axis, Coordinate
from tidy3d.exceptions import DataError
from tidy3d.log import log

from .path_integrals import (
    AxisAlignedPathIntegral,
    CurrentIntegralResultTypes,
    IntegralResultTypes,
    MonitorDataTypes,
    VoltageIntegralResultTypes,
    _make_base_result_data_array,
    _make_current_data_array,
    _make_voltage_data_array,
)

FieldParameter = Literal["E", "H"]


class CustomPathIntegral2D(CustomPathIntegral2DSpec):
    """Class for defining a custom path integral defined as a curve on an axis-aligned plane.

    Notes
    -----

    Given a set of vertices :math:`\\vec{r}_i`, this class approximates path integrals over
    vector fields of the form :math:`\\int{\\vec{F} \\cdot \\vec{dl}}`
    as :math:`\\sum_i{\\vec{F}(\\vec{r}_i) \\cdot \\vec{dl}_i}`,
    where the differential length :math:`\\vec{dl}` is approximated using central differences
    :math:`\\vec{dl}_i = \\frac{\\vec{r}_{i+1} - \\vec{r}_{i-1}}{2}`.
    If the path is not closed, forward and backward differences are used at the endpoints.
    """

    def compute_integral(
        self, field: FieldParameter, em_field: MonitorDataTypes
    ) -> IntegralResultTypes:
        """Computes the path integral defined by ``vertices`` given the input ``em_field``.

        Parameters
        ----------
        field : :class:`.FieldParameter`
            Can take the value of ``"E"`` or ``"H"``. Determines whether to perform the integral
            over electric or magnetic field.
        em_field : :class:`.MonitorDataTypes`
            The electromagnetic field data that will be used for integrating.

        Returns
        -------
        :class:`.IntegralResultTypes`
            Result of integral over remaining dimensions (frequency, time, mode indices).
        """

        (dim1, dim2, dim3) = self.local_dims

        h_field_name = f"{field}{dim1}"
        v_field_name = f"{field}{dim2}"

        # Validate that fields are present
        em_field._check_fields_stored([h_field_name, v_field_name])

        # Select fields lying on the plane
        plane_indexer = {dim3: self.position}
        field1 = em_field.field_components[h_field_name].sel(plane_indexer, method="nearest")
        field2 = em_field.field_components[v_field_name].sel(plane_indexer, method="nearest")

        # Although for users we use the convention that an axis is simply `popped`
        # internally we prefer a right-handed coordinate system where dimensions
        # keep a proper order. The only change is to swap 'x' and 'z' when the
        # normal axis is along  `y`
        # Dim 's' represents the parameterization of the line
        # 't' is likely used for time
        if self.main_axis == 1:
            x_path = xr.DataArray(self.vertices[:, 1], dims="s")
            y_path = xr.DataArray(self.vertices[:, 0], dims="s")
        else:
            x_path = xr.DataArray(self.vertices[:, 0], dims="s")
            y_path = xr.DataArray(self.vertices[:, 1], dims="s")

        path_indexer = {dim1: x_path, dim2: y_path}
        field1_interp = field1.interp(path_indexer, method="linear")
        field2_interp = field2.interp(path_indexer, method="linear")

        # Determine the differential length elements along the path
        dl_x = self._compute_dl_component(x_path, self.is_closed_contour)
        dl_y = self._compute_dl_component(y_path, self.is_closed_contour)
        dl_x = xr.DataArray(dl_x, dims="s")
        dl_y = xr.DataArray(dl_y, dims="s")

        # Compute the dot product between differential length element and vector field
        integrand = field1_interp * dl_x + field2_interp * dl_y
        # Integrate along the path
        result = integrand.integrate(coord="s")
        result = result.reset_coords(drop=True)
        return _make_base_result_data_array(result)

    @staticmethod
    def _compute_dl_component(coord_array: xr.DataArray, closed_contour=False) -> np.array:
        """Computes the differential length element along the integration path."""
        dl = np.gradient(coord_array)
        if closed_contour:
            # If the contour is closed, we can use central difference on the starting/end point
            # which will be more accurate than the default forward/backward choice in np.gradient
            grad_end = np.gradient([coord_array[-2], coord_array[0], coord_array[1]])
            dl[0] = dl[-1] = grad_end[1]
        return dl

    @classmethod
    def from_circular_path(
        cls, center: Coordinate, radius: float, num_points: int, normal_axis: Axis, clockwise: bool
    ) -> CustomPathIntegral2D:
        """Creates a ``CustomPathIntegral2D`` from a circular path given a desired number of points
        along the perimeter.

        Parameters
        ----------
        center : Coordinate
            The center of the circle.
        radius : float
            The radius of the circle.
        num_points : int
            THe number of equidistant points to use along the perimeter of the circle.
        normal_axis : Axis
            The axis normal to the defined circle.
        clockwise : bool
            When ``True``, the points will be ordered clockwise with respect to the positive
            direction of the ``normal_axis``.

        Returns
        -------
        :class:`.CustomPathIntegral2D`
            A path integral defined on a circular path.
        """

        def generate_circle_coordinates(radius: float, num_points: int, clockwise: bool):
            """Helper for generating x,y vertices around a circle in the local coordinate frame."""
            sign = 1.0
            if clockwise:
                sign = -1.0
            angles = np.linspace(0, sign * 2 * np.pi, num_points, endpoint=True)
            xt = radius * np.cos(angles)
            yt = radius * np.sin(angles)
            return (xt, yt)

        # Get transverse axes
        normal_center, trans_center = Geometry.pop_axis(center, normal_axis)

        # These x,y coordinates in the local coordinate frame
        if normal_axis == 1:
            # Handle special case when y is the axis that is popped
            clockwise = not clockwise
        xt, yt = generate_circle_coordinates(radius, num_points, clockwise)
        xt += trans_center[0]
        yt += trans_center[1]
        circle_vertices = np.column_stack((xt, yt))
        # Close the contour exactly
        circle_vertices[-1, :] = circle_vertices[0, :]
        return cls(axis=normal_axis, position=normal_center, vertices=circle_vertices)


class CustomVoltageIntegral2D(CustomPathIntegral2D, CustomVoltageIntegral2DSpec):
    """Class for computing the voltage between two points defined by a custom path.
    Computed voltage is :math:`V=V_b-V_a`, where position b is the final vertex in the supplied path.

    Notes
    -----

    Use :class:`.VoltageIntegralAxisAligned` if possible, since interpolation
    near conductors will not be accurate.

    .. TODO Improve by including extrapolate_to_endpoints field, non-trivial extension."""

    def compute_voltage(self, em_field: MonitorDataTypes) -> VoltageIntegralResultTypes:
        """Compute voltage along path defined by a line.

        Parameters
        ----------
        em_field : :class:`.MonitorDataTypes`
            The electromagnetic field data that will be used for integrating.

        Returns
        -------
        :class:`.VoltageIntegralResultTypes`
            Result of voltage computation over remaining dimensions (frequency, time, mode indices).
        """

        AxisAlignedPathIntegral._check_monitor_data_supported(em_field=em_field)
        voltage = -1.0 * self.compute_integral(field="E", em_field=em_field)
        return _make_voltage_data_array(voltage)


class CustomCurrentIntegral2D(CustomPathIntegral2D, CustomCurrentIntegral2DSpec):
    """Class for computing conduction current via Ampère's circuital law on a custom path.
    To compute the current flowing in the positive ``axis`` direction, the vertices should be
    ordered in a counterclockwise direction."""

    def compute_current(self, em_field: MonitorDataTypes) -> CurrentIntegralResultTypes:
        """Compute current flowing in a custom loop.

        Parameters
        ----------
        em_field : :class:`.MonitorDataTypes`
            The electromagnetic field data that will be used for integrating.

        Returns
        -------
        :class:`.CurrentIntegralResultTypes`
            Result of current computation over remaining dimensions (frequency, time, mode indices).
        """

        AxisAlignedPathIntegral._check_monitor_data_supported(em_field=em_field)
        current = self.compute_integral(field="H", em_field=em_field)
        return _make_current_data_array(current)


class CompositeCurrentIntegral(CompositeCurrentIntegralSpec):
    """Current integral comprising one or more disjoint paths"""

    def compute_current(self, em_field: MonitorDataTypes) -> IntegralResultTypes:
        """Compute current flowing in loop defined by the outer edge of a rectangle."""
        if isinstance(em_field, FieldTimeData) and self.sum_spec == "split":
            raise DataError(
                "Only frequency domain field data is supported when using the 'split' sum_spec. "
                "Either switch the sum_spec to 'sum' or supply frequency domain data."
            )

        from tidy3d.components.microwave.path_integrals.path_integral_factory import (
            make_current_integral,
        )

        current_integrals = [make_current_integral(path_spec) for path_spec in self.path_specs]

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
            return current_in_phase + current_out_phase

        # Check amplitude consistency across frequencies
        self._check_phase_amplitude_consistency(current_in_phase, current_out_phase)

        # For split mode, return the larger magnitude current
        current = xr.where(
            abs(current_in_phase) >= abs(current_out_phase), current_in_phase, current_out_phase
        )
        return _make_current_data_array(current)

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
