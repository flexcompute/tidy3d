"""Voltage integral classes"""

from __future__ import annotations

from tidy3d.components.data.data_array import (
    VoltageIntegralResultType,
    _make_voltage_data_array,
)
from tidy3d.components.microwave.path_integrals.integrals.base import (
    AxisAlignedPathIntegral,
    Custom2DPathIntegral,
    IntegrableMonitorDataType,
)
from tidy3d.components.microwave.path_integrals.specs.voltage import (
    AxisAlignedVoltageIntegralSpec,
    Custom2DVoltageIntegralSpec,
)


class AxisAlignedVoltageIntegral(AxisAlignedPathIntegral, AxisAlignedVoltageIntegralSpec):
    """Class for computing the voltage between two points defined by an axis-aligned line.

    Example
    -------
    >>> voltage = AxisAlignedVoltageIntegral(
    ...     center=(0, 0, 0),
    ...     size=(0, 0, 2),
    ...     sign="+",
    ...     extrapolate_to_endpoints=True,
    ...     snap_path_to_grid=True,
    ... )
    """

    def compute_voltage(self, em_field: IntegrableMonitorDataType) -> VoltageIntegralResultType:
        """Compute voltage along path defined by a line."""

        self._check_monitor_data_supported(em_field=em_field)
        e_component = "xyz"[self.main_axis]
        field_name = f"E{e_component}"
        # Validate that fields are present
        em_field._check_fields_stored([field_name])
        e_field = em_field.field_components[field_name]

        voltage = self.compute_integral(e_field)

        if self.sign == "+":
            voltage *= -1

        return _make_voltage_data_array(voltage)


class Custom2DVoltageIntegral(Custom2DPathIntegral, Custom2DVoltageIntegralSpec):
    """Class for computing the voltage between two points defined by a custom path.
    Computed voltage is :math:`V=V_b-V_a`, where position b is the final vertex in the supplied path.

    Notes
    -----

    Use :class:`.AxisAlignedVoltageIntegral` if possible, since interpolation
    near conductors will not be accurate.

    .. TODO Improve by including extrapolate_to_endpoints field, non-trivial extension.

    Example
    -------
    >>> import numpy as np
    >>> vertices = np.array([[0, 0], [0.5, 0.2], [1.0, 0.5]])
    >>> voltage = Custom2DVoltageIntegral(
    ...     axis=2,
    ...     position=0.0,
    ...     vertices=vertices,
    ... )
    """

    def compute_voltage(self, em_field: IntegrableMonitorDataType) -> VoltageIntegralResultType:
        """Compute voltage along path defined by a line.

        Parameters
        ----------
        em_field : :class:`.IntegrableMonitorDataType`
            The electromagnetic field data that will be used for integrating.

        Returns
        -------
        :class:`.VoltageIntegralResultType`
            Result of voltage computation over remaining dimensions (frequency, time, mode indices).
        """

        AxisAlignedPathIntegral._check_monitor_data_supported(em_field=em_field)
        voltage = -1.0 * self.compute_integral(field="E", em_field=em_field)
        return _make_voltage_data_array(voltage)
