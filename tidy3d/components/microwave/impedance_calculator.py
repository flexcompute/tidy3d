"""Class for computing characteristic impedance of transmission lines."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.data.data_array import (
    CurrentIntegralResultType,
    ImpedanceResultType,
    VoltageIntegralResultType,
    _make_current_data_array,
    _make_impedance_data_array,
    _make_voltage_data_array,
)
from tidy3d.components.data.monitor_data import FieldTimeData
from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.components.microwave.path_integrals.integrals.base import (
    AxisAlignedPathIntegral,
    IntegrableMonitorDataType,
)
from tidy3d.components.microwave.path_integrals.integrals.current import (
    AxisAlignedCurrentIntegral,
    CompositeCurrentIntegral,
    Custom2DCurrentIntegral,
)
from tidy3d.components.microwave.path_integrals.integrals.voltage import (
    AxisAlignedVoltageIntegral,
    Custom2DVoltageIntegral,
)
from tidy3d.components.monitor import ModeMonitor, ModeSolverMonitor
from tidy3d.exceptions import ValidationError

VoltageIntegralType = Union[AxisAlignedVoltageIntegral, Custom2DVoltageIntegral]
CurrentIntegralType = Union[
    AxisAlignedCurrentIntegral, Custom2DCurrentIntegral, CompositeCurrentIntegral
]


class ImpedanceCalculator(MicrowaveBaseModel):
    """Tool for computing the characteristic impedance of a transmission line.

    Example
    -------
    Create a calculator with both voltage and current path integrals defined.

    >>> v_int = AxisAlignedVoltageIntegral(
    ...     center=(0, 0, 0),
    ...     size=(0, 0, 2),
    ...     sign="+",
    ...     extrapolate_to_endpoints=True,
    ...     snap_path_to_grid=True,
    ... )
    >>> i_int = AxisAlignedCurrentIntegral(
    ...     center=(0, 0, 0),
    ...     size=(1, 1, 0),
    ...     sign="+",
    ...     extrapolate_to_endpoints=True,
    ...     snap_contour_to_grid=True,
    ... )
    >>> calc = ImpedanceCalculator(voltage_integral=v_int, current_integral=i_int)

    You can also define only one of the integrals. At least one is required.

    >>> _ = ImpedanceCalculator(voltage_integral=v_int)
    """

    voltage_integral: Optional[VoltageIntegralType] = pd.Field(
        None,
        title="Voltage Integral",
        description="Definition of path integral for computing voltage.",
    )

    current_integral: Optional[CurrentIntegralType] = pd.Field(
        None,
        title="Current Integral",
        description="Definition of contour integral for computing current.",
    )

    def compute_impedance(
        self, em_field: IntegrableMonitorDataType, return_voltage_and_current=False
    ) -> Union[
        ImpedanceResultType,
        tuple[ImpedanceResultType, VoltageIntegralResultType, CurrentIntegralResultType],
    ]:
        """Compute impedance for the supplied ``em_field`` using ``voltage_integral`` and
        ``current_integral``. If only a single integral has been defined, impedance is
        computed using the total flux in ``em_field``.

        Parameters
        ----------
        em_field : :class:`.IntegrableMonitorDataType`
            The electromagnetic field data that will be used for computing the characteristic
            impedance.
        return_voltage_and_current: bool
            When ``True``, returns additional :class:`.IntegralResultType` that represent the voltage
            and current associated with the supplied fields.

        Returns
        -------
        :class:`.IntegralResultType` or tuple[VoltageIntegralResultType, CurrentIntegralResultType, ImpedanceResultType]
            If ``return_voltage_and_current=False``, single result of impedance computation
            over remaining dimensions (frequency, time, mode indices). If ``return_voltage_and_current=True``,
            tuple of (impedance, voltage, current).
        """

        AxisAlignedPathIntegral._check_monitor_data_supported(em_field=em_field)

        voltage = None
        current = None
        # If both voltage and current integrals have been defined then impedance is computed directly
        if self.voltage_integral is not None:
            voltage = self.voltage_integral.compute_voltage(em_field)
        if self.current_integral is not None:
            current = self.current_integral.compute_current(em_field)

        # If only one of the integrals has been provided, then the computation falls back to using
        # total power (flux) with Ohm's law to compute the missing quantity. The input field should
        # cover an area large enough to render the flux computation accurate. If the input field is
        # a time signal, then it is real and flux corresponds to the instantaneous power. Otherwise
        # the input field is in frequency domain, where flux indicates the time-averaged power
        # 0.5*Re(V*conj(I)).
        # We explicitly take the real part, in case Bloch BCs were used in the simulation.
        flux_sign = 1.0
        # Determine flux sign
        if isinstance(em_field.monitor, ModeSolverMonitor):
            flux_sign = 1 if em_field.monitor.direction == "+" else -1
        if isinstance(em_field.monitor, ModeMonitor):
            flux_sign = 1 if em_field.monitor.store_fields_direction == "+" else -1

        if self.voltage_integral is None:
            flux = flux_sign * em_field.complex_flux
            if isinstance(em_field, FieldTimeData):
                impedance = flux / np.real(current) ** 2
            else:
                impedance = 2 * flux / (current * np.conj(current))
        elif self.current_integral is None:
            flux = flux_sign * em_field.complex_flux
            if isinstance(em_field, FieldTimeData):
                impedance = np.real(voltage) ** 2 / flux
            else:
                impedance = (voltage * np.conj(voltage)) / (2 * np.conj(flux))
        else:
            if isinstance(em_field, FieldTimeData):
                impedance = np.real(voltage) / np.real(current)
            else:
                impedance = voltage / current
        impedance = _make_impedance_data_array(impedance)
        if return_voltage_and_current:
            if voltage is None:
                voltage = _make_voltage_data_array(impedance * current)
            if current is None:
                current = _make_current_data_array(voltage / impedance)
            return (impedance, voltage, current)
        return impedance

    @pd.validator("current_integral", always=True)
    def check_voltage_or_current(cls, val, values):
        """Raise validation error if both ``voltage_integral`` and ``current_integral``
        are not provided."""
        if not values.get("voltage_integral") and not val:
            raise ValidationError(
                "At least one of 'voltage_integral' or 'current_integral' must be provided."
            )
        return val
