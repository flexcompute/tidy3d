"""Class for computing characteristic impedance of transmission lines."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pydantic.v1 as pd

from ...components.base import Tidy3dBaseModel
from ...components.data.monitor_data import FieldData, FieldTimeData, ModeSolverData
from ...exceptions import DataError, ValidationError
from .custom_path_integrals import CustomCurrentIntegral2D, CustomVoltageIntegral2D
from .path_integrals import (
    CurrentIntegralAxisAligned,
    IntegralResultTypes,
    MonitorDataTypes,
    VoltageIntegralAxisAligned,
)

VoltageIntegralTypes = Union[VoltageIntegralAxisAligned, CustomVoltageIntegral2D]
CurrentIntegralTypes = Union[CurrentIntegralAxisAligned, CustomCurrentIntegral2D]


class ImpedanceCalculator(Tidy3dBaseModel):
    """Tool for computing the characteristic impedance of a transmission line."""

    voltage_integral: Optional[VoltageIntegralTypes] = pd.Field(
        None,
        title="Voltage Integral",
        description="Definition of path integral for computing voltage.",
    )

    current_integral: Optional[CurrentIntegralTypes] = pd.Field(
        None,
        title="Current Integral",
        description="Definition of contour integral for computing current.",
    )

    def compute_impedance(self, em_field: MonitorDataTypes) -> IntegralResultTypes:
        """Compute impedance for the supplied ``em_field`` using ``voltage_integral`` and
        ``current_integral``. If only a single integral has been defined, impedance is
        computed using the total flux in ``em_field``.

        Parameters
        ----------
        em_field : :class:`.MonitorDataTypes`
            The electromagnetic field data that will be used for computing the characteristic
            impedance.

        Returns
        -------
        :class:`.IntegralResultTypes`
            Result of impedance computation over remaining dimensions (frequency, time, mode indices).
        """

        if not isinstance(em_field, (FieldData, FieldTimeData, ModeSolverData)):
            raise DataError("'em_field' type not supported by impedance calculator.")

        # If both voltage and current integrals have been defined then impedance is computed directly
        if self.voltage_integral:
            voltage = self.voltage_integral.compute_voltage(em_field)
        if self.current_integral:
            current = self.current_integral.compute_current(em_field)

        # If only one of the integrals has been provided then fall back to using total power (flux)
        # with Ohm's law. The input field should cover an area large enough to render the flux
        # computation accurate. If the input field is a time signal, then it is real and flux
        # corresponds to the instantaneous power. Otherwise the input field is in frequency domain,
        # where flux indicates the time-averaged power 0.5*Re(V*conj(I))
        if not self.voltage_integral:
            flux = em_field.flux
            if isinstance(em_field, FieldTimeData):
                voltage = flux / current
            else:
                voltage = 2 * flux / np.conj(current)
        if not self.current_integral:
            flux = em_field.flux
            if isinstance(em_field, FieldTimeData):
                current = flux / voltage
            else:
                current = np.conj(2 * flux / voltage)

        impedance = voltage / current
        return impedance

    @pd.validator("current_integral", always=True)
    def check_voltage_or_current(cls, val, values):
        """Raise validation error if both ``voltage_integral`` and ``current_integral``
        were not provided."""
        if not values.get("voltage_integral") and not val:
            raise ValidationError(
                "Atleast one of 'voltage_integral' or 'current_integral' must be provided."
            )
        return val
