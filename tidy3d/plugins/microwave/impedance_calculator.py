"""Class for computing characteristic impedance of transmission lines."""

from __future__ import annotations


import pydantic.v1 as pd
import numpy as np
from typing import Optional
from ...components.data.monitor_data import FieldData, FieldTimeData, ModeSolverData

from ...components.base import Tidy3dBaseModel
from ...exceptions import ValidationError

from .path_integrals import VoltageIntegralAA, CurrentIntegralAA


class ImpedanceCalculator(Tidy3dBaseModel):
    """Tool for computing the characteristic impedance of a transmission line."""

    voltage_integral: Optional[VoltageIntegralAA] = pd.Field(
        ...,
        title="Voltage Integral",
        description="Integral for computing voltage.",
    )

    current_integral: Optional[CurrentIntegralAA] = pd.Field(
        ...,
        title="",
        description="Integral for computing current.",
    )

    def compute_impedance(self, em_field: FieldData | ModeSolverData | FieldTimeData):
        # If both voltage and current integrals have been defined then impedance is computed directly
        if self.voltage_integral:
            voltage = self.voltage_integral.compute_voltage(em_field.field_components)
        if self.current_integral:
            current = self.current_integral.compute_current(em_field.field_components)

        # If only one of the integrals has been provided then fall back to using total power (flux)
        # with Ohm's law. The input field should cover an area large enough to render the flux computation accurate.
        # If the input field is a time signal, then it is real and flux corresponds to the instantaneous power.
        # Otherwise the input field is in frequency domain, where flux indicates the time-averaged power 0.5*Re(V*conj(I))
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

        return voltage / current

    @pd.validator("current_integral", always=True)
    def check_voltage_or_current(cls, val, values):
        """Ensure that 'voltage_integral' and/or 'current_integral' were provided."""
        if not values.get("voltage_integral") and not val:
            raise ValidationError(
                "Atleast one of 'voltage_integral' or 'current_integral' must be provided."
            )
        return val
