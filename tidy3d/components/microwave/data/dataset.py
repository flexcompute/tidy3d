"""Dataset for microwave and RF transmission line mode data, including impedance, voltage, and current coefficients."""

from __future__ import annotations

import pydantic.v1 as pd

from tidy3d.components.data.data_array import (
    CurrentFreqModeDataArray,
    ImpedanceFreqModeDataArray,
    VoltageFreqModeDataArray,
)
from tidy3d.components.data.dataset import ModeFreqDataset


class TransmissionLineDataset(ModeFreqDataset):
    """Holds mode data that is specific to transmission lines in microwave and RF applications,
    like characteristic impedance.

    Notes
    -----
        The data in this class is only calculated when a :class:`MicrowaveModeSpec`
        is provided to the :class:`ModeMonitor`, :class:`ModeSolverMonitor`, :class:`ModeSolver`,
        or :class:`ModeSimulation`.
    """

    Z0: ImpedanceFreqModeDataArray = pd.Field(
        ...,
        title="Characteristic Impedance",
        description="The characteristic impedance of the transmission line.",
    )

    voltage_coeffs: VoltageFreqModeDataArray = pd.Field(
        ...,
        title="Mode Voltage Coefficients",
        description="Quantity calculated for transmission lines, which associates "
        "a voltage-like quantity with each mode profile that scales linearly with the "
        "complex-valued mode amplitude.",
    )

    current_coeffs: CurrentFreqModeDataArray = pd.Field(
        ...,
        title="Mode Current Coefficients",
        description="Quantity calculated for transmission lines, which associates "
        "a current-like quantity with each mode profile that scales linearly with the "
        "complex-valued mode amplitude.",
    )
