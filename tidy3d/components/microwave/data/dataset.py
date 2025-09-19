"""Post-processing data and figures of merit for antennas, including radiation efficiency,
reflection efficiency, gain, and realized gain.
"""

from __future__ import annotations

from typing import Optional

import pydantic.v1 as pd

from tidy3d.components.data.data_array import (
    CurrentFreqModeDataArray,
    ImpedanceFreqModeDataArray,
    VoltageFreqModeDataArray,
)
from tidy3d.components.data.dataset import Dataset


class MicrowaveModeDataset(Dataset):
    """Holds mode data that is specific to microwave and RF applications, like characteristic impedance."""

    Z0: Optional[ImpedanceFreqModeDataArray] = pd.Field(
        None,
        title="Characteristic Impedance",
        description="Optional quantity calculated for transmission lines. "
        "The characteristic impedance is only calculated when a :class:`MicrowaveModeSpec` "
        "is provided to the :class:`ModeSpec` associated with this data.",
    )

    voltage_coeffs: Optional[VoltageFreqModeDataArray] = pd.Field(
        None,
        title="Mode Voltage Coefficients",
        description="Optional quantity calculated for transmission lines, which associates "
        "a voltage-like quantity with each mode profile that scales linearly with the "
        "complex-valued mode amplitude. The mode voltages are only calculated when a :class:`MicrowaveModeSpec` "
        "is provided to the :class:`ModeSpec` associated with this data.",
    )

    current_coeffs: Optional[CurrentFreqModeDataArray] = pd.Field(
        None,
        title="Mode Current Coefficients",
        description="Optional quantity calculated for transmission lines, which associates "
        "a current-like quantity with each mode profile that scales linearly with the "
        "complex-valued mode amplitude. The mode currents are only calculated when a :class:`MicrowaveModeSpec`"
        " is provided to the :class:`ModeSpec` associated with this data.",
    )
