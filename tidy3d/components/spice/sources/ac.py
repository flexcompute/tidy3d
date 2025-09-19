from __future__ import annotations

from typing import Optional

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import ArrayFloat1D
from tidy3d.constants import VOLT
from tidy3d.constants import inf as td_inf


class SSACVoltageSource(Tidy3dBaseModel):
    """
    Small-Signal AC (SSAC) voltage source.

    Notes
    -----
    This source represents a small-signal AC excitation defined by a DC operating point
    voltage and the amplitude of the small signal perturbation.

    The ``voltage`` refers to the DC operating point above the simulation ground.
    The ``amplitude`` defines the magnitude of the small-signal perturbation.
    Currently, full circuit simulation through electrical ports is not supported.

    Examples
    --------
    >>> import tidy3d as td
    >>> ssac_source = td.SSACVoltageSource(
    ...     name="VIN",
    ...     voltage=0.8,  # DC bias voltage
    ...     amplitude=1e-3  # Small signal amplitude
    ... )
    """

    name: Optional[str] = pd.Field(
        None,
        title="Name",
        description="Unique name for the SSAC voltage source.",
        min_length=1,
    )

    voltage: ArrayFloat1D = pd.Field(
        ...,
        title="DC Bias Voltages",
        description="List of DC operating point voltages (above ground) used with :class:`VoltageBC`.",
        units=VOLT,
    )

    amplitude: pd.FiniteFloat = pd.Field(
        default=1.0,
        title="Small Signal Amplitude",
        description="Amplitude of the small-signal perturbation for SSAC analysis.",
        units=VOLT,
    )

    @pd.validator("voltage")
    def validate_voltage(cls, val):
        for v in val:
            if v == td_inf:
                raise ValueError(f"Voltages must be finite. Current voltage={val}.")
        return val

    @pd.validator("amplitude")
    def validate_amplitude(cls, val):
        if val == td_inf:
            raise ValueError(f"Signal amplitude must be finite. Current amplitude={val}.")
        return val
