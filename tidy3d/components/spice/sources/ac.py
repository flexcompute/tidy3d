from __future__ import annotations

from pydantic import Field, FiniteFloat, field_validator

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

    name: str | None = Field(
        None,
        title="Name",
        description="Unique name for the SSAC voltage source.",
        min_length=1,
    )

    voltage: ArrayFloat1D = Field(
        title="DC Bias Voltages",
        description="List of DC operating point voltages (above ground) used with :class:`VoltageBC`.",
        json_schema_extra={"units": VOLT},
    )

    amplitude: FiniteFloat = Field(
        default=1.0,
        title="Small Signal Amplitude",
        description="Amplitude of the small-signal perturbation for SSAC analysis.",
        json_schema_extra={"units": VOLT},
    )

    @field_validator("voltage")
    @classmethod
    def validate_voltage(cls, val: ArrayFloat1D) -> ArrayFloat1D:
        for v in val:
            if v == td_inf:
                raise ValueError(f"Voltages must be finite. Current voltage={val}.")
        return val

    @field_validator("amplitude")
    @classmethod
    def validate_amplitude(cls, val: FiniteFloat) -> FiniteFloat:
        if val == td_inf:
            raise ValueError(f"Signal amplitude must be finite. Current amplitude={val}.")
        return val
