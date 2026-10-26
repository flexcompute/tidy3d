from __future__ import annotations

from flexcompute.core.validation import raise_first_error
from pydantic import Field, PositiveFloat, field_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import ArrayFloat1D
from tidy3d.constants import VOLT


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
        default=None,
        title="Name",
        description="Unique name for the SSAC voltage source.",
        min_length=1,
    )

    voltage: ArrayFloat1D = Field(
        title="DC Bias Voltages",
        description="List of DC operating point voltages (above ground) used with :class:`VoltageBC`.",
        json_schema_extra={"units": VOLT},
    )

    amplitude: PositiveFloat = Field(
        default=1.0,
        title="Small Signal Amplitude",
        description="Amplitude of the small-signal perturbation for SSAC analysis.",
        json_schema_extra={"units": VOLT},
        allow_inf_nan=False,
    )

    # Stays a field adapter, so a voltage error reports together with other field errors.
    @field_validator("voltage")
    @classmethod
    def validate_voltage(cls, val: ArrayFloat1D) -> ArrayFloat1D:
        from flexcompute.core._migration.spice.schema.sources import (
            SSACVoltageSource as SSACVoltageSourceSchema,
        )
        from flexcompute.core._migration.spice.validation import validate_ac_voltage

        source = SSACVoltageSourceSchema.model_construct(voltage=val)
        raise_first_error(validate_ac_voltage(source), ValueError)
        return val
