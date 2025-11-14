"""
Our DC sources ultimately need to follow this standard form if we want to enable full electrical integration.

```
11.3.2 .DC: DC Transfer Function

General form:

    .dc srcnam vstart vstop vincr [src2 start2 stop2 incr2]

Examples:

    .dc VIN 0.25 5.0 0.25
    .dc VDS 0 10 .5 VGS 0 5 1
    .dc VCE 0 10 .25 IB 0 10u 1u
    .dc RLoad 1k 2k 100
    .dc TEMP -15 75 5
```

"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, FiniteFloat, field_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import ArrayFloat1D
from tidy3d.constants import AMP, VOLT, inf


class DCVoltageSource(Tidy3dBaseModel):
    """
    DC voltage source in volts.

    Notes
    -----

        This voltage refers to potential above the equivalent simulation ground. Currently, electrical ports
        are not defined.

    Examples
    --------
    >>> import tidy3d as td
    >>> voltages = [-0.5, 0, 1, 2, 3, 4]
    >>> voltage_source = td.DCVoltageSource(voltage=voltages)
    """

    name: Optional[str] = Field(
        None,
        title="Name",
        description="Unique name for the DC voltage source",
        min_length=1,
    )

    voltage: ArrayFloat1D = Field(
        title="Voltage",
        description="DC voltage usually used as source in :class:`VoltageBC` boundary conditions.",
        units=VOLT,
    )

    # TODO: This should have always been in the field above but was introduced wrongly as a
    # standalone field. Keeping for compatibility, remove in 3.0.
    units: Literal[VOLT] = VOLT

    @field_validator("voltage")
    @classmethod
    def check_voltage(cls, val):
        for v in val:
            if v == inf:
                raise ValueError(f"Voltages must be finite. Currently  voltage={val}.")
        return val


class GroundVoltage(Tidy3dBaseModel):
    """
    Ground voltage source (0V reference).


    Notes
    -----
    This source explicitly sets the ground reference (0V) for the simulation.
    It is equivalent to :class:`DCVoltageSource(voltage=0)` but more explicit about
    establishing the ground reference.

    If no :class:`GroundVoltage` is specified, the smallest voltage among all
    sources will be considered as the ground reference. Note that the boundary
    conditions defined using a voltage array will be ignored during this
    process and cannot be used as a default ground.

    Example
    -------
    >>> import tidy3d as td
    >>> ground_source = td.GroundVoltage()
    >>> voltage_bc = td.VoltageBC(source=ground_source)
    """


class DCCurrentSource(Tidy3dBaseModel):
    """
    DC current source in amperes.

    Example
    -------
    >>> import tidy3d as td
    >>> current_source = td.DCCurrentSource(current=0.4)
    """

    name: Optional[str] = Field(
        None,
        title="Name",
        description="Unique name for the DC current source",
        min_length=1,
    )

    current: FiniteFloat = Field(
        title="Current",
        description="DC current usually used as source in :class:`CurrentBC` boundary conditions.",
        units=AMP,
    )

    # TODO: This should have always been in the field above but was introduced wrongly as a
    # standalone field. Keeping for compatibility, remove in 3.0.
    units: Literal[AMP] = AMP
