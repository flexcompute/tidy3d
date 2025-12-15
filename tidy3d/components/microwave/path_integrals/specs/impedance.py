"""Specification for impedance computation in transmission lines and waveguides."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from pydantic import Field, model_validator

from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.components.microwave.path_integrals.types import (
    CurrentPathSpecType,
    VoltagePathSpecType,
)
from tidy3d.exceptions import SetupError

if TYPE_CHECKING:
    from tidy3d.compat import Self


class AutoImpedanceSpec(MicrowaveBaseModel):
    """Specification for fully automatic transmission line impedance computation.

    This specification automatically calculates impedance by current
    paths based on the simulation geometry and conductors that intersect the mode plane.
    No user-defined path specifications are required.
    """


class CustomImpedanceSpec(MicrowaveBaseModel):
    """Specification for custom transmission line voltages and currents in mode solvers.

    The :class:`.CustomImpedanceSpec` class specifies how quantities related to transmission line
    modes are computed. It defines the paths for line integrals, which are used to
    compute voltage, current, and characteristic impedance of the transmission line.

    Users must supply at least one of voltage or current path specifications to control where these integrals
    are evaluated. Both voltage_spec and current_spec cannot be ``None`` simultaneously.

    Example
    -------
    >>> from tidy3d.components.microwave.path_integrals.specs.voltage import AxisAlignedVoltageIntegralSpec
    >>> from tidy3d.components.microwave.path_integrals.specs.current import AxisAlignedCurrentIntegralSpec
    >>> voltage_spec = AxisAlignedVoltageIntegralSpec(
    ...     center=(0, 0, 0), size=(0, 0, 1), sign="+"
    ... )
    >>> current_spec = AxisAlignedCurrentIntegralSpec(
    ...     center=(0, 0, 0), size=(2, 1, 0), sign="+"
    ... )
    >>> impedance_spec = CustomImpedanceSpec(
    ...     voltage_spec=voltage_spec,
    ...     current_spec=current_spec
    ... )
    """

    voltage_spec: Optional[VoltagePathSpecType] = Field(
        None,
        title="Voltage Integration Path",
        description="Path specification for computing the voltage associated with a mode profile.",
    )

    current_spec: Optional[CurrentPathSpecType] = Field(
        None,
        title="Current Integration Path",
        description="Path specification for computing the current associated with a mode profile.",
    )

    @model_validator(mode="after")
    def check_path_spec_combinations(self) -> Self:
        """Validate that at least one of voltage_spec or current_spec is provided.

        In order to define voltage/current/impedance, either a voltage or current path specification
        must be provided. Both cannot be ``None`` simultaneously.
        """
        val = self.current_spec
        voltage_spec = self.voltage_spec
        if val is None and voltage_spec is None:
            raise SetupError(
                "Not a valid 'CustomImpedanceSpec', the 'voltage_spec' and 'current_spec' cannot both be 'None'."
            )
        return self


ImpedanceSpecType = Union[AutoImpedanceSpec, CustomImpedanceSpec]
