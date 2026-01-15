"""Specification for impedance computation in transmission lines and waveguides."""

from __future__ import annotations

from typing import Optional, Union

import pydantic.v1 as pd

from tidy3d.components.base import skip_if_fields_missing
from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.components.microwave.path_integrals.types import (
    CurrentPathSpecType,
    VoltagePathSpecType,
)
from tidy3d.exceptions import SetupError


class AutoImpedanceSpec(MicrowaveBaseModel):
    """Specification for fully automatic transmission line impedance computation.

    Notes
    -----
        Automatically calculates impedance using paths based on simulation geometry
        and conductors that intersect the mode plane. No user-defined path
        specifications are required.
    """


class CustomImpedanceSpec(MicrowaveBaseModel):
    """Specification for custom transmission line voltages and currents in mode solvers.

    Notes
    -----
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

    voltage_spec: Optional[VoltagePathSpecType] = pd.Field(
        None,
        title="Voltage Integration Path",
        description="Path specification for computing the voltage associated with a mode profile.",
    )

    current_spec: Optional[CurrentPathSpecType] = pd.Field(
        None,
        title="Current Integration Path",
        description="Path specification for computing the current associated with a mode profile.",
    )

    @pd.validator("current_spec", always=True)
    @skip_if_fields_missing(["voltage_spec"])
    def check_path_spec_combinations(cls, val, values):
        """Validate that at least one of voltage_spec or current_spec is provided.

        In order to define voltage/current/impedance, either a voltage or current path specification
        must be provided. Both cannot be ``None`` simultaneously.
        """

        voltage_spec = values["voltage_spec"]
        if val is None and voltage_spec is None:
            raise SetupError(
                "Not a valid 'CustomImpedanceSpec', the 'voltage_spec' and 'current_spec' cannot both be 'None'."
            )
        return val


ImpedanceSpecType = Union[AutoImpedanceSpec, CustomImpedanceSpec]
