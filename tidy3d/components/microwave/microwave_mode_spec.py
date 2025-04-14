"""Specification for modes associated with transmission lines."""

from __future__ import annotations

from typing import Optional

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.exceptions import SetupError

from .path_integrals.types import CurrentPathSpecTypes, VoltagePathSpecTypes


class MicrowaveModeSpec(Tidy3dBaseModel):
    """Specification for computing transmission line voltages and currents in mode solvers.

    The :class:`.MicrowaveModeSpec` class specifies how quantities related to transmission line
    modes are computed. For example, it defines the paths for line integrals, which are used to
    compute voltage, current, and characteristic impedance of the transmission line.

    Users may supply their own voltage and current path specifications to control where these integrals
    are evaluated. If neither voltage nor current specifications are provided, an automatic choice of
    paths will be made based on the simulation geometry and context.

    TODO
        Validate that either all specs are voltage specs/ all current specs or all a pair of VI specs
    """

    voltage_spec: Optional[tuple[Optional[VoltagePathSpecTypes], ...]] = pd.Field(
        None,
        title="Voltage Integration Path",
        description="Path specification for computing the voltage associated with each mode. "
        "The number of path specifications should equal the 'num_modes' field "
        "in the 'ModeSpec'.",
    )

    current_spec: Optional[tuple[Optional[CurrentPathSpecTypes], ...]] = pd.Field(
        None,
        title="Current Integration Path",
        description="Path specification for computing the current associated with each mode. "
        "The number of path specifications should equal the 'num_modes' field "
        "in the 'ModeSpec'.",
    )

    @property
    def use_automatic_setup(self) -> bool:
        """Whether to setup the :class:`.MicrowaveModeSpec` automatically or use the supplied
        path specifications."""
        return self.voltage_spec is None and self.current_spec is None

    @property
    def num_voltage_specs(self) -> Optional[int]:
        """The number of voltage specifications supplied."""
        if type(self.voltage_spec) is tuple:
            return len(self.voltage_spec)
        return None

    @property
    def num_current_specs(self) -> Optional[int]:
        """The number of current specifications supplied."""
        if type(self.current_spec) is tuple:
            return len(self.current_spec)
        return None

    @pd.validator("current_spec", always=True)
    def check_path_spec_combinations(cls, val, values):
        """In order to define voltage/current/impedance, either a voltage or current path spec
        must be provided for each associated mode index.
        """

        voltage_specs = values["voltage_spec"]
        if val is None and voltage_specs is None:
            return val
        elif val is not None and voltage_specs is not None:
            if len(val) != len(voltage_specs):
                raise SetupError(
                    f"The length of 'voltage_spec' is {len(voltage_specs)}, which is not equal to the length "
                    f"of 'current_spec' {len(val)}. Please ensure that the same number of voltage and current "
                    "specifications have been supplied."
                )

        if val is None:
            current_specs = (None,) * len(voltage_specs)
        else:
            current_specs = val

        if voltage_specs is None:
            voltage_specs = (None,) * len(val)

        for current_spec, voltage_spec, index in zip(
            current_specs, voltage_specs, range(len(current_specs))
        ):
            if current_spec is None and voltage_spec is None:
                raise SetupError(
                    f"Both entries in 'voltage_spec' and 'current_spec' at position {index} are "
                    "'None'. Please ensure at least one of them is a valid path specification."
                )

        return val
