"""Specification for modes associated with transmission lines."""

from __future__ import annotations

import pydantic.v1 as pd

from tidy3d.components.base import cached_property
from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.components.microwave.path_integrals.specs.impedance import (
    AutoImpedanceSpec,
    ImpedanceSpecType,
)
from tidy3d.components.mode_spec import AbstractModeSpec
from tidy3d.components.types import annotate_type


class MicrowaveModeSpec(AbstractModeSpec, MicrowaveBaseModel):
    """
    The :class:`.MicrowaveModeSpec` class specifies how quantities related to transmission line
    modes and microwave waveguides are computed. For example, it defines the paths for line integrals, which are used to
    compute voltage, current, and characteristic impedance of the transmission line.

    Example
    -------
    >>> import tidy3d as td
    >>> # Using automatic impedance calculation (single spec, will be duplicated for all modes)
    >>> mode_spec_auto = td.MicrowaveModeSpec(
    ...     num_modes=2,
    ...     impedance_specs=td.AutoImpedanceSpec()
    ... )
    >>> # Using custom impedance specification for multiple modes
    >>> voltage_spec = td.AxisAlignedVoltageIntegralSpec(
    ...     center=(0, 0, 0), size=(0, 0, 1), sign="+"
    ... )
    >>> current_spec = td.AxisAlignedCurrentIntegralSpec(
    ...     center=(0, 0, 0), size=(2, 1, 0), sign="+"
    ... )
    >>> custom_impedance = td.CustomImpedanceSpec(
    ...     voltage_spec=voltage_spec, current_spec=current_spec
    ... )
    >>> mode_spec_custom = td.MicrowaveModeSpec(
    ...     num_modes=1,
    ...     impedance_specs=custom_impedance
    ... )
    """

    impedance_specs: (
        annotate_type(ImpedanceSpecType) | tuple[annotate_type(ImpedanceSpecType) | None, ...]
    ) = pd.Field(
        default_factory=AutoImpedanceSpec._default_without_license_warning,
        title="Impedance Specifications",
        description="Field controls how the impedance is calculated for each mode calculated by the mode solver. "
        "Can be a single impedance specification (which will be applied to all modes) or a tuple of specifications "
        "(one per mode). The number of impedance specifications should match the number of modes field. "
        "When an impedance specification of ``None`` is used, the impedance calculation will be "
        "ignored for the associated mode.",
    )

    @cached_property
    def _impedance_specs_as_tuple(self) -> tuple[ImpedanceSpecType | None]:
        """Gets the impedance_specs field converted to a tuple."""
        if isinstance(self.impedance_specs, tuple | list):
            return tuple(self.impedance_specs)
        return (self.impedance_specs,)

    @cached_property
    def _using_auto_current_spec(self) -> bool:
        """Checks whether at least one of the modes will require an auto setup of the current path specification."""
        return any(
            isinstance(impedance_spec, AutoImpedanceSpec)
            for impedance_spec in self._impedance_specs_as_tuple
        )

    @pd.validator("impedance_specs", always=True)
    def check_impedance_specs_consistent_with_num_modes(cls, val, values):
        """Check that the number of impedance specifications is equal to the number of modes.
        A single impedance spec is also permitted."""
        num_modes = values.get("num_modes")
        if isinstance(val, tuple | list):
            num_impedance_specs = len(val)
        else:
            return val

        # Otherwise, check that the count matches
        if num_impedance_specs != num_modes:
            from tidy3d.exceptions import SetupError

            raise SetupError(
                f"Given {num_impedance_specs} impedance specifications in the 'MicrowaveModeSpec', "
                f"but the number of modes requested is {num_modes}. Please ensure that the "
                "number of impedance specifications is equal to the number of modes, or provide "
                "a single specification to apply to all modes."
            )

        return val
