"""Specification for modes associated with transmission lines."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from pydantic import Field, model_validator

from tidy3d.components.base import cached_property
from tidy3d.components.geometry.base import Box
from tidy3d.components.geometry.bound_ops import bounds_contains
from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.components.microwave.path_integrals.specs.impedance import (
    AutoImpedanceSpec,
    ImpedanceSpecType,
)
from tidy3d.components.mode_spec import AbstractModeSpec
from tidy3d.constants import fp_eps
from tidy3d.exceptions import SetupError


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

    impedance_specs: Union[
        ImpedanceSpecType,
        tuple[Optional[ImpedanceSpecType], ...],
    ] = Field(
        default_factory=AutoImpedanceSpec._default_without_license_warning,
        title="Impedance Specifications",
        description="Field controls how the impedance is calculated for each mode calculated by the mode solver. "
        "Can be a single impedance specification (which will be applied to all modes) or a tuple of specifications "
        "(one per mode). The number of impedance specifications should match the number of modes field. "
        "When an impedance specification of ``None`` is used, the impedance calculation will be "
        "ignored for the associated mode.",
    )

    @cached_property
    def _impedance_specs_as_tuple(self) -> tuple[Optional[ImpedanceSpecType]]:
        """Gets the impedance_specs field converted to a tuple."""
        if isinstance(self.impedance_specs, Union[tuple, list]):
            return tuple(self.impedance_specs)
        return (self.impedance_specs,)

    @cached_property
    def _using_auto_current_spec(self) -> bool:
        """Checks whether at least one of the modes will require an auto setup of the current path specification."""
        return any(
            isinstance(impedance_spec, AutoImpedanceSpec)
            for impedance_spec in self._impedance_specs_as_tuple
        )

    @model_validator(mode="after")
    def check_impedance_specs_consistent_with_num_modes(self):
        """Check that the number of impedance specifications is equal to the number of modes.
        A single impedance spec is also permitted."""
        val = self.impedance_specs
        num_modes = self.num_modes
        if isinstance(val, Union[tuple, list]):
            num_impedance_specs = len(val)
        else:
            return self

        # Otherwise, check that the count matches
        if num_impedance_specs != num_modes:
            raise SetupError(
                f"Given {num_impedance_specs} impedance specifications in the 'MicrowaveModeSpec', "
                f"but the number of modes requested is {num_modes}. Please ensure that the "
                "number of impedance specifications is equal to the number of modes, or provide "
                "a single specification to apply to all modes."
            )

        return self

    def _check_path_integrals_within_box(self, box: Box):
        """Raise SetupError if a ``CustomImpedanceSpec`` includes a path specification
        defined outside a candidate box.
        """
        for impedance_ind, impedance_spec in enumerate(self._impedance_specs_as_tuple):
            if isinstance(impedance_spec, AutoImpedanceSpec) or impedance_spec is None:
                continue

            # Check both voltage and current specs using the same logic
            specs_to_check = [
                (impedance_spec.voltage_spec, "voltage"),
                (impedance_spec.current_spec, "current"),
            ]

            for spec, spec_type in specs_to_check:
                if spec is None:
                    continue

                box_bounds = box.bounds
                # If the box is a plane (one dimension is zero), we need to ignore
                # the bounds check along the normal axis
                if box.size.count(0.0) == 1:
                    normal_axis = box._normal_axis
                    # Convert tuple to list so we can modify it
                    box_bounds = [list(box_bounds[0]), list(box_bounds[1])]
                    # Set the bounds along normal axis to match the spec bounds
                    box_bounds[0][normal_axis] = spec.bounds[0][normal_axis]
                    box_bounds[1][normal_axis] = spec.bounds[1][normal_axis]
                    # Convert back to tuple for bounds_contains
                    box_bounds = (tuple(box_bounds[0]), tuple(box_bounds[1]))

                if not bounds_contains(
                    box_bounds, spec.bounds, fp_eps, np.finfo(np.float32).smallest_normal
                ):
                    raise SetupError(
                        "A 'MicrowaveModeSpec' must be setup with all path specifications defined within "
                        f"the bounds of the mode solving plane. The 'CustomImpedanceSpec' at index "
                        f"'{impedance_ind}' was provided with a {spec_type} path specification with bounds "
                        f"'{spec.bounds}', but the mode plane bounds are '{box.bounds}'."
                    )
