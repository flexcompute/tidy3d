"""Specification for impedance computation in transmission lines and waveguides."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from pydantic import Field, model_validator

from tidy3d.components.geometry.bound_ops import bounds_contains
from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.components.microwave.path_integrals.specs.current import AxisAlignedCurrentIntegralSpec
from tidy3d.components.microwave.path_integrals.types import (
    CurrentPathSpecType,
    VoltagePathSpecType,
)
from tidy3d.components.types.base import discriminated_union
from tidy3d.constants import fp_eps
from tidy3d.exceptions import SetupError

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.geometry.base import Box
    from tidy3d.components.microwave.types import ImpedanceDef
    from tidy3d.components.types.base import Direction


class AbstractImpedanceSpec(MicrowaveBaseModel):
    """Abstract base class for impedance specifications."""

    @abstractmethod
    def _check_path_integrals_within_box(self, box: Box) -> None:
        """Raise SetupError if a path specification is
        defined outside a candidate box.
        """


class AutoImpedanceSpec(AbstractImpedanceSpec):
    """Specification for fully automatic transmission line impedance computation.

    Notes
    -----
        Automatically calculates impedance using paths based on simulation geometry
        and conductors that intersect the mode plane. No user-defined path
        specifications are required.
    """

    def _check_path_integrals_within_box(self, box: Box) -> None:
        """Raise SetupError if a path specification is
        defined outside a candidate box.
        """


class CustomImpedanceSpec(AbstractImpedanceSpec):
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

    @property
    def impedance_definition(self) -> ImpedanceDef:
        """Determine the impedance definition based on provided path specifications.

        Returns
        -------
        ImpedanceDef
            The impedance definition type:
            - VI: Both voltage and current specs provided.
            - PI: Only current spec provided.
            - PV: Only voltage spec provided.
        """
        if self.voltage_spec is not None and self.current_spec is not None:
            return "VI"
        elif self.current_spec is not None:
            return "PI"
        else:
            return "PV"

    def _check_path_integrals_within_box(self, box: Box) -> None:
        """Raise 'SetupError' if a path specification is defined outside a candidate box."""
        for spec, spec_type in [
            (self.voltage_spec, "voltage"),
            (self.current_spec, "current"),
        ]:
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
                    "A 'CustomImpedanceSpec' must be setup with all path specifications defined within "
                    f"the bounds of the mode solving plane. The 'CustomImpedanceSpec' was provided with a {spec_type} path specification with bounds "
                    f"'{spec.bounds}', but the mode plane bounds are '{box.bounds}'."
                )

    @classmethod
    def from_bounding_box(
        cls, bounding_box: Box, current_sign: Direction = "+"
    ) -> CustomImpedanceSpec:
        """Create a custom impedance specification from a bounding box."""
        return cls(
            current_spec=AxisAlignedCurrentIntegralSpec(
                center=bounding_box.center,
                size=bounding_box.size,
                sign=current_sign,
                extrapolate_to_endpoints=False,
                snap_contour_to_grid=True,
            ),
            voltage_spec=None,
        )


ImpedanceSpecType = discriminated_union(Union[AutoImpedanceSpec, CustomImpedanceSpec])
