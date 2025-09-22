"""Specification for modes associated with transmission lines."""

from __future__ import annotations

from typing import Optional

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.microwave.path_integrals.impedance_spec import (
    AutoImpedanceSpec,
    ImpedanceSpecTypes,
)
from tidy3d.components.types import annotate_type


class MicrowaveModeSpec(Tidy3dBaseModel):
    """
    The :class:`.MicrowaveModeSpec` class specifies how quantities related to transmission line
    modes and microwave waveguides are computed. For example, it defines the paths for line integrals, which are used to
    compute voltage, current, and characteristic impedance of the transmission line.
    """

    impedance_spec: tuple[Optional[annotate_type(ImpedanceSpecTypes)], ...] = pd.Field(
        ...,
        title="Impedance Specification",
        description="Field controls how the impedance is calculated for each mode calculated by the mode solver.",
    )

    @cached_property
    def _using_auto_current_spec(self) -> bool:
        """Checks whether at least one of the modes will require an auto setup of the current path specification."""
        return any(
            isinstance(impedance_spec, AutoImpedanceSpec) for impedance_spec in self.impedance_spec
        )

    @cached_property
    def num_impedance_specs(self) -> int:
        """The number of impedance specifications to be used."""
        return len(self.impedance_spec)
