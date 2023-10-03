"""Defines lumped elements that should be included in the simulation."""


from __future__ import annotations
from abc import ABC
from typing import Union

import pydantic.v1 as pydantic

from .base import cached_property

from .geometry.base import Box
from .types import Axis
from .viz import PlotParams, plot_params_lumped_element
from ..constants import OHM
from ..components.validators import validate_name_str, assert_plane
from ..exceptions import ValidationError


class LumpedElement(Box, ABC):
    """Base class describing lumped elements."""

    name: str = pydantic.Field(
        ...,
        title="Name",
        description="Unique name for the lumped element.",
        min_length=1,
    )

    voltage_axis: Axis = pydantic.Field(
        ...,
        title="Voltage Drop Axis",
        description="Specifies the axis along which the component is oriented and along which the "
        "associated voltage drop will occur. Must be in the plane of the element.",
    )

    _name_validator = validate_name_str()
    _plane_validator = assert_plane()

    @cached_property
    def normal_axis(self):
        """Normal axis of the lumped element."""
        return self.size.index(0.0)

    @cached_property
    def plot_params(self) -> PlotParams:
        """Default parameters for plotting a LumpedElement object."""
        return plot_params_lumped_element

    @pydantic.validator("voltage_axis", always=True)
    def _voltage_axis_in_plane(cls, val, values):
        """Ensure voltage drop axis is in the plane of the lumped element."""
        name = values.get("name")
        size = values.get("size")
        if size and size.count(0.0) == 1 and size.index(0.0) == val:
            # if not planar, then a separate validator should be triggered, not this one
            raise ValidationError(
                f"'voltage_axis' must be in the plane of lumped element '{name}'."
            )
        return val


class LumpedResistor(LumpedElement):
    """Base class describing lumped elements."""

    resistance: pydantic.NonNegativeFloat = pydantic.Field(
        ...,
        title="Resistance",
        description="Resistance value in ohms.",
        unit=OHM,
    )

    @cached_property
    def sheet_conductance(self):
        """Effective sheet conductance."""
        lateral_axis = 3 - self.voltage_axis - self.normal_axis
        return self.size[self.voltage_axis] / self.size[lateral_axis] / self.resistance


# lumped elements allowed in Simulation.lumped_elements
LumpedElementType = Union[
    LumpedResistor,
]
