"""Defines lumped elements that should be included in the simulation."""
from __future__ import annotations
from abc import ABC
from typing import Union, List, Optional

import pydantic.v1 as pydantic

from .base import cached_property, skip_if_fields_missing

from .geometry.base import Box
from ..components.structure import MeshOverrideStructure
from .types import Axis
from .viz import PlotParams, plot_params_lumped_element
from ..constants import OHM
from ..components.validators import validate_name_str, assert_plane
from ..exceptions import ValidationError

DEFAULT_LUMPED_ELEMENT_NUM_CELLS = 3


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

    num_grid_cells: Optional[pydantic.PositiveInt] = pydantic.Field(
        DEFAULT_LUMPED_ELEMENT_NUM_CELLS,
        title="Lumped element grid cells",
        description="Number of mesh grid cells associated with the lumped element along each direction. "
        "Used in generating the suggested list of :class:`MeshOverrideStructure` objects."
        "A value of ``None`` will turn off mesh refinement suggestions.",
    )

    _name_validator = validate_name_str()
    _plane_validator = assert_plane()

    @cached_property
    def normal_axis(self):
        """Normal axis of the lumped element."""
        return self.size.index(0.0)

    @cached_property
    def plot_params(self) -> PlotParams:
        """Default parameters for plotting a :class:`LumpedElement` object."""
        return plot_params_lumped_element

    def to_mesh_overrides(self) -> List[MeshOverrideStructure]:
        """Creates a suggested :class:`MeshOverrideStructure` list that could be added to the :class:`Simulation`"""
        # Create a mesh override when refinement is needed.
        # An important use case is when a LumpedResistor is used with a LumpedPort
        # The port is a flat surface, but when computing the port current,
        # we'll eventually integrate the magnetic field just above and below
        # this surface, so the mesh override needs to ensure that the mesh
        # is fine enough not only in plane, but also in the normal direction.
        # So in the normal direction, we'll make sure there are at least
        # 2 cell layers above and below whose size is the same as the in-plane
        # cell size in the override region. Also, to ensure that the port itself
        # is aligned with a grid boundary in the normal direction, two separate
        # override regions are defined, one above and one below the analytical
        # port region.
        mesh_overrides = []
        if self.num_grid_cells:
            override_dl = [self.size[self.voltage_axis] / self.num_grid_cells] * 3
            override_dl[self.normal_axis] /= 2
            override_size = list(self.size)
            override_size[override_size.index(0)] = 2 * override_dl[self.normal_axis]
            override_center_above = list(self.center)
            override_center_above[self.normal_axis] += override_dl[self.normal_axis]
            override_center_below = list(self.center)
            override_center_below[self.normal_axis] -= override_dl[self.normal_axis]
            mesh_overrides.append(
                MeshOverrideStructure(
                    geometry=Box(center=override_center_below, size=override_size),
                    dl=override_dl,
                )
            )
            mesh_overrides.append(
                MeshOverrideStructure(
                    geometry=Box(center=override_center_above, size=override_size),
                    dl=override_dl,
                )
            )
        return mesh_overrides

    @pydantic.validator("voltage_axis", always=True)
    @skip_if_fields_missing(["name", "size"])
    def _voltage_axis_in_plane(cls, val, values):
        """Ensure voltage drop axis is in the plane of the lumped element."""
        name = values.get("name")
        size = values.get("size")
        if size.count(0.0) == 1 and size.index(0.0) == val:
            # if not planar, then a separate validator should be triggered, not this one
            raise ValidationError(
                f"'voltage_axis' must be in the plane of lumped element '{name}'."
            )
        return val


class LumpedResistor(LumpedElement):
    """Class representing a lumped resistor. Lumped resistors are appended to the list of structures in the simulation
    as :class:`Medium2D` with the appropriate conductivity given their size and voltage axis."""

    resistance: pydantic.PositiveFloat = pydantic.Field(
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
LumpedElementType = Union[LumpedResistor,]
