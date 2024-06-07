"""Defines lumped elements that should be included in the simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pydantic.v1 as pydantic

from ..components.medium import Medium, Medium2D
from ..components.structure import MeshOverrideStructure, Structure
from ..components.validators import assert_plane, validate_name_str
from ..constants import MICROMETER, OHM
from ..exceptions import ValidationError
from .base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from .geometry.base import Box, ClipOperation, Geometry
from .geometry.primitives import Cylinder
from .types import Axis, Coordinate
from .viz import PlotParams, plot_params_lumped_element

DEFAULT_LUMPED_ELEMENT_NUM_CELLS = 3


class LumpedElement(Tidy3dBaseModel, ABC):
    """Base class describing the interface all lumped elements obey."""

    name: str = pydantic.Field(
        ...,
        title="Name",
        description="Unique name for the lumped element.",
        min_length=1,
    )

    num_grid_cells: Optional[pydantic.PositiveInt] = pydantic.Field(
        DEFAULT_LUMPED_ELEMENT_NUM_CELLS,
        title="Lumped element grid cells",
        description="Number of mesh grid cells associated with the lumped element along each direction. "
        "Used in generating the suggested list of :class:`MeshOverrideStructure` objects. "
        "A value of ``None`` will turn off mesh refinement suggestions.",
    )

    _name_validator = validate_name_str()

    @cached_property
    def plot_params(self) -> PlotParams:
        """Default parameters for plotting a :class:`LumpedElement` object."""
        return plot_params_lumped_element

    @abstractmethod
    def to_mesh_overrides(self) -> list[MeshOverrideStructure]:
        """Creates a suggested :class:`MeshOverrideStructure` list
        that could be added to the :class:`Simulation`"""

    @abstractmethod
    def to_structure(self) -> Structure:
        """Converts the :class:`LumpedElement` object to a :class:`Structure`
        ready to be added to the :class:`Simulation`"""


class AbstractLumpedResistor(LumpedElement):
    """A lumped element representing a discrete resistance."""

    resistance: pydantic.PositiveFloat = pydantic.Field(
        ...,
        title="Resistance",
        description="Resistance value in ohms.",
        unit=OHM,
    )


class LumpedResistor(AbstractLumpedResistor, Box):
    """Class representing a rectangular lumped resistor. Lumped resistors are appended to the list
    of structures in the simulation as :class:`Medium2D` with the appropriate conductivity given
    their size and voltage axis."""

    voltage_axis: Axis = pydantic.Field(
        ...,
        title="Voltage Drop Axis",
        description="Specifies the axis along which the component is oriented and along which the "
        "associated voltage drop will occur. Must be in the plane of the element.",
    )

    _plane_validator = assert_plane()

    @cached_property
    def normal_axis(self):
        """Normal axis of the lumped element."""
        return self.size.index(0.0)

    def to_mesh_overrides(self) -> list[MeshOverrideStructure]:
        """Creates a suggested :class:`MeshOverrideStructure` list that could be added to the
        :class:`Simulation`.

        Note
        ----

        An important use case is when a :class:`LumpedResistor` is used with a :class:`LumpedPort`,
        where the mesh overrides may be automatically generated
        depending on ``num_grid_cells``. The port is a flat surface, but when computing the
        port current, we'll eventually integrate the magnetic field just above or below this
        surface. The mesh override needs to ensure that the mesh is fine enough not only in
        plane, but also in the normal direction. In the normal direction, we'll make sure there
        are at least 2 cell layers above and below whose size is the same as the in-plane cell
        size in the override region. Also, to ensure that the port itself is aligned with a grid
        boundary in the normal direction, two separate override regions are defined, one above
        and one below the analytical port region.
        """

        mesh_overrides = []
        if self.num_grid_cells:
            dl = self.size[self.voltage_axis] / self.num_grid_cells
            override_dl = Geometry.unpop_axis(dl, (dl, dl), axis=self.normal_axis)
            override_size = list(self.size)
            override_size[override_size.index(0)] = 2 * override_dl[self.normal_axis]

            override_center_below = list(self.center)
            override_center_below[self.normal_axis] -= dl
            mesh_overrides.append(
                MeshOverrideStructure(
                    geometry=Box(center=override_center_below, size=override_size),
                    dl=override_dl,
                )
            )

            override_center_above = list(self.center)
            override_center_above[self.normal_axis] += dl
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

    @cached_property
    def _sheet_conductance(self):
        """Effective sheet conductance."""
        lateral_axis = 3 - self.voltage_axis - self.normal_axis
        return self.size[self.voltage_axis] / self.size[lateral_axis] / self.resistance

    @cached_property
    def to_structure(self) -> Structure:
        """Converts the :class:`LumpedResistor` object to a :class:`Structure`
        ready to be added to the :class:`Simulation`"""
        conductivity = self._sheet_conductance
        medium_dict = {
            "tt": Medium(conductivity=conductivity),
            "ss": Medium(conductivity=conductivity),
        }
        return Structure(
            geometry=Box(size=self.size, center=self.center),
            medium=Medium2D(**medium_dict),
        )


class CoaxialLumpedResistor(AbstractLumpedResistor):
    """Class representing a coaxial lumped resistor. Lumped resistors are appended to the list of
    structures in the simulation as :class:`Medium2D` with the appropriate conductivity given their
    size and geometry."""

    center: Coordinate = pydantic.Field(
        (0.0, 0.0, 0.0),
        title="Center",
        description="Center of object in x, y, and z.",
        units=MICROMETER,
    )

    outer_diameter: pydantic.PositiveFloat = pydantic.Field(
        ...,
        title="Outer Diameter",
        description="Diameter of the outer concentric circle.",
        units=MICROMETER,
    )

    inner_diameter: pydantic.PositiveFloat = pydantic.Field(
        ...,
        title="Inner Diameter",
        description="Diameter of the inner concentric circle.",
        units=MICROMETER,
    )

    normal_axis: Axis = pydantic.Field(
        ...,
        title="Normal Axis",
        description="Specifies the normal axis, which defines "
        "the orientation of the circles making up the coaxial lumped element.",
    )

    def to_mesh_overrides(self) -> list[MeshOverrideStructure]:
        """Creates a suggested :class:`MeshOverrideStructure` list that could be added to the
        :class:`Simulation`.

        Note
        ----

        An important use case is when a :class:`CoaxialLumpedResistor` is used with
        a :class:`CoaxialLumpedPort`, where the mesh overrides may be automatically generated
        depending on ``num_grid_cells``. The port is a flat surface, but when computing the
        port current, we'll eventually integrate the magnetic field just above or below this
        surface. The mesh override needs to ensure that the mesh is fine enough not only in
        plane, but also in the normal direction. In the normal direction, we'll make sure there
        are at least 2 cell layers above and below whose size is the same as the in-plane cell
        size in the override region. Also, to ensure that the port itself is aligned with a grid
        boundary in the normal direction, two separate override regions are defined, one above
        and one below the analytical port region.
        """
        mesh_overrides = []

        if self.num_grid_cells:
            # Make sure the number of grid cells between inner and outer radius is `self.num_grid_cells`
            dl = (self.outer_diameter - self.inner_diameter) / self.num_grid_cells / 2
            override_dl = Geometry.unpop_axis(dl / 2, (dl, dl), axis=self.normal_axis)
            override_size = Geometry.unpop_axis(
                dl, (self.outer_diameter, self.outer_diameter), axis=self.normal_axis
            )

            override_center_below = list(self.center)
            override_center_below[self.normal_axis] -= dl / 2
            mesh_overrides.append(
                MeshOverrideStructure(
                    geometry=Box(center=override_center_below, size=override_size),
                    dl=override_dl,
                )
            )

            override_center_above = list(self.center)
            override_center_above[self.normal_axis] += dl / 2
            mesh_overrides.append(
                MeshOverrideStructure(
                    geometry=Box(center=override_center_above, size=override_size),
                    dl=override_dl,
                )
            )
        return mesh_overrides

    @pydantic.validator("center", always=True)
    def _center_not_inf(cls, val):
        """Make sure center is not infinitiy."""
        if any(np.isinf(v) for v in val):
            raise ValidationError("'center' can not contain 'td.inf' terms.")
        return val

    @pydantic.validator("inner_diameter", always=True)
    @skip_if_fields_missing(["outer_diameter"])
    def _ensure_inner_diameter_is_smaller(cls, val, values):
        """Ensures that the inner diameter is smaller than the outer diameter, so that the final shape is an annulus."""
        outer_diameter = values.get("outer_diameter")
        if val >= outer_diameter:
            raise ValidationError(
                f"The 'inner_diameter' {val} of a coaxial lumped element must be less than its 'outer_diameter' {outer_diameter}."
            )
        return val

    @cached_property
    def _sheet_conductance(self):
        """Effective sheet conductance for a coaxial resistor."""
        rout = self.outer_diameter / 2
        rin = self.inner_diameter / 2
        return 1 / (2 * np.pi * self.resistance) * (np.log(rout / rin))

    @cached_property
    def to_structure(self) -> Structure:
        """Converts the :class:`CoaxialLumpedResistor` object to a :class:`Structure`
        ready to be added to the :class:`Simulation`"""
        conductivity = self._sheet_conductance
        medium_dict = {
            "tt": Medium(conductivity=conductivity),
            "ss": Medium(conductivity=conductivity),
        }
        rout = self.outer_diameter / 2
        rin = self.inner_diameter / 2
        disk_out = Cylinder(axis=self.normal_axis, radius=rout, length=0, center=self.center)
        disk_in = Cylinder(axis=self.normal_axis, radius=rin, length=0, center=self.center)
        annulus = ClipOperation(operation="difference", geometry_a=disk_out, geometry_b=disk_in)
        return Structure(
            geometry=annulus,
            medium=Medium2D(**medium_dict),
        )


# lumped elements allowed in Simulation.lumped_elements
LumpedElementType = Union[
    LumpedResistor,
    CoaxialLumpedResistor,
]
