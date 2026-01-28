"""Specification for a single layer in a stackup."""

from __future__ import annotations

from typing import Optional

import pydantic.v1 as pydantic

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.medium import MediumType3D
from tidy3d.components.types import Axis


class LayerSpec(Tidy3dBaseModel):
    """Specification for a single layer in a stackup.

    Defines the physical properties of a layer including its z-bounds,
    material, and extrusion parameters.

    Example
    -------
    >>> from tidy3d import Medium, PEC
    >>> copper = PEC
    >>> fr4 = Medium(permittivity=4.2)
    >>>
    >>> top_copper = LayerSpec(
    ...     name="top_copper",
    ...     z_bounds=(0.0, 0.035),
    ...     medium=copper,
    ... )
    >>> substrate = LayerSpec(
    ...     name="substrate",
    ...     z_bounds=(-1.6, 0.0),
    ...     medium=fr4,
    ... )
    """

    name: str = pydantic.Field(
        ...,
        title="Name",
        description="Unique identifier for this layer. Used to match geometry layers to specs.",
    )

    z_bounds: tuple[float, float] = pydantic.Field(
        ...,
        title="Z Bounds",
        description="Z-coordinate bounds of the layer as (z_min, z_max).",
        units="um",
    )

    medium: Optional[MediumType3D] = pydantic.Field(
        None,
        title="Medium",
        description="Material for this layer. None represents air/void.",
    )

    sidewall_angle: float = pydantic.Field(
        0.0,
        title="Sidewall Angle",
        description="Angle of the sidewall for extruded shapes in radians. "
        "Positive angle creates a narrower top than bottom.",
        units="rad",
    )

    axis: Axis = pydantic.Field(
        2,
        title="Extrusion Axis",
        description="Axis perpendicular to the layer plane. Default is 2 (z-axis).",
    )

    @property
    def z_min(self) -> float:
        """Minimum z-coordinate of the layer (bottom surface)."""
        return self.z_bounds[0]

    @property
    def z_max(self) -> float:
        """Maximum z-coordinate of the layer (top surface)."""
        return self.z_bounds[1]

    @property
    def thickness(self) -> float:
        """Thickness of the layer."""
        return abs(self.z_max - self.z_min)

    @property
    def z_center(self) -> float:
        """Z-coordinate of the layer center."""
        return (self.z_min + self.z_max) / 2.0

    @property
    def slab_bounds(self) -> tuple[float, float]:
        """Returns (z_min, z_max) tuple for use with PolySlab."""
        return self.z_bounds

    @pydantic.validator("z_bounds")
    def _z_bounds_ordered(cls, val):
        """Ensure z_bounds[1] >= z_bounds[0]."""
        z_min, z_max = val
        if z_max < z_min:
            raise ValueError(f"z_bounds[1] ({z_max}) must be >= z_bounds[0] ({z_min})")
        return val
