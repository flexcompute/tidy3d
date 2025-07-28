"""Class and custom data array for representing a scattering matrix port based on waveguide modes."""

from __future__ import annotations

import pydantic.v1 as pd

from tidy3d.components.geometry.base import Box
from tidy3d.components.mode_spec import ModeSpec
from tidy3d.components.types import Direction


class Port(Box):
    """Specifies a port in the scattering matrix."""

    direction: Direction = pd.Field(
        ...,
        title="Direction",
        description="'+' or '-', defining which direction is considered 'input'.",
    )
    mode_spec: ModeSpec = pd.Field(
        ModeSpec(),
        title="Mode Specification",
        description="Specifies how the mode solver will solve for the modes of the port.",
    )
    name: str = pd.Field(
        ...,
        title="Name",
        description="Unique name for the port.",
        min_length=1,
    )
