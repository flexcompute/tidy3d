"""Abstract base class for all ports in the component and terminal component modelers."""

from __future__ import annotations

from abc import ABC

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.exceptions import SetupError


class AbstractBasePort(Tidy3dBaseModel, ABC):
    """Abstract base class representing a port excitation of a component."""

    name: str = pd.Field(
        ...,
        title="Name",
        description="Unique name for the port.",
        min_length=1,
    )

    @pd.validator("name")
    def _valid_port_name(cls, val):
        """Make sure port name does not include the '@' symbol, so that task names will always be unique."""
        if "@" in val:
            raise SetupError(f"Port names must not include the '@' symbol. Name given was '{val}'.")
        return val
