"""Defines specifications for source frames."""

from __future__ import annotations

from abc import ABC

import pydantic.v1 as pydantic

from tidy3d.components.base import Tidy3dBaseModel


class AbstractSourceFrame(Tidy3dBaseModel, ABC):
    """Abstract base class for all source frames."""

    length: int = pydantic.Field(
        2,
        title="Length",
        description="The length of the frame, specified as the number of cells along the source "
        "injection direction.",
        gt=0,
    )


class PECFrame(AbstractSourceFrame):
    """PEC source frame."""
