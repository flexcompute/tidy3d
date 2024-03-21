"""Defines semiconductor simulation material specifications"""
from __future__ import annotations

from abc import ABC

# import pydantic.v1 as pd

from ..types import Union
from ..base import Tidy3dBaseModel

# from ...constants import PERMITTIVITY


class AbstractSemiConDevSpec(ABC, Tidy3dBaseModel):
    """Abstract heat material specification."""


class InsulatingSpec(AbstractSemiConDevSpec):
    """Insulating medium.

    Example
    -------
    >>> solid = InsulatingSpec()
    """


class DeviceSpec(AbstractSemiConDevSpec):
    """Device medium.

    Example
    -------
    >>> solid = DeviceSpec(
    ...     permittivity=3,
    ... )
    """

    # no need for permittivity here since we already have it in AbstractMedium class
    # permittivity: pd.PositiveFloat = pd.Field(
    #     title="Relative permittivity of material.",
    #     description=f"Relative permittivity of material in unit of {PERMITTIVITY}.",
    #     units=PERMITTIVITY,
    # )


SemiConDevSpecType = Union[InsulatingSpec, DeviceSpec]
