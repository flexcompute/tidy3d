"""Defines heat material specifications"""

from __future__ import annotations

from pydantic import Field, NonNegativeFloat, PositiveFloat

from tidy3d.components.tcad.boundary.abstract import HeatChargeBC
from tidy3d.constants import HEAT_FLUX, HEAT_TRANSFER_COEFF, KELVIN


class TemperatureBC(HeatChargeBC):
    """Constant temperature thermal boundary conditions.

    Example
    -------
    >>> import tidy3d as td
    >>> bc = td.TemperatureBC(temperature=300)
    """

    temperature: PositiveFloat = Field(
        title="Temperature",
        description=f"Temperature value in units of {KELVIN}.",
        units=KELVIN,
    )


class HeatFluxBC(HeatChargeBC):
    """Constant flux thermal boundary conditions.

    Example
    -------
    >>> import tidy3d as td
    >>> bc = td.HeatFluxBC(flux=1)
    """

    flux: float = Field(
        title="Heat Flux",
        description=f"Heat flux value in units of {HEAT_FLUX}.",
        units=HEAT_FLUX,
    )


class ConvectionBC(HeatChargeBC):
    """Convective thermal boundary conditions.

    Example
    -------
    >>> import tidy3d as td
    >>> bc = td.ConvectionBC(ambient_temperature=300, transfer_coeff=1)
    """

    ambient_temperature: PositiveFloat = Field(
        title="Ambient Temperature",
        description=f"Ambient temperature value in units of {KELVIN}.",
        units=KELVIN,
    )

    transfer_coeff: NonNegativeFloat = Field(
        title="Heat Transfer Coefficient",
        description=f"Heat flux value in units of {HEAT_TRANSFER_COEFF}.",
        units=HEAT_TRANSFER_COEFF,
    )
