"""File containing classes required for the setup of a DEVSIM case."""

from abc import ABC
from typing import Optional

import pydantic.v1 as pd

from ...constants import VOLT
from ..base import Tidy3dBaseModel
from ..types import Union


class AbstractDevsimStruct(ABC, Tidy3dBaseModel):
    """"""


class ChargeToleranceSpec(Tidy3dBaseModel):
    """This class sets some Charge tolerance parameters.

    Example
    -------
    >>> import tidy3d as td
    >>> charge_settings = td.ChargeToleranceSpec(abs_tol=1e8, rel_tol=1e-10, max_iters=30)
    """

    abs_tol: Optional[pd.PositiveFloat] = pd.Field(
        default=1e10,
        title="Absolute tolerance.",
        description="Absolute tolerance used as stop criteria when converging towards a solution.",
    )

    rel_tol: Optional[pd.PositiveFloat] = pd.Field(
        default=1e-10,
        title="Relative tolerance.",
        description="Relative tolerance used as stop criteria when converging towards a solution.",
    )

    max_iters: Optional[pd.PositiveInt] = pd.Field(
        default=30,
        title="Maximum number of iterations.",
        description="Indicates the maximum number of iterations to be run. "
        "The solver will stop either when this maximum of iterations is met "
        "or when the tolerance criteria has been met.",
    )


class DCSpec(Tidy3dBaseModel):
    """This class sets parameters used in DC charge simulations.

    Example
    -------
    >>> import tidy3d as td
    >>> dc_spec = td.DCSpec(dv=0.1)
    """

    dv: Optional[pd.PositiveFloat] = pd.Field(
        1.0,
        title="Bias step.",
        description="By default, a solution is computed at 0 bias. "
        "If a bias different than 0 is requested, DEVSIM will start at 0 and increase bias "
        "at 'dV' intervals until the required bias is reached. ",
    )


# TODO: implement future classes for unsteady regimes SmallSignalSpec & NonlinearUnsteadySpec


# Mobility models
class CaugheyThomasMobility(Tidy3dBaseModel):
    """This class defines the parameters for the mobility model of Caughey and Thomas.
    NOTE: high electric field effects not yet supported.
    NOTE: Default values are those appropriate for Silicon."""

    # mobilities
    mu_n_min: pd.PositiveFloat = pd.Field(
        52.2,
        title="Minimum electron mobility",
        description="Minimum electron mobility at reference temperature (300K) in cm^2/V-s. ",
    )

    mu_n: pd.PositiveFloat = pd.Field(
        1471.0,
        title="Electron reference mobility",
        description="Reference electron mobility at reference temperature (300K) in cm^2/V-s",
    )

    mu_p_min: pd.PositiveFloat = pd.Field(
        44.9,
        title="Minimum hole mobility",
        description="Minimum hole mobility at reference temperature (300K) in cm^2/V-s. ",
    )

    mu_p: pd.PositiveFloat = pd.Field(
        470.5,
        title="Hole reference mobility",
        description="Reference hole mobility at reference temperature (300K) in cm^2/V-s",
    )

    # thermal exponent for reference mobility
    exp_t_mu: float = pd.Field(
        -2.33, title="Exponent for temperature dependent behavior of reference mobility"
    )

    # doping exponent
    exp_d_n: pd.PositiveFloat = pd.Field(
        0.68,
        title="Exponent for doping dependence of electron mobility.",
        description="Exponent for doping dependence of electron mobility at reference temperature (300K).",
    )

    exp_d_p: pd.PositiveFloat = pd.Field(
        0.719,
        title="Exponent for doping dependence of hole mobility.",
        description="Exponent for doping dependence of hole mobility at reference temperature (300K).",
    )

    # reference doping
    ref_N: pd.PositiveFloat = pd.Field(
        2.23e17,
        title="Reference doping",
        description="Reference doping at reference temperature (300K) in #/cm^3.",
    )

    # temperature exponent
    exp_t_mu_min: float = pd.Field(
        -0.57,
        title="Exponent of thermal dependence of minimum mobility.",
        description="Exponent of thermal dependence of minimum mobility.",
    )

    exp_t_d: float = pd.Field(
        2.4,
        title="Exponent of thermal dependence of reference doping.",
        description="Exponent of thermal dependence of reference doping.",
    )

    exp_t_d_exp: float = pd.Field(
        -0.146,
        title="Exponent of thermal dependence of the doping exponent effect.",
        description="Exponent of thermal dependence of the doping exponent effect.",
    )


# Generation-Recombination models
class AugerRecombination(Tidy3dBaseModel):
    """This class defines the parameters for the Auger recombination model.
    NOTE: default parameters are those appropriate for Silicon."""

    c_n: pd.PositiveFloat = pd.Field(
        2.8e-31, title="Constant for electrons", description="Constant for electrons in cm^6/s"
    )

    c_p: pd.PositiveFloat = pd.Field(
        9.9e-32, title="Constant for holes", description="Constant for holes in cm^6/s"
    )


class RadiativeRecombination(Tidy3dBaseModel):
    """This class is used to define the parameters for the radiative recombination model.
    NOTE: default values are those appropriate for Silicon."""

    r_const: float = pd.Field(
        1.6e-14,
        title="Radiation constant in cm^3/s",
        description="Radiation constant in cm^3/s",
    )


class ShockleyReedHallRecombination(Tidy3dBaseModel):
    """This class defines the parameters for the Shockley-Reed-Hall recombination model.
    NOTE: currently, lifetimes are considered constant (not dependent on temp. nor doping)
    NOTE: default values are those appropriate for Silicon."""

    tau_n: pd.PositiveFloat = pd.Field(
        3.3e-6, title="Electron lifetime.", description="Electron lifetime."
    )

    tau_p: pd.PositiveFloat = pd.Field(4e-6, title="Hole lifetime.", description="Hole lifetime.")


# Band-gap narrowing models
class SlotboomNarrowingBandGap(Tidy3dBaseModel):
    """This class specifies the parameters for the Slotboom model for band-gap narrowing.

    Reference
    ---------
    'UNIFIED APPARENT BANDGAP NARROWING IN n- AND p-TYPE SILICON'
    Solid-State Electronics Vol. 35, No. 2, pp. 125-129, 1992"""

    v1: pd.PositiveFloat = pd.Field(
        6.92 * 1e-3, title="V1 parameter", description=f"V1 parameter in {VOLT}", units=VOLT
    )

    n2: pd.PositiveFloat = pd.Field(
        1.3e17,
        title="n2 parameter",
        description="n2 parameter in cm^(-3)",
    )

    c2: float = pd.Field(
        0.5,
        title="c2 parameter",
        description="c2 parameter",
    )


ChargeToleranceType = Union[ChargeToleranceSpec]
ChargeRegimeType = Union[DCSpec]
MobilityModelType = Union[CaugheyThomasMobility]
RecombinationModelType = Union[
    AugerRecombination, RadiativeRecombination, ShockleyReedHallRecombination
]
BandgapNarrowingModelType = Union[SlotboomNarrowingBandGap]
