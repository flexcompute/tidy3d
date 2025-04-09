"""
This class defines standard SPICE electrical_analysis types (electrical simulations configurations).
"""

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel

from ....constants import KELVIN


class ChargeToleranceSpec(Tidy3dBaseModel):
    """
    Charge tolerance parameters relevant to multiple simulation analysis types.

    Example
    -------
    >>> import tidy3d as td
    >>> charge_settings = td.ChargeToleranceSpec(abs_tol=1e8, rel_tol=1e-10, max_iters=30)
    """

    abs_tol: pd.PositiveFloat = pd.Field(
        default=1e10,
        title="Absolute tolerance.",
        description="Absolute tolerance used as stop criteria when converging towards a solution.",
    )

    rel_tol: pd.PositiveFloat = pd.Field(
        default=1e-10,
        title="Relative tolerance.",
        description="Relative tolerance used as stop criteria when converging towards a solution.",
    )

    max_iters: pd.PositiveInt = pd.Field(
        default=30,
        title="Maximum number of iterations.",
        description="Indicates the maximum number of iterations to be run. "
        "The solver will stop either when this maximum of iterations is met "
        "or when the tolerance criteria has been met.",
    )

    ramp_up_iters: pd.PositiveInt = pd.Field(
        default=1,
        title="Ramp-up iterations.",
        description="In order to help in start up, quantities such as doping "
        "are ramped up until they reach their specified value. This parameter "
        "determines how many of this iterations it takes to reach full values.",
    )


class IsothermalSteadyChargeDCAnalysis(Tidy3dBaseModel):
    """
    Configures relevant steady-state DC simulation parameters for a charge simulation.
    """

    temperature: pd.PositiveFloat = pd.Field(
        300,
        title="Temperature",
        description="Lattice temperature. Assumed constant throughout the device. "
        "Carriers are assumed to be at thermodynamic equilibrium with the lattice.",
        units=KELVIN,
    )

    tolerance_settings: ChargeToleranceSpec = pd.Field(
        default=ChargeToleranceSpec(), title="Tolerance settings"
    )

    convergence_dv: pd.PositiveFloat = pd.Field(
        default=1.0,
        title="Bias step.",
        description="By default, a solution is computed at 0 bias. If a bias different than "
        "0 is requested through a voltage source, the charge solver will start at 0 and increase bias "
        "at `convergence_dv` intervals until the required bias is reached. This is, therefore, a "
        "convergence parameter in DC computations.",
    )

    fermi_dirac: bool = pd.Field(
        False,
        title="Fermi-Dirac statistics",
        description="Determines whether Fermi-Dirac statistics are used. When False, "
        "Boltzmann statistics will be used. This can provide more accurate results in situations "
        "where very high doping may lead the pseudo-Fermi energy level to approach "
        "either the conduction or valence energy bands.",
    )
