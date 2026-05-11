"""
This class defines standard SPICE electrical_analysis types (electrical simulations configurations).
"""

from __future__ import annotations

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import KELVIN
from tidy3d.log import log


class ChargeToleranceSpec(Tidy3dBaseModel):
    """
    Charge tolerance parameters relevant to multiple simulation analysis types.

    Example
    -------
    >>> import tidy3d as td
    >>> charge_settings = td.ChargeToleranceSpec(abs_tol=1e8, rel_tol=1e-10, max_iters=30)
    """

    abs_tol: PositiveFloat = Field(
        default=1e10,
        title="Absolute tolerance.",
        description="Absolute tolerance used as stop criteria when converging towards a solution.",
    )

    rel_tol: PositiveFloat = Field(
        default=1e-10,
        title="Relative tolerance.",
        description="Relative tolerance used as stop criteria when converging towards a solution.",
    )

    max_iters: PositiveInt = Field(
        default=30,
        title="Maximum number of iterations.",
        description="Indicates the maximum number of iterations to be run. "
        "The solver will stop either when this maximum of iterations is met "
        "or when the tolerance criteria has been met.",
    )

    ramp_up_iters: PositiveInt = Field(
        default=1,
        title="Ramp-up iterations.",
        description="In order to help in start up, quantities such as doping "
        "are ramped up until they reach their specified value. This parameter "
        "determines how many of this iterations it takes to reach full values.",
    )

    max_pseudo_steps: PositiveInt = Field(
        default=60,
        title="Maximum pseudo steps.",
        description="Maximum number of pseudo time steps used per physical step "
        "in the drift-diffusion solver.",
    )

    cfl_number: PositiveFloat = Field(
        default=1e9,
        title="CFL number.",
        description="CFL multiplier used in the drift-diffusion solver. "
        "Controls the pseudo time step size.",
    )

    preconditioner_iterations: PositiveInt = Field(
        default=50,
        title="Preconditioner iterations.",
        description="Maximum number of preconditioner iterations in "
        "the linear solver of the drift-diffusion solver.",
    )

    @model_validator(mode="after")
    def _warn_non_default_solver_params(self) -> ChargeToleranceSpec:
        """Warn when solver parameters differ from their defaults."""
        field_names = ("max_pseudo_steps", "cfl_number", "preconditioner_iterations")
        fields = type(self).model_fields
        changed = [name for name in field_names if getattr(self, name) != fields[name].default]
        if changed:
            log.warning(
                f"Non-default values detected for {', '.join(changed)} in "
                "'ChargeToleranceSpec'. Settings different than the defaults can lead to "
                "long simulation times, lack of convergence, and divergence."
            )
        return self


class SteadyChargeDCAnalysis(Tidy3dBaseModel):
    """
    Configures relevant steady-state DC simulation parameters for a charge simulation.
    """

    tolerance_settings: ChargeToleranceSpec = Field(
        default=ChargeToleranceSpec(),
        title="Tolerance settings",
        description="Charge tolerance parameters relevant to multiple simulation analysis types.",
    )

    convergence_dv: PositiveFloat = Field(
        default=1.0,
        title="Bias step.",
        description="By default, a solution is computed at 0 bias. If a bias different than "
        "0 is requested through a voltage source, the charge solver will start at 0 and increase bias "
        "at `convergence_dv` intervals until the required bias is reached. This is, therefore, a "
        "convergence parameter in DC computations.",
    )

    fermi_dirac: bool = Field(
        False,
        title="Fermi-Dirac statistics",
        description="Determines whether Fermi-Dirac statistics are used. When ``False``, "
        "Boltzmann statistics will be used. This can provide more accurate results in situations "
        "where very high doping may lead the pseudo-Fermi energy level to approach "
        "either the conduction or valence energy bands.",
    )


class IsothermalSteadyChargeDCAnalysis(SteadyChargeDCAnalysis):
    """
    Configures relevant Isothermal steady-state DC simulation parameters for a charge simulation.
    """

    temperature: PositiveFloat = Field(
        300,
        title="Temperature",
        description="Lattice temperature. Assumed constant throughout the device. "
        "Carriers are assumed to be at thermodynamic equilibrium with the lattice.",
        json_schema_extra={"units": KELVIN},
    )
