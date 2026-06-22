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
    >>> charge_settings = td.ChargeToleranceSpec()
    """

    abs_tol: PositiveFloat = Field(
        default=1e10,
        title="Absolute tolerance.",
        description="Absolute tolerance used as stop criteria when converging towards a solution. "
        "This is honored by the legacy solver only; on the accelerated (default) solver, "
        "``rel_tol`` is the effective convergence criterion.",
    )

    rel_tol: PositiveFloat = Field(
        default=1e-10,
        title="Relative tolerance.",
        description="Relative tolerance used as stop criteria when converging towards a solution.",
    )

    max_iters: PositiveInt = Field(
        default=120,
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
        "Controls the pseudo time step size and acts as the upper bound "
        "of the adaptive CFL controller.",
    )

    cfl_min: PositiveFloat | None = Field(
        default=None,
        title="Minimum CFL number.",
        description="Lower bound of the adaptive CFL controller in the drift-diffusion "
        "solver. When ``None`` (default), the solver uses ``cfl_number * 1e-6`` so "
        "behaviour matches setups that predate this field. Setting ``cfl_min`` equal "
        "to ``cfl_number`` effectively runs the solver with a constant CFL (no "
        "adaptive backoff).",
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
        field_names = (
            "max_pseudo_steps",
            "cfl_number",
            "cfl_min",
            "preconditioner_iterations",
        )
        fields = type(self).model_fields
        changed = [name for name in field_names if getattr(self, name) != fields[name].default]
        if changed:
            log.warning(
                f"Non-default values detected for {', '.join(changed)} in "
                "'ChargeToleranceSpec'. Settings different than the defaults can lead to "
                "long simulation times, lack of convergence, and divergence."
            )
        return self

    @model_validator(mode="after")
    def _warn_cfl_min_above_max(self) -> ChargeToleranceSpec:
        """Warn when ``cfl_min`` exceeds ``cfl_number`` (the adaptive-CFL upper bound)."""
        if self.cfl_min is not None and self.cfl_min > self.cfl_number:
            log.warning(
                f"'cfl_min' ({self.cfl_min}) is greater than 'cfl_number' "
                f"({self.cfl_number}) in 'ChargeToleranceSpec'. The adaptive CFL "
                "controller expects cfl_min <= cfl_number; with these bounds "
                "inverted the controller will clamp to the (smaller) upper bound "
                "every step and the solver may not behave as intended."
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
        description="Maximum bias step used to aid convergence in DC computations. "
        "The accelerated solver applies it only to multi-voltage sweeps: where the gap "
        "between consecutive sweep voltages exceeds `convergence_dv`, intermediate "
        "warm-start bias points are inserted (and excluded from the output); a "
        "single-voltage simulation is solved directly. The legacy solver instead ramps "
        "every requested bias from 0 in `convergence_dv` increments.",
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
