"""
This class defines standard SPICE electrical_analysis types (electrical simulations configurations).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.data.data_array import SpatialDataArray
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
        "solver. The pseudo-time term is the transport diagonal divided by the CFL, "
        "so a high CFL reduces the effect of the pseudo-time damping during the "
        "nonlinear iterations. When ``None`` (default), the solver uses 1. Setting "
        "``cfl_min`` equal to ``cfl_number`` runs the solver at a constant CFL (no "
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

    Notes
    -----
        By default (``temperature=None``) the analysis solves the full non-isothermal
        system: self-heating and the lattice temperature are solved together.

        Supplying ``temperature`` instead *prescribes* the lattice temperature. The
        temperature-dependent physics (mobility, intrinsic carrier concentration, SRH
        lifetimes, ...) is still evaluated, but the field is held fixed and no thermal
        solve runs, so there is no feedback from the device onto the temperature. This is
        the input half of a manual charge -> heat -> charge iteration; the output half is
        :class:`.SelfHeatingMonitor`.

        For a spatially uniform lattice temperature, use
        :class:`.IsothermalSteadyChargeDCAnalysis`, which takes a scalar instead.

    Example
    -------
    >>> import numpy as np
    >>> import tidy3d as td
    >>> T = td.SpatialDataArray(
    ...     300 + np.zeros((2, 2, 2)), coords=dict(x=[0, 1], y=[0, 1], z=[0, 1])
    ... )
    >>> analysis = td.SteadyChargeDCAnalysis(temperature=T)
    """

    # Overridden per family (see 'ac.py'): answering an SSAC user with the DC class would have
    # them drop 'freqs'. A name, not the class, because 'ac' imports this module -- naming
    # 'IsothermalSSACAnalysis' here would be a cycle.
    _isothermal_equivalent: ClassVar[str] = "IsothermalSteadyChargeDCAnalysis"

    temperature: SpatialDataArray | None = Field(
        default=None,
        title="Prescribed lattice temperature",
        description="Fixed, spatially varying lattice temperature. When ``None`` (the "
        "default) the lattice temperature is solved for self-consistently with the "
        "charge transport. When set, the temperature-dependent physics is evaluated on "
        "this field, but no thermal solve is performed and the field is never updated. "
        "Values outside the field's bounding box are clamped to the nearest value on its "
        "boundary. Requires the accelerated charge solver.",
        json_schema_extra={"units": KELVIN},
    )

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
        default=False,
        title="Fermi-Dirac statistics",
        description="Determines whether Fermi-Dirac statistics are used. When ``False``, "
        "Boltzmann statistics will be used. This can provide more accurate results in situations "
        "where very high doping may lead the pseudo-Fermi energy level to approach "
        "either the conduction or valence energy bands.",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_scalar_temperature(cls, data: Any) -> Any:
        """Point a scalar ``temperature`` at the isothermal spec that actually takes one.

        The bare pydantic union error names neither class. Inherited by every subclass, so
        it must exempt the isothermal specs, where a scalar is correct -- including
        ``IsothermalSSACAnalysis``, which reaches that exemption by inheritance.
        """
        if not isinstance(data, dict) or issubclass(cls, IsothermalSteadyChargeDCAnalysis):
            return data
        temperature = data.get("temperature")
        if isinstance(temperature, (int, float, np.number)) and not isinstance(temperature, bool):
            raise ValueError(
                f"'{cls.__name__}.temperature' prescribes a spatially varying lattice "
                f"temperature and expects a 'SpatialDataArray', but got the scalar "
                f"{temperature}. For a uniform lattice temperature use "
                f"'{cls._isothermal_equivalent}(temperature={temperature})' instead."
            )
        return data

    @model_validator(mode="after")
    def _check_prescribed_temperature_is_usable(self) -> SteadyChargeDCAnalysis:
        """A prescribed lattice temperature must be positive, finite and ascending.

        All three are about a field the solver can sample rather than about the physics, so
        they share one registration. Values first: kT/q and exp(-Eg/2kT) divide
        by T, so a stray zero would surface deep in the solver rather than here. Then the
        coordinates: ``SpatialDataArray`` accepts any coordinate order, but the solver
        locates each mesh node by bisection and clamps against the first and last entry of
        each axis, so on a descending axis every node compares as "left of the data" and
        takes that axis's first plane -- a plausible temperature that was never sampled.
        Rejected rather than sorted silently, because reversed axes usually mean the values
        are transposed with respect to the intended geometry too.
        """
        temperature = self.temperature
        # A scalar on the isothermal subclass is already constrained by 'PositiveFloat'.
        if temperature is None or not isinstance(temperature, SpatialDataArray):
            return self

        values = np.asarray(temperature.values)
        if not np.all(np.isfinite(values)):
            self._raise_validation_error_at_loc(
                "'temperature' contains non-finite values (NaN or infinity). A prescribed "
                "lattice temperature must be finite everywhere.",
                "temperature",
            )
        if np.any(values <= 0):
            hint = ""
            if np.isclose(values.min(), 0.0):
                # 'to_spatial_data_array' defaults to fill_value=0.0: right for a source
                # term, 0 K for a temperature.
                hint = (
                    " If this field came from 'to_spatial_data_array', pass a physical "
                    "'fill_value' (the ambient temperature): its default of 0.0 is meant "
                    "for source terms, not temperatures."
                )
            self._raise_validation_error_at_loc(
                f"'temperature' must be strictly positive everywhere, but its minimum is "
                f"{values.min():.4g} K. A prescribed lattice temperature is an absolute "
                f"temperature in Kelvin.{hint}",
                "temperature",
            )

        for axis in "xyz":
            coords = np.atleast_1d(np.asarray(temperature.coords[axis].values))
            if coords.size > 1 and np.any(coords[1:] <= coords[:-1]):
                self._raise_validation_error_at_loc(
                    f"'temperature' has a '{axis}' coordinate that is not strictly "
                    f"ascending. Sort the field along every axis before prescribing it, "
                    f"e.g. 'T.sortby(['x', 'y', 'z'])'.",
                    "temperature",
                )
        return self


class IsothermalSteadyChargeDCAnalysis(SteadyChargeDCAnalysis):
    """
    Configures relevant Isothermal steady-state DC simulation parameters for a charge simulation.

    Notes
    -----
        Narrows the base class's ``temperature`` to a scalar, so a uniform lattice
        temperature and a prescribed field are mutually exclusive by construction. Like a
        prescribed field, a uniform temperature runs no thermal solve.
    """

    temperature: PositiveFloat = Field(
        default=300,
        title="Temperature",
        description="Lattice temperature. Assumed constant throughout the device. "
        "Carriers are assumed to be at thermodynamic equilibrium with the lattice.",
        json_schema_extra={"units": KELVIN},
    )
