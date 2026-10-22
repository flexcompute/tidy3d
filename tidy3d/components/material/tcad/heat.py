"""Defines heat material specifications"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial import polynomial
from pydantic import Field, FiniteFloat, NonNegativeFloat, PositiveFloat

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.material.tensor import AbstractPrincipalAxesTensor
from tidy3d.components.types import Coordinate
from tidy3d.constants import (
    DENSITY,
    DYNAMIC_VISCOSITY,
    KELVIN,
    SPECIFIC_HEAT,
    SPECIFIC_HEAT_CAPACITY,
    THERMAL_CONDUCTIVITY,
    THERMAL_EXPANSIVITY,
    VELOCITY,
)
from tidy3d.exceptions import ValidationError

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.types.base import ArrayFloat

# The constant term is fixed at 1, so this is also the polynomial degree.
POLYNOMIAL_MODEL_MAX_COEFFICIENTS = 4


# Liquid class
class AbstractHeatMedium(ABC, Tidy3dBaseModel):
    """Abstract heat material specification."""

    name: str | None = Field(
        default=None, title="Name", description="Optional unique name for medium."
    )

    @property
    def heat(self) -> Self:
        """
        This means that a heat medium has been defined inherently within this solver medium.
        This provides interconnection with the `MultiPhysicsMedium` higher-dimensional classes.
        """
        return self

    @property
    def charge(self) -> None:
        raise ValueError(f"A `charge` medium does not exist in this Medium definition: {self}")

    @property
    def electrical(self) -> None:
        raise ValueError(f"An `electrical` medium does not exist in this Medium definition: {self}")

    @property
    def optical(self) -> None:
        raise ValueError(f"An `optical` medium does not exist in this Medium definition: {self}")


class FluidMedium(AbstractHeatMedium):
    """Fluid medium. Heat simulations will not solve for temperature
    in a structure that has a medium with this ``heat_spec``.


    Notes
    --------
    The full set of parameters is primarily intended for calculations involving natural
    convection, where they are used to determine the heat transfer coefficient.
    In the current version, these specific properties may not be utilized for
    other boundary condition types.

    Examples
    --------
    >>> # If you are using a boundary condition without a natural convection model,
    >>> # the specific properties of the fluid are not required. In this common
    >>> # scenario, you can instantiate the class without arguments.
    >>> air = FluidMedium()

    >>> # It is most convenient to define the fluid from standard SI units
    >>> # using the `from_si_units` classmethod.
    >>> # The following defines air at approximately 20°C.
    >>> air_from_si = FluidMedium.from_si_units(
    ...     thermal_conductivity=0.0257,  # Unit: W/(m*K)
    ...     viscosity=1.81e-5,          # Unit: Pa*s
    ...     specific_heat=1005,         # Unit: J/(kg*K)
    ...     density=1.204,              # Unit: kg/m^3
    ...     expansivity=1/293.15        # Unit: 1/K
    ... )

    >>> # One can also define the medium directly in Tidy3D units.
    >>> # The following is equivalent to the example above.
    >>> air_direct = FluidMedium(
    ...     thermal_conductivity=2.57e-8,
    ...     viscosity=1.81e-11,
    ...     specific_heat=1.005e+15,
    ...     density=1.204e-18,
    ...     expansivity=1/293.15
    ... )
    """

    thermal_conductivity: NonNegativeFloat | None = Field(
        default=None,
        title="Fluid Thermal Conductivity",
        description="Thermal conductivity (k) of the fluid.",
        json_schema_extra={"units": THERMAL_CONDUCTIVITY},
    )
    viscosity: NonNegativeFloat | None = Field(
        default=None,
        title="Fluid Dynamic Viscosity",
        description="Dynamic viscosity (μ) of the fluid.",
        json_schema_extra={"units": DYNAMIC_VISCOSITY},
    )
    specific_heat: NonNegativeFloat | None = Field(
        default=None,
        title="Fluid Specific Heat",
        description="Specific heat of the fluid at constant pressure, in units of "
        f"{SPECIFIC_HEAT}, which is 1e12 times the SI value in {SPECIFIC_HEAT_CAPACITY}. "
        "Prefer ``FluidMedium.from_si_units()``, which takes every property in SI units and "
        "applies the conversion.",
        json_schema_extra={"units": SPECIFIC_HEAT},
    )
    density: NonNegativeFloat | None = Field(
        default=None,
        title="Fluid Density",
        description="Density (ρ) of the fluid.",
        json_schema_extra={"units": DENSITY},
    )
    expansivity: NonNegativeFloat | None = Field(
        default=None,
        title="Fluid Thermal Expansivity",
        description="Thermal expansion coefficient (β) of the fluid.",
        json_schema_extra={"units": THERMAL_EXPANSIVITY},
    )

    @classmethod
    def from_si_units(
        cls,
        thermal_conductivity: NonNegativeFloat,
        viscosity: NonNegativeFloat,
        specific_heat: NonNegativeFloat,
        density: NonNegativeFloat,
        expansivity: NonNegativeFloat,
    ) -> Self:
        """Create a FluidMedium from SI units, converting each property to Tidy3D units.

        This is the recommended way to build a fluid, since every property except
        ``expansivity`` differs from its SI unit.
        """
        thermal_conductivity_tidy = thermal_conductivity / 1e6  # W/(m*K) -> W/(um*K)
        viscosity_tidy = viscosity / 1e6  # Pa*s -> kg/(um*s)
        specific_heat_tidy = specific_heat * 1e12  # J/(kg*K) -> um**2/(s**2*K)
        density_tidy = density / 1e18  # kg/m**3 -> kg/um**3
        expansivity_tidy = expansivity  # 1/K -> 1/K (no change)

        return cls(
            thermal_conductivity=thermal_conductivity_tidy,
            viscosity=viscosity_tidy,
            specific_heat=specific_heat_tidy,
            density=density_tidy,
            expansivity=expansivity_tidy,
        )


class FluidSpec(FluidMedium):
    """Fluid medium class for backwards compatibility"""


class AnisotropicConductivity(AbstractPrincipalAxesTensor):
    """Anisotropic (tensor) thermal conductivity of a solid medium.

    Notes
    -----
    Specified by the principal conductivities ``xx``, ``yy``, ``zz`` along local axes,
    plus an optional ``rotation`` that orients those axes in the global frame. The
    resulting conductivity tensor is symmetric positive-definite (SPD) by construction
    (positive principals rotated by an orthonormal matrix), and the heat flux is
    ``q = -K . grad(T)``.

    Only the SPD class is representable. Non-symmetric / nonreciprocal thermal transport
    (for example thermal Hall / Righi-Leduc effects under a magnetic field) and indefinite
    or active effective models are out of scope. Use :meth:`from_components` to build an
    instance from the six symmetric components ``[kxx, kyy, kzz, kxy, kxz, kyz]`` directly
    (they are diagonalized into the equivalent principal-axes-plus-rotation form).

    Example
    -------
    >>> aniso = AnisotropicConductivity(xx=1.0, yy=2.0, zz=3.0)
    >>> aniso = AnisotropicConductivity.from_components(kxx=2.0, kyy=3.0, kzz=4.0, kxy=0.5)
    """

    xx: PositiveFloat = Field(
        title="Principal conductivity xx",
        description="Principal thermal conductivity along the local x-axis, in units of "
        f"{THERMAL_CONDUCTIVITY}.",
        json_schema_extra={"units": THERMAL_CONDUCTIVITY},
    )
    yy: PositiveFloat = Field(
        title="Principal conductivity yy",
        description="Principal thermal conductivity along the local y-axis, in units of "
        f"{THERMAL_CONDUCTIVITY}.",
        json_schema_extra={"units": THERMAL_CONDUCTIVITY},
    )
    zz: PositiveFloat = Field(
        title="Principal conductivity zz",
        description="Principal thermal conductivity along the local z-axis, in units of "
        f"{THERMAL_CONDUCTIVITY}.",
        json_schema_extra={"units": THERMAL_CONDUCTIVITY},
    )

    @classmethod
    def from_components(
        cls,
        kxx: float,
        kyy: float,
        kzz: float,
        kxy: float = 0.0,
        kxz: float = 0.0,
        kyz: float = 0.0,
    ) -> AnisotropicConductivity:
        """Build from the six symmetric tensor components in the global frame.

        The components define the symmetric matrix ``[[kxx, kxy, kxz], [kxy, kyy, kyz],
        [kxz, kyz, kzz]]``, which must be positive-definite. It is diagonalized into
        principal conductivities (its eigenvalues) plus a ``rotation`` orienting the
        principal axes, i.e. the equivalent principal-axes-plus-rotation form. This adds no
        new physics: only symmetric positive-definite tensors are supported.
        The principal values come out sorted ascending, so ``xx`` is the smallest eigenvalue
        rather than ``kxx``.

        Parameters
        ----------
        kxx : float
            ``xx`` component of the conductivity tensor.
        kyy : float
            ``yy`` component of the conductivity tensor.
        kzz : float
            ``zz`` component of the conductivity tensor.
        kxy : float = 0.0
            Off-diagonal ``xy`` (= ``yx``) component.
        kxz : float = 0.0
            Off-diagonal ``xz`` (= ``zx``) component.
        kyz : float = 0.0
            Off-diagonal ``yz`` (= ``zy``) component.

        Example
        -------
        >>> aniso = AnisotropicConductivity.from_components(kxx=2.0, kyy=3.0, kzz=4.0, kxy=0.5)
        """
        eigvals, rotation = cls._diagonalize(kxx, kyy, kzz, kxy, kxz, kyz)
        # Symmetric by construction; require positive-definite (all eigenvalues > 0).
        if not np.all(eigvals > 0):
            raise ValidationError(
                "'AnisotropicConductivity.from_components' requires a positive-definite "
                f"conductivity tensor, but the provided components give eigenvalues {eigvals}. "
                "Only symmetric positive-definite conductivities are supported."
            )
        return cls(
            xx=float(eigvals[0]),
            yy=float(eigvals[1]),
            zz=float(eigvals[2]),
            rotation=rotation,
        )


class AbstractThermalPropertyModel(ABC, Tidy3dBaseModel):
    """Temperature-dependent thermal property of a solid: a reference value, the property at
    ``reference_temperature``, times a dimensionless factor ``f(T)`` with ``f = 1`` there.

    Notes
    -----
    The temperature is clamped into ``temperature_range`` before ``f`` is evaluated, so the
    model is never extrapolated. ``f`` must be positive over the whole range.
    """

    reference_temperature: PositiveFloat = Field(
        default=300.0,
        title="Reference temperature",
        description="Temperature at which the property equals its reference value. Must lie "
        "within ``temperature_range``.",
        json_schema_extra={"units": KELVIN},
    )

    temperature_range: tuple[NonNegativeFloat, PositiveFloat] = Field(
        title="Temperature range",
        description="Temperatures ``(T_min, T_max)`` over which the model is valid. Outside "
        "this range the property is held at its value at the nearest end.",
        json_schema_extra={"units": KELVIN},
    )

    @abstractmethod
    def _factor(self, temperature: np.ndarray) -> np.ndarray:
        """Factor at temperatures already clamped into ``temperature_range``."""

    def _clamped_factor(self, temperature: ArrayFloat) -> np.ndarray:
        """Factor at ``temperature`` (in K), clamped into ``temperature_range`` first."""
        clamped = np.clip(np.asarray(temperature, dtype=float), *self.temperature_range)
        return self._factor(clamped)


class AbstractPowerLawModel(AbstractThermalPropertyModel):
    """Power-law factor ``f(T) = (T / reference_temperature) ** exponent``."""

    exponent: FiniteFloat = Field(
        title="Exponent",
        description="Power-law exponent.",
    )

    def _factor(self, temperature: np.ndarray) -> np.ndarray:
        return (temperature / self.reference_temperature) ** self.exponent


class AbstractPolynomialModel(AbstractThermalPropertyModel):
    """Model built on ``p(T) = 1 + a1 dT + ... + a4 dT^4`` with
    ``dT = T - reference_temperature``, required to be positive over ``temperature_range``."""

    coefficients: tuple[FiniteFloat, ...] = Field(
        title="Temperature coefficients",
        description="Coefficients ``(a1, a2, ...)`` of increasing powers of "
        "``T - reference_temperature`` in the polynomial ``1 + a1 dT + ... + a4 dT^4``, at most "
        f"{POLYNOMIAL_MODEL_MAX_COEFFICIENTS} values. ``a_i`` has units of ``1/K^i``. For "
        "``InversePolynomialConductivity`` the polynomial is the denominator.",
        min_length=1,
        max_length=POLYNOMIAL_MODEL_MAX_COEFFICIENTS,
    )

    @property
    def _coefficients_in_dT(self) -> tuple[float, ...]:
        """Coefficients of ``p`` in increasing powers of ``T - reference_temperature``."""
        return (1.0, *self.coefficients)

    def _coefficients_in_T(self) -> tuple[float, ...]:
        """Coefficients of ``p`` in increasing powers of ``T`` itself."""
        shifted = polynomial.Polynomial(self._coefficients_in_dT)(
            polynomial.Polynomial([-self.reference_temperature, 1.0])
        )
        return tuple(float(c) for c in shifted.coef)

    def _polynomial(self, temperature: np.ndarray) -> np.ndarray:
        return polynomial.polyval(
            temperature - self.reference_temperature, self._coefficients_in_dT
        )

    def _factor(self, temperature: np.ndarray) -> np.ndarray:
        return self._polynomial(temperature)


class AbstractConductivityModel(AbstractThermalPropertyModel):
    """Temperature-dependent thermal conductivity ``K(T) = k_ref * f(T)``. A tensor ``k_ref``
    is scaled as a whole, so its anisotropy ratios and principal axes stay fixed."""

    k_ref: PositiveFloat | AnisotropicConductivity = Field(
        title="Reference conductivity",
        description="Thermal conductivity at ``reference_temperature`` in units of "
        f"{THERMAL_CONDUCTIVITY}. Either an isotropic scalar or an ``AnisotropicConductivity``.",
        json_schema_extra={"units": THERMAL_CONDUCTIVITY},
    )

    def conductivity(self, temperature: ArrayFloat) -> np.ndarray:
        """Evaluate the thermal conductivity.

        Parameters
        ----------
        temperature : ArrayFloat
            Temperature(s) in K. Values outside ``temperature_range`` are clamped into it.

        Returns
        -------
        np.ndarray
            ``K(T)`` in units of W/(um*K), shaped like ``temperature``. For an
            ``AnisotropicConductivity`` ``k_ref``, with a trailing axis of the six packed
            components ``[kxx, kyy, kzz, kxy, kxz, kyz]``.
        """
        factor = self._clamped_factor(temperature)
        if isinstance(self.k_ref, AnisotropicConductivity):
            return np.multiply.outer(factor, self.k_ref.to_tensor())
        return factor * self.k_ref


class AbstractHeatCapacityModel(AbstractThermalPropertyModel):
    """Temperature-dependent specific heat capacity ``cp(T) = cp_ref * f(T)``."""

    cp_ref: PositiveFloat = Field(
        title="Reference heat capacity",
        description="Specific heat capacity at ``reference_temperature`` in units of "
        f"{SPECIFIC_HEAT_CAPACITY}.",
        json_schema_extra={"units": SPECIFIC_HEAT_CAPACITY},
    )

    def heat_capacity(self, temperature: ArrayFloat) -> np.ndarray:
        """Evaluate the specific heat capacity.

        Parameters
        ----------
        temperature : ArrayFloat
            Temperature(s) in K. Values outside ``temperature_range`` are clamped into it.

        Returns
        -------
        np.ndarray
            ``cp(T)`` in units of J/(kg*K), shaped like ``temperature``.
        """
        return self._clamped_factor(temperature) * self.cp_ref


class PowerLawConductivity(AbstractConductivityModel, AbstractPowerLawModel):
    """Power-law thermal conductivity ``K(T) = k_ref * (T / reference_temperature) ** exponent``.

    Notes
    -----
    A negative exponent describes the falling conductivity of crystalline semiconductors,
    e.g. about ``-1.3`` for silicon near room temperature.

    Example
    -------
    >>> silicon_k = PowerLawConductivity(k_ref=1.48e-4, exponent=-1.3, temperature_range=(200, 800))
    >>> tensor_k = PowerLawConductivity(
    ...     k_ref=AnisotropicConductivity(xx=1e-4, yy=1e-4, zz=5e-5),
    ...     exponent=-1.0,
    ...     temperature_range=(200, 800),
    ... )
    """


class PolynomialConductivity(AbstractConductivityModel, AbstractPolynomialModel):
    """Polynomial thermal conductivity
    ``K(T) = k_ref * (1 + a1 dT + ... + a4 dT^4)``, with ``dT = T - reference_temperature``.

    Example
    -------
    >>> # 0.1% per K rise around 300 K
    >>> k_model = PolynomialConductivity(
    ...     k_ref=1e-6, coefficients=(1e-3,), temperature_range=(100, 1200)
    ... )
    """


class InversePolynomialConductivity(AbstractConductivityModel, AbstractPolynomialModel):
    """Thermal conductivity from a thermal-resistivity polynomial
    ``K(T) = k_ref / (1 + b1 dT + ... + b4 dT^4)``, with ``dT = T - reference_temperature``.

    Notes
    -----
    Suited to the common ``1 / (a + b T + c T^2)`` fits of crystalline semiconductors.

    Example
    -------
    >>> k_model = InversePolynomialConductivity(
    ...     k_ref=1.48e-4, coefficients=(3e-3, 2e-6), temperature_range=(250, 600)
    ... )
    >>> # Silicon from a published resistivity fit 0.03 + 1.56e-3 T + 1.65e-6 T^2 in cm*K/W
    >>> silicon_k = InversePolynomialConductivity.from_resistivity(
    ...     coefficients=(300, 15.6, 1.65e-2), temperature_range=(250, 600)
    ... )
    """

    @classmethod
    def from_resistivity(
        cls,
        coefficients: tuple[float, ...],
        temperature_range: tuple[float, float],
        reference_temperature: float = 300.0,
    ) -> Self:
        """Create from a thermal-resistivity fit ``W(T) = w0 + w1 T + ... + w4 T^4`` in powers
        of ``T`` itself, the form such fits are usually published in.

        Parameters
        ----------
        coefficients : tuple[float, ...]
            ``(w0, w1, ...)``, at most five values, with ``w_i`` in units of um*K/W/K^i
            (``1 cm*K/W = 1e4 um*K/W``).
        temperature_range : tuple[float, float]
            Temperatures ``(T_min, T_max)`` in K over which the fit is valid.
        reference_temperature : float = 300.0
            Temperature in K at which ``k_ref = 1 / W(reference_temperature)``.

        Returns
        -------
        :class:`InversePolynomialConductivity`
            Model with ``K(T) = 1 / W(T)`` over ``temperature_range``.
        """
        # Re-center W on the reference temperature and normalize it by W(reference_temperature).
        shifted = polynomial.Polynomial(coefficients)(
            polynomial.Polynomial([reference_temperature, 1.0])
        ).coef
        w_ref = float(shifted[0])
        if not w_ref > 0:
            raise ValidationError(
                "'InversePolynomialConductivity.from_resistivity' requires a positive "
                f"resistivity at the reference temperature, got W({reference_temperature}) = "
                f"{w_ref:.6g}."
            )
        return cls(
            k_ref=1.0 / w_ref,
            coefficients=tuple(float(c) / w_ref for c in shifted[1:]) or (0.0,),
            reference_temperature=reference_temperature,
            temperature_range=temperature_range,
        )

    def _factor(self, temperature: np.ndarray) -> np.ndarray:
        return 1.0 / self._polynomial(temperature)


class PowerLawHeatCapacity(AbstractHeatCapacityModel, AbstractPowerLawModel):
    """Power-law specific heat capacity
    ``cp(T) = cp_ref * (T / reference_temperature) ** exponent``.

    Example
    -------
    >>> cp_model = PowerLawHeatCapacity(cp_ref=700, exponent=0.3, temperature_range=(200, 800))
    """


class PolynomialHeatCapacity(AbstractHeatCapacityModel, AbstractPolynomialModel):
    """Polynomial specific heat capacity
    ``cp(T) = cp_ref * (1 + a1 dT + ... + a4 dT^4)``, with ``dT = T - reference_temperature``.

    Example
    -------
    >>> # 0.2% per K rise around 300 K
    >>> cp_model = PolynomialHeatCapacity(
    ...     cp_ref=700, coefficients=(2e-3,), temperature_range=(100, 1200)
    ... )
    """


ConductivityModelType = (
    PowerLawConductivity | PolynomialConductivity | InversePolynomialConductivity
)
HeatCapacityModelType = PowerLawHeatCapacity | PolynomialHeatCapacity


class SolidMedium(AbstractHeatMedium):
    """Solid medium for heat simulations.

    Example
    -------
    >>> solid = SolidMedium(
    ...     capacity=2,
    ...     conductivity=3,
    ... )

    >>> # It is most convenient to define the solid from standard SI units
    >>> # using the `from_si_units` classmethod. The following defines silicon.
    >>> silicon = SolidMedium.from_si_units(
    ...     conductivity=148,   # Unit: W/(m*K)
    ...     capacity=700,       # Unit: J/(kg*K)
    ...     density=2330,       # Unit: kg/m^3
    ... )

    >>> # One can also define the medium directly in Tidy3D units.
    >>> # The following is equivalent to the example above. Note that `capacity` is
    >>> # unchanged, since J/(kg*K) is already the Tidy3D convention for solids, while
    >>> # `conductivity` and `density` are scaled.
    >>> silicon_direct = SolidMedium(
    ...     conductivity=1.48e-4,
    ...     capacity=700,
    ...     density=2.33e-15,
    ... )

    >>> # Conductivity and capacity also accept temperature-dependent models. The following
    >>> # silicon has k ~ T^-1.3 around 300 K, from SI units.
    >>> silicon_hot = SolidMedium.from_si_units(
    ...     conductivity=PowerLawConductivity(
    ...         k_ref=148, exponent=-1.3, temperature_range=(200, 800)
    ...     ),
    ...     capacity=700,
    ...     density=2330,
    ... )
    """

    capacity: PositiveFloat | HeatCapacityModelType | None = Field(
        default=None,
        title="Heat capacity",
        description=f"Specific heat capacity in units of {SPECIFIC_HEAT_CAPACITY}, which is "
        "already the SI unit, so no conversion is needed for this field. Prefer "
        "``SolidMedium.from_si_units()``, which takes every property in SI units and converts "
        "``conductivity`` and ``density``, both of which do differ from their SI units. "
        "Either a constant, or a temperature-dependent model (``PowerLawHeatCapacity``, "
        "``PolynomialHeatCapacity``), used in unsteady heat simulations and in the convective "
        "term of an advecting solid. A non-isothermal charge analysis is steady, so a "
        "temperature-dependent model has no effect there; it still requires "
        "``use_accelerated_solver=True`` (the default).",
        json_schema_extra={"units": SPECIFIC_HEAT_CAPACITY},
    )

    conductivity: PositiveFloat | AnisotropicConductivity | ConductivityModelType = Field(
        title="Thermal conductivity",
        description="Thermal conductivity of material in units of "
        f"{THERMAL_CONDUCTIVITY}. Either an isotropic scalar, an "
        "``AnisotropicConductivity`` giving a symmetric positive-definite (optionally "
        "rotated) tensor, or a temperature-dependent model (``PowerLawConductivity``, "
        "``PolynomialConductivity``, ``InversePolynomialConductivity``) whose reference value "
        "is either of those. The tensor and temperature-dependent forms are applied by the "
        "heat solver, including when heat is coupled with electrical conduction. In "
        "non-isothermal charge (coupled charge+heat) simulations both require "
        "``use_accelerated_solver=True`` (the default).",
        json_schema_extra={"units": THERMAL_CONDUCTIVITY},
    )

    density: PositiveFloat | None = Field(
        default=None,
        title="Density",
        description=f"Mass density of material in units of {DENSITY}.",
        json_schema_extra={"units": DENSITY},
    )

    velocity: Coordinate | None = Field(
        default=None,
        title="Advection velocity",
        description="Constant advection velocity ``(vx, vy, vz)`` of the solid medium in units "
        f"of {VELOCITY}. When set to a nonzero value, the heat solver adds a convective "
        "transport term ``V . grad(H)`` for structures using this medium, where the enthalpy "
        "``H`` has ``dH/dT = density * capacity`` (``rho * cp * V . grad(T)`` for a constant "
        "``capacity``); both ``capacity`` and ``density`` must then be provided. Leave as "
        "``None`` (the default) for pure conduction. A velocity component normal to an "
        "interface between two solids is not allowed. This is not validated, and a simulation "
        "that has one gives physically inconsistent results. Velocities tangential to such an "
        "interface are fine. This term is applied by the heat solver, including when heat is coupled "
        "with electrical conduction and in non-isothermal charge simulations run by the "
        "accelerated charge solver. There a semiconductor cannot have a nonzero ``velocity``, "
        "since its carriers are not advected, and it is not supported with "
        "``use_accelerated_solver=False``.",
        json_schema_extra={"units": VELOCITY},
    )

    @property
    def _reference_conductivity(self) -> float | AnisotropicConductivity:
        """Conductivity without its temperature dependence: a model's ``k_ref``, otherwise
        ``conductivity`` itself."""
        if isinstance(self.conductivity, AbstractConductivityModel):
            return self.conductivity.k_ref
        return self.conductivity

    @property
    def _reference_capacity(self) -> float | None:
        """Heat capacity without its temperature dependence: a model's ``cp_ref``, otherwise
        ``capacity`` itself."""
        if isinstance(self.capacity, AbstractHeatCapacityModel):
            return self.capacity.cp_ref
        return self.capacity

    @classmethod
    def from_si_units(
        cls,
        conductivity: PositiveFloat | AnisotropicConductivity | ConductivityModelType,
        capacity: PositiveFloat | HeatCapacityModelType | None = None,
        density: PositiveFloat | None = None,
        velocity: Coordinate | None = None,
    ) -> Self:
        """Create a SolidMedium using SI units. ``conductivity`` may be a scalar, an
        ``AnisotropicConductivity`` or a temperature-dependent model given in SI units; the
        principals and a model's ``k_ref`` are converted too. ``capacity`` is passed through
        unchanged, since J/(kg*K) is both the SI unit and the Tidy3D convention for solids."""

        def to_tidy3d_conductivity(
            value: PositiveFloat | AnisotropicConductivity,
        ) -> PositiveFloat | AnisotropicConductivity:
            # W/(m*K) -> W/(um*K). For a tensor, scale the principals (rotation is unitless).
            if isinstance(value, AnisotropicConductivity):
                return value.updated_copy(
                    xx=value.xx * 1e-6, yy=value.yy * 1e-6, zz=value.zz * 1e-6
                )
            return value * 1e-6

        # A model's temperature coefficients are relative, so only its reference is scaled.
        if isinstance(conductivity, AbstractConductivityModel):
            new_conductivity = conductivity.updated_copy(
                k_ref=to_tidy3d_conductivity(conductivity.k_ref)
            )
        else:
            new_conductivity = to_tidy3d_conductivity(conductivity)
        new_capacity = capacity
        new_density = density
        new_velocity = velocity

        if density is not None:
            new_density = density * 1e-18

        if velocity is not None:
            new_velocity = tuple(v * 1e6 for v in velocity)  # Convert from m/s to um/s

        return cls(
            capacity=new_capacity,
            conductivity=new_conductivity,
            density=new_density,
            velocity=new_velocity,
        )


class SolidSpec(SolidMedium):
    """Solid medium class for backwards compatibility"""


ThermalSpecType = FluidSpec | SolidSpec | SolidMedium | FluidMedium
# Note this needs to remain here to avoid circular imports in the new medium structure.
