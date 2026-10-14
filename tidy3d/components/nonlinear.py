"""Defines nonlinear models and specifications"""

from __future__ import annotations

from abc import ABC

from pydantic import Field, NonNegativeFloat, PositiveFloat, PositiveInt

from tidy3d.constants import MICROMETER, SECOND, VOLT, WATT

from .base import Tidy3dBaseModel
from .types.base import discriminated_union

NONLINEAR_DEFAULT_NUM_ITERS = 5


class NonlinearModel(ABC, Tidy3dBaseModel):
    """Abstract model for a nonlinear material response.
    Used as part of a :class:`.NonlinearSpec`."""

    @property
    def complex_fields(self) -> bool:
        """Whether the model uses complex fields."""
        return False

    @property
    def aux_fields(self) -> list[str]:
        """List of available aux fields in this model."""
        return []


class NonlinearSusceptibility(NonlinearModel):
    """Model for an instantaneous nonlinear chi3 susceptibility.
    The expression for the instantaneous nonlinear polarization is given below.

    Notes
    -----

        This model uses real time-domain fields, so :math:`\\chi_3` must be real.

        .. math::

            P_{NL} = \\varepsilon_0 \\chi_3 |E|^2 E

        The nonlinear constitutive relation is solved iteratively; it may not converge
        for strong nonlinearities. Increasing :attr:`tidy3d.NonlinearSpec.num_iters` can
        help with convergence.

        For complex fields (e.g. when using Bloch boundary conditions), the nonlinearity
        is applied separately to the real and imaginary parts, so that the above equation
        holds when both :math:`E` and :math:`P_{NL}` are replaced by their real or imaginary parts.
        The nonlinearity is only applied to the real-valued fields since they are the
        physical fields.

        Different field components do not interact nonlinearly. For example,
        when calculating :math:`P_{NL, x}`, we approximate :math:`|E|^2 \\approx |E_x|^2`.
        This approximation is valid when the :math:`E` field is predominantly polarized along one
        of the ``x``, ``y``, or ``z`` axes.

        .. TODO add links to notebooks here.

    Example
    -------
    >>> nonlinear_susceptibility = NonlinearSusceptibility(chi3=1)
    """

    chi3: float = Field(
        default=0,
        title="Chi3",
        description=":math:`\\chi_3` nonlinear susceptibility.",
        json_schema_extra={"units": f"{MICROMETER}^2 / {VOLT}^2"},
    )

    numiters: PositiveInt | None = Field(
        default=None,
        title="Number of iterations",
        description="Deprecated. The old usage ``nonlinear_spec=model`` with ``model.numiters`` "
        "is deprecated and will be removed in a future release. The new usage is "
        "``nonlinear_spec=NonlinearSpec(models=[model], num_iters=num_iters)``. Under the new "
        "usage, this parameter is ignored, and ``NonlinearSpec.num_iters`` is used instead.",
    )


class TwoPhotonAbsorption(NonlinearModel):
    """Model for two-photon absorption (TPA) nonlinearity which gives an intensity-dependent
    absorption of the form :math:`\\alpha = \\alpha_0 + \\beta I`.
    Also includes free-carrier absorption (FCA) and free-carrier plasma dispersion (FCPD) effects.
    The expression for the nonlinear polarization is given below.

    Notes
    -----

        This model uses real time-domain fields, so :math:`\\beta` must be real.

        .. math::

            P_{NL} = P_{TPA} + P_{FCA} + P_{FCPD} \\\\
            P_{TPA} = -\\frac{4}{3}\\frac{c_0^2 \\varepsilon_0^2 n_0^2 \\beta}{2 i \\omega} |E|^2 E \\\\
            P_{FCA} = -\\frac{c_0 \\varepsilon_0 n_0 \\sigma N_f}{i \\omega} E \\\\
            \\frac{dN_f}{dt} = \\frac{8}{3}\\frac{c_0^2 \\varepsilon_0^2 n_0^2 \\beta}{8 q_e \\hbar \\omega} |E|^4 - \\frac{N_f}{\\tau} \\\\
            N_e = N_h = N_f \\\\
            P_{FCPD} = \\varepsilon_0 2 n_0 \\Delta n (N_f) E \\\\
            \\Delta n (N_f) = (c_e N_e^{e_e} + c_h N_h^{e_h})

        In these equations, :math:`n_0` means the real part of the linear
        refractive index of the medium.

        The nonlinear constitutive relation is solved iteratively; it may not converge
        for strong nonlinearities. Increasing :attr:`tidy3d.NonlinearSpec.num_iters` can
        help with convergence.

        For complex fields (e.g. when using Bloch boundary conditions), the nonlinearity
        is applied separately to the real and imaginary parts, so that the above equation
        holds when both :math:`E` and :math:`P_{NL}` are replaced by their real or imaginary parts.
        The nonlinearity is only applied to the real-valued fields since they are the
        physical fields.

        Different field components do not interact nonlinearly. For example,
        when calculating :math:`P_{NL, x}`, we approximate :math:`|E|^2 \\approx |E_x|^2`.
        This approximation is valid when the :math:`E` field is predominantly polarized along one
        of the ``x``, ``y``, or ``z`` axes.

        The implementation is described in::

            N. Suzuki, "FDTD Analysis of Two-Photon Absorption and Free-Carrier Absorption in Si
            High-Index-Contrast Waveguides," J. Light. Technol. 25, 9 (2007).

        .. TODO add links to notebooks here.

    Example
    -------
    >>> tpa_model = TwoPhotonAbsorption(beta=1)
    """

    beta: float = Field(
        default=0,
        title="TPA coefficient",
        description="Coefficient for two-photon absorption (TPA).",
        json_schema_extra={"units": f"{MICROMETER} / {WATT}"},
    )

    tau: NonNegativeFloat = Field(
        default=0,
        title="Carrier lifetime",
        description="Lifetime for the free carriers created by two-photon absorption (TPA).",
        json_schema_extra={"units": f"{SECOND}"},
    )

    sigma: NonNegativeFloat = Field(
        default=0,
        title="FCA cross section",
        description="Total cross section for free-carrier absorption (FCA). "
        "Contains contributions from electrons and from holes.",
        json_schema_extra={"units": f"{MICROMETER}^2"},
    )
    e_e: NonNegativeFloat = Field(
        default=1,
        title="Electron exponent",
        description="Exponent for the free electron refractive index shift in the free-carrier plasma dispersion (FCPD).",
    )
    e_h: NonNegativeFloat = Field(
        default=1,
        title="Hole exponent",
        description="Exponent for the free hole refractive index shift in the free-carrier plasma dispersion (FCPD).",
    )
    c_e: float = Field(
        default=0,
        title="Electron coefficient",
        description="Coefficient for the free electron refractive index shift in the free-carrier plasma dispersion (FCPD).",
        json_schema_extra={"units": f"{MICROMETER}^(3 e_e)"},
    )
    c_h: float = Field(
        default=0,
        title="Hole coefficient",
        description="Coefficient for the free hole refractive index shift in the free-carrier plasma dispersion (FCPD).",
        json_schema_extra={"units": f"{MICROMETER}^(3 e_h)"},
    )

    n0: float | None = Field(
        default=None,
        title="Linear refractive index",
        description="Real linear refractive index of the medium, computed for instance using "
        "'medium.nk_model'. If not provided, it is calculated automatically using the central "
        "frequencies of the simulation sources (as long as these are all equal).",
    )

    freq0: PositiveFloat | None = Field(
        default=None,
        title="Central frequency",
        description="Central frequency, used to calculate the energy of the free-carriers "
        "excited by two-photon absorption. If not provided, it is obtained automatically "
        "from the simulation sources (as long as these are all equal).",
    )

    @property
    def aux_fields(self) -> list[str]:
        """List of available aux fields in this model."""
        if self.tau == 0:
            return []
        return ["Nfx", "Nfy", "Nfz"]


class KerrNonlinearity(NonlinearModel):
    """Model for Kerr nonlinearity which gives an intensity-dependent refractive index
    of the form :math:`n = n_0 + n_2 I`. The expression for the nonlinear polarization
    is given below.

    Notes
    -----

        This model uses real time-domain fields, so :math:`\\n_2` must be real.

        This model is equivalent to a :class:`.NonlinearSusceptibility`; the
        relation between the parameters is given below.

        .. math::

            P_{NL} = \\varepsilon_0 \\chi_3 |E|^2 E \\\\
            n_2 = \\frac{3}{4 n_0^2 \\varepsilon_0 c_0} \\chi_3

        In these equations, :math:`n_0` means the real part of the linear
        refractive index of the medium.

        To simulate nonlinear loss, consider instead using a :class:`.TwoPhotonAbsorption`
        model, which implements a more physical dispersive loss of the form
        :math:`\\chi_{TPA} = i \\frac{c_0 n_0 \\beta}{\\omega} I`.

        The nonlinear constitutive relation is solved iteratively; it may not converge
        for strong nonlinearities. Increasing :attr:`tidy3d.NonlinearSpec.num_iters` can
        help with convergence.

        For complex fields (e.g. when using Bloch boundary conditions), the nonlinearity
        is applied separately to the real and imaginary parts, so that the above equation
        holds when both :math:`E` and :math:`P_{NL}` are replaced by their real or imaginary parts.
        The nonlinearity is only applied to the real-valued fields since they are the
        physical fields.

        Different field components do not interact nonlinearly. For example,
        when calculating :math:`P_{NL, x}`, we approximate :math:`|E|^2 \\approx |E_x|^2`.
        This approximation is valid when the :math:`E` field is predominantly polarized along one
        of the ``x``, ``y``, or ``z`` axes.

        .. TODO add links to notebooks here.

    Example
    -------
    >>> kerr_model = KerrNonlinearity(n2=1)
    """

    n2: float = Field(
        default=0,
        title="Nonlinear refractive index",
        description="Nonlinear refractive index in the Kerr nonlinearity.",
        json_schema_extra={"units": f"{MICROMETER}^2 / {WATT}"},
    )

    n0: float | None = Field(
        default=None,
        title="Complex linear refractive index",
        description="Complex linear refractive index of the medium, computed for instance using "
        "'medium.nk_model'. If not provided, it is calculated automatically using the central "
        "frequencies of the simulation sources (as long as these are all equal).",
    )


NonlinearModelType = NonlinearSusceptibility | TwoPhotonAbsorption | KerrNonlinearity


class NonlinearSpec(ABC, Tidy3dBaseModel):
    """Abstract specification for adding nonlinearities to a medium.

    Note
    ----
    The nonlinear constitutive relation is solved iteratively; it may not converge
    for strong nonlinearities. Increasing ``num_iters`` can help with convergence.

    Example
    -------
    >>> from tidy3d import Medium
    >>> nonlinear_susceptibility = NonlinearSusceptibility(chi3=1)
    >>> nonlinear_spec = NonlinearSpec(models=[nonlinear_susceptibility])
    >>> medium = Medium(permittivity=2, nonlinear_spec=nonlinear_spec)
    """

    models: tuple[discriminated_union(NonlinearModelType), ...] = Field(
        default=(),
        title="Nonlinear models",
        description="The nonlinear models present in this nonlinear spec. "
        "Nonlinear models of different types are additive. "
        "Multiple nonlinear models of the same type are not allowed.",
    )

    num_iters: PositiveInt = Field(
        default=NONLINEAR_DEFAULT_NUM_ITERS,
        title="Number of iterations",
        description="Number of iterations for solving nonlinear constitutive relation.",
    )

    @property
    def aux_fields(self) -> list[str]:
        """List of available aux fields in all present models."""
        fields = []
        for model in self.models:
            fields += model.aux_fields
        return fields


NonlinearSpecType = discriminated_union(NonlinearSpec | NonlinearSusceptibility)
