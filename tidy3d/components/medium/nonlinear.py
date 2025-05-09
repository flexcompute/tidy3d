from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import autograd.numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import Complex
from tidy3d.constants import MICROMETER, SECOND, VOLT, WATT
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log

from .constants import NONLINEAR_DEFAULT_NUM_ITERS, NONLINEAR_MAX_NUM_ITERS
from .dispersionless import Medium
from .dispersive import DispersiveMedium

if TYPE_CHECKING:
    from . import NonlinearModelType
    from .base import AbstractMedium


class NonlinearModel(ABC, Tidy3dBaseModel):
    """Abstract model for a nonlinear material response.
    Used as part of a :class:`.NonlinearSpec`."""

    def _validate_medium_type(self, medium: AbstractMedium):
        from .base import AbstractCustomMedium

        """Check that the model is compatible with the medium."""
        if isinstance(medium, AbstractCustomMedium):
            raise ValidationError(
                f"'NonlinearModel' of class '{type(self).__name__}' is not currently supported "
                f"for medium class '{type(medium).__name__}'."
            )
        if medium.is_time_modulated:
            raise ValidationError(
                f"'NonlinearModel' of class '{type(self).__name__}' is not currently supported "
                f"for time-modulated medium class '{type(medium).__name__}'."
            )
        if not isinstance(medium, (Medium, DispersiveMedium)):
            raise ValidationError(
                f"'NonlinearModel' of class '{type(self).__name__}' is not currently supported "
                f"for medium class '{type(medium).__name__}'."
            )

    def _validate_medium(self, medium: AbstractMedium):
        """Any additional validation that depends on the medium"""
        pass

    def _validate_medium_freqs(self, medium: AbstractMedium, freqs: List[pd.PositiveFloat]) -> None:
        """Any additional validation that depends on the central frequencies of the sources."""
        pass

    def _hardcode_medium_freqs(
        self, medium: AbstractMedium, freqs: List[pd.PositiveFloat]
    ) -> NonlinearSpec:
        """Update the nonlinear model to hardcode information on medium and freqs."""
        return self

    def _get_freq0(self, freq0, freqs: List[pd.PositiveFloat]) -> float:
        """Get a single value for freq0."""

        # freq0 is not specified; need to calculate it
        if freq0 is None:
            if not len(freqs):
                raise SetupError(
                    f"Class '{type(self).__name__}' cannot determine 'freq0' in the absence of "
                    f"sources. Please either specify 'freq0' in '{type(self).__name__}' "
                    "or add sources to the simulation."
                )
            if not all(np.isclose(freq, freqs[0]) for freq in freqs):
                raise SetupError(
                    f"Class '{type(self).__name__}' cannot determine 'freq0' because the source "
                    f"frequencies '{freqs}' are not all equal. "
                    f"Please specify 'freq0' in '{type(self).__name__}' "
                    "to match the desired source central frequency."
                )
            return freqs[0]

        # now, freq0 is specified; we use it, but warn if it might be inconsistent
        if not all(np.isclose(freq, freq0) for freq in freqs):
            log.warning(
                f"Class '{type(self).__name__}' given 'freq0={freq0}' which is different from "
                f"the source central frequencies '{freqs}'. In order "
                "to obtain correct nonlinearity parameters, the provided frequency "
                "should agree with the source central frequencies. The provided value of 'freq0' "
                "is being used; the resulting nonlinearity parameters "
                "may be incorrect for those sources whose central frequency "
                "is different from this value."
            )
        return freq0

    def _get_n0(
        self,
        n0: complex,
        medium: AbstractMedium,
        freqs: List[pd.PositiveFloat],
    ) -> complex:
        """Get a single value for n0."""
        if freqs is None:
            freqs = []
        freqs = np.array(freqs, dtype=float)
        ns, ks = medium.nk_model(freqs)
        nks = ns + 1j * ks

        # n0 not specified; need to calculate it
        if n0 is None:
            if not len(nks):
                raise SetupError(
                    f"Class '{type(self).__name__}' cannot determine 'n0' in the absence of "
                    f"sources. Please either specify 'n0' in '{type(self).__name__}' "
                    "or add sources to the simulation."
                )
            if not all(np.isclose(nk, nks[0]) for nk in nks):
                raise SetupError(
                    f"Class '{type(self).__name__}' cannot determine 'n0' because at the source "
                    f"frequencies '{freqs}' the complex refractive indices '{nks}' of the medium "
                    f"are not all equal. Please specify 'n0' in '{type(self).__name__}' "
                    "to match the complex refractive index of the medium at the desired "
                    "source central frequency."
                )
            return nks[0]

        # now, n0 is specified; we use it, but warn if it might be inconsistent
        if not all(np.isclose(nk, n0) for nk in nks):
            log.warning(
                f"Class '{type(self).__name__}' given 'n0={n0}'. At the source frequencies "
                f"'{freqs}' the medium has complex refractive indices '{nks}'. In order "
                "to obtain correct nonlinearity parameters, the provided refractive index "
                "should agree with the complex refractive index at the source frequencies. "
                "The provided value of 'n0' is being used; the resulting nonlinearity parameters "
                "may be incorrect for those sources where the complex refractive index of the "
                "medium is different from this value."
            )
        return n0

    @property
    def complex_fields(self) -> bool:
        """Whether the model uses complex fields."""
        return False

    @property
    def aux_fields(self) -> List[str]:
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

    chi3: float = pd.Field(
        0,
        title="Chi3",
        description="Chi3 nonlinear susceptibility.",
        units=f"{MICROMETER}^2 / {VOLT}^2",
    )

    numiters: pd.PositiveInt = pd.Field(
        None,
        title="Number of iterations",
        description="Deprecated. The old usage 'nonlinear_spec=model' with 'model.numiters' "
        "is deprecated and will be removed in a future release. The new usage is "
        r"'nonlinear_spec=NonlinearSpec(models=\[model], num_iters=num_iters)'. Under the new "
        "usage, this parameter is ignored, and 'NonlinearSpec.num_iters' is used instead.",
    )

    @pd.validator("numiters", always=True)
    def _validate_numiters(cls, val):
        """Check that numiters is not too large."""
        if val is None:
            return val
        if val > NONLINEAR_MAX_NUM_ITERS:
            raise ValidationError(
                "'NonlinearSusceptibility.numiters' must be less than "
                f"{NONLINEAR_MAX_NUM_ITERS}, currently {val}."
            )
        return val


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

    use_complex_fields: bool = pd.Field(
        False,
        title="Use complex fields",
        description="Whether to use the old deprecated complex-fields implementation. "
        "The default real-field implementation is more physical and is always "
        "recommended; this option is only available for backwards compatibility "
        "with Tidy3D version < 2.8 and may be removed in a future release.",
    )

    beta: Union[float, Complex] = pd.Field(
        0,
        title="TPA coefficient",
        description="Coefficient for two-photon absorption (TPA).",
        units=f"{MICROMETER} / {WATT}",
    )

    tau: pd.NonNegativeFloat = pd.Field(
        0,
        title="Carrier lifetime",
        description="Lifetime for the free carriers created by two-photon absorption (TPA).",
        units=f"{SECOND}",
    )

    sigma: pd.NonNegativeFloat = pd.Field(
        0,
        title="FCA cross section",
        description="Total cross section for free-carrier absorption (FCA). "
        "Contains contributions from electrons and from holes.",
        units=f"{MICROMETER}^2",
    )
    e_e: pd.NonNegativeFloat = pd.Field(
        1,
        title="Electron exponent",
        description="Exponent for the free electron refractive index shift in the free-carrier plasma dispersion (FCPD).",
    )
    e_h: pd.NonNegativeFloat = pd.Field(
        1,
        title="Hole exponent",
        description="Exponent for the free hole refractive index shift in the free-carrier plasma dispersion (FCPD).",
    )
    c_e: float = pd.Field(
        0,
        title="Electron coefficient",
        description="Coefficient for the free electron refractive index shift in the free-carrier plasma dispersion (FCPD).",
        units=f"{MICROMETER}^(3 e_e)",
    )
    c_h: float = pd.Field(
        0,
        title="Hole coefficient",
        description="Coefficient for the free hole refractive index shift in the free-carrier plasma dispersion (FCPD).",
        units=f"{MICROMETER}^(3 e_h)",
    )

    n0: Optional[Complex] = pd.Field(
        None,
        title="Complex linear refractive index",
        description="Complex linear refractive index of the medium, computed for instance using "
        "'medium.nk_model'. If not provided, it is calculated automatically using the central "
        "frequencies of the simulation sources (as long as these are all equal).",
    )

    freq0: Optional[pd.PositiveFloat] = pd.Field(
        None,
        title="Central frequency",
        description="Central frequency, used to calculate the energy of the free-carriers "
        "excited by two-photon absorption. If not provided, it is obtained automatically "
        "from the simulation sources (as long as these are all equal).",
    )

    @pd.validator("beta", always=True)
    def _validate_beta_real(cls, val, values):
        """Check that beta is real and give a useful error if it is not."""
        use_complex_fields = values.get("use_complex_fields")
        if use_complex_fields:
            return val
        if not np.isreal(val):
            raise SetupError(
                "Complex values of 'beta' in 'TwoPhotonAbsorption' are not "
                "supported; the implementation uses the "
                "physical real-valued fields."
            )
        return val

    def _validate_medium_freqs(self, medium: AbstractMedium, freqs: List[pd.PositiveFloat]) -> None:
        """Any validation that depends on knowing the central frequencies of the sources.
        This includes passivity checking, if necessary."""
        n0 = self._get_n0(self.n0, medium, freqs)
        if freqs is not None:
            _ = self._get_freq0(self.freq0, freqs)
        beta = self.beta
        if not medium.allow_gain:
            chi_imag = np.real(beta * n0 * np.real(n0))
            if chi_imag < 0:
                raise ValidationError(
                    "For passive medium, 'beta' in 'TwoPhotonAbsorption' must satisfy "
                    f"'Re(beta * n0 * Re(n0)) >= 0'. Currently, this quantity equals '{chi_imag}', "
                    f"and the linear index is 'n0={n0}'. To simulate gain medium, please set "
                    "'allow_gain=True' in the medium class. Caution: simulations containing "
                    "gain medium are unstable, and are likely to diverge."
                )

    def _hardcode_medium_freqs(
        self, medium: AbstractMedium, freqs: List[pd.PositiveFloat]
    ) -> TwoPhotonAbsorption:
        """Update the nonlinear model to hardcode information on medium and freqs."""
        n0 = self._get_n0(n0=self.n0, medium=medium, freqs=freqs)
        freq0 = self._get_freq0(freq0=self.freq0, freqs=freqs)
        return self.updated_copy(n0=n0, freq0=freq0)

    def _validate_medium(self, medium: AbstractMedium):
        """Check that the model is compatible with the medium."""
        # if n0 is specified, we can go ahead and validate passivity
        if self.n0 is not None:
            self._validate_medium_freqs(medium, None)

    @property
    def complex_fields(self) -> bool:
        """Whether the model uses complex fields."""
        return self.use_complex_fields

    @property
    def aux_fields(self) -> List[str]:
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

    use_complex_fields: bool = pd.Field(
        False,
        title="Use complex fields",
        description="Whether to use the old deprecated complex-fields implementation. "
        "The default real-field implementation is more physical and is always "
        "recommended; this option is only available for backwards compatibility "
        "with Tidy3D version < 2.8 and may be removed in a future release.",
    )

    n2: Complex = pd.Field(
        0,
        title="Nonlinear refractive index",
        description="Nonlinear refractive index in the Kerr nonlinearity.",
        units=f"{MICROMETER}^2 / {WATT}",
    )

    n0: Optional[Complex] = pd.Field(
        None,
        title="Complex linear refractive index",
        description="Complex linear refractive index of the medium, computed for instance using "
        "'medium.nk_model'. If not provided, it is calculated automatically using the central "
        "frequencies of the simulation sources (as long as these are all equal).",
    )

    @pd.validator("n2", always=True)
    def _validate_n2_real(cls, val, values):
        """Check that n2 is real and give a useful error if it is not."""
        use_complex_fields = values.get("use_complex_fields")
        if use_complex_fields:
            return val
        if not np.isreal(val):
            raise SetupError(
                "Complex values of 'n2' in 'KerrNonlinearity' are not "
                "supported; the implementation uses the "
                "physical real-valued fields. "
                "To simulate nonlinear loss, consider instead using a "
                "'TwoPhotonAbsorption' model, which implements a "
                "more physical dispersive loss of the form "
                "'chi_{TPA} = i (c_0 n_0 beta / omega) I'."
            )
        return val

    def _validate_medium_freqs(self, medium: AbstractMedium, freqs: List[pd.PositiveFloat]) -> None:
        """Any validation that depends on knowing the central frequencies of the sources.
        This includes passivity checking, if necessary."""
        n0 = self._get_n0(self.n0, medium, freqs)
        n2 = self.n2
        if not self.use_complex_fields:
            return
        if not medium.allow_gain:
            chi_imag = np.imag(n2 * n0 * np.real(n0))
            if chi_imag < 0:
                raise ValidationError(
                    "For passive medium, 'n2' in 'KerrNonlinearity' must satisfy "
                    f"'Im(n2 * n0 * Re(n0)) >= 0'. Currently, this quantity equals '{chi_imag}', "
                    f"and the linear index is 'n0={n0}'. To simulate gain medium, please set "
                    "'allow_gain=True' in the medium class. Caution: simulations containing "
                    "gain medium are unstable, and are likely to diverge."
                )

    def _validate_medium(self, medium: AbstractMedium):
        """Check that the model is compatible with the medium."""
        # if n0 is specified, we can go ahead and validate passivity
        if self.n0 is not None:
            self._validate_medium_freqs(medium, [])

    def _hardcode_medium_freqs(
        self, medium: AbstractMedium, freqs: List[pd.PositiveFloat]
    ) -> KerrNonlinearity:
        """Update the nonlinear model to hardcode information on medium and freqs."""
        n0 = self._get_n0(n0=self.n0, medium=medium, freqs=freqs)
        return self.updated_copy(n0=n0)

    @property
    def complex_fields(self) -> bool:
        """Whether the model uses complex fields."""
        return self.use_complex_fields


class NonlinearSpec(ABC, Tidy3dBaseModel):
    """Abstract specification for adding nonlinearities to a medium.

    Note
    ----
    The nonlinear constitutive relation is solved iteratively; it may not converge
    for strong nonlinearities. Increasing ``num_iters`` can help with convergence.

    Example
    -------
    >>> nonlinear_susceptibility = NonlinearSusceptibility(chi3=1)
    >>> nonlinear_spec = NonlinearSpec(models=[nonlinear_susceptibility])
    >>> medium = Medium(permittivity=2, nonlinear_spec=nonlinear_spec)
    """

    models: Tuple[NonlinearModelType, ...] = pd.Field(
        (),
        title="Nonlinear models",
        description="The nonlinear models present in this nonlinear spec. "
        "Nonlinear models of different types are additive. "
        "Multiple nonlinear models of the same type are not allowed.",
    )

    num_iters: pd.PositiveInt = pd.Field(
        NONLINEAR_DEFAULT_NUM_ITERS,
        title="Number of iterations",
        description="Number of iterations for solving nonlinear constitutive relation.",
    )

    @pd.validator("models", always=True)
    def _no_duplicate_models(cls, val):
        """Ensure each type of model appears at most once."""
        if val is None:
            return val
        models = [model.__class__ for model in val]
        models_unique = set(models)
        if len(models) != len(models_unique):
            raise ValidationError(
                "Multiple 'NonlinearModels' of the same type "
                "were found in a single 'NonlinearSpec'. Please ensure that "
                "each type of 'NonlinearModel' appears at most once in a single 'NonlinearSpec'."
            )
        return val

    @pd.validator("models", always=True)
    def _consistent_old_complex_fields(cls, val):
        """Ensure that old complex fields implementation is used consistently."""
        if val is None:
            return val
        use_complex_fields = False
        for model in val:
            if isinstance(model, (KerrNonlinearity, TwoPhotonAbsorption)):
                if model.use_complex_fields:
                    use_complex_fields = True
                elif use_complex_fields:
                    # if one model uses complex fields, they all should
                    raise SetupError(
                        "Some of the nonlinear models have "
                        "'use_complex_fields=True' and some have "
                        "'use_complex_fields=False'. This option "
                        "is only available for backwards compatibility "
                        "with Tidy3D version < 2.8 "
                        "and it must be consistent across the nonlinear "
                        "models in a given 'NonlinearSpec'."
                    )
        return val

    @pd.validator("num_iters", always=True)
    def _validate_num_iters(cls, val, values):
        """Check that num_iters is not too large."""
        if val > NONLINEAR_MAX_NUM_ITERS:
            raise ValidationError(
                "'NonlinearSpec.num_iters' must be less than "
                f"{NONLINEAR_MAX_NUM_ITERS}, currently {val}."
            )
        return val

    def _hardcode_medium_freqs(
        self, medium: AbstractMedium, freqs: List[pd.PositiveFloat]
    ) -> NonlinearSpec:
        """Update the nonlinear spec to hardcode information on medium and freqs."""
        new_models = []
        for model in self.models:
            new_model = model._hardcode_medium_freqs(medium=medium, freqs=freqs)
            new_models.append(new_model)
        return self.updated_copy(models=new_models)

    @property
    def aux_fields(self) -> List[str]:
        """List of available aux fields in all present models."""
        fields = []
        for model in self.models:
            fields += model.aux_fields
        return fields
