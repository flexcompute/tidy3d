"""Defines properties of the medium / materials"""

from __future__ import annotations

import functools
import warnings
from abc import ABC, abstractmethod
from math import isclose
from typing import Callable, Dict, List, Optional, Tuple, Union

import autograd as ag
import autograd.numpy as np

# TODO: it's hard to figure out which functions need this, for now all get it
import numpy as npo
import pydantic.v1 as pd
import xarray as xr
from numpy.typing import NDArray
from scipy import signal

from tidy3d.components.material.tcad.heat import ThermalSpecType

from ..constants import (
    C_0,
    CONDUCTIVITY,
    EPSILON_0,
    ETA_0,
    HBAR,
    HERTZ,
    MICROMETER,
    MU_0,
    PERMITTIVITY,
    RADPERSEC,
    SECOND,
    VOLT,
    WATT,
    fp_eps,
    pec_val,
)
from ..exceptions import SetupError, ValidationError
from ..log import log
from .autograd.derivative_utils import DerivativeInfo, integrate_within_bounds
from .autograd.types import AutogradFieldMap, TracedFloat, TracedPoleAndResidue, TracedPositiveFloat
from .base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from .data.data_array import DATA_ARRAY_MAP, ScalarFieldDataArray, SpatialDataArray
from .data.dataset import (
    ElectromagneticFieldDataset,
    PermittivityDataset,
)
from .data.unstructured.base import UnstructuredGridDataset
from .data.utils import (
    CustomSpatialDataType,
    CustomSpatialDataTypeAnnotated,
    _check_same_coordinates,
    _get_numpy_array,
    _ones_like,
    _zeros_like,
)
from .data.validators import validate_no_nans
from .dispersion_fitter import (
    LOSS_CHECK_MAX,
    LOSS_CHECK_MIN,
    LOSS_CHECK_NUM,
    fit,
    imag_resp_extrema_locs,
)
from .geometry.base import Geometry
from .grid.grid import Coords, Grid
from .parameter_perturbation import (
    IndexPerturbation,
    ParameterPerturbation,
    PermittivityPerturbation,
)
from .time_modulation import ModulationSpec
from .transformation import RotationType
from .types import (
    TYPE_TAG_STR,
    ArrayComplex1D,
    ArrayComplex3D,
    ArrayFloat1D,
    Ax,
    Axis,
    Bound,
    Complex,
    FreqBound,
    InterpMethod,
    Literal,
    PermittivityComponent,
    PoleAndResidue,
    TensorReal,
)
from .validators import _warn_potential_error, validate_name_str, validate_parameter_perturbation
from .viz import VisualizationSpec, add_ax_if_none

# evaluate frequency as this number (Hz) if inf
FREQ_EVAL_INF = 1e50

# extrapolation option in custom medium
FILL_VALUE = "extrapolate"

# cap on number of nonlinear iterations
NONLINEAR_MAX_NUM_ITERS = 100
NONLINEAR_DEFAULT_NUM_ITERS = 5

# Lossy metal
LOSSY_METAL_DEFAULT_SAMPLING_FREQUENCY = 20
LOSSY_METAL_SCALED_REAL_PART = 10.0
LOSSY_METAL_DEFAULT_MAX_POLES = 5
LOSSY_METAL_DEFAULT_TOLERANCE_RMS = 1e-3


def ensure_freq_in_range(eps_model: Callable[[float], complex]) -> Callable[[float], complex]:
    """Decorate ``eps_model`` to log warning if frequency supplied is out of bounds."""

    @functools.wraps(eps_model)
    def _eps_model(self, frequency: float) -> complex:
        """New eps_model function."""
        # evaluate infs and None as FREQ_EVAL_INF
        is_inf_scalar = isinstance(frequency, float) and np.isinf(frequency)
        if frequency is None or is_inf_scalar:
            frequency = FREQ_EVAL_INF

        if isinstance(frequency, np.ndarray):
            frequency = frequency.astype(float)
            frequency[np.where(np.isinf(frequency))] = FREQ_EVAL_INF

        # if frequency range not present just return original function
        if self.frequency_range is None:
            return eps_model(self, frequency)

        fmin, fmax = self.frequency_range
        # don't warn for evaluating infinite frequency
        if is_inf_scalar:
            return eps_model(self, frequency)
        if np.any(frequency < fmin * (1 - fp_eps)) or np.any(frequency > fmax * (1 + fp_eps)):
            log.warning(
                "frequency passed to 'Medium.eps_model()'"
                f"is outside of 'Medium.frequency_range' = {self.frequency_range}",
                capture=False,
            )
        return eps_model(self, frequency)

    return _eps_model


""" Medium Definitions """


class NonlinearModel(ABC, Tidy3dBaseModel):
    """Abstract model for a nonlinear material response.
    Used as part of a :class:`.NonlinearSpec`."""

    def _validate_medium_type(self, medium: AbstractMedium):
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


NonlinearModelType = Union[NonlinearSusceptibility, TwoPhotonAbsorption, KerrNonlinearity]


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


class AbstractMedium(ABC, Tidy3dBaseModel):
    """A medium within which electromagnetic waves propagate."""

    name: str = pd.Field(None, title="Name", description="Optional unique name for medium.")

    frequency_range: FreqBound = pd.Field(
        None,
        title="Frequency Range",
        description="Optional range of validity for the medium.",
        units=(HERTZ, HERTZ),
    )

    allow_gain: bool = pd.Field(
        False,
        title="Allow gain medium",
        description="Allow the medium to be active. Caution: "
        "simulations with a gain medium are unstable, and are likely to diverge."
        "Simulations where 'allow_gain' is set to 'True' will still be charged even if "
        "diverged. Monitor data up to the divergence point will still be returned and can be "
        "useful in some cases.",
    )

    nonlinear_spec: Union[NonlinearSpec, NonlinearSusceptibility] = pd.Field(
        None,
        title="Nonlinear Spec",
        description="Nonlinear spec applied on top of the base medium properties.",
    )

    modulation_spec: ModulationSpec = pd.Field(
        None,
        title="Modulation Spec",
        description="Modulation spec applied on top of the base medium properties.",
    )

    viz_spec: Optional[VisualizationSpec] = pd.Field(
        None,
        title="Visualization Specification",
        description="Plotting specification for visualizing medium.",
    )

    @cached_property
    def _nonlinear_models(self) -> List:
        """The nonlinear models in the nonlinear_spec."""
        if self.nonlinear_spec is None:
            return []
        if isinstance(self.nonlinear_spec, NonlinearModel):
            return [self.nonlinear_spec]
        if self.nonlinear_spec.models is None:
            return []
        return list(self.nonlinear_spec.models)

    @cached_property
    def _nonlinear_num_iters(self) -> pd.PositiveInt:
        """The num_iters of the nonlinear_spec."""
        if self.nonlinear_spec is None:
            return 0
        if isinstance(self.nonlinear_spec, NonlinearModel):
            if self.nonlinear_spec.numiters is None:
                return 1  # old default value for backwards compatibility
            return self.nonlinear_spec.numiters
        return self.nonlinear_spec.num_iters

    def _post_init_validators(self) -> None:
        """Call validators taking ``self`` that get run after init."""
        self._validate_nonlinear_spec()
        self._validate_modulation_spec_post_init()

    def _validate_nonlinear_spec(self):
        """Check compatibility with nonlinear_spec."""
        if self.__class__.__name__ == "AnisotropicMedium" and any(
            comp.nonlinear_spec is not None for comp in [self.xx, self.yy, self.zz]
        ):
            raise ValidationError(
                "Nonlinearities are not currently supported for the components "
                "of an anisotropic medium."
            )
        if self.__class__.__name__ == "Medium2D" and any(
            comp.nonlinear_spec is not None for comp in [self.ss, self.tt]
        ):
            raise ValidationError(
                "Nonlinearities are not currently supported for the components " "of a 2D medium."
            )

        if self.nonlinear_spec is None:
            return
        if isinstance(self.nonlinear_spec, NonlinearModel):
            log.warning(
                "The API for 'nonlinear_spec' has changed. "
                "The old usage 'nonlinear_spec=model' is deprecated and will be removed "
                "in a future release. The new usage is "
                r"'nonlinear_spec=NonlinearSpec(models=\[model])'."
            )
        for model in self._nonlinear_models:
            model._validate_medium_type(self)
            model._validate_medium(self)
            if (
                isinstance(self.nonlinear_spec, NonlinearSpec)
                and isinstance(model, NonlinearSusceptibility)
                and model.numiters is not None
            ):
                raise ValidationError(
                    "'NonlinearSusceptibility.numiters' is deprecated. "
                    "Please use 'NonlinearSpec.num_iters' instead."
                )

    def _validate_modulation_spec_post_init(self):
        """Check compatibility with nonlinear_spec."""
        if self.__class__.__name__ == "Medium2D" and any(
            comp.modulation_spec is not None for comp in [self.ss, self.tt]
        ):
            raise ValidationError(
                "Time modulation is not currently supported for the components " "of a 2D medium."
            )

    heat_spec: Optional[ThermalSpecType] = pd.Field(
        None,
        title="Heat Specification",
        description="DEPRECATED: Use `td.MultiPhysicsMedium`. Specification of the medium heat properties. They are "
        "used for solving the heat equation via the ``HeatSimulation`` interface. Such simulations can be"
        "used for investigating the influence of heat propagation on the properties of optical systems. "
        "Once the temperature distribution in the system is found using ``HeatSimulation`` object, "
        "``Simulation.perturbed_mediums_copy()`` can be used to convert mediums with perturbation "
        "models defined into spatially dependent custom mediums. "
        "Otherwise, the ``heat_spec`` does not directly affect the running of an optical "
        "``Simulation``.",
        discriminator=TYPE_TAG_STR,
    )

    @property
    def charge(self):
        return None

    @property
    def electrical(self):
        return None

    @property
    def heat(self):
        return self.heat_spec

    @property
    def optical(self):
        return None

    @pd.validator("modulation_spec", always=True)
    @skip_if_fields_missing(["nonlinear_spec"])
    def _validate_modulation_spec(cls, val, values):
        """Check compatibility with modulation_spec."""
        nonlinear_spec = values.get("nonlinear_spec")
        if val is not None and nonlinear_spec is not None:
            raise ValidationError(
                f"For medium class {cls}, 'modulation_spec' of class {type(val)} and "
                f"'nonlinear_spec' of class {type(nonlinear_spec)} are "
                "not simultaneously supported."
            )
        return val

    _name_validator = validate_name_str()

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        return True

    @cached_property
    def is_time_modulated(self) -> bool:
        """Whether any component of the medium is time modulated."""
        return self.modulation_spec is not None and self.modulation_spec.applied_modulation

    @cached_property
    def is_nonlinear(self) -> bool:
        """Whether the medium is nonlinear."""
        return self.nonlinear_spec is not None

    @cached_property
    def is_custom(self) -> bool:
        """Whether the medium is custom."""
        return isinstance(self, AbstractCustomMedium)

    @cached_property
    def is_fully_anisotropic(self) -> bool:
        """Whether the medium is fully anisotropic."""
        return isinstance(self, FullyAnisotropicMedium)

    @cached_property
    def _incompatible_material_types(self) -> List[str]:
        """A list of material properties present which may lead to incompatibilities."""
        properties = [
            self.is_time_modulated,
            self.is_nonlinear,
            self.is_custom,
            self.is_fully_anisotropic,
        ]
        names = ["time modulated", "nonlinear", "custom", "fully anisotropic"]
        types = [name for name, prop in zip(names, properties) if prop]
        return types

    @cached_property
    def _has_incompatibilities(self) -> bool:
        """Whether the medium has incompatibilities. Certain medium types are incompatible
        with certain others, and such pairs are not allowed to intersect in a simulation."""
        return len(self._incompatible_material_types) > 0

    def _compatible_with(self, other: AbstractMedium) -> bool:
        """Whether these two media are compatible if in structures that intersect."""
        if not (self._has_incompatibilities and other._has_incompatibilities):
            return True
        for med1, med2 in [(self, other), (other, self)]:
            if med1.is_custom:
                # custom and fully_anisotropic is OK
                if med2.is_nonlinear or med2.is_time_modulated:
                    return False
            if med1.is_fully_anisotropic:
                if med2.is_nonlinear or med2.is_time_modulated:
                    return False
            if med1.is_nonlinear:
                if med2.is_time_modulated:
                    return False
        return True

    @abstractmethod
    def eps_model(self, frequency: float) -> complex:
        # TODO this should be moved out of here into FDTD Simulation Mediums?
        """Complex-valued permittivity as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
            Complex-valued relative permittivity evaluated at ``frequency``.
        """

    def nk_model(self, frequency: float) -> Tuple[float, float]:
        """Real and imaginary parts of the refactive index as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[float, float]
            Real part (n) and imaginary part (k) of refractive index of medium.
        """
        eps_complex = self.eps_model(frequency=frequency)
        return self.eps_complex_to_nk(eps_complex)

    def loss_tangent_model(self, frequency: float) -> Tuple[float, float]:
        """Permittivity and loss tangent as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[float, float]
            Real part of permittivity and loss tangent.
        """
        eps_complex = self.eps_model(frequency=frequency)
        return self.eps_complex_to_eps_loss_tangent(eps_complex)

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[complex, complex, complex]
            The diagonal elements of the relative permittivity tensor evaluated at ``frequency``.
        """

        # This only needs to be overwritten for anisotropic materials
        eps = self.eps_model(frequency)
        return (eps, eps, eps)

    def eps_diagonal_numerical(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor for numerical considerations
        such as meshing and runtime estimation.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[complex, complex, complex]
            The diagonal elements of relative permittivity tensor relevant for numerical
            considerations evaluated at ``frequency``.
        """

        if self.is_pec:
            # also 1 for lossy metal and Medium2D, but let's handle them in the subclass.
            return (1.0 + 0j,) * 3

        return self.eps_diagonal(frequency)

    def eps_comp(self, row: Axis, col: Axis, frequency: float) -> complex:
        """Single component of the complex-valued permittivity tensor as a function of frequency.

        Parameters
        ----------
        row : int
            Component's row in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        col : int
            Component's column in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
           Element of the relative permittivity tensor evaluated at ``frequency``.
        """

        # This only needs to be overwritten for anisotropic materials
        if row == col:
            return self.eps_model(frequency)
        return 0j

    def _eps_plot(
        self, frequency: float, eps_component: Optional[PermittivityComponent] = None
    ) -> float:
        """Returns real part of epsilon for plotting. A specific component of the epsilon tensor can
        be selected for anisotropic medium.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at.
        eps_component : PermittivityComponent
            Component of the permittivity tensor to plot
            e.g. ``"xx"``, ``"yy"``, ``"zz"``, ``"xy"``, ``"yz"``, ...
            Defaults to ``None``, which returns the average of the diagonal values.

        Returns
        -------
        float
            Element ``eps_component`` of the relative permittivity tensor evaluated at ``frequency``.
        """
        # Assumes the material is isotropic
        # Will need to be overridden for anisotropic materials
        return self.eps_model(frequency).real

    @cached_property
    @abstractmethod
    def n_cfl(self):
        # TODO this should be moved out of here into FDTD Simulation Mediums?
        """To ensure a stable FDTD simulation, it is essential to select an appropriate
        time step size in accordance with the CFL condition. The maximal time step
        size is inversely proportional to the speed of light in the medium, and thus
        proportional to the index of refraction. However, for dispersive medium,
        anisotropic medium, and other more complicated media, there are complications in
        deciding on the choice of the index of refraction.

        This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.
        """

    @add_ax_if_none
    def plot(self, freqs: float, ax: Ax = None) -> Ax:
        """Plot n, k of a :class:`Medium` as a function of frequency.

        Parameters
        ----------
        freqs: float
            Frequencies (Hz) to evaluate the medium properties at.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        freqs = np.array(freqs)
        eps_complex = np.array([self.eps_model(freq) for freq in freqs])
        n, k = AbstractMedium.eps_complex_to_nk(eps_complex)

        freqs_thz = freqs / 1e12
        ax.plot(freqs_thz, n, label="n")
        ax.plot(freqs_thz, k, label="k")
        ax.set_xlabel("frequency (THz)")
        ax.set_title("medium dispersion")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    """ Conversion helper functions """

    @staticmethod
    def nk_to_eps_complex(n: float, k: float = 0.0) -> complex:
        """Convert n, k to complex permittivity.

        Parameters
        ----------
        n : float
            Real part of refractive index.
        k : float = 0.0
            Imaginary part of refrative index.

        Returns
        -------
        complex
            Complex-valued relative permittivity.
        """
        eps_real = n**2 - k**2
        eps_imag = 2 * n * k
        return eps_real + 1j * eps_imag

    @staticmethod
    def eps_complex_to_nk(eps_c: complex) -> Tuple[float, float]:
        """Convert complex permittivity to n, k values.

        Parameters
        ----------
        eps_c : complex
            Complex-valued relative permittivity.

        Returns
        -------
        Tuple[float, float]
            Real and imaginary parts of refractive index (n & k).
        """
        eps_c = np.array(eps_c)
        ref_index = np.sqrt(eps_c)
        return np.real(ref_index), np.imag(ref_index)

    @staticmethod
    def nk_to_eps_sigma(n: float, k: float, freq: float) -> Tuple[float, float]:
        """Convert ``n``, ``k`` at frequency ``freq`` to permittivity and conductivity values.

        Parameters
        ----------
        n : float
            Real part of refractive index.
        k : float = 0.0
            Imaginary part of refrative index.
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[float, float]
            Real part of relative permittivity & electric conductivity.
        """
        eps_complex = AbstractMedium.nk_to_eps_complex(n, k)
        eps_real, eps_imag = eps_complex.real, eps_complex.imag
        omega = 2 * np.pi * freq
        sigma = omega * eps_imag * EPSILON_0
        return eps_real, sigma

    @staticmethod
    def eps_sigma_to_eps_complex(eps_real: float, sigma: float, freq: float) -> complex:
        """convert permittivity and conductivity to complex permittivity at freq

        Parameters
        ----------
        eps_real : float
            Real-valued relative permittivity.
        sigma : float
            Conductivity.
        freq : float
            Frequency to evaluate permittivity at (Hz).
            If not supplied, returns real part of permittivity (limit as frequency -> infinity.)

        Returns
        -------
        complex
            Complex-valued relative permittivity.
        """
        if freq is None:
            return eps_real
        omega = 2 * np.pi * freq

        return eps_real + 1j * sigma / omega / EPSILON_0

    @staticmethod
    def eps_complex_to_eps_sigma(eps_complex: complex, freq: float) -> Tuple[float, float]:
        """Convert complex permittivity at frequency ``freq``
        to permittivity and conductivity values.

        Parameters
        ----------
        eps_complex : complex
            Complex-valued relative permittivity.
        freq : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[float, float]
            Real part of relative permittivity & electric conductivity.
        """
        eps_real, eps_imag = eps_complex.real, eps_complex.imag
        omega = 2 * np.pi * freq
        sigma = omega * eps_imag * EPSILON_0
        return eps_real, sigma

    @staticmethod
    def eps_complex_to_eps_loss_tangent(eps_complex: complex) -> Tuple[float, float]:
        """Convert complex permittivity to permittivity and loss tangent.

        Parameters
        ----------
        eps_complex : complex
            Complex-valued relative permittivity.

        Returns
        -------
        Tuple[float, float]
            Real part of relative permittivity & loss tangent
        """
        eps_real, eps_imag = eps_complex.real, eps_complex.imag
        return eps_real, eps_imag / eps_real

    @staticmethod
    def eps_loss_tangent_to_eps_complex(eps_real: float, loss_tangent: float) -> complex:
        """Convert permittivity and loss tangent to complex permittivity.

        Parameters
        ----------
        eps_real : float
            Real part of relative permittivity
        loss_tangent : float
            Loss tangent

        Returns
        -------
        eps_complex : complex
            Complex-valued relative permittivity.
        """
        return eps_real * (1 + 1j * loss_tangent)

    @staticmethod
    def eV_to_angular_freq(f_eV: float):
        """Convert frequency in unit of eV to rad/s.

        Parameters
        ----------
        f_eV : float
            Frequency in unit of eV
        """
        return f_eV / HBAR

    @staticmethod
    def angular_freq_to_eV(f_rad: float):
        """Convert frequency in unit of rad/s to eV.

        Parameters
        ----------
        f_rad : float
            Frequency in unit of rad/s
        """
        return f_rad * HBAR

    @staticmethod
    def angular_freq_to_Hz(f_rad: float):
        """Convert frequency in unit of rad/s to Hz.

        Parameters
        ----------
        f_rad : float
            Frequency in unit of rad/s
        """
        return f_rad / 2 / np.pi

    @staticmethod
    def Hz_to_angular_freq(f_hz: float):
        """Convert frequency in unit of Hz to rad/s.

        Parameters
        ----------
        f_hz : float
            Frequency in unit of Hz
        """
        return f_hz * 2 * np.pi

    @ensure_freq_in_range
    def sigma_model(self, freq: float) -> complex:
        """Complex-valued conductivity as a function of frequency.

        Parameters
        ----------
        freq: float
            Frequency to evaluate conductivity at (Hz).

        Returns
        -------
        complex
            Complex conductivity at this frequency.
        """
        omega = freq * 2 * np.pi
        eps_complex = self.eps_model(freq)
        eps_inf = self.eps_model(np.inf)
        sigma = (eps_inf - eps_complex) * 1j * omega * EPSILON_0
        return sigma

    @cached_property
    def is_pec(self):
        """Whether the medium is a PEC."""
        return False

    def sel_inside(self, bounds: Bound) -> AbstractMedium:
        """Return a new medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        AbstractMedium
            Medium with reduced data.
        """

        if self.modulation_spec is not None:
            modulation_reduced = self.modulation_spec.sel_inside(bounds)
            return self.updated_copy(modulation_spec=modulation_reduced)

        return self

    """ Autograd code """

    def compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute the adjoint derivatives for this object."""
        raise NotImplementedError(f"Can't compute derivative for 'Medium': '{type(self)}'.")

    def derivative_eps_sigma_volume(
        self, E_der_map: ElectromagneticFieldDataset, bounds: Bound, freqs: NDArray
    ) -> dict[str, xr.DataArray]:
        """Get the derivative w.r.t permittivity and conductivity in the volume."""

        vjp_eps_complex = self.derivative_eps_complex_volume(
            E_der_map=E_der_map, bounds=bounds, freqs=freqs
        )

        values = vjp_eps_complex.values

        eps_vjp, sigma_vjp = self.eps_complex_to_eps_sigma(eps_complex=values, freq=freqs)

        eps_vjp = np.sum(eps_vjp)
        sigma_vjp = np.sum(sigma_vjp)

        return dict(permittivity=eps_vjp, conductivity=sigma_vjp)

    def derivative_eps_complex_volume(
        self, E_der_map: ElectromagneticFieldDataset, bounds: Bound, freqs: NDArray
    ) -> xr.DataArray:
        """Get the derivative w.r.t complex-valued permittivity in the volume."""

        vjp_value = 0.0
        for field_name in ("Ex", "Ey", "Ez"):
            fld = E_der_map[field_name].sel(f=freqs)
            vjp_value_fld = integrate_within_bounds(
                arr=fld,
                dims=("x", "y", "z"),
                bounds=bounds,
            )
            vjp_value += vjp_value_fld

        return vjp_value.sum("f")

    def __repr__(self):
        """If the medium has a name, use it as the representation. Otherwise, use the default representation."""
        if self.name:
            return self.name
        else:
            return super().__repr__()


class AbstractCustomMedium(AbstractMedium, ABC):
    """A spatially varying medium."""

    interp_method: InterpMethod = pd.Field(
        "nearest",
        title="Interpolation method",
        description="Interpolation method to obtain permittivity values "
        "that are not supplied at the Yee grids; For grids outside the range "
        "of the supplied data, extrapolation will be applied. When the extrapolated "
        "value is smaller (greater) than the minimal (maximal) of the supplied data, "
        "the extrapolated value will take the minimal (maximal) of the supplied data.",
    )

    subpixel: bool = pd.Field(
        False,
        title="Subpixel averaging",
        description="If ``True``, apply the subpixel averaging method specified by "
        "``Simulation``'s field ``subpixel`` for this type of material on the "
        "interface of the structure, including exterior boundary and "
        "intersection interfaces with other structures.",
    )

    @cached_property
    @abstractmethod
    def is_isotropic(self) -> bool:
        """The medium is isotropic or anisotropic."""

    def _interp_method(self, comp: Axis) -> InterpMethod:
        """Interpolation method applied to comp."""
        return self.interp_method

    @abstractmethod
    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """

    def eps_diagonal_on_grid(
        self,
        frequency: float,
        coords: Coords,
    ) -> Tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]:
        """Spatial profile of main diagonal of the complex-valued permittivity
        at ``frequency`` interpolated at the supplied coordinates.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).
        coords : :class:`.Coords`
            The grid point coordinates over which interpolation is performed.

        Returns
        -------
        Tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]
            The complex-valued permittivity tensor at ``frequency`` interpolated
            at the supplied coordinate.
        """
        eps_spatial = self.eps_dataarray_freq(frequency)
        if self.is_isotropic:
            eps_interp = _get_numpy_array(
                coords.spatial_interp(eps_spatial[0], self._interp_method(0))
            )
            return (eps_interp, eps_interp, eps_interp)
        return tuple(
            _get_numpy_array(coords.spatial_interp(eps_comp, self._interp_method(comp)))
            for comp, eps_comp in enumerate(eps_spatial)
        )

    def eps_comp_on_grid(
        self,
        row: Axis,
        col: Axis,
        frequency: float,
        coords: Coords,
    ) -> ArrayComplex3D:
        """Spatial profile of a single component of the complex-valued permittivity tensor at
        ``frequency`` interpolated at the supplied coordinates.

        Parameters
        ----------
        row : int
            Component's row in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        col : int
            Component's column in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        frequency : float
            Frequency to evaluate permittivity at (Hz).
        coords : :class:`.Coords`
            The grid point coordinates over which interpolation is performed.

        Returns
        -------
        ArrayComplex3D
            Single component of the complex-valued permittivity tensor at ``frequency`` interpolated
            at the supplied coordinates.
        """

        if row == col:
            return self.eps_diagonal_on_grid(frequency, coords)[row]
        return 0j

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued spatially averaged permittivity as a function of frequency."""
        if self.is_isotropic:
            return np.mean(_get_numpy_array(self.eps_dataarray_freq(frequency)[0]))
        return np.mean(
            [np.mean(_get_numpy_array(eps_comp)) for eps_comp in self.eps_dataarray_freq(frequency)]
        )

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor
        at ``frequency``. Spatially, we take max{||eps||}, so that autoMesh generation
        works appropriately.
        """
        eps_spatial = self.eps_dataarray_freq(frequency)
        if self.is_isotropic:
            eps_comp = _get_numpy_array(eps_spatial[0]).ravel()
            eps = eps_comp[np.argmax(np.abs(eps_comp))]
            return (eps, eps, eps)
        eps_spatial_array = (_get_numpy_array(eps_comp).ravel() for eps_comp in eps_spatial)
        return tuple(eps_comp[np.argmax(np.abs(eps_comp))] for eps_comp in eps_spatial_array)

    def _get_real_vals(self, x: np.ndarray) -> np.ndarray:
        """Grab the real part of the values in array.
        Used for _eps_bounds()
        """
        return _get_numpy_array(np.real(x)).ravel()

    def _eps_bounds(
        self, frequency: float = None, eps_component: Optional[PermittivityComponent] = None
    ) -> Tuple[float, float]:
        """Returns permittivity bounds for setting the color bounds when plotting.

        Parameters
        ----------
        frequency : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        eps_component : Optional[PermittivityComponent] = None
            Component of the permittivity tensor to plot for anisotropic materials,
            e.g. ``"xx"``, ``"yy"``, ``"zz"``, ``"xy"``, ``"yz"``, ...
            Defaults to ``None``, which returns the average of the diagonal values.

        Returns
        -------
        Tuple[float, float]
            The min and max values of the permittivity for the selected component and evaluated at ``frequency``.
        """
        eps_dataarray = self.eps_dataarray_freq(frequency)
        all_eps = np.concatenate(self._get_real_vals(eps_comp) for eps_comp in eps_dataarray)
        return (np.min(all_eps), np.max(all_eps))

    @staticmethod
    def _validate_isreal_dataarray(dataarray: CustomSpatialDataType) -> bool:
        """Validate that the dataarray is real"""
        return np.all(np.isreal(_get_numpy_array(dataarray)))

    @staticmethod
    def _validate_isreal_dataarray_tuple(
        dataarray_tuple: Tuple[CustomSpatialDataType, ...],
    ) -> bool:
        """Validate that the dataarray is real"""
        return np.all([AbstractCustomMedium._validate_isreal_dataarray(f) for f in dataarray_tuple])

    @abstractmethod
    def _sel_custom_data_inside(self, bounds: Bound):
        """Return a new medium that contains the minimal amount custom data necessary to cover
        a spatial region defined by ``bounds``."""

    def sel_inside(self, bounds: Bound) -> AbstractCustomMedium:
        """Return a new medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        AbstractMedium
            Medium with reduced data.
        """

        self_mod_data_reduced = super().sel_inside(bounds)

        return self_mod_data_reduced._sel_custom_data_inside(bounds)

    @staticmethod
    def _not_loaded(field):
        """Check whether data was not loaded."""
        if isinstance(field, str) and field in DATA_ARRAY_MAP:
            return True
        # attempting to construct an UnstructuredGridDataset from a dict
        elif isinstance(field, dict) and field.get("type") in (
            "TriangularGridDataset",
            "TetrahedralGridDataset",
        ):
            return any(
                isinstance(subfield, str) and subfield in DATA_ARRAY_MAP
                for subfield in [field["points"], field["cells"], field["values"]]
            )
        # attempting to pass an UnstructuredGridDataset with zero points
        elif isinstance(field, UnstructuredGridDataset):
            return any(len(subfield) == 0 for subfield in [field.points, field.cells, field.values])

    def _derivative_field_cmp(
        self,
        E_der_map: ElectromagneticFieldDataset,
        eps_data: PermittivityDataset,
        dim: str,
        freqs: NDArray,
    ) -> np.ndarray:
        coords_interp = {key: val for key, val in eps_data.coords.items() if len(val) > 1}
        dims_sum = {dim for dim in eps_data.coords.keys() if dim not in coords_interp}

        # compute sizes along each of the interpolation dimensions
        sizes_list = []
        for _, coords in coords_interp.items():
            num_coords = len(coords)
            coords = np.array(coords)

            # compute distances between midpoints for all internal coords
            mid_points = (coords[1:] + coords[:-1]) / 2.0
            dists = np.diff(mid_points)
            sizes = np.zeros(num_coords)
            sizes[1:-1] = dists

            # estimate the sizes on the edges using 2 x the midpoint distance
            sizes[0] = 2 * abs(mid_points[0] - coords[0])
            sizes[-1] = 2 * abs(coords[-1] - mid_points[-1])

            sizes_list.append(sizes)

        # turn this into a volume element, should be re-sizeable to the gradient shape
        if sizes_list:
            d_vol = functools.reduce(np.outer, sizes_list)
        else:
            # if sizes_list is empty, then reduce() fails
            d_vol = np.array(1.0)

        # TODO: probably this could be more robust. eg if the DataArray has weird edge cases
        E_der_dim = E_der_map[f"E{dim}"].sel(f=freqs)
        E_der_dim_interp = (
            E_der_dim.interp(**coords_interp, assume_sorted=True).fillna(0.0).sum(dims_sum).sum("f")
        )
        vjp_array = np.array(E_der_dim_interp.values).astype(complex)
        vjp_array = vjp_array.reshape(eps_data.shape)

        # multiply by volume elements (if possible, being defensive here..)
        try:
            vjp_array *= d_vol.reshape(vjp_array.shape)
        except ValueError:
            log.warning(
                "Skipping volume element normalization of 'CustomMedium' gradients. "
                f"Could not reshape the volume elements of shape {d_vol.shape} "
                f"to the shape of the gradient {vjp_array.shape}. "
                "If you encounter this warning, gradient direction will be accurate but the norm "
                "will be inaccurate. Please raise an issue on the tidy3d front end with this "
                "message and some information about your simulation setup and we will investigate. "
            )
        return vjp_array


""" Dispersionless Medium """


# PEC keyword
class PECMedium(AbstractMedium):
    """Perfect electrical conductor class.

    Note
    ----

        To avoid confusion from duplicate PECs, must import ``tidy3d.PEC`` instance directly.



    """

    @pd.validator("modulation_spec", always=True)
    def _validate_modulation_spec(cls, val):
        """Check compatibility with modulation_spec."""
        if val is not None:
            raise ValidationError(
                f"A 'modulation_spec' of class {type(val)} is not "
                f"currently supported for medium class {cls}."
            )
        return val

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        # return something like frequency with value of pec_val + 0j
        return 0j * frequency + pec_val

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.
        """
        return 1.0

    @cached_property
    def is_pec(self):
        """Whether the medium is a PEC."""
        return True


# PEC builtin instance
PEC = PECMedium(name="PEC")


class Medium(AbstractMedium):
    """Dispersionless medium. Mediums define the optical properties of the materials within the simulation.

    Notes
    -----

        In a dispersion-less medium, the displacement field :math:`D(t)` reacts instantaneously to the applied
        electric field :math:`E(t)`.

        .. math::

            D(t) = \\epsilon E(t)

    Example
    -------
    >>> dielectric = Medium(permittivity=4.0, name='my_medium')
    >>> eps = dielectric.eps_model(200e12)

    See Also
    --------

    **Notebooks**
        * `Introduction on Tidy3D working principles <../../notebooks/Primer.html#Mediums>`_
        * `Index <../../notebooks/docs/features/medium.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_

    **GUI**
        * `Mediums <https://www.flexcompute.com/tidy3d/learning-center/tidy3d-gui/Lecture-2-Mediums/>`_

    """

    permittivity: TracedFloat = pd.Field(
        1.0, ge=1.0, title="Permittivity", description="Relative permittivity.", units=PERMITTIVITY
    )

    conductivity: TracedFloat = pd.Field(
        0.0,
        title="Conductivity",
        description="Electric conductivity. Defined such that the imaginary part of the complex "
        "permittivity at angular frequency omega is given by conductivity/omega.",
        units=CONDUCTIVITY,
    )

    @pd.validator("conductivity", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if ``allow_gain`` is False."""
        if not values.get("allow_gain") and val < 0:
            raise ValidationError(
                "For passive medium, 'conductivity' must be non-negative. "
                "To simulate a gain medium, please set 'allow_gain=True'. "
                "Caution: simulations with a gain medium are unstable, and are likely to diverge."
            )
        return val

    @pd.validator("permittivity", always=True)
    @skip_if_fields_missing(["modulation_spec"])
    def _permittivity_modulation_validation(cls, val, values):
        """Assert modulated permittivity cannot be <= 0."""
        modulation = values.get("modulation_spec")
        if modulation is None or modulation.permittivity is None:
            return val

        min_eps_inf = np.min(_get_numpy_array(val))
        if min_eps_inf - modulation.permittivity.max_modulation <= 0:
            raise ValidationError(
                "The minimum permittivity value with modulation applied was found to be negative."
            )
        return val

    @pd.validator("conductivity", always=True)
    @skip_if_fields_missing(["modulation_spec", "allow_gain"])
    def _passivity_modulation_validation(cls, val, values):
        """Assert passive medium if ``allow_gain`` is False."""
        modulation = values.get("modulation_spec")
        if modulation is None or modulation.conductivity is None:
            return val

        min_sigma = np.min(_get_numpy_array(val))
        if not values.get("allow_gain") and min_sigma - modulation.conductivity.max_modulation < 0:
            raise ValidationError(
                "For passive medium, 'conductivity' must be non-negative at any time."
                "With conductivity modulation, this medium can sometimes be active. "
                "Please set 'allow_gain=True'. "
                "Caution: simulations with a gain medium are unstable, "
                "and are likely to diverge."
            )
        return val

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For dispersiveless medium, it equals ``sqrt(permittivity)``.
        """
        permittivity = self.permittivity
        if self.modulation_spec is not None and self.modulation_spec.permittivity is not None:
            permittivity -= self.modulation_spec.permittivity.max_modulation
        n, _ = self.eps_complex_to_nk(permittivity)
        return n

    @staticmethod
    def _eps_model(permittivity: float, conductivity: float, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        return AbstractMedium.eps_sigma_to_eps_complex(permittivity, conductivity, frequency)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        return self._eps_model(self.permittivity, self.conductivity, frequency)

    @classmethod
    def from_nk(cls, n: float, k: float, freq: float, **kwargs):
        """Convert ``n`` and ``k`` values at frequency ``freq`` to :class:`Medium`.

        Parameters
        ----------
        n : float
            Real part of refractive index.
        k : float = 0
            Imaginary part of refrative index.
        freq : float
            Frequency to evaluate permittivity at (Hz).
        kwargs: dict
            Keyword arguments passed to the medium construction.

        Returns
        -------
        :class:`Medium`
            medium containing the corresponding ``permittivity`` and ``conductivity``.
        """
        eps, sigma = AbstractMedium.nk_to_eps_sigma(n, k, freq)
        if eps < 1:
            raise ValidationError(
                "Dispersiveless medium must have 'permittivity>=1`. "
                "Please use 'Lorentz.from_nk()' to covert to a Lorentz medium, or the utility "
                "function 'td.medium_from_nk()' to automatically return the proper medium type."
            )
        return cls(permittivity=eps, conductivity=sigma, **kwargs)

    def compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute the adjoint derivatives for this object."""

        # get vjps w.r.t. permittivity and conductivity of the bulk
        vjps_volume = self.derivative_eps_sigma_volume(
            E_der_map=derivative_info.E_der_map,
            bounds=derivative_info.bounds,
            freqs=np.atleast_1d(derivative_info.frequency),
        )

        # store the fields asked for by ``field_paths``
        derivative_map = {}
        for field_path in derivative_info.paths:
            field_name, *_ = field_path
            if field_name in vjps_volume:
                derivative_map[field_path] = vjps_volume[field_name]

        return derivative_map

    def derivative_eps_sigma_volume(
        self, E_der_map: ElectromagneticFieldDataset, bounds: Bound, freqs: NDArray
    ) -> dict[str, xr.DataArray]:
        """Get the derivative w.r.t permittivity and conductivity in the volume."""

        vjp_eps_complex = self.derivative_eps_complex_volume(
            E_der_map=E_der_map, bounds=bounds, freqs=freqs
        )

        values = vjp_eps_complex.values

        # vjp of eps_complex_to_eps_sigma
        omegas = 2 * np.pi * freqs
        eps_vjp = np.real(values)
        sigma_vjp = -np.imag(values) / omegas / EPSILON_0

        eps_vjp = np.sum(eps_vjp)
        sigma_vjp = np.sum(sigma_vjp)

        return dict(permittivity=eps_vjp, conductivity=sigma_vjp)

    def derivative_eps_complex_volume(
        self, E_der_map: ElectromagneticFieldDataset, bounds: Bound, freqs: NDArray
    ) -> xr.DataArray:
        """Get the derivative w.r.t complex-valued permittivity in the volume."""

        vjp_value = 0.0
        for field_name in ("Ex", "Ey", "Ez"):
            fld = E_der_map[field_name].sel(f=freqs)
            vjp_value_fld = integrate_within_bounds(
                arr=fld,
                dims=("x", "y", "z"),
                bounds=bounds,
            )
            vjp_value += vjp_value_fld

        return vjp_value.sum("f")


class CustomIsotropicMedium(AbstractCustomMedium, Medium):
    """:class:`.Medium` with user-supplied permittivity distribution.
    (This class is for internal use in v2.0; it will be renamed as `CustomMedium` in v3.0.)

    Example
    -------
    >>> Nx, Ny, Nz = 10, 9, 8
    >>> X = np.linspace(-1, 1, Nx)
    >>> Y = np.linspace(-1, 1, Ny)
    >>> Z = np.linspace(-1, 1, Nz)
    >>> coords = dict(x=X, y=Y, z=Z)
    >>> permittivity= SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> conductivity= SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> dielectric = CustomIsotropicMedium(permittivity=permittivity, conductivity=conductivity)
    >>> eps = dielectric.eps_model(200e12)
    """

    permittivity: CustomSpatialDataTypeAnnotated = pd.Field(
        ...,
        title="Permittivity",
        description="Relative permittivity.",
        units=PERMITTIVITY,
    )

    conductivity: Optional[CustomSpatialDataTypeAnnotated] = pd.Field(
        None,
        title="Conductivity",
        description="Electric conductivity. Defined such that the imaginary part of the complex "
        "permittivity at angular frequency omega is given by conductivity/omega.",
        units=CONDUCTIVITY,
    )

    _no_nans_eps = validate_no_nans("permittivity")
    _no_nans_sigma = validate_no_nans("conductivity")

    @pd.validator("permittivity", always=True)
    def _eps_inf_greater_no_less_than_one(cls, val):
        """Assert any eps_inf must be >=1"""

        if not CustomIsotropicMedium._validate_isreal_dataarray(val):
            raise SetupError("'permittivity' must be real.")

        if np.any(_get_numpy_array(val) < 1):
            raise SetupError("'permittivity' must be no less than one.")

        return val

    @pd.validator("conductivity", always=True)
    @skip_if_fields_missing(["permittivity"])
    def _conductivity_real_and_correct_shape(cls, val, values):
        """Assert conductivity is real and of right shape."""

        if val is None:
            return val

        if not CustomIsotropicMedium._validate_isreal_dataarray(val):
            raise SetupError("'conductivity' must be real.")

        if not _check_same_coordinates(values["permittivity"], val):
            raise SetupError("'permittivity' and 'conductivity' must have the same coordinates.")
        return val

    @pd.validator("conductivity", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if ``allow_gain`` is False."""
        if val is None:
            return val
        if not values.get("allow_gain") and np.any(_get_numpy_array(val) < 0):
            raise ValidationError(
                "For passive medium, 'conductivity' must be non-negative. "
                "To simulate a gain medium, please set 'allow_gain=True'. "
                "Caution: simulations with a gain medium are unstable, and are likely to diverge."
            )
        return val

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        if self.conductivity is None:
            return self.permittivity.is_uniform
        return self.permittivity.is_uniform and self.conductivity.is_uniform

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For dispersiveless medium, it equals ``sqrt(permittivity)``.
        """
        permittivity = np.min(_get_numpy_array(self.permittivity))
        if self.modulation_spec is not None and self.modulation_spec.permittivity is not None:
            permittivity -= self.modulation_spec.permittivity.max_modulation
        n, _ = self.eps_complex_to_nk(permittivity)
        return n

    @cached_property
    def is_isotropic(self):
        """Whether the medium is isotropic."""
        return True

    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """
        conductivity = self.conductivity
        if conductivity is None:
            conductivity = _zeros_like(self.permittivity)
        eps = self.eps_sigma_to_eps_complex(self.permittivity, conductivity, frequency)
        return (eps, eps, eps)

    def _sel_custom_data_inside(self, bounds: Bound):
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        CustomMedium
            CustomMedium with reduced data.
        """
        if not self.permittivity.does_cover(bounds=bounds):
            log.warning(
                "Permittivity spatial data array does not fully cover the requested region."
            )
        perm_reduced = self.permittivity.sel_inside(bounds=bounds)
        cond_reduced = None
        if self.conductivity is not None:
            if not self.conductivity.does_cover(bounds=bounds):
                log.warning(
                    "Conductivity spatial data array does not fully cover the requested region."
                )
            cond_reduced = self.conductivity.sel_inside(bounds=bounds)

        return self.updated_copy(
            permittivity=perm_reduced,
            conductivity=cond_reduced,
        )


class CustomMedium(AbstractCustomMedium):
    """:class:`.Medium` with user-supplied permittivity distribution.

    Example
    -------
    >>> Nx, Ny, Nz = 10, 9, 8
    >>> X = np.linspace(-1, 1, Nx)
    >>> Y = np.linspace(-1, 1, Ny)
    >>> Z = np.linspace(-1, 1, Nz)
    >>> coords = dict(x=X, y=Y, z=Z)
    >>> permittivity= SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> conductivity= SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> dielectric = CustomMedium(permittivity=permittivity, conductivity=conductivity)
    >>> eps = dielectric.eps_model(200e12)
    """

    eps_dataset: Optional[PermittivityDataset] = pd.Field(
        None,
        title="Permittivity Dataset",
        description="[To be deprecated] User-supplied dataset containing complex-valued "
        "permittivity as a function of space. Permittivity distribution over the Yee-grid "
        "will be interpolated based on ``interp_method``.",
    )

    permittivity: Optional[CustomSpatialDataTypeAnnotated] = pd.Field(
        None,
        title="Permittivity",
        description="Spatial profile of relative permittivity.",
        units=PERMITTIVITY,
    )

    conductivity: Optional[CustomSpatialDataTypeAnnotated] = pd.Field(
        None,
        title="Conductivity",
        description="Spatial profile Electric conductivity. Defined such "
        "that the imaginary part of the complex permittivity at angular "
        "frequency omega is given by conductivity/omega.",
        units=CONDUCTIVITY,
    )

    _no_nans_eps_dataset = validate_no_nans("eps_dataset")
    _no_nans_permittivity = validate_no_nans("permittivity")
    _no_nans_sigma = validate_no_nans("conductivity")

    @pd.root_validator(pre=True)
    def _warn_if_none(cls, values):
        """Warn if the data array fails to load, and return a vacuum medium."""
        eps_dataset = values.get("eps_dataset")
        permittivity = values.get("permittivity")
        conductivity = values.get("conductivity")
        fail_load = False
        if cls._not_loaded(permittivity):
            log.warning(
                "Loading 'permittivity' without data; constructing a vacuum medium instead."
            )
            fail_load = True
        if cls._not_loaded(conductivity):
            log.warning(
                "Loading 'conductivity' without data; constructing a vacuum medium instead."
            )
            fail_load = True
        if isinstance(eps_dataset, dict):
            if any((v in DATA_ARRAY_MAP for _, v in eps_dataset.items() if isinstance(v, str))):
                log.warning(
                    "Loading 'eps_dataset' without data; constructing a vacuum medium instead."
                )
                fail_load = True
        if fail_load:
            eps_real = SpatialDataArray(np.ones((1, 1, 1)), coords=dict(x=[0], y=[0], z=[0]))
            return dict(permittivity=eps_real)
        return values

    @pd.root_validator(pre=True)
    def _deprecation_dataset(cls, values):
        """Raise deprecation warning if dataset supplied and convert to dataset."""

        eps_dataset = values.get("eps_dataset")
        permittivity = values.get("permittivity")
        conductivity = values.get("conductivity")

        # Incomplete custom medium definition.
        if eps_dataset is None and permittivity is None and conductivity is None:
            raise SetupError("Missing spatial profiles of 'permittivity' or 'eps_dataset'.")
        if eps_dataset is None and permittivity is None:
            raise SetupError("Missing spatial profiles of 'permittivity'.")

        # Definition racing
        if eps_dataset is not None and (permittivity is not None or conductivity is not None):
            raise SetupError(
                "Please either define 'permittivity' and 'conductivity', or 'eps_dataset', "
                "but not both simultaneously."
            )

        if eps_dataset is None:
            return values

        # TODO: sometime before 3.0, uncomment these lines to warn users to start using new API
        # if isinstance(eps_dataset, dict):
        #     eps_components = [eps_dataset[f"eps_{dim}{dim}"] for dim in "xyz"]
        # else:
        #     eps_components = [eps_dataset.eps_xx, eps_dataset.eps_yy, eps_dataset.eps_zz]

        # is_isotropic = eps_components[0] == eps_components[1] == eps_components[2]

        # if is_isotropic:
        #     # deprecation warning for isotropic custom medium
        #     log.warning(
        #         "For spatially varying isotropic medium, the 'eps_dataset' field "
        #         "is being replaced by 'permittivity' and 'conductivity' in v3.0. "
        #         "We recommend you change your scripts to be compatible with the new API."
        #     )
        # else:
        #     # deprecation warning for anisotropic custom medium
        #     log.warning(
        #         "For spatially varying anisotropic medium, this class is being replaced "
        #         "by 'CustomAnisotropicMedium' in v3.0. "
        #         "We recommend you change your scripts to be compatible with the new API."
        #     )

        return values

    @pd.validator("eps_dataset", always=True)
    def _eps_dataset_single_frequency(cls, val):
        """Assert only one frequency supplied."""
        if val is None:
            return val

        for name, eps_dataset_component in val.field_components.items():
            freqs = eps_dataset_component.f
            if len(freqs) != 1:
                raise SetupError(
                    f"'eps_dataset.{name}' must have a single frequency, "
                    f"but it contains {len(freqs)} frequencies."
                )
        return val

    @pd.validator("eps_dataset", always=True)
    @skip_if_fields_missing(["modulation_spec", "allow_gain"])
    def _eps_dataset_eps_inf_greater_no_less_than_one_sigma_positive(cls, val, values):
        """Assert any eps_inf must be >=1"""
        if val is None:
            return val
        modulation = values.get("modulation_spec")

        for comp in ["eps_xx", "eps_yy", "eps_zz"]:
            eps_real, sigma = CustomMedium.eps_complex_to_eps_sigma(
                val.field_components[comp], val.field_components[comp].f
            )
            if np.any(_get_numpy_array(eps_real) < 1):
                raise SetupError(
                    "Permittivity at infinite frequency at any spatial point "
                    "must be no less than one."
                )

            if modulation is not None and modulation.permittivity is not None:
                if np.any(_get_numpy_array(eps_real) - modulation.permittivity.max_modulation <= 0):
                    raise ValidationError(
                        "The minimum permittivity value with modulation applied "
                        "was found to be negative."
                    )

            if not values.get("allow_gain") and np.any(_get_numpy_array(sigma) < 0):
                raise ValidationError(
                    "For passive medium, imaginary part of permittivity must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )

            if (
                not values.get("allow_gain")
                and modulation is not None
                and modulation.conductivity is not None
                and np.any(_get_numpy_array(sigma) - modulation.conductivity.max_modulation <= 0)
            ):
                raise ValidationError(
                    "For passive medium, imaginary part of permittivity must be non-negative "
                    "at any time. "
                    "With conductivity modulation, this medium can sometimes be active. "
                    "Please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return val

    @pd.validator("permittivity", always=True)
    @skip_if_fields_missing(["modulation_spec"])
    def _eps_inf_greater_no_less_than_one(cls, val, values):
        """Assert any eps_inf must be >=1"""
        if val is None:
            return val

        if not CustomMedium._validate_isreal_dataarray(val):
            raise SetupError("'permittivity' must be real.")

        if np.any(_get_numpy_array(val) < 1):
            raise SetupError("'permittivity' must be no less than one.")

        modulation = values.get("modulation_spec")
        if modulation is None or modulation.permittivity is None:
            return val

        if np.any(_get_numpy_array(val) - modulation.permittivity.max_modulation <= 0):
            raise ValidationError(
                "The minimum permittivity value with modulation applied was found to be negative."
            )

        return val

    @pd.validator("conductivity", always=True)
    @skip_if_fields_missing(["permittivity", "allow_gain"])
    def _conductivity_non_negative_correct_shape(cls, val, values):
        """Assert conductivity>=0"""

        if val is None:
            return val

        if not CustomMedium._validate_isreal_dataarray(val):
            raise SetupError("'conductivity' must be real.")

        if not values.get("allow_gain") and np.any(_get_numpy_array(val) < 0):
            raise ValidationError(
                "For passive medium, 'conductivity' must be non-negative. "
                "To simulate a gain medium, please set 'allow_gain=True'. "
                "Caution: simulations with a gain medium are unstable, "
                "and are likely to diverge."
            )

        if not _check_same_coordinates(values["permittivity"], val):
            raise SetupError("'permittivity' and 'conductivity' must have the same coordinates.")

        return val

    @pd.validator("conductivity", always=True)
    @skip_if_fields_missing(["eps_dataset", "modulation_spec", "allow_gain"])
    def _passivity_modulation_validation(cls, val, values):
        """Assert passive medium at any time during modulation if ``allow_gain`` is False."""

        # validated already when the data is supplied through `eps_dataset`
        if values.get("eps_dataset"):
            return val

        # permittivity defined with ``permittivity`` and ``conductivity``
        modulation = values.get("modulation_spec")
        if values.get("allow_gain") or modulation is None or modulation.conductivity is None:
            return val
        if val is None or np.any(
            _get_numpy_array(val) - modulation.conductivity.max_modulation < 0
        ):
            raise ValidationError(
                "For passive medium, 'conductivity' must be non-negative at any time. "
                "With conductivity modulation, this medium can sometimes be active. "
                "Please set 'allow_gain=True'. "
                "Caution: simulations with a gain medium are unstable, "
                "and are likely to diverge."
            )
        return val

    @pd.validator("permittivity", "conductivity", always=True)
    def _check_permittivity_conductivity_interpolate(cls, val, values, field):
        """Check that the custom medium 'SpatialDataArrays' can be interpolated."""

        if isinstance(val, SpatialDataArray):
            val._interp_validator(field.name)

        return val

    @cached_property
    def is_isotropic(self) -> bool:
        """Check if the medium is isotropic or anisotropic."""
        if self.eps_dataset is None:
            return True
        if self.eps_dataset.eps_xx == self.eps_dataset.eps_yy == self.eps_dataset.eps_zz:
            return True
        return False

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        return self._medium.is_spatially_uniform

    @cached_property
    def freqs(self) -> np.ndarray:
        """float array of frequencies.
        This field is to be deprecated in v3.0.
        """
        # return dummy values in this case
        if self.eps_dataset is None:
            return np.array([0, 0, 0])
        return np.array(
            [
                self.eps_dataset.eps_xx.coords["f"],
                self.eps_dataset.eps_yy.coords["f"],
                self.eps_dataset.eps_zz.coords["f"],
            ]
        )

    @cached_property
    def _medium(self):
        """Internal representation in the form of
        either `CustomIsotropicMedium` or `CustomAnisotropicMedium`.
        """
        self_dict = self.dict(exclude={"type", "eps_dataset"})
        # isotropic
        if self.eps_dataset is None:
            self_dict.update({"permittivity": self.permittivity, "conductivity": self.conductivity})
            return CustomIsotropicMedium.parse_obj(self_dict)

        def get_eps_sigma(eps_complex: SpatialDataArray, freq: float) -> tuple:
            """Convert a complex permittivity to real permittivity and conductivity."""
            eps_values = np.array(eps_complex.values)

            eps_real, sigma = CustomMedium.eps_complex_to_eps_sigma(eps_values, freq)
            coords = eps_complex.coords

            eps_real = ScalarFieldDataArray(eps_real, coords=coords)
            sigma = ScalarFieldDataArray(sigma, coords=coords)

            eps_real = SpatialDataArray(eps_real.squeeze(dim="f", drop=True))
            sigma = SpatialDataArray(sigma.squeeze(dim="f", drop=True))

            return eps_real, sigma

        # isotropic, but with `eps_dataset`
        if self.is_isotropic:
            eps_complex = self.eps_dataset.eps_xx
            eps_real, sigma = get_eps_sigma(eps_complex, freq=self.freqs[0])

            self_dict.update({"permittivity": eps_real, "conductivity": sigma})
            return CustomIsotropicMedium.parse_obj(self_dict)

        # anisotropic
        mat_comp = {"interp_method": self.interp_method}
        for freq, comp in zip(self.freqs, ["xx", "yy", "zz"]):
            eps_complex = self.eps_dataset.field_components["eps_" + comp]
            eps_real, sigma = get_eps_sigma(eps_complex, freq=freq)

            comp_dict = self_dict.copy()
            comp_dict.update({"permittivity": eps_real, "conductivity": sigma})
            mat_comp.update({comp: CustomIsotropicMedium.parse_obj(comp_dict)})
        return CustomAnisotropicMediumInternal(**mat_comp)

    def _interp_method(self, comp: Axis) -> InterpMethod:
        """Interpolation method applied to comp."""
        return self._medium._interp_method(comp)

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl```.

        For dispersiveless custom medium, it equals ``min[sqrt(eps_inf)]``, where ``min``
        is performed over all components and spatial points.
        """
        return self._medium.n_cfl

    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``. ()

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """
        return self._medium.eps_dataarray_freq(frequency)

    def eps_diagonal_on_grid(
        self,
        frequency: float,
        coords: Coords,
    ) -> Tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]:
        """Spatial profile of main diagonal of the complex-valued permittivity
        at ``frequency`` interpolated at the supplied coordinates.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).
        coords : :class:`.Coords`
            The grid point coordinates over which interpolation is performed.

        Returns
        -------
        Tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]
            The complex-valued permittivity tensor at ``frequency`` interpolated
            at the supplied coordinate.
        """
        return self._medium.eps_diagonal_on_grid(frequency, coords)

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor
        at ``frequency``. Spatially, we take max{|eps|}, so that autoMesh generation
        works appropriately.
        """
        return self._medium.eps_diagonal(frequency)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Spatial and polarizaiton average of complex-valued permittivity
        as a function of frequency.
        """
        return self._medium.eps_model(frequency)

    @classmethod
    def from_eps_raw(
        cls,
        eps: Union[ScalarFieldDataArray, CustomSpatialDataType],
        freq: float = None,
        interp_method: InterpMethod = "nearest",
        **kwargs,
    ) -> CustomMedium:
        """Construct a :class:`.CustomMedium` from datasets containing raw permittivity values.

        Parameters
        ----------
        eps : Union[
                :class:`.SpatialDataArray`,
                :class:`.ScalarFieldDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ]
            Dataset containing complex-valued permittivity as a function of space.
        freq : float, optional
            Frequency at which ``eps`` are defined.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain permittivity values that are not supplied
            at the Yee grids.

        Notes
        -----

            For lossy medium that has a complex-valued ``eps``, if ``eps`` is supplied through
            :class:`.SpatialDataArray`, which doesn't contain frequency information,
            the ``freq`` kwarg will be used to evaluate the permittivity and conductivity.
            Alternatively, ``eps`` can be supplied through :class:`.ScalarFieldDataArray`,
            which contains a frequency coordinate.
            In this case, leave ``freq`` kwarg as the default of ``None``.

        Returns
        -------
        :class:`.CustomMedium`
            Medium containing the spatially varying permittivity data.
        """
        if isinstance(eps, CustomSpatialDataType.__args__):
            # purely real, not need to know `freq`
            if CustomMedium._validate_isreal_dataarray(eps):
                return cls(permittivity=eps, interp_method=interp_method, **kwargs)
            # complex permittivity, needs to know `freq`
            if freq is None:
                raise SetupError(
                    "For a complex 'eps', 'freq' at which 'eps' is defined must be supplied",
                )
            eps_real, sigma = CustomMedium.eps_complex_to_eps_sigma(eps, freq)
            return cls(
                permittivity=eps_real, conductivity=sigma, interp_method=interp_method, **kwargs
            )

        # eps is ScalarFieldDataArray
        # contradictory definition of frequency
        freq_data = eps.coords["f"].data[0]
        if freq is not None and not isclose(freq, freq_data):
            raise SetupError(
                "'freq' value is inconsistent with the coordinate 'f'"
                "in 'eps' DataArray. It's unclear at which frequency 'eps' "
                "is defined. Please leave 'freq=None' to use the frequency "
                "value in the DataArray."
            )
        eps_real, sigma = CustomMedium.eps_complex_to_eps_sigma(eps, freq_data)
        eps_real = SpatialDataArray(eps_real.squeeze(dim="f", drop=True))
        sigma = SpatialDataArray(sigma.squeeze(dim="f", drop=True))
        return cls(permittivity=eps_real, conductivity=sigma, interp_method=interp_method, **kwargs)

    @classmethod
    def from_nk(
        cls,
        n: Union[ScalarFieldDataArray, CustomSpatialDataType],
        k: Optional[Union[ScalarFieldDataArray, CustomSpatialDataType]] = None,
        freq: float = None,
        interp_method: InterpMethod = "nearest",
        **kwargs,
    ) -> CustomMedium:
        """Construct a :class:`.CustomMedium` from datasets containing n and k values.

        Parameters
        ----------
        n : Union[
                :class:`.SpatialDataArray`,
                :class:`.ScalarFieldDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ]
            Real part of refractive index.
        k : Union[
                :class:`.SpatialDataArray`,
                :class:`.ScalarFieldDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ], optional
            Imaginary part of refrative index for lossy medium.
        freq : float, optional
            Frequency at which ``n`` and ``k`` are defined.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain permittivity values that are not supplied
            at the Yee grids.
        kwargs: dict
            Keyword arguments passed to the medium construction.

        Note
        ----
        For lossy medium, if both ``n`` and ``k`` are supplied through
        :class:`.SpatialDataArray`, which doesn't contain frequency information,
        the ``freq`` kwarg will be used to evaluate the permittivity and conductivity.
        Alternatively, ``n`` and ``k`` can be supplied through :class:`.ScalarFieldDataArray`,
        which contains a frequency coordinate.
        In this case, leave ``freq`` kwarg as the default of ``None``.

        Returns
        -------
        :class:`.CustomMedium`
            Medium containing the spatially varying permittivity data.
        """
        # lossless
        if k is None:
            if isinstance(n, ScalarFieldDataArray):
                n = SpatialDataArray(n.squeeze(dim="f", drop=True))
            freq = 0  # dummy value
            eps_real, _ = CustomMedium.nk_to_eps_sigma(n, 0 * n, freq)
            return cls(permittivity=eps_real, interp_method=interp_method, **kwargs)

        # lossy case
        if not _check_same_coordinates(n, k):
            raise SetupError("'n' and 'k' must be of the same type and must have same coordinates.")

        # k is a SpatialDataArray
        if isinstance(k, CustomSpatialDataType.__args__):
            if freq is None:
                raise SetupError(
                    "For a lossy medium, must supply 'freq' at which to convert 'n' "
                    "and 'k' to a complex valued permittivity."
                )
            eps_real, sigma = CustomMedium.nk_to_eps_sigma(n, k, freq)
            return cls(
                permittivity=eps_real, conductivity=sigma, interp_method=interp_method, **kwargs
            )

        # k is a ScalarFieldDataArray
        freq_data = k.coords["f"].data[0]
        if freq is not None and not isclose(freq, freq_data):
            raise SetupError(
                "'freq' value is inconsistent with the coordinate 'f'"
                "in 'k' DataArray. It's unclear at which frequency 'k' "
                "is defined. Please leave 'freq=None' to use the frequency "
                "value in the DataArray."
            )

        eps_real, sigma = CustomMedium.nk_to_eps_sigma(n, k, freq_data)
        eps_real = SpatialDataArray(eps_real.squeeze(dim="f", drop=True))
        sigma = SpatialDataArray(sigma.squeeze(dim="f", drop=True))
        return cls(permittivity=eps_real, conductivity=sigma, interp_method=interp_method, **kwargs)

    def grids(self, bounds: Bound) -> Dict[str, Grid]:
        """Make a :class:`.Grid` corresponding to the data in each ``eps_ii`` component.
        The min and max coordinates along each dimension are bounded by ``bounds``."""

        rmin, rmax = bounds
        pt_mins = dict(zip("xyz", rmin))
        pt_maxs = dict(zip("xyz", rmax))

        def make_grid(scalar_field: Union[ScalarFieldDataArray, SpatialDataArray]) -> Grid:
            """Make a grid for a single dataset."""

            def make_bound_coords(coords: np.ndarray, pt_min: float, pt_max: float) -> List[float]:
                """Convert user supplied coords into boundary coords to use in :class:`.Grid`."""

                # get coordinates of the bondaries halfway between user-supplied data
                coord_bounds = (coords[1:] + coords[:-1]) / 2.0

                # res-set coord boundaries that lie outside geometry bounds to the boundary (0 vol.)
                coord_bounds[coord_bounds <= pt_min] = pt_min
                coord_bounds[coord_bounds >= pt_max] = pt_max

                # add the geometry bounds in explicitly
                return [pt_min] + coord_bounds.tolist() + [pt_max]

            # grab user supplied data long this dimension
            coords = {key: np.array(val) for key, val in scalar_field.coords.items()}
            spatial_coords = {key: coords[key] for key in "xyz"}

            # convert each spatial coord to boundary coords
            bound_coords = {}
            for key, coords in spatial_coords.items():
                pt_min = pt_mins[key]
                pt_max = pt_maxs[key]
                bound_coords[key] = make_bound_coords(coords=coords, pt_min=pt_min, pt_max=pt_max)

            # construct grid
            boundaries = Coords(**bound_coords)
            return Grid(boundaries=boundaries)

        grids = {}
        for field_name in ("eps_xx", "eps_yy", "eps_zz"):
            # grab user supplied data long this dimension
            scalar_field = self.eps_dataset.field_components[field_name]

            # feed it to make_grid
            grids[field_name] = make_grid(scalar_field)

        return grids

    def _sel_custom_data_inside(self, bounds: Bound):
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        CustomMedium
            CustomMedium with reduced data.
        """

        perm_reduced = None
        if self.permittivity is not None:
            if not self.permittivity.does_cover(bounds=bounds):
                log.warning(
                    "Permittivity spatial data array does not fully cover the requested region."
                )
            perm_reduced = self.permittivity.sel_inside(bounds=bounds)

        cond_reduced = None
        if self.conductivity is not None:
            if not self.conductivity.does_cover(bounds=bounds):
                log.warning(
                    "Conductivity spatial data array does not fully cover the requested region."
                )
            cond_reduced = self.conductivity.sel_inside(bounds=bounds)

        eps_reduced = None
        if self.eps_dataset is not None:
            eps_reduced_dict = {}
            for key, comp in self.eps_dataset.field_components.items():
                if not comp.does_cover(bounds=bounds):
                    log.warning(
                        f"{key} spatial data array does not fully cover the requested region."
                    )
                eps_reduced_dict[key] = comp.sel_inside(bounds=bounds)
            eps_reduced = PermittivityDataset(**eps_reduced_dict)

        return self.updated_copy(
            permittivity=perm_reduced,
            conductivity=cond_reduced,
            eps_dataset=eps_reduced,
        )

    def compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute the adjoint derivatives for this object."""

        vjps = {}

        for field_path in derivative_info.paths:
            if field_path == ("permittivity",):
                vjp_array = 0.0
                for dim in "xyz":
                    vjp_array += self._derivative_field_cmp(
                        E_der_map=derivative_info.E_der_map,
                        eps_data=self.permittivity,
                        dim=dim,
                        freqs=np.atleast_1d(derivative_info.frequency),
                    )
                vjps[field_path] = vjp_array

            elif field_path[0] == "eps_dataset":
                key = field_path[1]
                dim = key[-1]
                vjps[field_path] = self._derivative_field_cmp(
                    E_der_map=derivative_info.E_der_map,
                    eps_data=self.eps_dataset.field_components[key],
                    dim=dim,
                    freqs=np.atleast_1d(derivative_info.frequency),
                )

            else:
                raise NotImplementedError(
                    f"No derivative defined for 'CustomMedium' field: {field_path}."
                )

        return vjps

    def _derivative_field_cmp(
        self,
        E_der_map: ElectromagneticFieldDataset,
        eps_data: PermittivityDataset,
        dim: str,
        freqs: NDArray,
    ) -> np.ndarray:
        """Compute derivative with respect to the ``dim`` components within the custom medium."""

        coords_interp = {key: eps_data.coords[key] for key in "xyz"}
        coords_interp = {key: val for key, val in coords_interp.items() if len(val) > 1}

        E_der_dim_interp = E_der_map[f"E{dim}"].sel(f=freqs)

        for dim_ in "xyz":
            if dim_ not in coords_interp:
                bound_max = np.max(E_der_dim_interp.coords[dim_])
                bound_min = np.min(E_der_dim_interp.coords[dim_])
                dimension_size = bound_max - bound_min

                if dimension_size > 0.0:
                    E_der_dim_interp = E_der_dim_interp.integrate(dim_)

        # compute sizes along each of the interpolation dimensions
        sizes_list = []
        for _, coords in coords_interp.items():
            num_coords = len(coords)
            coords = np.array(coords)

            # compute distances between midpoints for all internal coords
            mid_points = (coords[1:] + coords[:-1]) / 2.0
            dists = np.diff(mid_points)
            sizes = np.zeros(num_coords)
            sizes[1:-1] = dists

            # estimate the sizes on the edges using 2 x the midpoint distance
            sizes[0] = 2 * abs(mid_points[0] - coords[0])
            sizes[-1] = 2 * abs(coords[-1] - mid_points[-1])

            sizes_list.append(sizes)

        # turn this into a volume element, should be re-sizeable to the gradient shape
        if sizes_list:
            d_vol = functools.reduce(np.outer, sizes_list)
        else:
            # if sizes_list is empty, then reduce() fails
            d_vol = np.array(1.0)

        # TODO: probably this could be more robust. eg if the DataArray has weird edge cases
        E_der_dim_interp = (
            E_der_dim_interp.interp(**coords_interp, assume_sorted=True).fillna(0.0).real.sum("f")
        )

        try:
            E_der_dim_interp = E_der_dim_interp * d_vol.reshape(E_der_dim_interp.shape)
        except ValueError:
            log.warning(
                "Skipping volume element normalization of 'CustomMedium' gradients. "
                f"Could not reshape the volume elements of shape {d_vol.shape} "
                f"to the shape of the fields {E_der_dim_interp.shape}. "
                "If you encounter this warning, gradient direction will be accurate but the norm "
                "will be inaccurate. Please raise an issue on the tidy3d front end with this "
                "message and some information about your simulation setup and we will investigate. "
            )
        vjp_array = E_der_dim_interp.values
        vjp_array = vjp_array.reshape(eps_data.shape)

        return vjp_array


""" Dispersive Media """


class DispersiveMedium(AbstractMedium, ABC):
    """
    A Medium with dispersion: field propagation characteristics depend on frequency.

    Notes
    -----

        In dispersive mediums, the displacement field :math:`D(t)` depends on the previous electric field :math:`E(
        t')` and time-dependent permittivity :math:`\\epsilon` changes.

        .. math::

            D(t) = \\int \\epsilon(t - t') E(t') \\delta t'

        Dispersive mediums can be defined in three ways:

        - Imported from our `material library <../material_library.html>`_.
        - Defined directly by specifying the parameters in the `various supplied dispersive models <../mediums.html>`_.
        - Fitted to optical n-k data using the `dispersion fitting tool plugin <../plugins/dispersion.html>`_.

        It is important to keep in mind that dispersive materials are inevitably slower to simulate than their
        dispersion-less counterparts, with complexity increasing with the number of poles included in the dispersion
        model. For simulations with a narrow range of frequencies of interest, it may sometimes be faster to define
        the material through its real and imaginary refractive index at the center frequency.


    See Also
    --------

    :class:`CustomPoleResidue`:
        A spatially varying dispersive medium described by the pole-residue pair model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    @staticmethod
    def _permittivity_modulation_validation():
        """Assert modulated permittivity cannot be <= 0 at any time."""

        @pd.validator("eps_inf", allow_reuse=True, always=True)
        @skip_if_fields_missing(["modulation_spec"])
        def _validate_permittivity_modulation(cls, val, values):
            """Assert modulated permittivity cannot be <= 0."""
            modulation = values.get("modulation_spec")
            if modulation is None or modulation.permittivity is None:
                return val

            min_eps_inf = np.min(_get_numpy_array(val))
            if min_eps_inf - modulation.permittivity.max_modulation <= 0:
                raise ValidationError(
                    "The minimum permittivity value with modulation applied was found to be negative."
                )
            return val

        return _validate_permittivity_modulation

    @staticmethod
    def _conductivity_modulation_validation():
        """Assert passive medium at any time if not ``allow_gain``."""

        @pd.validator("modulation_spec", allow_reuse=True, always=True)
        @skip_if_fields_missing(["allow_gain"])
        def _validate_conductivity_modulation(cls, val, values):
            """With conductivity modulation, the medium can exhibit gain during the cycle.
            So `allow_gain` must be True when the conductivity is modulated.
            """
            if val is None or val.conductivity is None:
                return val

            if not values.get("allow_gain"):
                raise ValidationError(
                    "For passive medium, 'conductivity' must be non-negative at any time. "
                    "With conductivity modulation, this medium can sometimes be active. "
                    "Please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, and are likely to diverge."
                )
            return val

        return _validate_conductivity_modulation

    @abstractmethod
    def _pole_residue_dict(self) -> Dict:
        """Dict representation of Medium as a pole-residue model."""

    @cached_property
    def pole_residue(self):
        """Representation of Medium as a pole-residue model."""
        return PoleResidue(**self._pole_residue_dict(), allow_gain=self.allow_gain)

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For PoleResidue model, it equals ``sqrt(eps_inf)``
        [https://ieeexplore.ieee.org/document/9082879].
        """
        permittivity = self.pole_residue.eps_inf
        if self.modulation_spec is not None and self.modulation_spec.permittivity is not None:
            permittivity -= self.modulation_spec.permittivity.max_modulation
        n, _ = self.eps_complex_to_nk(permittivity)
        return n

    @staticmethod
    def tuple_to_complex(value: Tuple[float, float]) -> complex:
        """Convert a tuple of real and imaginary parts to complex number."""

        val_r, val_i = value
        return val_r + 1j * val_i

    @staticmethod
    def complex_to_tuple(value: complex) -> Tuple[float, float]:
        """Convert a complex number to a tuple of real and imaginary parts."""

        return (value.real, value.imag)


class CustomDispersiveMedium(AbstractCustomMedium, DispersiveMedium, ABC):
    """A spatially varying dispersive medium."""

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For PoleResidue model, it equals ``sqrt(eps_inf)``
        [https://ieeexplore.ieee.org/document/9082879].
        """
        permittivity = np.min(_get_numpy_array(self.pole_residue.eps_inf))
        if self.modulation_spec is not None and self.modulation_spec.permittivity is not None:
            permittivity -= self.modulation_spec.permittivity.max_modulation
        n, _ = self.eps_complex_to_nk(permittivity)
        return n

    @cached_property
    def is_isotropic(self):
        """Whether the medium is isotropic."""
        return True

    @cached_property
    def pole_residue(self):
        """Representation of Medium as a pole-residue model."""
        return CustomPoleResidue(
            **self._pole_residue_dict(),
            interp_method=self.interp_method,
            allow_gain=self.allow_gain,
            subpixel=self.subpixel,
        )

    @staticmethod
    def _warn_if_data_none(nested_tuple_field: str):
        """Warn if any of `eps_inf` and nested_tuple_field are not loaded,
        and return a vacuum with eps_inf = 1.
        """

        @pd.root_validator(pre=True, allow_reuse=True)
        def _warn_if_none(cls, values):
            """Warn if any of `eps_inf` and nested_tuple_field are not load."""
            eps_inf = values.get("eps_inf")
            coeffs = values.get(nested_tuple_field)
            fail_load = False

            if AbstractCustomMedium._not_loaded(eps_inf):
                log.warning("Loading 'eps_inf' without data; constructing a vacuum medium instead.")
                fail_load = True
            for coeff in coeffs:
                if fail_load:
                    break
                for coeff_i in coeff:
                    if AbstractCustomMedium._not_loaded(coeff_i):
                        log.warning(
                            f"Loading '{nested_tuple_field}' without data; "
                            "constructing a vacuum medium instead."
                        )
                        fail_load = True
                        break

            if fail_load and eps_inf is None:
                return {nested_tuple_field: ()}
            if fail_load:
                eps_inf = SpatialDataArray(np.ones((1, 1, 1)), coords=dict(x=[0], y=[0], z=[0]))
                return {"eps_inf": eps_inf, nested_tuple_field: ()}
            return values

        return _warn_if_none


class PoleResidue(DispersiveMedium):
    """A dispersive medium described by the pole-residue pair model.

    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(\\omega) = \\epsilon_\\infty - \\sum_i
            \\left[\\frac{c_i}{j \\omega + a_i} +
            \\frac{c_i^*}{j \\omega + a_i^*}\\right]

    Example
    -------
    >>> pole_res = PoleResidue(eps_inf=2.0, poles=[((-1+2j), (3+4j)), ((-5+6j), (7+8j))])
    >>> eps = pole_res.eps_model(200e12)

    See Also
    --------

    :class:`CustomPoleResidue`:
        A spatially varying dispersive medium described by the pole-residue pair model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: TracedPositiveFloat = pd.Field(
        1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    poles: Tuple[TracedPoleAndResidue, ...] = pd.Field(
        (),
        title="Poles",
        description="Tuple of complex-valued (:math:`a_i, c_i`) poles for the model.",
        units=(RADPERSEC, RADPERSEC),
    )

    @pd.validator("poles", always=True)
    def _causality_validation(cls, val):
        """Assert causal medium."""
        for a, _ in val:
            if np.any(np.real(_get_numpy_array(a)) > 0):
                raise SetupError("For stable medium, 'Re(a_i)' must be non-positive.")
        return val

    _validate_permittivity_modulation = DispersiveMedium._permittivity_modulation_validation()
    _validate_conductivity_modulation = DispersiveMedium._conductivity_modulation_validation()

    @staticmethod
    def _eps_model(
        eps_inf: pd.PositiveFloat, poles: Tuple[PoleAndResidue, ...], frequency: float
    ) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        omega = 2 * np.pi * frequency
        eps = eps_inf + 0 * frequency + 0.0j
        for a, c in poles:
            a_cc = np.conj(a)
            c_cc = np.conj(c)
            eps = eps - c / (1j * omega + a)
            eps = eps - c_cc / (1j * omega + a_cc)
        return eps

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""
        return self._eps_model(eps_inf=self.eps_inf, poles=self.poles, frequency=frequency)

    def _pole_residue_dict(self) -> Dict:
        """Dict representation of Medium as a pole-residue model."""

        return dict(
            eps_inf=self.eps_inf,
            poles=self.poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )

    def __str__(self):
        """string representation"""
        return (
            f"td.PoleResidue("
            f"\n\teps_inf={self.eps_inf}, "
            f"\n\tpoles={self.poles}, "
            f"\n\tfrequency_range={self.frequency_range})"
        )

    @classmethod
    def from_medium(cls, medium: Medium) -> PoleResidue:
        """Convert a :class:`.Medium` to a pole residue model.

        Parameters
        ----------
        medium: :class:`.Medium`
            The medium with permittivity and conductivity to convert.

        Returns
        -------
        :class:`.PoleResidue`
            The pole residue equivalent.
        """
        poles = [(0, medium.conductivity / (2 * EPSILON_0))]
        return PoleResidue(
            eps_inf=medium.permittivity, poles=poles, frequency_range=medium.frequency_range
        )

    def to_medium(self) -> Medium:
        """Convert to a :class:`.Medium`.
        Requires the pole residue model to only have a pole at 0 frequency,
        corresponding to a constant conductivity term.

        Returns
        -------
        :class:`.Medium`
            The non-dispersive equivalent with constant permittivity and conductivity.
        """
        res = 0
        for a, c in self.poles:
            if abs(a) > fp_eps:
                raise ValidationError("Cannot convert dispersive 'PoleResidue' to 'Medium'.")
            res = res + (c + np.conj(c)) / 2
        sigma = res * 2 * EPSILON_0
        return Medium(
            permittivity=self.eps_inf,
            conductivity=np.real(sigma),
            frequency_range=self.frequency_range,
        )

    @staticmethod
    def lo_to_eps_model(
        poles: Tuple[Tuple[float, float, float, float], ...],
        eps_inf: pd.PositiveFloat,
        frequency: float,
    ) -> complex:
        """Complex permittivity as a function of frequency for a given set of LO-TO coefficients.
        See ``from_lo_to`` in :class:`.PoleResidue` for the detailed form of the model
        and a reference paper.

        Parameters
        ----------
        poles : Tuple[Tuple[float, float, float, float], ...]
            The LO-TO poles, given as list of tuples of the form
            (omega_LO, gamma_LO, omega_TO, gamma_TO).
        eps_inf: pd.PositiveFloat
            The relative permittivity at infinite frequency.
        frequency: float
            Frequency at which to evaluate the permittivity.

        Returns
        -------
        complex
            The complex permittivity of the given LO-TO model at the given frequency.
        """
        omega = 2 * np.pi * frequency
        eps = eps_inf
        for omega_lo, gamma_lo, omega_to, gamma_to in poles:
            eps *= omega_lo**2 - omega**2 - 1j * omega * gamma_lo
            eps /= omega_to**2 - omega**2 - 1j * omega * gamma_to
        return eps

    @classmethod
    def from_lo_to(
        cls, poles: Tuple[Tuple[float, float, float, float], ...], eps_inf: pd.PositiveFloat = 1
    ) -> PoleResidue:
        """Construct a pole residue model from the LO-TO form
        (longitudinal and transverse optical modes).
        The LO-TO form is :math:`\\epsilon_\\infty \\prod_{i=1}^l \\frac{\\omega_{LO, i}^2 - \\omega^2 - i \\omega \\gamma_{LO, i}}{\\omega_{TO, i}^2 - \\omega^2 - i \\omega \\gamma_{TO, i}}` as given in the paper:

            M. Schubert, T. E. Tiwald, and C. M. Herzinger,
            "Infrared dielectric anisotropy and phonon modes of sapphire,"
            Phys. Rev. B 61, 8187 (2000).

        Parameters
        ----------
        poles : Tuple[Tuple[float, float, float, float], ...]
            The LO-TO poles, given as list of tuples of the form
            (omega_LO, gamma_LO, omega_TO, gamma_TO).
        eps_inf: pd.PositiveFloat
            The relative permittivity at infinite frequency.

        Returns
        -------
        :class:`.PoleResidue`
            The pole residue equivalent of the LO-TO form provided.
        """

        omegas_lo, gammas_lo, omegas_to, gammas_to = map(np.array, zip(*poles))

        # discriminants of quadratic factors of denominator
        discs = 2 * npo.emath.sqrt((gammas_to / 2) ** 2 - omegas_to**2)

        # require nondegenerate TO poles
        if len({(omega_to, gamma_to) for (_, _, omega_to, gamma_to) in poles}) != len(poles) or any(
            disc == 0 for disc in discs
        ):
            raise ValidationError(
                "Unable to construct a pole residue model "
                "from an LO-TO form with degenerate TO poles. Consider adding a "
                "perturbation to split the poles, or using "
                "'PoleResidue.lo_to_eps_model' and fitting with the 'FastDispersionFitter'."
            )

        # roots of denominator, in pairs
        roots = []
        for gamma_to, disc in zip(gammas_to, discs):
            roots.append(-gamma_to / 2 + disc / 2)
            roots.append(-gamma_to / 2 - disc / 2)

        # interpolants
        interpolants = eps_inf * np.ones(len(roots), dtype=complex)
        for i, a in enumerate(roots):
            for omega_lo, gamma_lo in zip(omegas_lo, gammas_lo):
                interpolants[i] *= omega_lo**2 + a**2 + a * gamma_lo
            for j, a2 in enumerate(roots):
                if j != i:
                    interpolants[i] /= a - a2

        a_coeffs = []
        c_coeffs = []

        for i in range(0, len(roots), 2):
            if not np.isreal(roots[i]):
                a_coeffs.append(roots[i])
                c_coeffs.append(interpolants[i])
            else:
                a_coeffs.append(roots[i])
                a_coeffs.append(roots[i + 1])
                # factor of two from adding conjugate pole of real pole
                c_coeffs.append(interpolants[i] / 2)
                c_coeffs.append(interpolants[i + 1] / 2)

        return PoleResidue(eps_inf=eps_inf, poles=list(zip(a_coeffs, c_coeffs)))

    @staticmethod
    def imag_ep_extrema(poles: Tuple[PoleAndResidue, ...]) -> ArrayFloat1D:
        """Extrema of Im[eps] in the same unit as poles.

        Parameters
        ----------
        poles: Tuple[PoleAndResidue, ...]
            Tuple of complex-valued (``a_i, c_i``) poles for the model.
        """

        poles_a = [a for (a, _) in poles]
        poles_c = [c for (_, c) in poles]
        return imag_resp_extrema_locs(poles=poles_a, residues=poles_c)

    def _imag_ep_extrema_with_samples(self) -> ArrayFloat1D:
        """Provide a list of frequencies (in unit of rad/s) to probe the possible lower and
        upper bound of Im[eps] within the ``frequency_range``. If ``frequency_range`` is None,
        it checks the entire frequency range. The returned frequencies include not only extrema,
        but also a list of sampled frequencies.
        """

        # extrema frequencies: in the intermediate stage, convert to the unit eV for
        # better numerical handling, since those quantities will be ~ 1 in photonics
        extrema_freq = self.imag_ep_extrema(self.angular_freq_to_eV(np.array(self.poles)))
        extrema_freq = self.eV_to_angular_freq(extrema_freq)

        # let's check a big range in addition to the imag_extrema
        if self.frequency_range is None:
            range_ev = np.logspace(LOSS_CHECK_MIN, LOSS_CHECK_MAX, LOSS_CHECK_NUM)
            range_omega = self.eV_to_angular_freq(range_ev)
        else:
            fmin, fmax = self.frequency_range
            fmin = max(fmin, fp_eps)
            range_freq = np.logspace(np.log10(fmin), np.log10(fmax), LOSS_CHECK_NUM)
            range_omega = self.Hz_to_angular_freq(range_freq)

            extrema_freq = extrema_freq[
                np.logical_and(extrema_freq > range_omega[0], extrema_freq < range_omega[-1])
            ]
        return np.concatenate((range_omega, extrema_freq))

    @cached_property
    def loss_upper_bound(self) -> float:
        """Upper bound of Im[eps] in `frequency_range`"""
        freq_list = self.angular_freq_to_Hz(self._imag_ep_extrema_with_samples())
        ep = self.eps_model(freq_list)
        # filter `NAN` in case some of freq_list are exactly at the pole frequency
        # of Sellmeier-type poles.
        ep = ep[~np.isnan(ep)]
        return max(ep.imag)

    def compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute adjoint derivatives for each of the ``fields`` given the multiplied E and D."""

        # compute all derivatives beforehand
        dJ_deps = self.derivative_eps_complex_volume(
            E_der_map=derivative_info.E_der_map,
            bounds=derivative_info.bounds,
            freqs=np.atleast_1d(derivative_info.frequency),
        )

        dJ_deps = complex(dJ_deps)

        # TODO: fix for multi-frequency
        frequency = derivative_info.frequency
        poles_complex = [(complex(a), complex(c)) for a, c in self.poles]
        poles_complex = np.stack(poles_complex, axis=0)

        # compute gradients of eps_model with respect to eps_inf and poles
        grad_eps_model = ag.holomorphic_grad(self._eps_model, argnum=(0, 1))
        with warnings.catch_warnings():
            # ignore warnings about holmorphic grad being passed a non-complex input (poles)
            warnings.simplefilter("ignore")
            deps_deps_inf, deps_dpoles = grad_eps_model(
                complex(self.eps_inf), poles_complex, complex(frequency)
            )

        # multiply with partial dJ/deps to give full gradients

        dJ_deps_inf = dJ_deps * deps_deps_inf
        dJ_dpoles = [(dJ_deps * a, dJ_deps * c) for a, c in deps_dpoles]

        # get vjps w.r.t. permittivity and conductivity of the bulk
        derivative_map = {}
        for field_path in derivative_info.paths:
            field_name, *rest = field_path

            if field_name == "eps_inf":
                derivative_map[field_path] = float(np.real(dJ_deps_inf))

            elif field_name == "poles":
                pole_index, a_or_c = rest
                derivative_map[field_path] = complex(dJ_dpoles[pole_index][a_or_c])

        return derivative_map

    @classmethod
    def _real_partial_fraction_decomposition(
        cls, a: np.ndarray, b: np.ndarray, tol: pd.PositiveFloat = 1e-2
    ) -> tuple[list[tuple[Complex, Complex]], np.ndarray]:
        """Computes the complex conjugate pole residue pairs given a rational expression with
        real coefficients.

        Parameters
        ----------

        a : np.ndarray
            Coefficients of the numerator polynomial in increasing monomial order.
        b : np.ndarray
            Coefficients of the denominator polynomial in increasing monomial order.
        tol : pd.PositiveFloat
            Tolerance for pole finding. Two poles are considered equal, if their spacing is less
            than ``tol``.

        Returns
        -------
        tuple[list[tuple[Complex, Complex]], np.ndarray]
            The list of complex conjugate poles and their associated residues. The second element of the
            ``tuple`` is an array of coefficients representing any direct polynomial term.

        """

        if a.ndim != 1 or np.any(np.iscomplex(a)):
            raise ValidationError(
                "Numerator coefficients must be a one-dimensional array of real numbers."
            )
        if b.ndim != 1 or np.any(np.iscomplex(b)):
            raise ValidationError(
                "Denominator coefficients must be a one-dimensional array of real numbers."
            )

        # Compute residues and poles using scipy
        (r, p, k) = signal.residue(np.flip(a), np.flip(b), tol=tol, rtype="avg")

        # Assuming real coefficients for the polynomials, the poles should be real or come as
        # complex conjugate pairs
        r_filtered = []
        p_filtered = []
        for res, (idx, pole) in zip(list(r), enumerate(list(p))):
            # Residue equal to zero interpreted as rational expression was not
            # in simplest form. So skip this pole.
            if res == 0:
                continue
            # Causal and stability check
            if np.real(pole) > 0:
                raise ValidationError("Transfer function is invalid. It is non-causal.")
            # Check for higher order pole, which come in consecutive order
            if idx > 0 and p[idx - 1] == pole:
                raise ValidationError(
                    "Transfer function is invalid. A higher order pole was detected. Try reducing ``tol``, "
                    "or ensure that the rational expression does not have repeated poles. "
                )
            if np.imag(pole) == 0:
                r_filtered.append(res / 2)
                p_filtered.append(pole)
            else:
                pair_found = len(np.argwhere(np.array(p) == np.conj(pole))) == 1
                if not pair_found:
                    raise ValueError(
                        "Failed to find complex-conjugate of pole in poles computed by SciPy."
                    )
                previously_added = len(np.argwhere(np.array(p_filtered) == np.conj(pole))) == 1
                if not previously_added:
                    r_filtered.append(res)
                    p_filtered.append(pole)

        poles_residues = list(zip(p_filtered, r_filtered))
        k_increasing_order = np.flip(k)
        return (poles_residues, k_increasing_order)

    @classmethod
    def from_admittance_coeffs(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        eps_inf: pd.PositiveFloat = 1,
        pole_tol: pd.PositiveFloat = 1e-2,
    ) -> PoleResidue:
        """Construct a :class:`.PoleResidue` model from an admittance function defining the
        relationship between the electric field and the polarization current density in the
        Laplace domain.

        Parameters
        ----------
        a : np.ndarray
            Coefficients of the numerator polynomial in increasing monomial order.
        b : np.ndarray
            Coefficients of the denominator polynomial in increasing monomial order.
        eps_inf: pd.PositiveFloat
            The relative permittivity at infinite frequency.
        pole_tol: pd.PositiveFloat
            Tolerance for the pole finding algorithm in Hertz. Two poles are considered equal, if their
            spacing is closer than ``pole_tol`.
        Returns
        -------
        :class:`.PoleResidue`
            The pole residue equivalent.

        Notes
        -----

            The supplied admittance function relates the electric field to the polarization current density
            in the Laplace domain and is equivalent to a frequency-dependent complex conductivity
            :math:`\\sigma(\\omega)`.

            .. math::
                J_p(s) = Y(s)E(s)

            .. math::
                Y(s) = \\frac{a_0 + a_1 s + \\dots + a_M s^M}{b_0 + b_1 s + \\dots + b_N s^N}

            An equivalent :class:`.PoleResidue` medium is constructed using an equivalent frequency-dependent
            complex permittivity defined as

            .. math::
                \\epsilon(s) = \\epsilon_\\infty - \\frac{1}{\\epsilon_0 s}
                \\frac{a_0 + a_1 s + \\dots + a_M s^M}{b_0 + b_1 s + \\dots + b_N s^N}.
        """

        if a.ndim != 1 or np.any(np.logical_or(np.iscomplex(a), a < 0)):
            raise ValidationError(
                "Numerator coefficients must be a one-dimensional array of non-negative real numbers."
            )
        if b.ndim != 1 or np.any(np.logical_or(np.iscomplex(b), b < 0)):
            raise ValidationError(
                "Denominator coefficients must be a one-dimensional array of non-negative real numbers."
            )

        # Trim any trailing zeros, so that length corresponds with polynomial order
        a = np.trim_zeros(a, "b")
        b = np.trim_zeros(b, "b")

        # Validate that transfer function will result in a proper transfer function, once converted to
        # the complex permittivity version
        # Let q equal the order of the numerator polynomial, and p equal the order
        # of the denominator polynomal. Then, q < p is strictly proper rational transfer function (RTF)
        # q <= p is a proper RTF, and q > p is an improper RTF.
        q = len(a) - 1
        p = len(b) - 1

        if q > p + 1:
            raise ValidationError(
                "Transfer function is improper, the order of the numerator polynomial must be at most "
                "one greater than the order of the denominator polynomial."
            )

        # Modify the transfer function defining a complex conductivity to match the complex
        # frequency-dependent portion of the pole residue model
        # Meaning divide by -j*omega*epsilon (s*epsilon)
        b = np.concatenate(([0], b * EPSILON_0))

        poles_and_residues, k = cls._real_partial_fraction_decomposition(
            a=a, b=b, tol=pole_tol * 2 * np.pi
        )

        # A direct polynomial term of zeroth order is interpreted as an additional contribution to eps_inf.
        # So we only handle that special case.
        if len(k) == 1:
            if np.iscomplex(k[0]) or k[0] < 0:
                raise ValidationError(
                    "Transfer function is invalid. Direct polynomial term must be real and positive for "
                    "conversion to an equivalent 'PoleResidue' medium."
                )
            else:
                # A pure capacitance will translate to an increased permittivity at infinite frequency.
                eps_inf = eps_inf + k[0]

        pole_residue_from_transfer = PoleResidue(eps_inf=eps_inf, poles=poles_and_residues)

        # Check passivity
        ang_freqs = PoleResidue._imag_ep_extrema_with_samples(pole_residue_from_transfer)
        freq_list = PoleResidue.angular_freq_to_Hz(ang_freqs)
        ep = pole_residue_from_transfer.eps_model(freq_list)
        # filter `NAN` in case some of freq_list are exactly at the pole frequency
        ep = ep[~np.isnan(ep)]

        if np.any(np.imag(ep) < -fp_eps):
            log.warning(
                "Generated 'PoleResidue' medium is not passive. Please raise an issue on the "
                "Tidy3d frontend with this message and some information about your "
                "simulation setup and we will investigate."
            )

        return pole_residue_from_transfer


class CustomPoleResidue(CustomDispersiveMedium, PoleResidue):
    """A spatially varying dispersive medium described by the pole-residue pair model.

    Notes
    -----

        In this method, the frequency-dependent permittivity :math:`\\epsilon(\\omega)` is expressed as a sum of
        resonant material poles _`[1]`.

        .. math::

            \\epsilon(\\omega) = \\epsilon_\\infty - \\sum_i
            \\left[\\frac{c_i}{j \\omega + a_i} +
            \\frac{c_i^*}{j \\omega + a_i^*}\\right]

        For each of these resonant poles identified by the index :math:`i`, an auxiliary differential equation is
        used to relate the auxiliary current :math:`J_i(t)` to the applied electric field :math:`E(t)`.
        The sum of all these auxiliary current contributions describes the total dielectric response of the material.

        .. math::

            \\frac{d}{dt} J_i (t) - a_i J_i (t) = \\epsilon_0 c_i \\frac{d}{dt} E (t)

        Hence, the computational cost increases with the number of poles.

        **References**

        .. [1]   M. Han, R.W. Dutton and S. Fan, IEEE Microwave and Wireless Component Letters, 16, 119 (2006).

        .. TODO add links to notebooks using this.

    Example
    -------
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 6)
    >>> z = np.linspace(-1, 1, 7)
    >>> coords = dict(x=x, y=y, z=z)
    >>> eps_inf = SpatialDataArray(np.ones((5, 6, 7)), coords=coords)
    >>> a1 = SpatialDataArray(-np.random.random((5, 6, 7)), coords=coords)
    >>> c1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> a2 = SpatialDataArray(-np.random.random((5, 6, 7)), coords=coords)
    >>> c2 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> pole_res = CustomPoleResidue(eps_inf=eps_inf, poles=[(a1, c1), (a2, c2)])
    >>> eps = pole_res.eps_model(200e12)

    See Also
    --------

    **Notebooks**

    * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**

    * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: CustomSpatialDataTypeAnnotated = pd.Field(
        ...,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    poles: Tuple[Tuple[CustomSpatialDataTypeAnnotated, CustomSpatialDataTypeAnnotated], ...] = (
        pd.Field(
            (),
            title="Poles",
            description="Tuple of complex-valued (:math:`a_i, c_i`) poles for the model.",
            units=(RADPERSEC, RADPERSEC),
        )
    )

    _no_nans_eps_inf = validate_no_nans("eps_inf")
    _no_nans_poles = validate_no_nans("poles")
    _warn_if_none = CustomDispersiveMedium._warn_if_data_none("poles")

    @pd.validator("eps_inf", always=True)
    def _eps_inf_positive(cls, val):
        """eps_inf must be positive"""
        if not CustomDispersiveMedium._validate_isreal_dataarray(val):
            raise SetupError("'eps_inf' must be real.")
        if np.any(_get_numpy_array(val) < 0):
            raise SetupError("'eps_inf' must be positive.")
        return val

    @pd.validator("poles", always=True)
    @skip_if_fields_missing(["eps_inf"])
    def _poles_correct_shape(cls, val, values):
        """poles must have the same shape."""

        for coeffs in val:
            for coeff in coeffs:
                if not _check_same_coordinates(coeff, values["eps_inf"]):
                    raise SetupError(
                        "All pole coefficients 'a' and 'c' must have the same coordinates; "
                        "The coordinates must also be consistent with 'eps_inf'."
                    )
        return val

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        if not self.eps_inf.is_uniform:
            return False

        for coeffs in self.poles:
            for coeff in coeffs:
                if not coeff.is_uniform:
                    return False
        return True

    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """
        eps = PoleResidue.eps_model(self, frequency)
        return (eps, eps, eps)

    def poles_on_grid(self, coords: Coords) -> Tuple[Tuple[ArrayComplex3D, ArrayComplex3D], ...]:
        """Spatial profile of poles interpolated at the supplied coordinates.

        Parameters
        ----------
        coords : :class:`.Coords`
            The grid point coordinates over which interpolation is performed.

        Returns
        -------
        Tuple[Tuple[ArrayComplex3D, ArrayComplex3D], ...]
            The poles interpolated at the supplied coordinate.
        """

        def fun_interp(input_data: SpatialDataArray) -> ArrayComplex3D:
            return _get_numpy_array(coords.spatial_interp(input_data, self.interp_method))

        return tuple((fun_interp(a), fun_interp(c)) for (a, c) in self.poles)

    @classmethod
    def from_medium(cls, medium: CustomMedium) -> CustomPoleResidue:
        """Convert a :class:`.CustomMedium` to a pole residue model.

        Parameters
        ----------
        medium: :class:`.CustomMedium`
            The medium with permittivity and conductivity to convert.

        Returns
        -------
        :class:`.CustomPoleResidue`
            The pole residue equivalent.
        """
        poles = [(_zeros_like(medium.conductivity), medium.conductivity / (2 * EPSILON_0))]
        medium_dict = medium.dict(exclude={"type", "eps_dataset", "permittivity", "conductivity"})
        medium_dict.update({"eps_inf": medium.permittivity, "poles": poles})
        return CustomPoleResidue.parse_obj(medium_dict)

    def to_medium(self) -> CustomMedium:
        """Convert to a :class:`.CustomMedium`.
        Requires the pole residue model to only have a pole at 0 frequency,
        corresponding to a constant conductivity term.

        Returns
        -------
        :class:`.CustomMedium`
            The non-dispersive equivalent with constant permittivity and conductivity.
        """
        res = 0
        for a, c in self.poles:
            if np.any(abs(_get_numpy_array(a)) > fp_eps):
                raise ValidationError(
                    "Cannot convert dispersive 'CustomPoleResidue' to 'CustomMedium'."
                )
            res = res + (c + np.conj(c)) / 2
        sigma = res * 2 * EPSILON_0

        self_dict = self.dict(exclude={"type", "eps_inf", "poles"})
        self_dict.update({"permittivity": self.eps_inf, "conductivity": np.real(sigma)})
        return CustomMedium.parse_obj(self_dict)

    @cached_property
    def loss_upper_bound(self) -> float:
        """Not implemented yet."""
        raise SetupError("To be implemented.")

    def _sel_custom_data_inside(self, bounds: Bound):
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        CustomPoleResidue
            CustomPoleResidue with reduced data.
        """
        if not self.eps_inf.does_cover(bounds=bounds):
            log.warning("eps_inf spatial data array does not fully cover the requested region.")
        eps_inf_reduced = self.eps_inf.sel_inside(bounds=bounds)
        poles_reduced = []
        for pole, residue in self.poles:
            if not pole.does_cover(bounds=bounds):
                log.warning("Pole spatial data array does not fully cover the requested region.")

            if not residue.does_cover(bounds=bounds):
                log.warning("Residue spatial data array does not fully cover the requested region.")

            poles_reduced.append((pole.sel_inside(bounds), residue.sel_inside(bounds)))

        return self.updated_copy(eps_inf=eps_inf_reduced, poles=poles_reduced)

    def compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute adjoint derivatives for each of the ``fields`` given the multiplied E and D."""

        dJ_deps = 0.0
        for dim in "xyz":
            dJ_deps += self._derivative_field_cmp(
                E_der_map=derivative_info.E_der_map,
                eps_data=self.eps_inf,
                dim=dim,
                freqs=np.atleast_1d(derivative_info.frequency),
            )

        # TODO: fix for multi-frequency
        frequency = derivative_info.frequency

        poles_complex = [
            (np.array(a.values, dtype=complex), np.array(c.values, dtype=complex))
            for a, c in self.poles
        ]
        poles_complex = np.stack(poles_complex, axis=0)

        def eps_model_r(
            eps_inf: complex, poles: list[tuple[complex, complex]], frequency: float
        ) -> float:
            """Real part of ``eps_model`` evaluated on ``self`` fields."""
            return np.real(self._eps_model(eps_inf, poles, frequency))

        def eps_model_i(
            eps_inf: complex, poles: list[tuple[complex, complex]], frequency: float
        ) -> float:
            """Real part of ``eps_model`` evaluated on ``self`` fields."""
            return np.imag(self._eps_model(eps_inf, poles, frequency))

        # compute the gradients w.r.t. each real and imaginary parts for eps_inf and poles
        grad_eps_model_r = ag.elementwise_grad(eps_model_r, argnum=(0, 1))
        grad_eps_model_i = ag.elementwise_grad(eps_model_i, argnum=(0, 1))
        deps_deps_inf_r, deps_dpoles_r = grad_eps_model_r(
            self.eps_inf.values, poles_complex, frequency
        )
        deps_deps_inf_i, deps_dpoles_i = grad_eps_model_i(
            self.eps_inf.values, poles_complex, frequency
        )

        # multiply with dJ_deps partial derivative to give full gradients

        deps_deps_inf = deps_deps_inf_r + 1j * deps_deps_inf_i
        dJ_deps_inf = dJ_deps * deps_deps_inf / 3.0  # mysterious 3

        dJ_dpoles = []
        for (da_r, dc_r), (da_i, dc_i) in zip(deps_dpoles_r, deps_dpoles_i):
            da = da_r + 1j * da_i
            dc = dc_r + 1j * dc_i
            dJ_da = dJ_deps * da / 2.0  # mysterious 2
            dJ_dc = dJ_deps * dc / 2.0  # mysterious 2
            dJ_dpoles.append((dJ_da, dJ_dc))

        derivative_map = {}
        for field_path in derivative_info.paths:
            field_name, *rest = field_path

            if field_name == "eps_inf":
                derivative_map[field_path] = np.real(dJ_deps_inf)

            elif field_name == "poles":
                pole_index, a_or_c = rest
                derivative_map[field_path] = dJ_dpoles[pole_index][a_or_c]

        return derivative_map


class Sellmeier(DispersiveMedium):
    """A dispersive medium described by the Sellmeier model.

    Notes
    -----

        The frequency-dependence of the refractive index is described by:

        .. math::

            n(\\lambda)^2 = 1 + \\sum_i \\frac{B_i \\lambda^2}{\\lambda^2 - C_i}

        For lossless, weakly dispersive materials, the best way to incorporate the dispersion without doing
        complicated fits and without slowing the simulation down significantly is to provide the value of the
        refractive index dispersion :math:`\\frac{dn}{d\\lambda}` in :meth:`tidy3d.Sellmeier.from_dispersion`. The
        value is assumed to be at the central frequency or wavelength (whichever is provided), and a one-pole model
        for the material is generated.

    Example
    -------
    >>> sellmeier_medium = Sellmeier(coeffs=[(1,2), (3,4)])
    >>> eps = sellmeier_medium.eps_model(200e12)

    See Also
    --------

    :class:`CustomSellmeier`
        A spatially varying dispersive medium described by the Sellmeier model.

    **Notebooks**

    * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**

    * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    coeffs: Tuple[Tuple[float, pd.PositiveFloat], ...] = pd.Field(
        title="Coefficients",
        description="List of Sellmeier (:math:`B_i, C_i`) coefficients.",
        units=(None, MICROMETER + "^2"),
    )

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if `allow_gain` is False."""
        if values.get("allow_gain"):
            return val
        for B, _ in val:
            if B < 0:
                raise ValidationError(
                    "For passive medium, 'B_i' must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return val

    @pd.validator("modulation_spec", always=True)
    def _validate_permittivity_modulation(cls, val):
        """Assert modulated permittivity cannot be <= 0."""

        if val is None or val.permittivity is None:
            return val

        min_eps_inf = 1.0
        if min_eps_inf - val.permittivity.max_modulation <= 0:
            raise ValidationError(
                "The minimum permittivity value with modulation applied was found to be negative."
            )
        return val

    _validate_conductivity_modulation = DispersiveMedium._conductivity_modulation_validation()

    def _n_model(self, frequency: float) -> complex:
        """Complex-valued refractive index as a function of frequency."""

        wvl = C_0 / np.array(frequency)
        wvl2 = wvl**2
        n_squared = 1.0
        for B, C in self.coeffs:
            n_squared = n_squared + B * wvl2 / (wvl2 - C)
        return np.sqrt(n_squared + 0j)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        n = self._n_model(frequency)
        return AbstractMedium.nk_to_eps_complex(n)

    def _pole_residue_dict(self) -> Dict:
        """Dict representation of Medium as a pole-residue model"""
        poles = []
        for B, C in self.coeffs:
            beta = 2 * np.pi * C_0 / np.sqrt(C)
            alpha = -0.5 * beta * B
            a = 1j * beta
            c = 1j * alpha
            poles.append((a, c))
        return dict(eps_inf=1, poles=poles, frequency_range=self.frequency_range, name=self.name)

    @staticmethod
    def _from_dispersion_to_coeffs(n: float, freq: float, dn_dwvl: float):
        """Compute Sellmeier coefficients from dispersion."""
        wvl = C_0 / np.array(freq)
        nsqm1 = n**2 - 1
        c_coeff = -(wvl**3) * n * dn_dwvl / (nsqm1 - wvl * n * dn_dwvl)
        b_coeff = (wvl**2 - c_coeff) / wvl**2 * nsqm1
        return [(b_coeff, c_coeff)]

    @classmethod
    def from_dispersion(cls, n: float, freq: float, dn_dwvl: float = 0, **kwargs):
        """Convert ``n`` and wavelength dispersion ``dn_dwvl`` values at frequency ``freq`` to
        a single-pole :class:`Sellmeier` medium.

        Parameters
        ----------
        n : float
            Real part of refractive index. Must be larger than or equal to one.
        dn_dwvl : float = 0
            Derivative of the refractive index with wavelength (1/um). Must be negative.
        freq : float
            Frequency at which ``n`` and ``dn_dwvl`` are sampled.

        Returns
        -------
        :class:`Sellmeier`
            Single-pole Sellmeier medium with the prvoided refractive index and index dispersion
            valuesat at the prvoided frequency.
        """

        if dn_dwvl >= 0:
            raise ValidationError("Dispersion ``dn_dwvl`` must be smaller than zero.")
        if n < 1:
            raise ValidationError("Refractive index ``n`` cannot be smaller than one.")
        return cls(coeffs=cls._from_dispersion_to_coeffs(n, freq, dn_dwvl), **kwargs)


class CustomSellmeier(CustomDispersiveMedium, Sellmeier):
    """A spatially varying dispersive medium described by the Sellmeier model.

    Notes
    -----

        The frequency-dependence of the refractive index is described by:

        .. math::

            n(\\lambda)^2 = 1 + \\sum_i \\frac{B_i \\lambda^2}{\\lambda^2 - C_i}

    Example
    -------
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 6)
    >>> z = np.linspace(-1, 1, 7)
    >>> coords = dict(x=x, y=y, z=z)
    >>> b1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> c1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> sellmeier_medium = CustomSellmeier(coeffs=[(b1,c1),])
    >>> eps = sellmeier_medium.eps_model(200e12)

    See Also
    --------

    :class:`Sellmeier`
        A dispersive medium described by the Sellmeier model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    coeffs: Tuple[Tuple[CustomSpatialDataTypeAnnotated, CustomSpatialDataTypeAnnotated], ...] = (
        pd.Field(
            ...,
            title="Coefficients",
            description="List of Sellmeier (:math:`B_i, C_i`) coefficients.",
            units=(None, MICROMETER + "^2"),
        )
    )

    _no_nans = validate_no_nans("coeffs")

    _warn_if_none = CustomDispersiveMedium._warn_if_data_none("coeffs")

    @pd.validator("coeffs", always=True)
    def _correct_shape_and_sign(cls, val):
        """every term in coeffs must have the same shape, and B>=0 and C>0."""
        if len(val) == 0:
            return val
        for B, C in val:
            if not _check_same_coordinates(B, val[0][0]) or not _check_same_coordinates(
                C, val[0][0]
            ):
                raise SetupError("Every term in 'coeffs' must have the same coordinates.")
            if not CustomDispersiveMedium._validate_isreal_dataarray_tuple((B, C)):
                raise SetupError("'B' and 'C' must be real.")
            if np.any(_get_numpy_array(C) <= 0):
                raise SetupError("'C' must be positive.")
        return val

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if `allow_gain` is False."""
        if values.get("allow_gain"):
            return val
        for B, _ in val:
            if np.any(_get_numpy_array(B) < 0):
                raise ValidationError(
                    "For passive medium, 'B_i' must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return val

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        for coeffs in self.coeffs:
            for coeff in coeffs:
                if not coeff.is_uniform:
                    return False
        return True

    def _pole_residue_dict(self) -> Dict:
        """Dict representation of Medium as a pole-residue model."""
        poles_dict = Sellmeier._pole_residue_dict(self)
        if len(self.coeffs) > 0:
            poles_dict.update({"eps_inf": _ones_like(self.coeffs[0][0])})
        return poles_dict

    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """
        eps = Sellmeier.eps_model(self, frequency)
        # if `eps` is simply a float, convert it to a SpatialDataArray ; this is possible when
        # `coeffs` is empty.
        if isinstance(eps, (int, float, complex)):
            eps = SpatialDataArray(eps * np.ones((1, 1, 1)), coords=dict(x=[0], y=[0], z=[0]))
        return (eps, eps, eps)

    @classmethod
    def from_dispersion(
        cls,
        n: CustomSpatialDataType,
        freq: float,
        dn_dwvl: CustomSpatialDataType,
        interp_method="nearest",
        **kwargs,
    ):
        """Convert ``n`` and wavelength dispersion ``dn_dwvl`` values at frequency ``freq`` to
        a single-pole :class:`CustomSellmeier` medium.

        Parameters
        ----------
        n : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ]
            Real part of refractive index. Must be larger than or equal to one.
        dn_dwvl : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ]
            Derivative of the refractive index with wavelength (1/um). Must be negative.
        freq : float
            Frequency at which ``n`` and ``dn_dwvl`` are sampled.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain permittivity values that are not supplied
            at the Yee grids.

        Returns
        -------
        :class:`.CustomSellmeier`
            Single-pole Sellmeier medium with the prvoided refractive index and index dispersion
            valuesat at the prvoided frequency.
        """

        if not _check_same_coordinates(n, dn_dwvl):
            raise ValidationError("'n' and'dn_dwvl' must have the same dimension.")
        if np.any(_get_numpy_array(dn_dwvl) >= 0):
            raise ValidationError("Dispersion ``dn_dwvl`` must be smaller than zero.")
        if np.any(_get_numpy_array(n) < 1):
            raise ValidationError("Refractive index ``n`` cannot be smaller than one.")
        return cls(
            coeffs=cls._from_dispersion_to_coeffs(n, freq, dn_dwvl),
            interp_method=interp_method,
            **kwargs,
        )

    def _sel_custom_data_inside(self, bounds: Bound):
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        CustomSellmeier
            CustomSellmeier with reduced data.
        """
        coeffs_reduced = []
        for b_coeff, c_coeff in self.coeffs:
            if not b_coeff.does_cover(bounds=bounds):
                log.warning(
                    "Sellmeier B coeff spatial data array does not fully cover the requested region."
                )

            if not c_coeff.does_cover(bounds=bounds):
                log.warning(
                    "Sellmeier C coeff spatial data array does not fully cover the requested region."
                )

            coeffs_reduced.append((b_coeff.sel_inside(bounds), c_coeff.sel_inside(bounds)))

        return self.updated_copy(coeffs=coeffs_reduced)


class Lorentz(DispersiveMedium):
    """A dispersive medium described by the Lorentz model.

    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(f) = \\epsilon_\\infty + \\sum_i
            \\frac{\\Delta\\epsilon_i f_i^2}{f_i^2 - 2jf\\delta_i - f^2}

    Example
    -------
    >>> lorentz_medium = Lorentz(eps_inf=2.0, coeffs=[(1,2,3), (4,5,6)])
    >>> eps = lorentz_medium.eps_model(200e12)

    See Also
    --------

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: pd.PositiveFloat = pd.Field(
        1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    coeffs: Tuple[Tuple[float, float, pd.NonNegativeFloat], ...] = pd.Field(
        ...,
        title="Coefficients",
        description="List of (:math:`\\Delta\\epsilon_i, f_i, \\delta_i`) values for model.",
        units=(PERMITTIVITY, HERTZ, HERTZ),
    )

    @pd.validator("coeffs", always=True)
    def _coeffs_unequal_f_delta(cls, val):
        """f**2 and delta**2 cannot be exactly the same."""
        for _, f, delta in val:
            if f**2 == delta**2:
                raise SetupError("'f' and 'delta' cannot take equal values.")
        return val

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if ``allow_gain`` is False."""
        if values.get("allow_gain"):
            return val
        for del_ep, _, _ in val:
            if del_ep < 0:
                raise ValidationError(
                    "For passive medium, 'Delta epsilon_i' must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return val

    _validate_permittivity_modulation = DispersiveMedium._permittivity_modulation_validation()
    _validate_conductivity_modulation = DispersiveMedium._conductivity_modulation_validation()

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        eps = self.eps_inf + 0.0j
        for de, f, delta in self.coeffs:
            eps = eps + (de * f**2) / (f**2 - 2j * frequency * delta - frequency**2)
        return eps

    def _pole_residue_dict(self) -> Dict:
        """Dict representation of Medium as a pole-residue model."""

        poles = []
        for de, f, delta in self.coeffs:
            w = 2 * np.pi * f
            d = 2 * np.pi * delta

            if self._all_larger(d**2, w**2):
                r = np.sqrt(d * d - w * w) + 0j
                a0 = -d + r
                c0 = de * w**2 / 4 / r
                a1 = -d - r
                c1 = -c0
                poles.extend(((a0, c0), (a1, c1)))
            else:
                r = np.sqrt(w * w - d * d)
                a = -d - 1j * r
                c = 1j * de * w**2 / 2 / r
                poles.append((a, c))

        return dict(
            eps_inf=self.eps_inf,
            poles=poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )

    @staticmethod
    def _all_larger(coeff_a, coeff_b) -> bool:
        """``coeff_a`` and ``coeff_b`` can be either float or SpatialDataArray."""
        if isinstance(coeff_a, CustomSpatialDataType.__args__):
            return np.all(_get_numpy_array(coeff_a) > _get_numpy_array(coeff_b))
        return coeff_a > coeff_b

    @classmethod
    def from_nk(cls, n: float, k: float, freq: float, **kwargs):
        """Convert ``n`` and ``k`` values at frequency ``freq`` to a single-pole Lorentz
        medium.

        Parameters
        ----------
        n : float
            Real part of refractive index.
        k : float = 0
            Imaginary part of refrative index.
        freq : float
            Frequency to evaluate permittivity at (Hz).
        kwargs: dict
            Keyword arguments passed to the medium construction.

        Returns
        -------
        :class:`Lorentz`
            Lorentz medium having refractive index n+ik at frequency ``freq``.
        """
        eps_complex = AbstractMedium.nk_to_eps_complex(n, k)
        eps_r, eps_i = eps_complex.real, eps_complex.imag
        if eps_r >= 1:
            log.warning(
                "For 'permittivity>=1', it is more computationally efficient to "
                "use a dispersiveless medium constructed from 'Medium.from_nk()'."
            )
        # first, lossless medium
        if isclose(eps_i, 0):
            if eps_r < 1:
                fp = np.sqrt((eps_r - 1) / (eps_r - 2)) * freq
                return cls(
                    eps_inf=1,
                    coeffs=[
                        (1, fp, 0),
                    ],
                )
            return cls(
                eps_inf=1,
                coeffs=[
                    ((eps_r - 1) / 2, np.sqrt(2) * freq, 0),
                ],
            )
        # lossy medium
        alpha = (eps_r - 1) / eps_i
        delta_p = freq / 2 / (alpha**2 - alpha + 1)
        fp = np.sqrt((alpha**2 + 1) / (alpha**2 - alpha + 1)) * freq
        return cls(
            eps_inf=1,
            coeffs=[
                (eps_i, fp, delta_p),
            ],
            **kwargs,
        )


class CustomLorentz(CustomDispersiveMedium, Lorentz):
    """A spatially varying dispersive medium described by the Lorentz model.

    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(f) = \\epsilon_\\infty + \\sum_i
            \\frac{\\Delta\\epsilon_i f_i^2}{f_i^2 - 2jf\\delta_i - f^2}

    Example
    -------
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 6)
    >>> z = np.linspace(-1, 1, 7)
    >>> coords = dict(x=x, y=y, z=z)
    >>> eps_inf = SpatialDataArray(np.ones((5, 6, 7)), coords=coords)
    >>> d_epsilon = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> f = SpatialDataArray(1+np.random.random((5, 6, 7)), coords=coords)
    >>> delta = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> lorentz_medium = CustomLorentz(eps_inf=eps_inf, coeffs=[(d_epsilon,f,delta),])
    >>> eps = lorentz_medium.eps_model(200e12)

    See Also
    --------

    :class:`CustomPoleResidue`:
        A spatially varying dispersive medium described by the pole-residue pair model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: CustomSpatialDataTypeAnnotated = pd.Field(
        ...,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    coeffs: Tuple[
        Tuple[
            CustomSpatialDataTypeAnnotated,
            CustomSpatialDataTypeAnnotated,
            CustomSpatialDataTypeAnnotated,
        ],
        ...,
    ] = pd.Field(
        ...,
        title="Coefficients",
        description="List of (:math:`\\Delta\\epsilon_i, f_i, \\delta_i`) values for model.",
        units=(PERMITTIVITY, HERTZ, HERTZ),
    )

    _no_nans_eps_inf = validate_no_nans("eps_inf")
    _no_nans_coeffs = validate_no_nans("coeffs")

    _warn_if_none = CustomDispersiveMedium._warn_if_data_none("coeffs")

    @pd.validator("eps_inf", always=True)
    def _eps_inf_positive(cls, val):
        """eps_inf must be positive"""
        if not CustomDispersiveMedium._validate_isreal_dataarray(val):
            raise SetupError("'eps_inf' must be real.")
        if np.any(_get_numpy_array(val) < 0):
            raise SetupError("'eps_inf' must be positive.")
        return val

    @pd.validator("coeffs", always=True)
    def _coeffs_unequal_f_delta(cls, val):
        """f and delta cannot be exactly the same.
        Not needed for now because we have a more strict
        validator `_coeffs_delta_all_smaller_or_larger_than_fi`.
        """
        return val

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["eps_inf"])
    def _coeffs_correct_shape(cls, val, values):
        """coeffs must have consistent shape."""
        for de, f, delta in val:
            if (
                not _check_same_coordinates(de, values["eps_inf"])
                or not _check_same_coordinates(f, values["eps_inf"])
                or not _check_same_coordinates(delta, values["eps_inf"])
            ):
                raise SetupError(
                    "All terms in 'coeffs' must have the same coordinates; "
                    "The coordinates must also be consistent with 'eps_inf'."
                )
            if not CustomDispersiveMedium._validate_isreal_dataarray_tuple((de, f, delta)):
                raise SetupError("All terms in 'coeffs' must be real.")
        return val

    @pd.validator("coeffs", always=True)
    def _coeffs_delta_all_smaller_or_larger_than_fi(cls, val):
        """We restrict either all f**2>delta**2 or all f**2<delta**2 for now."""
        for _, f, delta in val:
            f2 = f**2
            delta2 = delta**2
            if not (Lorentz._all_larger(f2, delta2) or Lorentz._all_larger(delta2, f2)):
                raise SetupError(
                    "Coefficients in 'coeffs' are restricted to have "
                    "either all 'delta**2'<'f**2' or all 'delta**2'>'f**2'."
                )
        return val

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if ``allow_gain`` is False."""
        allow_gain = values.get("allow_gain")
        for del_ep, _, delta in val:
            if np.any(_get_numpy_array(delta) < 0):
                raise ValidationError("For stable medium, 'delta_i' must be non-negative.")
            if not allow_gain and np.any(_get_numpy_array(del_ep) < 0):
                raise ValidationError(
                    "For passive medium, 'Delta epsilon_i' must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return val

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        if not self.eps_inf.is_uniform:
            return False
        for coeffs in self.coeffs:
            for coeff in coeffs:
                if not coeff.is_uniform:
                    return False
        return True

    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """
        eps = Lorentz.eps_model(self, frequency)
        return (eps, eps, eps)

    def _sel_custom_data_inside(self, bounds: Bound):
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        CustomLorentz
            CustomLorentz with reduced data.
        """
        if not self.eps_inf.does_cover(bounds=bounds):
            log.warning("Eps inf spatial data array does not fully cover the requested region.")
        eps_inf_reduced = self.eps_inf.sel_inside(bounds=bounds)
        coeffs_reduced = []
        for de, f, delta in self.coeffs:
            if not de.does_cover(bounds=bounds):
                log.warning(
                    "Lorentz 'de' spatial data array does not fully cover the requested region."
                )

            if not f.does_cover(bounds=bounds):
                log.warning(
                    "Lorentz 'f' spatial data array does not fully cover the requested region."
                )

            if not delta.does_cover(bounds=bounds):
                log.warning(
                    "Lorentz 'delta' spatial data array does not fully cover the requested region."
                )

            coeffs_reduced.append(
                (de.sel_inside(bounds), f.sel_inside(bounds), delta.sel_inside(bounds))
            )

        return self.updated_copy(eps_inf=eps_inf_reduced, coeffs=coeffs_reduced)


class Drude(DispersiveMedium):
    """A dispersive medium described by the Drude model.

    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(f) = \\epsilon_\\infty - \\sum_i
            \\frac{ f_i^2}{f^2 + jf\\delta_i}

    Example
    -------
    >>> drude_medium = Drude(eps_inf=2.0, coeffs=[(1,2), (3,4)])
    >>> eps = drude_medium.eps_model(200e12)

    See Also
    --------

    :class:`CustomDrude`:
        A spatially varying dispersive medium described by the Drude model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: pd.PositiveFloat = pd.Field(
        1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    coeffs: Tuple[Tuple[float, pd.PositiveFloat], ...] = pd.Field(
        ...,
        title="Coefficients",
        description="List of (:math:`f_i, \\delta_i`) values for model.",
        units=(HERTZ, HERTZ),
    )

    _validate_permittivity_modulation = DispersiveMedium._permittivity_modulation_validation()
    _validate_conductivity_modulation = DispersiveMedium._conductivity_modulation_validation()

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        eps = self.eps_inf + 0.0j
        for f, delta in self.coeffs:
            eps = eps - (f**2) / (frequency**2 + 1j * frequency * delta)
        return eps

    def _pole_residue_dict(self) -> Dict:
        """Dict representation of Medium as a pole-residue model."""

        poles = []

        for f, delta in self.coeffs:
            w = 2 * np.pi * f
            d = 2 * np.pi * delta

            c0 = (w**2) / 2 / d + 0j
            c1 = -c0
            a1 = -d + 0j

            if isinstance(c0, complex):
                a0 = 0j
            else:
                a0 = 0 * c0

            poles.extend(((a0, c0), (a1, c1)))

        return dict(
            eps_inf=self.eps_inf,
            poles=poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )


class CustomDrude(CustomDispersiveMedium, Drude):
    """A spatially varying dispersive medium described by the Drude model.


    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(f) = \\epsilon_\\infty - \\sum_i
            \\frac{ f_i^2}{f^2 + jf\\delta_i}

    Example
    -------
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 6)
    >>> z = np.linspace(-1, 1, 7)
    >>> coords = dict(x=x, y=y, z=z)
    >>> eps_inf = SpatialDataArray(np.ones((5, 6, 7)), coords=coords)
    >>> f1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> delta1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> drude_medium = CustomDrude(eps_inf=eps_inf, coeffs=[(f1,delta1),])
    >>> eps = drude_medium.eps_model(200e12)

    See Also
    --------

    :class:`Drude`:
        A dispersive medium described by the Drude model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: CustomSpatialDataTypeAnnotated = pd.Field(
        ...,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    coeffs: Tuple[Tuple[CustomSpatialDataTypeAnnotated, CustomSpatialDataTypeAnnotated], ...] = (
        pd.Field(
            ...,
            title="Coefficients",
            description="List of (:math:`f_i, \\delta_i`) values for model.",
            units=(HERTZ, HERTZ),
        )
    )

    _no_nans_eps_inf = validate_no_nans("eps_inf")
    _no_nans_coeffs = validate_no_nans("coeffs")

    _warn_if_none = CustomDispersiveMedium._warn_if_data_none("coeffs")

    @pd.validator("eps_inf", always=True)
    def _eps_inf_positive(cls, val):
        """eps_inf must be positive"""
        if not CustomDispersiveMedium._validate_isreal_dataarray(val):
            raise SetupError("'eps_inf' must be real.")
        if np.any(_get_numpy_array(val) < 0):
            raise SetupError("'eps_inf' must be positive.")
        return val

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["eps_inf"])
    def _coeffs_correct_shape_and_sign(cls, val, values):
        """coeffs must have consistent shape and sign."""
        for f, delta in val:
            if not _check_same_coordinates(f, values["eps_inf"]) or not _check_same_coordinates(
                delta, values["eps_inf"]
            ):
                raise SetupError(
                    "All terms in 'coeffs' must have the same coordinates; "
                    "The coordinates must also be consistent with 'eps_inf'."
                )
            if not CustomDispersiveMedium._validate_isreal_dataarray_tuple((f, delta)):
                raise SetupError("All terms in 'coeffs' must be real.")
            if np.any(_get_numpy_array(delta) <= 0):
                raise SetupError("For stable medium, 'delta' must be positive.")
        return val

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        if not self.eps_inf.is_uniform:
            return False
        for coeffs in self.coeffs:
            for coeff in coeffs:
                if not coeff.is_uniform:
                    return False
        return True

    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """
        eps = Drude.eps_model(self, frequency)
        return (eps, eps, eps)

    def _sel_custom_data_inside(self, bounds: Bound):
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        CustomDrude
            CustomDrude with reduced data.
        """
        if not self.eps_inf.does_cover(bounds=bounds):
            log.warning("Eps inf spatial data array does not fully cover the requested region.")
        eps_inf_reduced = self.eps_inf.sel_inside(bounds=bounds)
        coeffs_reduced = []
        for f, delta in self.coeffs:
            if not f.does_cover(bounds=bounds):
                log.warning(
                    "Drude 'f' spatial data array does not fully cover the requested region."
                )

            if not delta.does_cover(bounds=bounds):
                log.warning(
                    "Drude 'delta' spatial data array does not fully cover the requested region."
                )

            coeffs_reduced.append((f.sel_inside(bounds), delta.sel_inside(bounds)))

        return self.updated_copy(eps_inf=eps_inf_reduced, coeffs=coeffs_reduced)


class Debye(DispersiveMedium):
    """A dispersive medium described by the Debye model.

    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(f) = \\epsilon_\\infty + \\sum_i
            \\frac{\\Delta\\epsilon_i}{1 - jf\\tau_i}

    Example
    -------
    >>> debye_medium = Debye(eps_inf=2.0, coeffs=[(1,2),(3,4)])
    >>> eps = debye_medium.eps_model(200e12)

    See Also
    --------

    :class:`CustomDebye`
        A spatially varying dispersive medium described by the Debye model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: pd.PositiveFloat = pd.Field(
        1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    coeffs: Tuple[Tuple[float, pd.PositiveFloat], ...] = pd.Field(
        ...,
        title="Coefficients",
        description="List of (:math:`\\Delta\\epsilon_i, \\tau_i`) values for model.",
        units=(PERMITTIVITY, SECOND),
    )

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if `allow_gain` is False."""
        if values.get("allow_gain"):
            return val
        for del_ep, _ in val:
            if del_ep < 0:
                raise ValidationError(
                    "For passive medium, 'Delta epsilon_i' must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return val

    _validate_permittivity_modulation = DispersiveMedium._permittivity_modulation_validation()
    _validate_conductivity_modulation = DispersiveMedium._conductivity_modulation_validation()

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        eps = self.eps_inf + 0.0j
        for de, tau in self.coeffs:
            eps = eps + de / (1 - 1j * frequency * tau)
        return eps

    def _pole_residue_dict(self):
        """Dict representation of Medium as a pole-residue model."""

        poles = []
        for de, tau in self.coeffs:
            a = -2 * np.pi / tau + 0j
            c = -0.5 * de * a

            poles.append((a, c))

        return dict(
            eps_inf=self.eps_inf,
            poles=poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )


class CustomDebye(CustomDispersiveMedium, Debye):
    """A spatially varying dispersive medium described by the Debye model.

    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(f) = \\epsilon_\\infty + \\sum_i
            \\frac{\\Delta\\epsilon_i}{1 - jf\\tau_i}

    Example
    -------
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 6)
    >>> z = np.linspace(-1, 1, 7)
    >>> coords = dict(x=x, y=y, z=z)
    >>> eps_inf = SpatialDataArray(1+np.random.random((5, 6, 7)), coords=coords)
    >>> eps1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> tau1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> debye_medium = CustomDebye(eps_inf=eps_inf, coeffs=[(eps1,tau1),])
    >>> eps = debye_medium.eps_model(200e12)

    See Also
    --------

    :class:`Debye`
        A dispersive medium described by the Debye model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: CustomSpatialDataTypeAnnotated = pd.Field(
        ...,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    coeffs: Tuple[Tuple[CustomSpatialDataTypeAnnotated, CustomSpatialDataTypeAnnotated], ...] = (
        pd.Field(
            ...,
            title="Coefficients",
            description="List of (:math:`\\Delta\\epsilon_i, \\tau_i`) values for model.",
            units=(PERMITTIVITY, SECOND),
        )
    )

    _no_nans_eps_inf = validate_no_nans("eps_inf")
    _no_nans_coeffs = validate_no_nans("coeffs")

    _warn_if_none = CustomDispersiveMedium._warn_if_data_none("coeffs")

    @pd.validator("eps_inf", always=True)
    def _eps_inf_positive(cls, val):
        """eps_inf must be positive"""
        if not CustomDispersiveMedium._validate_isreal_dataarray(val):
            raise SetupError("'eps_inf' must be real.")
        if np.any(_get_numpy_array(val) < 0):
            raise SetupError("'eps_inf' must be positive.")
        return val

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["eps_inf"])
    def _coeffs_correct_shape(cls, val, values):
        """coeffs must have consistent shape."""
        for de, tau in val:
            if not _check_same_coordinates(de, values["eps_inf"]) or not _check_same_coordinates(
                tau, values["eps_inf"]
            ):
                raise SetupError(
                    "All terms in 'coeffs' must have the same coordinates; "
                    "The coordinates must also be consistent with 'eps_inf'."
                )
            if not CustomDispersiveMedium._validate_isreal_dataarray_tuple((de, tau)):
                raise SetupError("All terms in 'coeffs' must be real.")
        return val

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if ``allow_gain`` is False."""
        allow_gain = values.get("allow_gain")
        for del_ep, tau in val:
            if np.any(_get_numpy_array(tau) <= 0):
                raise SetupError("For stable medium, 'tau_i' must be positive.")
            if not allow_gain and np.any(_get_numpy_array(del_ep) < 0):
                raise ValidationError(
                    "For passive medium, 'Delta epsilon_i' must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return val

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        if not self.eps_inf.is_uniform:
            return False
        for coeffs in self.coeffs:
            for coeff in coeffs:
                if not coeff.is_uniform:
                    return False
        return True

    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """
        eps = Debye.eps_model(self, frequency)
        return (eps, eps, eps)

    def _sel_custom_data_inside(self, bounds: Bound):
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        CustomDebye
            CustomDebye with reduced data.
        """
        if not self.eps_inf.does_cover(bounds=bounds):
            log.warning("Eps inf spatial data array does not fully cover the requested region.")
        eps_inf_reduced = self.eps_inf.sel_inside(bounds=bounds)
        coeffs_reduced = []
        for de, tau in self.coeffs:
            if not de.does_cover(bounds=bounds):
                log.warning(
                    "Debye 'f' spatial data array does not fully cover the requested region."
                )

            if not tau.does_cover(bounds=bounds):
                log.warning(
                    "Debye 'tau' spatial data array does not fully cover the requested region."
                )

            coeffs_reduced.append((de.sel_inside(bounds), tau.sel_inside(bounds)))

        return self.updated_copy(eps_inf=eps_inf_reduced, coeffs=coeffs_reduced)


class SurfaceImpedanceFitterParam(Tidy3dBaseModel):
    """Advanced parameters for fitting surface impedance of a :class:`.LossyMetalMedium`.
    Internally, the quantity to be fitted is surface impedance divided by ``-1j * \\omega``.
    """

    max_num_poles: pd.PositiveInt = pd.Field(
        LOSSY_METAL_DEFAULT_MAX_POLES,
        title="Maximal Number Of Poles",
        description="Maximal number of poles in complex-conjugate pole residue model for "
        "fitting surface impedance.",
    )

    tolerance_rms: pd.NonNegativeFloat = pd.Field(
        LOSSY_METAL_DEFAULT_TOLERANCE_RMS,
        title="Tolerance In Fitting",
        description="Tolerance in fitting.",
    )

    frequency_sampling_points: pd.PositiveInt = pd.Field(
        LOSSY_METAL_DEFAULT_SAMPLING_FREQUENCY,
        title="Number Of Sampling Frequencies",
        description="Number of sampling frequencies used in fitting.",
    )

    log_sampling: bool = pd.Field(
        True,
        title="Frequencies Sampling In Log Scale",
        description="Whether to sample frequencies logarithmically (``True``),  "
        "or linearly (``False``).",
    )


class AbstractSurfaceRoughness(Tidy3dBaseModel):
    """Abstract class for modeling surface roughness of lossy metal."""

    @abstractmethod
    def roughness_correction_factor(
        self, frequency: ArrayFloat1D, skin_depths: ArrayFloat1D
    ) -> ArrayComplex1D:
        """Complex-valued roughness correction factor applied to surface impedance.

        Notes
        -----
            The roughness correction factor should be causal. It is multiplied to the
            surface impedance of the lossy metal to account for the effects of surface roughness.

        Parameters
        ----------
        frequency : ArrayFloat1D
            Frequency to evaluate roughness correction factor at (Hz).
        skin_depths : ArrayFloat1D
            Skin depths of the lossy metal that is frequency-dependent.

        Returns
        -------
        ArrayComplex1D
            The causal roughness correction factor evaluated at ``frequency``.
        """


class HammerstadSurfaceRoughness(AbstractSurfaceRoughness):
    """Modified Hammerstad surface roughness model. It's a popular model that works well
    under 5 GHz for surface roughness below 2 micrometer RMS.

    Note
    ----

        The power loss compared to smooth surface is described by:

        .. math::

            1 + (RF-1) \\frac{2}{\\pi}\\arctan(1.4\\frac{R_q^2}{\\delta^2})

        where :math:`\\delta` is skin depth, :math:`R_q` the RMS peak-to-vally height, and RF
        roughness factor.

    Note
    ----
    This model is based on:

        Y. Shlepnev, C. Nwachukwu, "Roughness characterization for interconnect analysis",
        2011 IEEE International Symposium on Electromagnetic Compatibility,
        (DOI: 10.1109/ISEMC.2011.6038367), 2011.

        V. Dmitriev-Zdorov, B. Simonovich, I. Kochikov, "A Causal Conductor Roughness Model
        and its Effect on Transmission Line Characteristics", Signal Integrity Journal, 2018.
    """

    rq: pd.PositiveFloat = pd.Field(
        ...,
        title="RMS Peak-to-Valley Height",
        description="RMS peak-to-valley height (Rq) of the surface roughness.",
        units=MICROMETER,
    )

    roughness_factor: float = pd.Field(
        2.0,
        title="Roughness Factor",
        description="Expected maximal increase in conductor losses due to roughness effect. "
        "Value 2 gives the classic Hammerstad equation.",
        gt=1.0,
    )

    def roughness_correction_factor(
        self, frequency: ArrayFloat1D, skin_depths: ArrayFloat1D
    ) -> ArrayComplex1D:
        """Complex-valued roughness correction factor applied to surface impedance.

        Notes
        -----
            The roughness correction factor should be causal. It is multiplied to the
            surface impedance of the lossy metal to account for the effects of surface roughness.

        Parameters
        ----------
        frequency : ArrayFloat1D
            Frequency to evaluate roughness correction factor at (Hz).
        skin_depths : ArrayFloat1D
            Skin depths of the lossy metal that is frequency-dependent.

        Returns
        -------
        ArrayComplex1D
            The causal roughness correction factor evaluated at ``frequency``.
        """
        normalized_laplace = -1.4j * (self.rq / skin_depths) ** 2
        sqrt_normalized_laplace = np.sqrt(normalized_laplace)
        causal_response = np.log(
            1 + 2 * sqrt_normalized_laplace / (1 + normalized_laplace)
        ) + 2 * np.arctan(sqrt_normalized_laplace)
        return 1 + (self.roughness_factor - 1) / np.pi * causal_response


class HuraySurfaceRoughness(AbstractSurfaceRoughness):
    """Huray surface roughness model.

    Note
    ----

        The power loss compared to smooth surface is described by:

        .. math::

            \\frac{A_{matte}}{A_{flat}} + \\frac{3}{2}\\sum_i f_i/[1+\\frac{\\delta}{r_i}+\\frac{\\delta^2}{2r_i^2}]

        where :math:`\\delta` is skin depth, :math:`r_i` the radius of sphere,
        :math:`\\frac{A_{matte}}{A_{flat}}` the relative area of the matte compared to flat surface,
        and :math:`f_i=N_i4\\pi r_i^2/A_{flat}` the ratio of total sphere
        surface area (number of spheres :math:`N_i` times the individual sphere surface area)
        to the flat surface area.

    Note
    ----
    This model is based on:

        J. Eric Bracken, "A Causal Huray Model for Surface Roughness", DesignCon, 2012.
    """

    relative_area: pd.PositiveFloat = pd.Field(
        1,
        title="Relative Area",
        description="Relative area of the matte base compared to a flat surface",
    )

    coeffs: Tuple[Tuple[pd.PositiveFloat, pd.PositiveFloat], ...] = pd.Field(
        ...,
        title="Coefficients for surface ratio and sphere radius",
        description="List of (:math:`f_i, r_i`) values for model, where :math:`f_i` is "
        "the ratio of total sphere surface area to the flat surface area, and :math:`r_i` "
        "the radius of the sphere.",
        units=(None, MICROMETER),
    )

    @classmethod
    def from_cannonball_huray(cls, radius: float) -> HuraySurfaceRoughness:
        """Construct a Cannonball-Huray model.

        Note
        ----

            The power loss compared to smooth surface is described by:

            .. math::

                1 + \\frac{7\\pi}{3} \\frac{1}{1+\\frac{\\delta}{r}+\\frac{\\delta^2}{2r^2}}

        Parameters
        ----------
        radius : float
            Radius of the sphere.

        Returns
        -------
        HuraySurfaceRoughness
            The Huray surface roughness model.
        """
        return cls(relative_area=1, coeffs=[(14.0 / 9 * np.pi, radius)])

    def roughness_correction_factor(
        self, frequency: ArrayFloat1D, skin_depths: ArrayFloat1D
    ) -> ArrayComplex1D:
        """Complex-valued roughness correction factor applied to surface impedance.

        Notes
        -----
            The roughness correction factor should be causal. It is multiplied to the
            surface impedance of the lossy metal to account for the effects of surface roughness.

        Parameters
        ----------
        frequency : ArrayFloat1D
            Frequency to evaluate roughness correction factor at (Hz).
        skin_depths : ArrayFloat1D
            Skin depths of the lossy metal that is frequency-dependent.

        Returns
        -------
        ArrayComplex1D
            The causal roughness correction factor evaluated at ``frequency``.
        """

        correction = self.relative_area
        for f, r in self.coeffs:
            normalized_laplace = -2j * (r / skin_depths) ** 2
            sqrt_normalized_laplace = np.sqrt(normalized_laplace)
            correction += 1.5 * f / (1 + 1 / sqrt_normalized_laplace)
        return correction


SurfaceRoughnessType = Union[HammerstadSurfaceRoughness, HuraySurfaceRoughness]


class LossyMetalMedium(Medium):
    """Lossy metal that can be modeled with a surface impedance boundary condition (SIBC).

    Notes
    -----

        SIBC is most accurate when the skin depth is much smaller than the structure feature size.
        If not the case, please use a regular medium instead, or set ``simulation.subpixel.lossy_metal``
        to ``td.VolumetricAveraging()`` or ``td.Staircasing()``.

    Example
    -------
    >>> lossy_metal = LossyMetalMedium(conductivity=10, frequency_range=(9e9, 10e9))

    """

    allow_gain: Literal[False] = pd.Field(
        False,
        title="Allow gain medium",
        description="Allow the medium to be active. Caution: "
        "simulations with a gain medium are unstable, and are likely to diverge."
        "Simulations where 'allow_gain' is set to 'True' will still be charged even if "
        "diverged. Monitor data up to the divergence point will still be returned and can be "
        "useful in some cases.",
    )

    permittivity: Literal[1] = pd.Field(
        1.0, title="Permittivity", description="Relative permittivity.", units=PERMITTIVITY
    )

    roughness: SurfaceRoughnessType = pd.Field(
        None,
        title="Surface Roughness Model",
        description="Surface roughness model that applies a frequency-dependent scaling "
        "factor to surface impedance.",
        discriminator=TYPE_TAG_STR,
    )

    frequency_range: FreqBound = pd.Field(
        ...,
        title="Frequency Range",
        description="Frequency range of validity for the medium.",
        units=(HERTZ, HERTZ),
    )

    fit_param: SurfaceImpedanceFitterParam = pd.Field(
        SurfaceImpedanceFitterParam(),
        title="Fitting Parameters For Surface Impedance",
        description="Parameters for fitting surface impedance divided by (-1j * omega) over "
        "the frequency range using pole-residue pair model.",
    )

    @pd.validator("frequency_range")
    def _validate_frequency_range(cls, val):
        """Validate that frequency range is finite and non-zero."""
        for freq in val:
            if not np.isfinite(freq):
                raise ValidationError("Values in 'frequency_range' must be finite.")
            if freq <= 0:
                raise ValidationError("Values in 'frequency_range' must be positive.")
        return val

    @pd.validator("conductivity", always=True)
    def _positive_conductivity(cls, val):
        """Assert conductivity>0."""
        if val <= 0:
            raise ValidationError("For lossy metal, 'conductivity' must be positive. ")
        return val

    @cached_property
    def _fitting_result(self) -> Tuple[PoleResidue, float]:
        """Fitted scaled surface impedance and residue."""

        omega_data = self.Hz_to_angular_freq(self.sampling_frequencies)
        surface_impedance = self.surface_impedance(self.sampling_frequencies)
        scaled_impedance = surface_impedance / (-1j * omega_data)

        # let's use scaled quantity in fitting: minimal real part equals ``SCALED_REAL_PART``
        min_real = np.min(scaled_impedance.real)
        if min_real <= 0:
            raise SetupError(
                "The real part of scaled surface impedance must be positive. "
                "Please create a github issue so that the problem can be investigated. "
                "In the meantime, make sure the material is passive."
            )

        scaling_factor = LOSSY_METAL_SCALED_REAL_PART / min_real
        scaled_impedance *= scaling_factor

        (res_inf, poles, residues), error = fit(
            omega_data=omega_data,
            resp_data=scaled_impedance,
            min_num_poles=0,
            max_num_poles=self.fit_param.max_num_poles,
            resp_inf=None,
            tolerance_rms=self.fit_param.tolerance_rms,
            scale_factor=1.0 / np.max(omega_data),
        )

        res_inf /= scaling_factor
        residues /= scaling_factor
        return PoleResidue(eps_inf=res_inf, poles=list(zip(poles, residues))), error

    @cached_property
    def scaled_surface_impedance_model(self) -> PoleResidue:
        """Fitted surface impedance divided by (-j \\omega) using pole-residue pair model within ``frequency_range``."""
        return self._fitting_result[0]

    @cached_property
    def num_poles(self) -> int:
        """Number of poles in the fitted model."""
        return len(self.scaled_surface_impedance_model.poles)

    def surface_impedance(self, frequencies: ArrayFloat1D):
        """Computing surface impedance including surface roughness effects."""
        # compute complex-valued skin depth
        n, k = self.nk_model(frequencies)

        # with surface roughness effects
        correction = 1.0
        if self.roughness is not None:
            skin_depths = 1 / np.sqrt(np.pi * frequencies * MU_0 * self.conductivity)
            correction = self.roughness.roughness_correction_factor(frequencies, skin_depths)

        return correction * ETA_0 / (n + 1j * k)

    @cached_property
    def sampling_frequencies(self) -> ArrayFloat1D:
        """Sampling frequencies used in fitting."""
        if self.fit_param.frequency_sampling_points < 2:
            return np.array([np.mean(self.frequency_range)])

        if self.fit_param.log_sampling:
            return np.logspace(
                np.log10(self.frequency_range[0]),
                np.log10(self.frequency_range[1]),
                self.fit_param.frequency_sampling_points,
            )
        return np.linspace(
            self.frequency_range[0],
            self.frequency_range[1],
            self.fit_param.frequency_sampling_points,
        )

    def eps_diagonal_numerical(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor for numerical considerations
        such as meshing and runtime estimation.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[complex, complex, complex]
            The diagonal elements of relative permittivity tensor relevant for numerical
            considerations evaluated at ``frequency``.
        """
        return (1.0 + 0j,) * 3

    @add_ax_if_none
    def plot(
        self,
        ax: Ax = None,
    ) -> Ax:
        """Make plot of complex-valued surface imepdance model vs fitted model, at sampling frequencies.
        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes = None
            Axes to plot the data on, if None, a new one is created.
        Returns
        -------
        matplotlib.axis.Axes
            Matplotlib axis corresponding to plot.
        """
        frequencies = self.sampling_frequencies
        surface_impedance = self.surface_impedance(frequencies)

        ax.plot(frequencies, surface_impedance.real, "x", label="Real")
        ax.plot(frequencies, surface_impedance.imag, "+", label="Imag")

        surface_impedance_model = (
            -1j
            * self.Hz_to_angular_freq(frequencies)
            * self.scaled_surface_impedance_model.eps_model(frequencies)
        )
        ax.plot(frequencies, surface_impedance_model.real, label="Real (model)")
        ax.plot(frequencies, surface_impedance_model.imag, label="Imag (model)")

        ax.set_ylabel(r"Surface impedance ($\Omega$)")
        ax.set_xlabel("Frequency (Hz)")
        ax.legend()

        return ax


IsotropicUniformMediumType = Union[
    Medium, LossyMetalMedium, PoleResidue, Sellmeier, Lorentz, Debye, Drude, PECMedium
]
IsotropicCustomMediumType = Union[
    CustomPoleResidue,
    CustomSellmeier,
    CustomLorentz,
    CustomDebye,
    CustomDrude,
]
IsotropicCustomMediumInternalType = Union[IsotropicCustomMediumType, CustomIsotropicMedium]
IsotropicMediumType = Union[IsotropicCustomMediumType, IsotropicUniformMediumType]


class AnisotropicMedium(AbstractMedium):
    """Diagonally anisotropic medium.

    Notes
    -----

        Only diagonal anisotropy is currently supported.

    Example
    -------
    >>> medium_xx = Medium(permittivity=4.0)
    >>> medium_yy = Medium(permittivity=4.1)
    >>> medium_zz = Medium(permittivity=3.9)
    >>> anisotropic_dielectric = AnisotropicMedium(xx=medium_xx, yy=medium_yy, zz=medium_zz)

    See Also
    --------

    :class:`CustomAnisotropicMedium`
        Diagonally anisotropic medium with spatially varying permittivity in each component.

    :class:`FullyAnisotropicMedium`
        Fully anisotropic medium including all 9 components of the permittivity and conductivity tensors.

    **Notebooks**
        * `Broadband polarizer assisted by anisotropic metamaterial <../../notebooks/SWGBroadbandPolarizer.html>`_
        * `Thin film lithium niobate adiabatic waveguide coupler <../../notebooks/AdiabaticCouplerLN.html>`_
    """

    xx: IsotropicUniformMediumType = pd.Field(
        ...,
        title="XX Component",
        description="Medium describing the xx-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    yy: IsotropicUniformMediumType = pd.Field(
        ...,
        title="YY Component",
        description="Medium describing the yy-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    zz: IsotropicUniformMediumType = pd.Field(
        ...,
        title="ZZ Component",
        description="Medium describing the zz-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    allow_gain: bool = pd.Field(
        None,
        title="Allow gain medium",
        description="This field is ignored. Please set ``allow_gain`` in each component",
    )

    @pd.validator("modulation_spec", always=True)
    def _validate_modulation_spec(cls, val):
        """Check compatibility with modulation_spec."""
        if val is not None:
            raise ValidationError(
                f"A 'modulation_spec' of class {type(val)} is not "
                f"currently supported for medium class {cls}. "
                "Please add modulation to each component."
            )
        return val

    @pd.root_validator(pre=True)
    def _ignored_fields(cls, values):
        """The field is ignored."""
        if values.get("xx") is not None and values.get("allow_gain") is not None:
            log.warning(
                "The field 'allow_gain' is ignored. Please set 'allow_gain' in each component."
            )
        return values

    @cached_property
    def components(self) -> Dict[str, Medium]:
        """Dictionary of diagonal medium components."""
        return dict(xx=self.xx, yy=self.yy, zz=self.zz)

    @cached_property
    def is_time_modulated(self) -> bool:
        """Whether any component of the medium is time modulated."""
        return any(mat.is_time_modulated for mat in self.components.values())

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For this medium, it takes the minimal of ``n_clf`` in all components.
        """
        return min(mat_component.n_cfl for mat_component in self.components.values())

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        return np.mean(self.eps_diagonal(frequency), axis=0)

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor as a function of frequency."""

        eps_xx = self.xx.eps_model(frequency)
        eps_yy = self.yy.eps_model(frequency)
        eps_zz = self.zz.eps_model(frequency)
        return (eps_xx, eps_yy, eps_zz)

    def eps_comp(self, row: Axis, col: Axis, frequency: float) -> complex:
        """Single component the complex-valued permittivity tensor as a function of frequency.

        Parameters
        ----------
        row : int
            Component's row in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        col : int
            Component's column in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
           Element of the relative permittivity tensor evaluated at ``frequency``.
        """

        if row != col:
            return 0j
        cmp = "xyz"[row]
        field_name = cmp + cmp
        return self.components[field_name].eps_model(frequency)

    def _eps_plot(
        self, frequency: float, eps_component: Optional[PermittivityComponent] = None
    ) -> float:
        """Returns real part of epsilon for plotting. A specific component of the epsilon tensor can
        be selected for anisotropic medium.

        Parameters
        ----------
        frequency : float
        eps_component : PermittivityComponent

        Returns
        -------
        float
            Element ``eps_component`` of the relative permittivity tensor evaluated at ``frequency``.
        """
        if eps_component is None:
            # return the average of the diag
            return self.eps_model(frequency).real
        elif eps_component in ["xx", "yy", "zz"]:
            # return the requested diagonal component
            comp2indx = {"x": 0, "y": 1, "z": 2}
            return self.eps_comp(
                row=comp2indx[eps_component[0]],
                col=comp2indx[eps_component[1]],
                frequency=frequency,
            ).real
        else:
            raise ValueError(
                f"Plotting component '{eps_component}' of a diagonally-anisotropic permittivity tensor is not supported."
            )

    @add_ax_if_none
    def plot(self, freqs: float, ax: Ax = None) -> Ax:
        """Plot n, k of a :class:`.Medium` as a function of frequency."""

        freqs = np.array(freqs)
        freqs_thz = freqs / 1e12

        for label, medium_component in self.elements.items():
            eps_complex = medium_component.eps_model(freqs)
            n, k = AbstractMedium.eps_complex_to_nk(eps_complex)
            ax.plot(freqs_thz, n, label=f"n, eps_{label}")
            ax.plot(freqs_thz, k, label=f"k, eps_{label}")

        ax.set_xlabel("frequency (THz)")
        ax.set_title("medium dispersion")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    @property
    def elements(self) -> Dict[str, IsotropicUniformMediumType]:
        """The diagonal elements of the medium as a dictionary."""
        return dict(xx=self.xx, yy=self.yy, zz=self.zz)

    @cached_property
    def is_pec(self):
        """Whether the medium is a PEC."""
        return any(self.is_comp_pec(i) for i in range(3))

    def is_comp_pec(self, comp: Axis):
        """Whether the medium is a PEC."""
        return isinstance(self.components[["xx", "yy", "zz"][comp]], PECMedium)

    def sel_inside(self, bounds: Bound):
        """Return a new medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        AnisotropicMedium
            AnisotropicMedium with reduced data.
        """

        new_comps = [comp.sel_inside(bounds) for comp in [self.xx, self.yy, self.zz]]

        return self.updated_copy(**dict(zip(["xx", "yy", "zz"], new_comps)))


class AnisotropicMediumFromMedium2D(AnisotropicMedium):
    """The same as ``AnisotropicMedium``, but converted from Medium2D.
    (This class is for internal use only)
    """


class FullyAnisotropicMedium(AbstractMedium):
    """Fully anisotropic medium including all 9 components of the permittivity and conductivity
    tensors.

    Notes
    -----

        Provided permittivity tensor and the symmetric part of the conductivity tensor must
        have coinciding main directions. A non-symmetric conductivity tensor can be used to model
        magneto-optic effects. Note that dispersive properties and subpixel averaging are currently not
        supported for fully anisotropic materials.

    Note
    ----

        Simulations involving fully anisotropic materials are computationally more intensive, thus,
        they take longer time to complete. This increase strongly depends on the filling fraction of
        the simulation domain by fully anisotropic materials, varying approximately in the range from
        1.5 to 5. The cost of running a simulation is adjusted correspondingly.

    Example
    -------
    >>> perm = [[2, 0, 0], [0, 1, 0], [0, 0, 3]]
    >>> cond = [[0.1, 0, 0], [0, 0, 0], [0, 0, 0]]
    >>> anisotropic_dielectric = FullyAnisotropicMedium(permittivity=perm, conductivity=cond)

    See Also
    --------

    :class:`CustomAnisotropicMedium`
        Diagonally anisotropic medium with spatially varying permittivity in each component.

    :class:`AnisotropicMedium`
        Diagonally anisotropic medium.

    **Notebooks**
        * `Broadband polarizer assisted by anisotropic metamaterial <../../notebooks/SWGBroadbandPolarizer.html>`_
        * `Thin film lithium niobate adiabatic waveguide coupler <../../notebooks/AdiabaticCouplerLN.html>`_
        * `Defining fully anisotropic materials <../../notebooks/FullyAnisotropic.html>`_
    """

    permittivity: TensorReal = pd.Field(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        title="Permittivity",
        description="Relative permittivity tensor.",
        units=PERMITTIVITY,
    )

    conductivity: TensorReal = pd.Field(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        title="Conductivity",
        description="Electric conductivity tensor. Defined such that the imaginary part "
        "of the complex permittivity at angular frequency omega is given by conductivity/omega.",
        units=CONDUCTIVITY,
    )

    @pd.validator("modulation_spec", always=True)
    def _validate_modulation_spec(cls, val):
        """Check compatibility with modulation_spec."""
        if val is not None:
            raise ValidationError(
                f"A 'modulation_spec' of class {type(val)} is not "
                f"currently supported for medium class {cls}."
            )
        return val

    @pd.validator("permittivity", always=True)
    def permittivity_spd_and_ge_one(cls, val):
        """Check that provided permittivity tensor is symmetric positive definite
        with eigenvalues >= 1.
        """

        if not np.allclose(val, np.transpose(val), atol=fp_eps):
            raise ValidationError("Provided permittivity tensor is not symmetric.")

        if np.any(np.linalg.eigvals(val) < 1 - fp_eps):
            raise ValidationError("Main diagonal of provided permittivity tensor is not >= 1.")

        return val

    @pd.validator("conductivity", always=True)
    @skip_if_fields_missing(["permittivity"])
    def conductivity_commutes(cls, val, values):
        """Check that the symmetric part of conductivity tensor commutes with permittivity tensor
        (that is, simultaneously diagonalizable).
        """

        perm = values.get("permittivity")
        cond_sym = 0.5 * (val + val.T)
        comm_diff = np.abs(np.matmul(perm, cond_sym) - np.matmul(cond_sym, perm))

        if not np.allclose(comm_diff, 0, atol=fp_eps):
            raise ValidationError(
                "Main directions of conductivity and permittivity tensor do not coincide."
            )

        return val

    @pd.validator("conductivity", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if ``allow_gain`` is False."""
        if values.get("allow_gain"):
            return val

        cond_sym = 0.5 * (val + val.T)
        if np.any(np.linalg.eigvals(cond_sym) < -fp_eps):
            raise ValidationError(
                "For passive medium, main diagonal of provided conductivity tensor "
                "must be non-negative. "
                "To simulate a gain medium, please set 'allow_gain=True'. "
                "Caution: simulations with a gain medium are unstable, and are likely to diverge."
            )
        return val

    @classmethod
    def from_diagonal(cls, xx: Medium, yy: Medium, zz: Medium, rotation: RotationType):
        """Construct a fully anisotropic medium by rotating a diagonally anisotropic medium.

        Parameters
        ----------
        xx : :class:`.Medium`
            Medium describing the xx-component of the diagonal permittivity tensor.
        yy : :class:`.Medium`
            Medium describing the yy-component of the diagonal permittivity tensor.
        zz : :class:`.Medium`
            Medium describing the zz-component of the diagonal permittivity tensor.
        rotation : Union[:class:`.RotationAroundAxis`]
                Rotation applied to diagonal permittivity tensor.

        Returns
        -------
        :class:`FullyAnisotropicMedium`
            Resulting fully anisotropic medium.
        """

        if any(comp.nonlinear_spec is not None for comp in [xx, yy, zz]):
            raise ValidationError(
                "Nonlinearities are not currently supported for the components "
                "of a fully anisotropic medium."
            )

        if any(comp.modulation_spec is not None for comp in [xx, yy, zz]):
            raise ValidationError(
                "Modulation is not currently supported for the components "
                "of a fully anisotropic medium."
            )

        permittivity_diag = np.diag([comp.permittivity for comp in [xx, yy, zz]]).tolist()
        conductivity_diag = np.diag([comp.conductivity for comp in [xx, yy, zz]]).tolist()

        permittivity = rotation.rotate_tensor(permittivity_diag)
        conductivity = rotation.rotate_tensor(conductivity_diag)

        return cls(permittivity=permittivity, conductivity=conductivity)

    @cached_property
    def _to_diagonal(self) -> AnisotropicMedium:
        """Construct a diagonally anisotropic medium from main components.

        Returns
        -------
        :class:`AnisotropicMedium`
            Resulting diagonally anisotropic medium.
        """

        perm, cond, _ = self.eps_sigma_diag

        return AnisotropicMedium(
            xx=Medium(permittivity=perm[0], conductivity=cond[0]),
            yy=Medium(permittivity=perm[1], conductivity=cond[1]),
            zz=Medium(permittivity=perm[2], conductivity=cond[2]),
        )

    @cached_property
    def eps_sigma_diag(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], TensorReal]:
        """Main components of permittivity and conductivity tensors and their directions."""

        perm_diag, vecs = np.linalg.eig(self.permittivity)
        cond_diag = np.diag(np.matmul(np.transpose(vecs), np.matmul(self.conductivity, vecs)))

        return (perm_diag, cond_diag, vecs)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""
        perm_diag, cond_diag, _ = self.eps_sigma_diag

        if not np.isscalar(frequency):
            perm_diag = perm_diag[:, None]
            cond_diag = cond_diag[:, None]
        eps_diag = AbstractMedium.eps_sigma_to_eps_complex(perm_diag, cond_diag, frequency)
        return np.mean(eps_diag)

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor as a function of frequency."""

        perm_diag, cond_diag, _ = self.eps_sigma_diag

        if not np.isscalar(frequency):
            perm_diag = perm_diag[:, None]
            cond_diag = cond_diag[:, None]
        return AbstractMedium.eps_sigma_to_eps_complex(perm_diag, cond_diag, frequency)

    def eps_comp(self, row: Axis, col: Axis, frequency: float) -> complex:
        """Single component the complex-valued permittivity tensor as a function of frequency.

        Parameters
        ----------
        row : int
            Component's row in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        col : int
            Component's column in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
           Element of the relative permittivity tensor evaluated at ``frequency``.
        """

        eps = self.permittivity[row][col]
        sig = self.conductivity[row][col]
        return AbstractMedium.eps_sigma_to_eps_complex(eps, sig, frequency)

    def _eps_plot(
        self, frequency: float, eps_component: Optional[PermittivityComponent] = None
    ) -> float:
        """Returns real part of epsilon for plotting. A specific component of the epsilon tensor can
        be selected for anisotropic medium.

        Parameters
        ----------
        frequency : float
        eps_component : PermittivityComponent

        Returns
        -------
        float
            Element ``eps_component`` of the relative permittivity tensor evaluated at ``frequency``.
        """
        if eps_component is None:
            # return the average of the diag
            return self.eps_model(frequency).real

        # return the requested component
        comp2indx = {"x": 0, "y": 1, "z": 2}
        return self.eps_comp(
            row=comp2indx[eps_component[0]], col=comp2indx[eps_component[1]], frequency=frequency
        ).real

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For this medium, it take the minimal of ``sqrt(permittivity)`` for main directions.
        """

        perm_diag, _, _ = self.eps_sigma_diag
        return min(np.sqrt(perm_diag))

    @add_ax_if_none
    def plot(self, freqs: float, ax: Ax = None) -> Ax:
        """Plot n, k of a :class:`FullyAnisotropicMedium` as a function of frequency."""

        diagonal_medium = self._to_diagonal
        ax = diagonal_medium.plot(freqs=freqs, ax=ax)
        _, _, directions = self.eps_sigma_diag

        # rename components from xx, yy, zz to 1, 2, 3 to avoid misleading
        # and add their directions
        for label, n_line, k_line, direction in zip(
            ("1", "2", "3"), ax.lines[-6::2], ax.lines[-5::2], directions.T
        ):
            direction_str = f"({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})"
            k_line.set_label(f"k, eps_{label} {direction_str}")
            n_line.set_label(f"n, eps_{label} {direction_str}")

        ax.legend()
        return ax


class CustomAnisotropicMedium(AbstractCustomMedium, AnisotropicMedium):
    """Diagonally anisotropic medium with spatially varying permittivity in each component.

    Note
    ----
        Only diagonal anisotropy is currently supported.

    Example
    -------
    >>> Nx, Ny, Nz = 10, 9, 8
    >>> x = np.linspace(-1, 1, Nx)
    >>> y = np.linspace(-1, 1, Ny)
    >>> z = np.linspace(-1, 1, Nz)
    >>> coords = dict(x=x, y=y, z=z)
    >>> permittivity= SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> conductivity= SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> medium_xx = CustomMedium(permittivity=permittivity, conductivity=conductivity)
    >>> medium_yy = CustomMedium(permittivity=permittivity, conductivity=conductivity)
    >>> d_epsilon = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> f = SpatialDataArray(1+np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> delta = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> medium_zz = CustomLorentz(eps_inf=permittivity, coeffs=[(d_epsilon,f,delta),])
    >>> anisotropic_dielectric = CustomAnisotropicMedium(xx=medium_xx, yy=medium_yy, zz=medium_zz)

    See Also
    --------

    :class:`AnisotropicMedium`
        Diagonally anisotropic medium.

    **Notebooks**
        * `Broadband polarizer assisted by anisotropic metamaterial <../../notebooks/SWGBroadbandPolarizer.html>`_
        * `Thin film lithium niobate adiabatic waveguide coupler <../../notebooks/AdiabaticCouplerLN.html>`_
        * `Defining fully anisotropic materials <../../notebooks/FullyAnisotropic.html>`_
    """

    xx: Union[IsotropicCustomMediumType, CustomMedium] = pd.Field(
        ...,
        title="XX Component",
        description="Medium describing the xx-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    yy: Union[IsotropicCustomMediumType, CustomMedium] = pd.Field(
        ...,
        title="YY Component",
        description="Medium describing the yy-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    zz: Union[IsotropicCustomMediumType, CustomMedium] = pd.Field(
        ...,
        title="ZZ Component",
        description="Medium describing the zz-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    interp_method: Optional[InterpMethod] = pd.Field(
        None,
        title="Interpolation method",
        description="When the value is 'None', each component will follow its own "
        "interpolation method. When the value is other than 'None', the interpolation "
        "method specified by this field will override the one in each component.",
    )

    allow_gain: bool = pd.Field(
        None,
        title="Allow gain medium",
        description="This field is ignored. Please set ``allow_gain`` in each component",
    )

    subpixel: bool = pd.Field(
        None,
        title="Subpixel averaging",
        description="This field is ignored. Please set ``subpixel`` in each component",
    )

    @pd.validator("xx", always=True)
    def _isotropic_xx(cls, val):
        """If it's `CustomMedium`, make sure it's isotropic."""
        if isinstance(val, CustomMedium) and not val.is_isotropic:
            raise SetupError("The xx-component medium type is not isotropic.")
        return val

    @pd.validator("yy", always=True)
    def _isotropic_yy(cls, val):
        """If it's `CustomMedium`, make sure it's isotropic."""
        if isinstance(val, CustomMedium) and not val.is_isotropic:
            raise SetupError("The yy-component medium type is not isotropic.")
        return val

    @pd.validator("zz", always=True)
    def _isotropic_zz(cls, val):
        """If it's `CustomMedium`, make sure it's isotropic."""
        if isinstance(val, CustomMedium) and not val.is_isotropic:
            raise SetupError("The zz-component medium type is not isotropic.")
        return val

    @pd.root_validator(pre=True)
    def _ignored_fields(cls, values):
        """The field is ignored."""
        if values.get("xx") is not None:
            if values.get("allow_gain") is not None:
                log.warning(
                    "The field 'allow_gain' is ignored. Please set 'allow_gain' in each component."
                )
            if values.get("subpixel") is not None:
                log.warning(
                    "The field 'subpixel' is ignored. Please set 'subpixel' in each component."
                )
        return values

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        return any(comp.is_spatially_uniform for comp in self.components.values())

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For this medium, it takes the minimal of ``n_clf`` in all components.
        """
        return min(mat_component.n_cfl for mat_component in self.components.values())

    @cached_property
    def is_isotropic(self):
        """Whether the medium is isotropic."""
        return False

    def _interp_method(self, comp: Axis) -> InterpMethod:
        """Interpolation method applied to comp."""
        # override `interp_method` in components if self.interp_method is not None
        if self.interp_method is not None:
            return self.interp_method
        # use component's interp_method
        comp_map = ["xx", "yy", "zz"]
        return self.components[comp_map[comp]].interp_method

    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """
        return tuple(
            mat_component.eps_dataarray_freq(frequency)[ind]
            for ind, mat_component in enumerate(self.components.values())
        )

    def _eps_bounds(
        self, frequency: float = None, eps_component: Optional[PermittivityComponent] = None
    ) -> Tuple[float, float]:
        """Returns permittivity bounds for setting the color bounds when plotting.

        Parameters
        ----------
        frequency : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        eps_component : Optional[PermittivityComponent] = None
            Component of the permittivity tensor to plot for anisotropic materials,
            e.g. ``"xx"``, ``"yy"``, ``"zz"``, ``"xy"``, ``"yz"``, ...
            Defaults to ``None``, which returns the average of the diagonal values.

        Returns
        -------
        Tuple[float, float]
            The min and max values of the permittivity for the selected component and evaluated at ``frequency``.
        """
        comps = ["xx", "yy", "zz"]
        if eps_component in comps:
            # Return the bounds of a specific component
            eps_dataarray = self.eps_dataarray_freq(frequency)
            eps = self._get_real_vals(eps_dataarray[comps.index(eps_component)])
            return (np.min(eps), np.max(eps))
        elif eps_component is None:
            # Returns the bounds across all components
            return super()._eps_bounds(frequency=frequency)
        else:
            raise ValueError(
                f"Plotting component '{eps_component}' of a diagonally-anisotropic permittivity tensor is not supported."
            )

    def _sel_custom_data_inside(self, bounds: Bound):
        return self


class CustomAnisotropicMediumInternal(CustomAnisotropicMedium):
    """Diagonally anisotropic medium with spatially varying permittivity in each component.

    Notes
    -----

        Only diagonal anisotropy is currently supported.

    Example
    -------
    >>> Nx, Ny, Nz = 10, 9, 8
    >>> X = np.linspace(-1, 1, Nx)
    >>> Y = np.linspace(-1, 1, Ny)
    >>> Z = np.linspace(-1, 1, Nz)
    >>> coords = dict(x=X, y=Y, z=Z)
    >>> permittivity= SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> conductivity= SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> medium_xx = CustomMedium(permittivity=permittivity, conductivity=conductivity)
    >>> medium_yy = CustomMedium(permittivity=permittivity, conductivity=conductivity)
    >>> d_epsilon = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> f = SpatialDataArray(1+np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> delta = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> medium_zz = CustomLorentz(eps_inf=permittivity, coeffs=[(d_epsilon,f,delta),])
    >>> anisotropic_dielectric = CustomAnisotropicMedium(xx=medium_xx, yy=medium_yy, zz=medium_zz)
    """

    xx: Union[IsotropicCustomMediumInternalType, CustomMedium] = pd.Field(
        ...,
        title="XX Component",
        description="Medium describing the xx-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    yy: Union[IsotropicCustomMediumInternalType, CustomMedium] = pd.Field(
        ...,
        title="YY Component",
        description="Medium describing the yy-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    zz: Union[IsotropicCustomMediumInternalType, CustomMedium] = pd.Field(
        ...,
        title="ZZ Component",
        description="Medium describing the zz-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )


""" Medium perturbation classes """


class AbstractPerturbationMedium(ABC, Tidy3dBaseModel):
    """Abstract class for medium perturbation."""

    subpixel: bool = pd.Field(
        True,
        title="Subpixel averaging",
        description="This value will be transferred to the resulting custom medium. That is, "
        "if ``True``, the subpixel averaging will be applied to the custom medium. The type "
        "of subpixel averaging method applied is specified in ``Simulation``'s field ``subpixel``. "
        "If the resulting medium is not a custom medium (no perturbations), this field does not "
        "have an effect.",
    )

    perturbation_spec: Optional[Union[PermittivityPerturbation, IndexPerturbation]] = pd.Field(
        None,
        title="Perturbation Spec",
        description="Specification of medium perturbation as one of predefined types.",
        discriminator=TYPE_TAG_STR,
    )

    @abstractmethod
    def perturbed_copy(
        self,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
        interp_method: InterpMethod = "linear",
    ) -> Union[AbstractMedium, AbstractCustomMedium]:
        """Sample perturbations on provided heat and/or charge data and create a custom medium.
        Any of ``temperature``, ``electron_density``, and ``hole_density`` can be ``None``.
        If all passed arguments are ``None`` then a non-custom medium is returned.
        All provided fields must have identical coords.

        Parameters
        ----------
        temperature : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Temperature field data.
        electron_density : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Electron density field data.
        hole_density : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Hole density field data.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain heat and/or charge values that are not supplied
            at the Yee grids.

        Returns
        -------
        Union[AbstractMedium, AbstractCustomMedium]
            Medium specification after application of heat and/or charge data.
        """

    @classmethod
    def from_unperturbed(
        cls,
        medium: Union[Medium, DispersiveMedium],
        subpixel: bool = True,
        perturbation_spec: Union[PermittivityPerturbation, IndexPerturbation] = None,
        **kwargs,
    ) -> AbstractPerturbationMedium:
        """Construct a medium with pertubation models from an unpertubed one.

        Parameters
        ----------
        medium : Union[
                :class:`.Medium`,
                :class:`.DispersiveMedium`,
            ]
            A medium with no perturbation models.
        subpixel : bool = True
            Subpixel averaging of derivative custom medium.
        perturbation_spec : Union[
                :class:`.PermittivityPerturbation`,
                :class:`.IndexPerturbation`,
            ] = None
            Perturbation model specification.

        Returns
        -------
        :class:`.AbstractPerturbationMedium`
            Resulting medium with perturbation model.
        """

        new_dict = medium.dict(
            exclude={
                "type",
            }
        )

        new_dict["perturbation_spec"] = perturbation_spec
        new_dict["subpixel"] = subpixel

        new_dict.update(kwargs)

        return cls.parse_obj(new_dict)


class PerturbationMedium(Medium, AbstractPerturbationMedium):
    """Dispersionless medium with perturbations. Perturbation model can be defined either directly
    through providing ``permittivity_perturbation`` and ``conductivity_perturbation`` or via
    providing a specific perturbation model (:class:`PermittivityPerturbation`,
    :class:`IndexPerturbation`) as ``perturbaiton_spec``.

    Example
    -------
    >>> from tidy3d import ParameterPerturbation, LinearHeatPerturbation
    >>> dielectric = PerturbationMedium(
    ...     permittivity=4.0,
    ...     permittivity_perturbation=ParameterPerturbation(
    ...         heat=LinearHeatPerturbation(temperature_ref=300, coeff=0.0001),
    ...     ),
    ...     name='my_medium',
    ... )
    """

    permittivity_perturbation: Optional[ParameterPerturbation] = pd.Field(
        None,
        title="Permittivity Perturbation",
        description="List of heat and/or charge perturbations to permittivity.",
        units=PERMITTIVITY,
    )

    conductivity_perturbation: Optional[ParameterPerturbation] = pd.Field(
        None,
        title="Permittivity Perturbation",
        description="List of heat and/or charge perturbations to permittivity.",
        units=CONDUCTIVITY,
    )

    _permittivity_perturbation_validator = validate_parameter_perturbation(
        "permittivity_perturbation",
        "permittivity",
        allowed_real_range=[(1.0, None)],
        allowed_imag_range=[None],
        allowed_complex=False,
    )

    _conductivity_perturbation_validator = validate_parameter_perturbation(
        "conductivity_perturbation",
        "conductivity",
        allowed_real_range=[(0.0, None)],
        allowed_imag_range=[None],
        allowed_complex=False,
    )

    @pd.root_validator(pre=True)
    def _check_overdefining(cls, values):
        """Check that perturbation model is provided either directly or through
        ``perturbation_spec``, but not both.
        """

        perm_p = values.get("permittivity_perturbation") is not None
        cond_p = values.get("conductivity_perturbation") is not None
        p_spec = values.get("perturbation_spec") is not None

        if p_spec and (perm_p or cond_p):
            raise SetupError(
                "Must provide perturbation model either as 'perturbation_spec' or as "
                "'permittivity_perturbation' and 'conductivity_perturbation', "
                "but not in both ways simultaneously."
            )

        return values

    @pd.root_validator(skip_on_failure=True)
    def _check_perturbation_spec_ranges(cls, values):
        """Check perturbation ranges if defined as ``perturbation_spec``."""
        p_spec = values["perturbation_spec"]
        if p_spec is None:
            return values

        perm = values["permittivity"]
        cond = values["conductivity"]

        if isinstance(p_spec, IndexPerturbation):
            eps_complex = Medium._eps_model(
                permittivity=perm, conductivity=cond, frequency=p_spec.freq
            )
            n, k = Medium.eps_complex_to_nk(eps_c=eps_complex)
            delta_eps_range, delta_sigma_range = p_spec._delta_eps_delta_sigma_ranges(n, k)
        elif isinstance(p_spec, PermittivityPerturbation):
            delta_eps_range, delta_sigma_range = p_spec._delta_eps_delta_sigma_ranges()
        else:
            raise SetupError("Unknown type of 'perturbation_spec'.")

        _warn_potential_error(
            field_name="permittivity",
            base_value=perm,
            val_change_range=delta_eps_range,
            allowed_real_range=(1.0, None),
            allowed_imag_range=None,
        )

        _warn_potential_error(
            field_name="conductivity",
            base_value=cond,
            val_change_range=delta_sigma_range,
            allowed_real_range=(0.0, None),
            allowed_imag_range=None,
        )
        return values

    def perturbed_copy(
        self,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
        interp_method: InterpMethod = "linear",
    ) -> Union[Medium, CustomMedium]:
        """Sample perturbations on provided heat and/or charge data and return 'CustomMedium'.
        Any of temperature, electron_density, and hole_density can be 'None'. If all passed
        arguments are 'None' then a 'Medium' object is returned. All provided fields must have
        identical coords.

        Parameters
        ----------
        temperature : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Temperature field data.
        electron_density : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Electron density field data.
        hole_density : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Hole density field data.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain heat and/or charge values that are not supplied
            at the Yee grids.

        Returns
        -------
        Union[Medium, CustomMedium]
            Medium specification after application of heat and/or charge data.
        """

        new_dict = self.dict(
            exclude={
                "permittivity_perturbation",
                "conductivity_perturbation",
                "perturbation_spec",
                "type",
            }
        )

        if all(x is None for x in [temperature, electron_density, hole_density]):
            new_dict.pop("subpixel")
            return Medium.parse_obj(new_dict)

        permittivity_field = self.permittivity + ParameterPerturbation._zeros_like(
            temperature, electron_density, hole_density
        )

        delta_eps = None
        delta_sigma = None

        if self.perturbation_spec is not None:
            pspec = self.perturbation_spec
            if isinstance(pspec, PermittivityPerturbation):
                delta_eps, delta_sigma = pspec._sample_delta_eps_delta_sigma(
                    temperature, electron_density, hole_density
                )
            elif isinstance(pspec, IndexPerturbation):
                n, k = self.nk_model(frequency=pspec.freq)
                delta_eps, delta_sigma = pspec._sample_delta_eps_delta_sigma(
                    n, k, temperature, electron_density, hole_density
                )
        else:
            if self.permittivity_perturbation is not None:
                delta_eps = self.permittivity_perturbation.apply_data(
                    temperature, electron_density, hole_density
                )

            if self.conductivity_perturbation is not None:
                delta_sigma = self.conductivity_perturbation.apply_data(
                    temperature, electron_density, hole_density
                )

        if delta_eps is not None:
            permittivity_field = permittivity_field + delta_eps

        conductivity_field = None
        if delta_sigma is not None:
            conductivity_field = self.conductivity + delta_sigma

        new_dict["permittivity"] = permittivity_field
        new_dict["conductivity"] = conductivity_field
        new_dict["interp_method"] = interp_method

        return CustomMedium.parse_obj(new_dict)


class PerturbationPoleResidue(PoleResidue, AbstractPerturbationMedium):
    """A dispersive medium described by the pole-residue pair model with perturbations.
    Perturbation model can be defined either directly
    through providing ``eps_inf_perturbation`` and ``poles_perturbation`` or via
    providing a specific perturbation model (:class:`PermittivityPerturbation`,
    :class:`IndexPerturbation`) as ``perturbaiton_spec``.

    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(\\omega) = \\epsilon_\\infty - \\sum_i
            \\left[\\frac{c_i}{j \\omega + a_i} +
            \\frac{c_i^*}{j \\omega + a_i^*}\\right]

    Example
    -------
    >>> from tidy3d import ParameterPerturbation, LinearHeatPerturbation
    >>> c0_perturbation = ParameterPerturbation(
    ...     heat=LinearHeatPerturbation(temperature_ref=300, coeff=0.0001),
    ... )
    >>> pole_res = PerturbationPoleResidue(
    ...     eps_inf=2.0,
    ...     poles=[((-1+2j), (3+4j)), ((-5+6j), (7+8j))],
    ...     poles_perturbation=[(None, c0_perturbation), (None, None)],
    ... )
    """

    eps_inf_perturbation: Optional[ParameterPerturbation] = pd.Field(
        None,
        title="Perturbation of Epsilon at Infinity",
        description="Perturbations to relative permittivity at infinite frequency "
        "(:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    poles_perturbation: Optional[
        Tuple[Tuple[Optional[ParameterPerturbation], Optional[ParameterPerturbation]], ...]
    ] = pd.Field(
        None,
        title="Perturbations of Poles",
        description="Perturbations to poles of the model.",
        units=(RADPERSEC, RADPERSEC),
    )

    _eps_inf_perturbation_validator = validate_parameter_perturbation(
        "eps_inf_perturbation",
        "eps_inf",
        allowed_real_range=[(0.0, None)],
        allowed_imag_range=[None],
        allowed_complex=False,
    )

    _poles_perturbation_validator = validate_parameter_perturbation(
        "poles_perturbation",
        "poles",
        allowed_real_range=[(None, 0.0), (None, None)],
        allowed_imag_range=[None, None],
    )

    @pd.root_validator(pre=True)
    def _check_overdefining(cls, values):
        """Check that perturbation model is provided either directly or through
        ``perturbation_spec``, but not both.
        """

        eps_i_p = values.get("eps_inf_perturbation") is not None
        poles_p = values.get("poles_perturbation") is not None
        p_spec = values.get("perturbation_spec") is not None

        if p_spec and (eps_i_p or poles_p):
            raise SetupError(
                "Must provide perturbation model either as 'perturbation_spec' or as "
                "'eps_inf_perturbation' and 'poles_perturbation', "
                "but not in both ways simultaneously."
            )

        return values

    @pd.root_validator(skip_on_failure=True)
    def _check_perturbation_spec_ranges(cls, values):
        """Check perturbation ranges if defined as ``perturbation_spec``."""
        p_spec = values["perturbation_spec"]
        if p_spec is None:
            return values

        eps_inf = values["eps_inf"]
        poles = values["poles"]

        if isinstance(p_spec, IndexPerturbation):
            eps_complex = PoleResidue._eps_model(
                eps_inf=eps_inf, poles=poles, frequency=p_spec.freq
            )
            n, k = Medium.eps_complex_to_nk(eps_c=eps_complex)
            delta_eps_range, _ = p_spec._delta_eps_delta_sigma_ranges(n, k)
        elif isinstance(p_spec, PermittivityPerturbation):
            delta_eps_range, _ = p_spec._delta_eps_delta_sigma_ranges()
        else:
            raise SetupError("Unknown type of 'perturbation_spec'.")

        _warn_potential_error(
            field_name="eps_inf",
            base_value=eps_inf,
            val_change_range=delta_eps_range,
            allowed_real_range=(0.0, None),
            allowed_imag_range=None,
        )

        return values

    def perturbed_copy(
        self,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
        interp_method: InterpMethod = "linear",
    ) -> Union[PoleResidue, CustomPoleResidue]:
        """Sample perturbations on provided heat and/or charge data and return 'CustomPoleResidue'.
        Any of temperature, electron_density, and hole_density can be 'None'. If all passed
        arguments are 'None' then a 'PoleResidue' object is returned. All provided fields must have
        identical coords.

        Parameters
        ----------
        temperature : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Temperature field data.
        electron_density : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Electron density field data.
        hole_density : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Hole density field data.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain heat and/or charge values that are not supplied
            at the Yee grids.

        Returns
        -------
        Union[PoleResidue, CustomPoleResidue]
            Medium specification after application of heat and/or charge data.
        """

        new_dict = self.dict(
            exclude={"eps_inf_perturbation", "poles_perturbation", "perturbation_spec", "type"}
        )

        if all(x is None for x in [temperature, electron_density, hole_density]):
            new_dict.pop("subpixel")
            return PoleResidue.parse_obj(new_dict)

        zeros = ParameterPerturbation._zeros_like(temperature, electron_density, hole_density)

        eps_inf_field = self.eps_inf + zeros
        poles_field = [[a + zeros, c + zeros] for a, c in self.poles]

        if self.perturbation_spec is not None:
            pspec = self.perturbation_spec
            if isinstance(pspec, PermittivityPerturbation):
                delta_eps, delta_sigma = pspec._sample_delta_eps_delta_sigma(
                    temperature, electron_density, hole_density
                )
            elif isinstance(pspec, IndexPerturbation):
                n, k = self.nk_model(frequency=pspec.freq)
                delta_eps, delta_sigma = pspec._sample_delta_eps_delta_sigma(
                    n, k, temperature, electron_density, hole_density
                )

            if delta_eps is not None:
                eps_inf_field = eps_inf_field + delta_eps

            if delta_sigma is not None:
                poles_field = poles_field + [[zeros, 0.5 * delta_sigma / EPSILON_0]]
        else:
            # sample eps_inf
            if self.eps_inf_perturbation is not None:
                eps_inf_field = eps_inf_field + self.eps_inf_perturbation.apply_data(
                    temperature, electron_density, hole_density
                )

            # sample poles
            if self.poles_perturbation is not None:
                for ind, ((a_perturb, c_perturb), (a_field, c_field)) in enumerate(
                    zip(self.poles_perturbation, poles_field)
                ):
                    if a_perturb is not None:
                        a_field = a_field + a_perturb.apply_data(
                            temperature, electron_density, hole_density
                        )
                    if c_perturb is not None:
                        c_field = c_field + c_perturb.apply_data(
                            temperature, electron_density, hole_density
                        )
                    poles_field[ind] = [a_field, c_field]

        new_dict["eps_inf"] = eps_inf_field
        new_dict["poles"] = poles_field
        new_dict["interp_method"] = interp_method

        return CustomPoleResidue.parse_obj(new_dict)


# types of mediums that can be used in Simulation and Structures


MediumType3D = Union[
    Medium,
    AnisotropicMedium,
    PECMedium,
    PoleResidue,
    Sellmeier,
    Lorentz,
    Debye,
    Drude,
    FullyAnisotropicMedium,
    CustomMedium,
    CustomPoleResidue,
    CustomSellmeier,
    CustomLorentz,
    CustomDebye,
    CustomDrude,
    CustomAnisotropicMedium,
    PerturbationMedium,
    PerturbationPoleResidue,
    LossyMetalMedium,
]


class Medium2D(AbstractMedium):
    """2D diagonally anisotropic medium.

    Notes
    -----

        Only diagonal anisotropy is currently supported.

    Example
    -------
    >>> drude_medium = Drude(eps_inf=2.0, coeffs=[(1,2), (3,4)])
    >>> medium2d = Medium2D(ss=drude_medium, tt=drude_medium)

    """

    ss: IsotropicUniformMediumType = pd.Field(
        ...,
        title="SS Component",
        description="Medium describing the ss-component of the diagonal permittivity tensor. "
        "The ss-component refers to the in-plane dimension of the medium that is the first "
        "component in order of 'x', 'y', 'z'. "
        "If the 2D material is normal to the y-axis, for example, then this determines the "
        "xx-component of the corresponding 3D medium.",
        discriminator=TYPE_TAG_STR,
    )

    tt: IsotropicUniformMediumType = pd.Field(
        ...,
        title="TT Component",
        description="Medium describing the tt-component of the diagonal permittivity tensor. "
        "The tt-component refers to the in-plane dimension of the medium that is the second "
        "component in order of 'x', 'y', 'z'. "
        "If the 2D material is normal to the y-axis, for example, then this determines the "
        "zz-component of the corresponding 3D medium.",
        discriminator=TYPE_TAG_STR,
    )

    @pd.validator("modulation_spec", always=True)
    def _validate_modulation_spec(cls, val):
        """Check compatibility with modulation_spec."""
        if val is not None:
            raise ValidationError(
                f"A 'modulation_spec' of class {type(val)} is not "
                f"currently supported for medium class {cls}."
            )
        return val

    @skip_if_fields_missing(["ss"])
    @pd.validator("tt", always=True)
    def _validate_inplane_pec(cls, val, values):
        """ss/tt components must be both PEC or non-PEC."""
        if isinstance(val, PECMedium) != isinstance(values["ss"], PECMedium):
            raise ValidationError(
                "Materials describing ss- and tt-components must be "
                "either both 'PECMedium', or non-'PECMedium'."
            )
        return val

    @classmethod
    def _weighted_avg(
        cls, meds: List[IsotropicUniformMediumType], weights: List[float]
    ) -> Union[PoleResidue, PECMedium]:
        """Average ``meds`` with weights ``weights``."""
        eps_inf = 1
        poles = []
        for med, weight in zip(meds, weights):
            if isinstance(med, DispersiveMedium):
                pole_res = med.pole_residue
                eps_inf += weight * (med.pole_residue.eps_inf - 1)
            elif isinstance(med, Medium):
                pole_res = PoleResidue.from_medium(med)
                eps_inf += weight * (med.permittivity - 1)
            elif isinstance(med, PECMedium):
                # special treatment for PEC
                return med
            else:
                raise ValidationError("Invalid medium type for the components of 'Medium2D'.")
            poles += [(a, weight * c) for (a, c) in pole_res.poles if c != 0.0]
        return PoleResidue(eps_inf=np.real(eps_inf), poles=poles)

    def volumetric_equivalent(
        self,
        axis: Axis,
        adjacent_media: Tuple[MediumType3D, MediumType3D],
        adjacent_dls: Tuple[float, float],
    ) -> AnisotropicMedium:
        """Produces a 3D volumetric equivalent medium. The new medium has thickness equal to
        the average of the ``dls`` in the ``axis`` direction.
        The ss and tt components of the 2D material are mapped in order onto the xx, yy, and
        zz components of the 3D material, excluding the ``axis`` component. The conductivity
        and residues (in the case of a dispersive 2D material) are rescaled by ``1/dl``.
        The neighboring media ``neighbors`` enter in as a background for the resulting
        volumetric equivalent.


        Parameters
        ----------
        axis : Axis
            Index (0, 1, or 2 for x, y, or z respectively) of the normal direction to the
            2D material.
        adjacent_media : Tuple[MediumType3D, MediumType3D]
            The neighboring media on either side of the 2D material.
            The first element is directly on the - side of the 2D material in the supplied axis,
            and the second element is directly on the + side.
        adjacent_dls : Tuple[float, float]
            Each dl represents twice the thickness of the desired volumetric model on the
            respective side of the 2D material.

        Returns
        -------
        :class:`.AnisotropicMedium`
            The 3D material corresponding to this 2D material.
        """

        def get_component(med: MediumType3D, comp: Axis) -> IsotropicUniformMediumType:
            """Extract the ``comp`` component of ``med``."""
            if isinstance(med, AnisotropicMedium):
                dim = "xyz"[comp]
                element_name = dim + dim
                return med.elements[element_name]
            return med

        def get_background(comp: Axis) -> PoleResidue:
            """Get the background medium appropriate for the ``comp`` component."""
            meds = [get_component(med=med, comp=comp) for med in adjacent_media]
            # the Yee site for the E field in the normal direction is fully contained
            # in the medium on the + side
            if comp == axis:
                return meds[1]
            weights = np.array(adjacent_dls) / np.sum(adjacent_dls)
            return self._weighted_avg(meds, weights)

        dl = (adjacent_dls[0] + adjacent_dls[1]) / 2
        media_bg = [get_background(comp=i) for i in range(3)]

        # perform weighted average of planar media transverse dimensions with the
        # respective background media
        media_fg_plane = list(self.elements.values())
        _, media_bg_plane = Geometry.pop_axis(media_bg, axis=axis)
        media_fg_weighted = [
            self._weighted_avg([media_bg, media_fg], [1, 1 / dl])
            for media_bg, media_fg in zip(media_bg_plane, media_fg_plane)
        ]

        # combine the two weighted, planar media with the background medium and put in the xyz basis
        media_3d = Geometry.unpop_axis(
            ax_coord=media_bg[axis], plane_coords=media_fg_weighted, axis=axis
        )
        media_3d_kwargs = {dim + dim: medium for dim, medium in zip("xyz", media_3d)}
        return AnisotropicMediumFromMedium2D(
            **media_3d_kwargs, frequency_range=self.frequency_range
        )

    def to_anisotropic_medium(self, axis: Axis, thickness: float) -> AnisotropicMedium:
        """Generate a 3D :class:`.AnisotropicMedium` equivalent of a given thickness.

        Parameters
        ----------
        axis: Axis
            The normal axis to the 2D medium.
        thickness: float
            The thickness of the desired 3D medium.

        Returns
        -------
        :class:`.AnisotropicMedium`
            The 3D equivalent of this 2D medium.
        """
        media = list(self.elements.values())
        media_weighted = [self._weighted_avg([medium], [1 / thickness]) for medium in media]
        media_3d = Geometry.unpop_axis(ax_coord=Medium(), plane_coords=media_weighted, axis=axis)
        media_3d_kwargs = {dim + dim: medium for dim, medium in zip("xyz", media_3d)}
        return AnisotropicMedium(**media_3d_kwargs, frequency_range=self.frequency_range)

    def to_pole_residue(self, thickness: float) -> PoleResidue:
        """Generate a :class:`.PoleResidue` equivalent of a given thickness.
        The 2D medium to be isotropic in-plane (otherwise the components are averaged).

        Parameters
        ----------
        thickness: float
            The thickness of the desired 3D medium.

        Returns
        -------
        :class:`.PoleResidue`
            The 3D equivalent pole residue model of this 2D medium.
        """
        return self._weighted_avg(
            [self.ss, self.tt], [1 / (2 * thickness), 1 / (2 * thickness)]
        ).updated_copy(frequency_range=self.frequency_range)

    def to_medium(self, thickness: float) -> Medium:
        """Generate a :class:`.Medium` equivalent of a given thickness.
        The 2D medium must be isotropic in-plane (otherwise the components are averaged)
        and non-dispersive besides a constant conductivity.

        Parameters
        ----------
        thickness: float
            The thickness of the desired 3D medium.

        Returns
        -------
        :class:`.Medium`
            The 3D equivalent of this 2D medium.
        """
        if self.is_pec:
            return PEC
        return self.to_pole_residue(thickness=thickness).to_medium()

    @classmethod
    def from_medium(cls, medium: Medium, thickness: float) -> Medium2D:
        """Generate a :class:`.Medium2D` equivalent of a :class:`.Medium`
        with a given thickness.

        Parameters
        ----------
        medium: :class:`.Medium`
            The 3D medium to convert.
        thickness : float
            The thickness of the 3D material.

        Returns
        -------
        :class:`.Medium2D`
            The 2D equivalent of the given 3D medium.
        """
        med = cls._weighted_avg([medium], [thickness])
        return Medium2D(ss=med, tt=med, frequency_range=medium.frequency_range)

    @classmethod
    def from_dispersive_medium(cls, medium: DispersiveMedium, thickness: float) -> Medium2D:
        """Generate a :class:`.Medium2D` equivalent of a :class:`.DispersiveMedium`
        with a given thickness.

        Parameters
        ----------
        medium: :class:`.DispersiveMedium`
            The 3D dispersive medium to convert.
        thickness : float
            The thickness of the 3D material.

        Returns
        -------
        :class:`.Medium2D`
            The 2D equivalent of the given 3D medium.
        """
        med = cls._weighted_avg([medium], [thickness])
        return Medium2D(ss=med, tt=med, frequency_range=medium.frequency_range, name=medium.name)

    @classmethod
    def from_anisotropic_medium(
        cls, medium: AnisotropicMedium, axis: Axis, thickness: float
    ) -> Medium2D:
        """Generate a :class:`.Medium2D` equivalent of a :class:`.AnisotropicMedium`
        with given normal axis and thickness. The ``ss`` and ``tt`` components of the resulting
        2D medium correspond to the first of the ``xx``, ``yy``, and ``zz`` components of
        the 3D medium, with the ``axis`` component removed.

        Parameters
        ----------
        medium: :class:`.AnisotropicMedium`
            The 3D anisotropic medium to convert.
        axis: :class:`.Axis`
            The normal axis to the 2D material.
        thickness : float
            The thickness of the 3D material.

        Returns
        -------
        :class:`.Medium2D`
            The 2D equivalent of the given 3D medium.
        """
        media = list(medium.elements.values())
        _, media_plane = Geometry.pop_axis(media, axis=axis)
        media_plane_scaled = []
        for _, med in enumerate(media_plane):
            media_plane_scaled.append(cls._weighted_avg([med], [thickness]))
        media_kwargs = {dim + dim: medium for dim, medium in zip("st", media_plane_scaled)}
        return Medium2D(**media_kwargs, frequency_range=medium.frequency_range)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""
        return np.mean(self.eps_diagonal(frequency=frequency), axis=0)

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor as a function of frequency."""
        log.warning(
            "The permittivity of a 'Medium2D' is unphysical. "
            "Use 'Medium2D.to_anisotropic_medium' or 'Medium2D.to_pole_residue' first "
            "to obtain the physical refractive index."
        )

        eps_ss = self.ss.eps_model(frequency)
        eps_tt = self.tt.eps_model(frequency)
        return (eps_ss, eps_tt)

    def eps_diagonal_numerical(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor for numerical considerations
        such as meshing and runtime estimation.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[complex, complex, complex]
            The diagonal elements of relative permittivity tensor relevant for numerical
            considerations evaluated at ``frequency``.
        """
        return (1.0 + 0j,) * 3

    @add_ax_if_none
    def plot(self, freqs: float, ax: Ax = None) -> Ax:
        """Plot n, k of a :class:`.Medium` as a function of frequency."""
        log.warning(
            "The refractive index of a 'Medium2D' is unphysical. "
            "Use 'Medium2D.plot_sigma' instead to plot surface conductivity, or call "
            "'Medium2D.to_anisotropic_medium' or 'Medium2D.to_pole_residue' first "
            "to obtain the physical refractive index."
        )

        freqs = np.array(freqs)
        freqs_thz = freqs / 1e12

        for label, medium_component in self.elements.items():
            eps_complex = medium_component.eps_model(freqs)
            n, k = AbstractMedium.eps_complex_to_nk(eps_complex)
            ax.plot(freqs_thz, n, label=f"n, eps_{label}")
            ax.plot(freqs_thz, k, label=f"k, eps_{label}")

        ax.set_xlabel("frequency (THz)")
        ax.set_title("medium dispersion")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    @add_ax_if_none
    def plot_sigma(self, freqs: float, ax: Ax = None) -> Ax:
        """Plot the surface conductivity of the 2D material."""
        freqs = np.array(freqs)
        freqs_thz = freqs / 1e12

        for label, medium_component in self.elements.items():
            sigma = medium_component.sigma_model(freqs)
            ax.plot(freqs_thz, np.real(sigma) * 1e6, label=f"Re($\\sigma$) ($\\mu$S), eps_{label}")
            ax.plot(freqs_thz, np.imag(sigma) * 1e6, label=f"Im($\\sigma$) ($\\mu$S), eps_{label}")

        ax.set_xlabel("frequency (THz)")
        ax.set_title("surface conductivity")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    @ensure_freq_in_range
    def sigma_model(self, freq: float) -> complex:
        """Complex-valued conductivity as a function of frequency.

        Parameters
        ----------
        freq: float
            Frequency to evaluate conductivity at (Hz).

        Returns
        -------
        complex
            Complex conductivity at this frequency.
        """
        return np.mean([self.ss.sigma_model(freq), self.tt.sigma_model(freq)], axis=0)

    @property
    def elements(self) -> Dict[str, IsotropicUniformMediumType]:
        """The diagonal elements of the 2D medium as a dictionary."""
        return dict(ss=self.ss, tt=self.tt)

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.
        """
        return 1.0

    @cached_property
    def is_pec(self):
        """Whether the medium is a PEC."""
        return any(isinstance(comp, PECMedium) for comp in self.elements.values())

    def is_comp_pec_2d(self, comp: Axis, axis: Axis):
        """Whether the medium is a PEC."""
        elements_3d = Geometry.unpop_axis(
            ax_coord=Medium(), plane_coords=self.elements.values(), axis=axis
        )
        return isinstance(elements_3d[comp], PECMedium)


PEC2D = Medium2D(ss=PEC, tt=PEC)

# types of mediums that can be used in Simulation and Structures

MediumType = Union[MediumType3D, Medium2D, AnisotropicMediumFromMedium2D]


# Utility function
def medium_from_nk(n: float, k: float, freq: float, **kwargs) -> Union[Medium, Lorentz]:
    """Convert ``n`` and ``k`` values at frequency ``freq`` to :class:`Medium` if ``Re[epsilon]>=1``,
    or :class:`Lorentz` if if ``Re[epsilon]<1``.

    Parameters
    ----------
    n : float
        Real part of refractive index.
    k : float = 0
        Imaginary part of refrative index.
    freq : float
        Frequency to evaluate permittivity at (Hz).
    kwargs: dict
        Keyword arguments passed to the medium construction.

    Returns
    -------
    Union[:class:`Medium`, :class:`Lorentz`]
        Dispersionless medium or Lorentz medium having refractive index n+ik at frequency ``freq``.
    """
    eps_complex = AbstractMedium.nk_to_eps_complex(n, k)
    if eps_complex.real >= 1:
        return Medium.from_nk(n, k, freq, **kwargs)
    return Lorentz.from_nk(n, k, freq, **kwargs)
