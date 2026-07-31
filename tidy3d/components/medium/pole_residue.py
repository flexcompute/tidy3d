"""Pole-residue and shared dispersive medium models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import autograd.numpy as np
import numpy as npo
from autograd.differential_operators import tensor_jacobian_product
from pydantic import Field, field_validator, model_validator

from tidy3d.components.autograd.path_utils import (
    AutogradRoute,
    format_traced_paths,
    traced_paths,
)
from tidy3d.components.autograd.types import PathType, TracedPolesAndResidues, TracedPositiveFloat
from tidy3d.components.autograd.utils import pack_complex_vec
from tidy3d.components.base import cached_property
from tidy3d.components.data.utils import (
    _get_numpy_array,
)
from tidy3d.components.dispersion_fitter import (
    LOSS_CHECK_MAX,
    LOSS_CHECK_MIN,
    LOSS_CHECK_NUM,
    imag_resp_extrema_locs,
)
from tidy3d.components.validators import (
    call_wrapped_validator,
)
from tidy3d.constants import (
    EPSILON_0,
    LARGEST_FP_NUMBER,
    PERMITTIVITY,
    RADPERSEC,
    fp_eps,
)
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from autograd.numpy.numpy_boxes import ArrayBox
    from numpy.typing import NDArray
    from pydantic import PositiveFloat

    from tidy3d.compat import Self
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.autograd.types import AutogradFieldMap, TracedFloat
    from tidy3d.components.types import (
        ArrayFloat1D,
        Complex,
    )
    from tidy3d.components.types.base import PolesAndResidues

    from .base import ArrayComplex, ArrayFloat, ComplexArrayOrScalar


def _medium_numerics() -> Any:
    """Import shared medium kernels after Pydantic model rebuilds finish."""
    from flex_em.numerical.raw import medium as medium_numerics

    return medium_numerics


from .base import AbstractMedium, ensure_freq_in_range  # noqa: E402
from .isotropic import Medium  # noqa: E402

T = TypeVar("T")


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

    _traced_indexed_root: ClassVar[str | None] = None

    @staticmethod
    def _permittivity_modulation_validation() -> Callable[[T], T]:
        """Assert modulated permittivity cannot be <= 0 at any time."""

        @model_validator(mode="after")
        def _validate_permittivity_modulation(self: T) -> T:
            """Assert modulated permittivity cannot be <= 0."""
            val = self.eps_inf
            modulation = self.modulation_spec
            if modulation is None or modulation.permittivity is None:
                return self

            min_eps_inf = np.min(_get_numpy_array(val))
            if min_eps_inf - modulation.permittivity.max_modulation <= 0:
                raise ValidationError(
                    "The minimum permittivity value with modulation applied was found to be negative."
                )
            return self

        return _validate_permittivity_modulation

    @staticmethod
    def _conductivity_modulation_validation() -> Callable[[T], T]:
        """Assert passive medium at any time if not ``allow_gain``."""

        @model_validator(mode="after")
        def _validate_conductivity_modulation(self: T) -> T:
            """With conductivity modulation, the medium can exhibit gain during the cycle.
            So `allow_gain` must be True when the conductivity is modulated.
            """
            val = self.modulation_spec
            if val is None or val.conductivity is None:
                return self

            if not self.allow_gain:
                raise ValidationError(
                    "For passive medium, 'conductivity' must be non-negative at any time. "
                    "With conductivity modulation, this medium can sometimes be active. "
                    "Please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, and are likely to diverge."
                )
            return self

        return _validate_conductivity_modulation

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        super()._run_after_validators()
        if "eps_inf" in type(self).model_fields:
            call_wrapped_validator(DispersiveMedium._permittivity_modulation_validation, self)
        call_wrapped_validator(DispersiveMedium._conductivity_modulation_validation, self)
        return self

    @abstractmethod
    def _pole_residue_dict(self) -> dict:
        """Dict representation of Medium as a pole-residue model."""

    @cached_property
    def pole_residue(self) -> PoleResidue:
        """Representation of Medium as a pole-residue model."""
        return PoleResidue(**self._pole_residue_dict(), allow_gain=self.allow_gain)

    @cached_property
    def n_cfl(self) -> float:
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
    def tuple_to_complex(value: tuple[float, float]) -> complex:
        """Convert a tuple of real and imaginary parts to complex number."""

        val_r, val_i = value
        return val_r + 1j * val_i

    @staticmethod
    def complex_to_tuple(value: complex) -> tuple[float, float]:
        """Convert a complex number to a tuple of real and imaginary parts."""

        return (value.real, value.imag)

    # --- shared autograd helpers for dispersive models ---
    def _tjp_inputs(self, derivative_info: DerivativeInfo) -> tuple[NDArray, ArrayFloat | ArrayBox]:
        """Prepare shared inputs for TJP: frequencies and packed adjoint vector."""
        dJ = self._derivative_eps_complex_volume(
            E_der_map=derivative_info.E_der_map, bounds=derivative_info.bounds
        )
        freqs = np.asarray(derivative_info.frequencies, float)
        dJv = np.asarray(getattr(dJ, "values", dJ))
        return freqs, pack_complex_vec(dJv)

    @staticmethod
    def _tjp_grad(
        theta0: ArrayFloat,
        eps_vec_fn: Callable[[ArrayFloat], ArrayComplex | ArrayBox],
        vec: ArrayComplex | ArrayBox,
    ) -> ArrayFloat:
        """Run a tensor-Jacobian-product to get J^T @ vec."""
        return tensor_jacobian_product(eps_vec_fn)(theta0, vec)

    @staticmethod
    def _map_grad_real(
        g: TracedFloat,
        paths: set[tuple],
        mapping: Sequence[tuple[tuple, int]],
    ) -> AutogradFieldMap:
        """Map flat gradient to model paths, taking the real part."""
        out = {}
        for k, idx in mapping:
            if k in paths:
                out[k] = np.real(g[idx])
        return out

    @classmethod
    def _traced_autograd_supported_parameters(cls) -> tuple[str, ...]:
        """Return user-facing supported parameter names for setup validation."""
        parameters = format_traced_paths(cls._traced_supported_paths)
        root = cls._traced_indexed_root
        if root is None:
            return parameters
        return (*parameters, f"{root}[index][component]")

    def _resolve_autograd_route(self, field_path: tuple[Any, ...]) -> AutogradRoute:
        """Resolve and validate one traced dispersive medium path for adjoint routing."""
        if field_path in self._traced_supported_paths:
            return AutogradRoute(local_path=field_path)

        root = self._traced_indexed_root
        # Stripped paths are produced from model containers; for indexed material coefficients
        # we only need to check the derivative-map path shape here, not duplicate model bounds.
        if root is not None and len(field_path) == 3 and field_path[0] == root:
            return AutogradRoute(local_path=field_path)

        self._raise_unsupported_traced_path(field_path)


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

    _traced_indexed_root: ClassVar[str] = "poles"
    _traced_supported_paths: ClassVar[tuple[PathType, ...]] = traced_paths("eps_inf")

    eps_inf: TracedPositiveFloat = Field(
        1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        json_schema_extra={"units": PERMITTIVITY},
    )

    poles: TracedPolesAndResidues = Field(
        (),
        title="Poles",
        description="Tuple of complex-valued (:math:`a_i, c_i`) poles for the model.",
        json_schema_extra={"units": (RADPERSEC, RADPERSEC)},
    )

    @field_validator("poles")
    @classmethod
    def _causality_validation(cls, val: TracedPolesAndResidues) -> TracedPolesAndResidues:
        """Assert causal medium."""
        for a, _ in val:
            if np.any(np.real(_get_numpy_array(a)) > 0):
                raise SetupError("For stable medium, 'Re(a_i)' must be non-positive.")
        return val

    @field_validator("poles")
    @classmethod
    def _poles_largest_value(cls, val: TracedPolesAndResidues) -> TracedPolesAndResidues:
        """Assert pole parameters are not too large."""
        for a, c in val:
            if np.any(abs(_get_numpy_array(a)) > LARGEST_FP_NUMBER):
                raise ValidationError(
                    "The value of some 'a_i' is too large. They are unlikely to contribute to material dispersion."
                )
            if np.any(abs(_get_numpy_array(c)) > LARGEST_FP_NUMBER):
                raise ValidationError("The value of some 'c_i' is too large.")
        return val

    @staticmethod
    def _eps_model(eps_inf: PositiveFloat, poles: PolesAndResidues, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        return _medium_numerics().pole_residue_eps_model(eps_inf, poles, frequency)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""
        return self._eps_model(eps_inf=self.eps_inf, poles=self.poles, frequency=frequency)

    def _pole_residue_dict(self) -> dict:
        """Dict representation of Medium as a pole-residue model."""

        return {
            "eps_inf": self.eps_inf,
            "poles": self.poles,
            "frequency_range": self.frequency_range,
            "name": self.name,
        }

    def __str__(self) -> str:
        """string representation"""
        return (
            f"td.PoleResidue("
            f"\n\teps_inf={self.eps_inf}, "
            f"\n\tpoles={self.poles}, "
            f"\n\tfrequency_range={self.frequency_range})"
        )

    @classmethod
    def from_medium(cls, medium: Medium) -> Self:
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
        poles: tuple[tuple[float, float, float, float], ...],
        eps_inf: PositiveFloat,
        frequency: float,
    ) -> complex:
        """Complex permittivity as a function of frequency for a given set of LO-TO coefficients.
        See ``from_lo_to`` in :class:`.PoleResidue` for the detailed form of the model
        and a reference paper.

        Parameters
        ----------
        poles : tuple[tuple[float, float, float, float], ...]
            The LO-TO poles, given as list of tuples of the form
            (omega_LO, gamma_LO, omega_TO, gamma_TO).
        eps_inf: PositiveFloat
            The relative permittivity at infinite frequency.
        frequency: float
            Frequency at which to evaluate the permittivity.

        Returns
        -------
        complex
            The complex permittivity of the given LO-TO model at the given frequency.
        """
        return _medium_numerics().lo_to_eps_model(poles, eps_inf, frequency)

    @classmethod
    def from_lo_to(
        cls, poles: tuple[tuple[float, float, float, float], ...], eps_inf: PositiveFloat = 1
    ) -> Self:
        """Construct a pole residue model from the LO-TO form
        (longitudinal and transverse optical modes).
        The LO-TO form is :math:`\\epsilon_\\infty \\prod_{i=1}^l \\frac{\\omega_{LO, i}^2 - \\omega^2 - i \\omega \\gamma_{LO, i}}{\\omega_{TO, i}^2 - \\omega^2 - i \\omega \\gamma_{TO, i}}` as given in the paper:

            M. Schubert, T. E. Tiwald, and C. M. Herzinger,
            "Infrared dielectric anisotropy and phonon modes of sapphire,"
            Phys. Rev. B 61, 8187 (2000).

        Parameters
        ----------
        poles : tuple[tuple[float, float, float, float], ...]
            The LO-TO poles, given as list of tuples of the form
            (omega_LO, gamma_LO, omega_TO, gamma_TO).
        eps_inf: PositiveFloat
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
    def imag_ep_extrema(poles: PolesAndResidues) -> ArrayFloat1D:
        """Extrema of Im[eps] in the same unit as poles.

        Parameters
        ----------
        poles: PolesAndResidues
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

    @staticmethod
    def _get_vjps_from_params(
        dJ_deps_complex: ComplexArrayOrScalar,
        poles_vals: list[tuple[ComplexArrayOrScalar, ComplexArrayOrScalar]],
        omega: float,
        requested_paths: list[tuple],
        project_real: bool = False,
    ) -> AutogradFieldMap:
        """
        Static helper to compute VJPs from parameters using the analytical chain rule.

        Parameters
        - dJ_deps_complex: Complex adjoint sensitivity w.r.t. epsilon at a single frequency.
        - poles_vals: Sequence of (a_i, c_i) pole parameters to differentiate with respect to.
        - omega: Angular frequency for this VJP evaluation.
        - requested_paths: Paths requested by the caller; used to filter outputs.
        - project_real: If True, project pole-parameter VJPs to their real part.
          Use True for uniform PoleResidue to match real-valued objectives; use False for
          CustomPoleResidue where parameters are complex and complex VJPs are required.
        """
        jw = 1j * omega
        vjps = {}

        if ("eps_inf",) in requested_paths:
            vjps[("eps_inf",)] = np.real(dJ_deps_complex)

        for i, (a_val, c_val) in enumerate(poles_vals):
            if any(path[1] == i for path in requested_paths if path[0] == "poles"):
                if ("poles", i, 0) in requested_paths:
                    deps_da = c_val / (jw + a_val) ** 2
                    dJ_da = dJ_deps_complex * deps_da
                    vjps[("poles", i, 0)] = np.real(dJ_da) if project_real else dJ_da
                if ("poles", i, 1) in requested_paths:
                    deps_dc = -1 / (jw + a_val)
                    dJ_dc = dJ_deps_complex * deps_dc
                    vjps[("poles", i, 1)] = np.real(dJ_dc) if project_real else dJ_dc

        return vjps

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute adjoint derivatives by preparing scalar data and calling the static helper."""

        dJ_deps_complex = self._derivative_eps_complex_volume(
            E_der_map=derivative_info.E_der_map,
            bounds=derivative_info.bounds,
        )

        poles_vals = [(complex(a), complex(c)) for a, c in self.poles]

        freqs = dJ_deps_complex.coords["f"].values
        vjps_total = {}

        for freq in freqs:
            dJ_deps_complex_f = dJ_deps_complex.sel(f=freq)
            vjps_f = self._get_vjps_from_params(
                dJ_deps_complex=complex(dJ_deps_complex_f.item()),
                poles_vals=poles_vals,
                omega=2 * np.pi * freq,
                requested_paths=derivative_info.paths,
                project_real=True,
            )
            for path, vjp in vjps_f.items():
                if path not in vjps_total:
                    vjps_total[path] = vjp
                else:
                    vjps_total[path] += vjp

        return vjps_total

    @classmethod
    def _real_partial_fraction_decomposition(
        cls, a: ArrayFloat, b: ArrayFloat, tol: PositiveFloat = 1e-2
    ) -> tuple[list[tuple[Complex, Complex]], ArrayFloat]:
        """Computes the complex conjugate pole residue pairs given a rational expression with
        real coefficients.

        Parameters
        ----------

        a : np.ndarray
            Coefficients of the numerator polynomial in increasing monomial order.
        b : np.ndarray
            Coefficients of the denominator polynomial in increasing monomial order.
        tol : PositiveFloat
            Tolerance for pole finding. Two poles are considered equal, if their spacing is less
            than ``tol``.

        Returns
        -------
        tuple[list[tuple[Complex, Complex]], np.ndarray]
            The list of complex conjugate poles and their associated residues. The second element of the
            ``tuple`` is an array of coefficients representing any direct polynomial term.

        """
        from scipy import signal

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

        poles_residues = tuple(zip(p_filtered, r_filtered))
        k_increasing_order = np.flip(k)
        return (poles_residues, k_increasing_order)

    @classmethod
    def from_admittance_coeffs(
        cls,
        a: ArrayFloat,
        b: ArrayFloat,
        eps_inf: PositiveFloat = 1,
        pole_tol: PositiveFloat = 1e-2,
    ) -> Self:
        """Construct a :class:`.PoleResidue` model from an admittance function defining the
        relationship between the electric field and the polarization current density in the
        Laplace domain.

        Parameters
        ----------
        a : np.ndarray
            Coefficients of the numerator polynomial in increasing monomial order.
        b : np.ndarray
            Coefficients of the denominator polynomial in increasing monomial order.
        eps_inf: PositiveFloat
            The relative permittivity at infinite frequency.
        pole_tol: PositiveFloat
            Tolerance for the pole finding algorithm in Hertz. Two poles are considered equal, if their
            spacing is closer than ``pole_tol``.
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
