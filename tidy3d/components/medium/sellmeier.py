"""Sellmeier medium models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import autograd.numpy as np
from pydantic import (
    Field,
    PositiveFloat,
    field_validator,
    model_validator,
)

from tidy3d.components.autograd.utils import pack_complex_vec
from tidy3d.components.data.utils import (
    _get_numpy_array,
    _ones_like,
)
from tidy3d.components.validators import (
    call_wrapped_validator,
)
from tidy3d.constants import (
    C_0,
    MICROMETER,
)
from tidy3d.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from autograd.numpy.numpy_boxes import ArrayBox
    from numpy.typing import NDArray

    from tidy3d.compat import Self
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.autograd.types import AutogradFieldMap
    from tidy3d.components.time_modulation import ModulationSpec

    from .base import ArrayFloat


def _medium_numerics() -> Any:
    """Import shared medium kernels after Pydantic model rebuilds finish."""
    from flex_em.numerical.raw import medium as medium_numerics

    return medium_numerics


from .base import AbstractMedium, ensure_freq_in_range  # noqa: E402
from .pole_residue import DispersiveMedium  # noqa: E402


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

    _traced_indexed_root: ClassVar[str] = "coeffs"

    coeffs: tuple[tuple[float, PositiveFloat], ...] = Field(
        title="Coefficients",
        description="List of Sellmeier (:math:`B_i, C_i`) coefficients.",
        json_schema_extra={"units": (None, MICROMETER + "^2")},
    )

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        AbstractMedium._run_after_validators(self)
        self._passivity_validation()
        call_wrapped_validator(DispersiveMedium._conductivity_modulation_validation, self)
        return self

    def _passivity_validation(self) -> Self:
        """Assert passive medium if `allow_gain` is False."""
        val = self.coeffs
        if self.allow_gain:
            return self
        for B, _ in val:
            if B < 0:
                self._raise_validation_error_at_loc(
                    ValidationError(
                        "For passive medium, 'B_i' must be non-negative. "
                        "To simulate a gain medium, please set 'allow_gain=True'. "
                        "Caution: simulations with a gain medium are unstable, "
                        "and are likely to diverge."
                    ),
                    "coeffs",
                )
        return self

    @field_validator("modulation_spec")
    @classmethod
    def _validate_permittivity_modulation(cls, val: ModulationSpec | None) -> ModulationSpec | None:
        """Assert modulated permittivity cannot be <= 0."""

        if val is None or val.permittivity is None:
            return val

        min_eps_inf = 1.0
        if min_eps_inf - val.permittivity.max_modulation <= 0:
            raise ValidationError(
                "The minimum permittivity value with modulation applied was found to be negative."
            )
        return val

    def _n_model(self, frequency: float) -> complex:
        """Complex-valued refractive index as a function of frequency."""

        return _medium_numerics().sellmeier_n_model(self.coeffs, frequency)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        return _medium_numerics().sellmeier_eps_model(self.coeffs, frequency)

    def _pole_residue_dict(self) -> dict:
        """Dict representation of Medium as a pole-residue model"""
        poles = []
        eps_inf = _ones_like(self.coeffs[0][0])
        for B, C in self.coeffs:
            # for small C, it's equivalent to modifying eps_inf
            if np.any(np.isclose(_get_numpy_array(C), 0)):
                eps_inf += B
            else:
                beta = 2 * np.pi * C_0 / np.sqrt(C)
                alpha = -0.5 * beta * B
                a = 1j * beta
                c = 1j * alpha
                poles.append((a, c))
        return {
            "eps_inf": eps_inf,
            "poles": poles,
            "frequency_range": self.frequency_range,
            "name": self.name,
        }

    @staticmethod
    def _from_dispersion_to_coeffs(
        n: float, freq: ArrayFloat, dn_dwvl: float
    ) -> list[tuple[float, float]]:
        """Compute Sellmeier coefficients from dispersion."""
        wvl = C_0 / np.array(freq)
        nsqm1 = n**2 - 1
        c_coeff = -(wvl**3) * n * dn_dwvl / (nsqm1 - wvl * n * dn_dwvl)
        b_coeff = (wvl**2 - c_coeff) / wvl**2 * nsqm1
        return [(b_coeff, c_coeff)]

    @classmethod
    def from_dispersion(cls, n: float, freq: float, dn_dwvl: float = 0, **kwargs: Any) -> Self:
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

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Adjoint derivatives for Sellmeier params via TJP through eps_model()."""

        freqs, vec = self._tjp_inputs(derivative_info)
        N = len(self.coeffs)
        if N == 0:
            return {}

        # pack parameters into flat vector [B..., C...]
        B0 = np.array([b for (b, _c) in self.coeffs])
        C0 = np.array([c for (_b, c) in self.coeffs])
        theta0 = np.concatenate([B0, C0])

        def _eps_vec(theta: Sequence[PositiveFloat]) -> NDArray | ArrayBox:
            B = theta[:N]
            C = theta[N : 2 * N]
            coeffs = tuple((B[i], C[i]) for i in range(N))
            eps = self.updated_copy(coeffs=coeffs, validate=False).eps_model(freqs)
            return pack_complex_vec(eps)

        g = self._tjp_grad(theta0, _eps_vec, vec)

        mapping = []
        mapping += [(("coeffs", i, 0), i) for i in range(N)]
        mapping += [(("coeffs", i, 1), N + i) for i in range(N)]
        return self._map_grad_real(g, derivative_info.paths, mapping)

    @staticmethod
    def _lam2(
        freq: float | ArrayFloat,
    ) -> float | ArrayFloat:
        return (C_0 / freq) ** 2

    @staticmethod
    def _sellmeier_den(
        lam2: float | ArrayFloat,
        C: float | ArrayFloat,
    ) -> float | ArrayFloat:
        return lam2 - C

    # frequency weights for custom Sellmeier
    @staticmethod
    def _w_B(
        freq: float | ArrayFloat,
        C: float | ArrayFloat,
    ) -> float | ArrayFloat:
        lam2 = Sellmeier._lam2(freq)
        return lam2 / Sellmeier._sellmeier_den(lam2, C)

    @staticmethod
    def _w_C(
        freq: float | ArrayFloat,
        B: float | ArrayFloat,
        C: float | ArrayFloat,
    ) -> float | ArrayFloat:
        lam2 = Sellmeier._lam2(freq)
        den = Sellmeier._sellmeier_den(lam2, C)
        return B * lam2 / (den**2)
