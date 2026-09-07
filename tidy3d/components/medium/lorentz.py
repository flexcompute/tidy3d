"""Lorentz medium models."""

from __future__ import annotations

from math import isclose
from typing import TYPE_CHECKING, Any, ClassVar

import autograd.numpy as np
from pydantic import (
    Field,
    NonNegativeFloat,
    PositiveFloat,
    field_validator,
    model_validator,
)

from tidy3d.components.autograd.path_utils import (
    traced_paths,
)
from tidy3d.components.autograd.types import (
    PathType,
)
from tidy3d.components.autograd.utils import pack_complex_vec
from tidy3d.components.data.utils import (
    CustomSpatialDataType,
    _get_numpy_array,
)
from tidy3d.constants import (
    HERTZ,
    PERMITTIVITY,
)
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log

if TYPE_CHECKING:
    from collections.abc import Sequence

    from autograd.numpy.numpy_boxes import ArrayBox
    from numpy.typing import NDArray

    from tidy3d.compat import Self
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.autograd.types import AutogradFieldMap

    from .base import ArrayComplex, ArrayFloat


def _medium_numerics() -> Any:
    """Import shared medium kernels after Pydantic model rebuilds finish."""
    from flex_em.numerical.raw import medium as medium_numerics

    return medium_numerics


from .base import AbstractMedium, ensure_freq_in_range  # noqa: E402
from .pole_residue import DispersiveMedium  # noqa: E402


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

    _traced_indexed_root: ClassVar[str] = "coeffs"
    _traced_supported_paths: ClassVar[tuple[PathType, ...]] = traced_paths("eps_inf")

    eps_inf: PositiveFloat = Field(
        default=1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        json_schema_extra={"units": PERMITTIVITY},
    )

    coeffs: tuple[tuple[float, float, NonNegativeFloat], ...] = Field(
        title="Coefficients",
        description="List of (:math:`\\Delta\\epsilon_i, f_i, \\delta_i`) values for model.",
        json_schema_extra={"units": (PERMITTIVITY, HERTZ, HERTZ)},
    )

    @field_validator("coeffs")
    @classmethod
    def _coeffs_unequal_f_delta(
        cls, val: tuple[tuple[float, float, NonNegativeFloat], ...]
    ) -> tuple[tuple[float, float, NonNegativeFloat], ...]:
        """f**2 and delta**2 cannot be exactly the same."""
        for _, f, delta in val:
            if f**2 == delta**2:
                raise SetupError("'f' and 'delta' cannot take equal values.")
        return val

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        super()._run_after_validators()
        self._validate_coeffs_shape()
        self._passivity_validation()
        return self

    def _passivity_validation(self) -> Self:
        """Assert passive medium if ``allow_gain`` is False."""
        val = self.coeffs
        if self.allow_gain:
            return self
        for del_ep, _, _ in val:
            if del_ep < 0:
                self._raise_validation_error_at_loc(
                    ValidationError(
                        "For passive medium, 'Delta epsilon_i' must be non-negative. "
                        "To simulate a gain medium, please set 'allow_gain=True'. "
                        "Caution: simulations with a gain medium are unstable, "
                        "and are likely to diverge."
                    ),
                    "coeffs",
                )
        return self

    def _validate_coeffs_shape(self) -> Self:
        """Hook for subclasses that need coeff shape checks."""
        return self

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        return _medium_numerics().lorentz_eps_model(self.eps_inf, self.coeffs, frequency)

    def _pole_residue_dict(self) -> dict:
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

        return {
            "eps_inf": self.eps_inf,
            "poles": poles,
            "frequency_range": self.frequency_range,
            "name": self.name,
        }

    @staticmethod
    def _all_larger(
        coeff_a: tuple[tuple[CustomSpatialDataType, CustomSpatialDataType], ...],
        coeff_b: tuple[tuple[CustomSpatialDataType, CustomSpatialDataType], ...],
    ) -> bool:
        """``coeff_a`` and ``coeff_b`` can be either float or SpatialDataArray."""
        if isinstance(coeff_a, CustomSpatialDataType.__args__):
            return np.all(_get_numpy_array(coeff_a) > _get_numpy_array(coeff_b))
        return coeff_a > coeff_b

    @classmethod
    def from_nk(cls, n: float, k: float, freq: float, **kwargs: Any) -> Self:
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

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Adjoint derivatives for Lorentz params via TJP through eps_model()."""

        f, vec = self._tjp_inputs(derivative_info)

        N = len(self.coeffs)
        if N == 0 and ("eps_inf",) not in derivative_info.paths:
            return {}

        # pack into flat [eps_inf, de..., f0..., delta...]
        eps_inf0 = self.eps_inf
        de0 = np.array([de for (de, _f, _d) in self.coeffs]) if N else np.array([])
        f0 = np.array([fi for (_de, fi, _d) in self.coeffs]) if N else np.array([])
        d0 = np.array([dd for (_de, _f, dd) in self.coeffs]) if N else np.array([])
        theta0 = np.concatenate([np.array([eps_inf0]), de0, f0, d0])

        def _eps_vec(theta: Sequence[PositiveFloat]) -> NDArray | ArrayBox:
            eps_inf = theta[0]
            de = theta[1 : 1 + N]
            fi = theta[1 + N : 1 + 2 * N]
            dd = theta[1 + 2 * N : 1 + 3 * N]
            coeffs = tuple((de[i], fi[i], dd[i]) for i in range(N))
            eps = self.updated_copy(eps_inf=eps_inf, coeffs=coeffs, validate=False).eps_model(f)
            return pack_complex_vec(eps)

        g = self._tjp_grad(theta0, _eps_vec, vec)

        mapping = [(("eps_inf",), 0)]
        base = 1
        mapping += [(("coeffs", i, 0), base + i) for i in range(N)]
        mapping += [(("coeffs", i, 1), base + N + i) for i in range(N)]
        mapping += [(("coeffs", i, 2), base + 2 * N + i) for i in range(N)]
        return self._map_grad_real(g, derivative_info.paths, mapping)

    @staticmethod
    def _den(
        freq: float | ArrayFloat,
        f0: float | ArrayFloat,
        delta: float | ArrayFloat,
    ) -> complex | ArrayComplex:
        return (f0**2) - 2j * (freq * delta) - (freq**2)

    # frequency weights for custom Lorentz
    @staticmethod
    def _w_de(
        freq: float | ArrayFloat,
        f0: float | ArrayFloat,
        delta: float | ArrayFloat,
    ) -> complex | ArrayComplex:
        return (f0**2) / Lorentz._den(freq, f0, delta)

    @staticmethod
    def _w_f0(
        freq: float | ArrayFloat,
        de: float | ArrayFloat,
        f0: float | ArrayFloat,
        delta: float | ArrayFloat,
    ) -> complex | ArrayComplex:
        den = Lorentz._den(freq, f0, delta)
        return (2.0 * de * f0 * (den - f0**2)) / (den**2)

    @staticmethod
    def _w_delta(
        freq: float | ArrayFloat,
        de: float | ArrayFloat,
        f0: float | ArrayFloat,
        delta: float | ArrayFloat,
    ) -> complex | ArrayComplex:
        den = Lorentz._den(freq, f0, delta)
        return (2j * freq * de * (f0**2)) / (den**2)
