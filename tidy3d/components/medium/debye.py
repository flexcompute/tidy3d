"""Debye medium models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import autograd.numpy as np
from pydantic import (
    Field,
    PositiveFloat,
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
    _get_numpy_array,
)
from tidy3d.constants import (
    LARGEST_FP_NUMBER,
    PERMITTIVITY,
    SECOND,
)
from tidy3d.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from autograd.numpy.numpy_boxes import ArrayBox
    from numpy.typing import NDArray

    from tidy3d.compat import Self
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.autograd.types import AutogradFieldMap
    from tidy3d.components.types import FreqBound

    from .base import ArrayComplex, ArrayFloat


def _medium_numerics() -> Any:
    """Import shared medium kernels after Pydantic model rebuilds finish."""
    from flexcompute.core._migration.em.numerical.raw import medium as medium_numerics

    return medium_numerics


from .base import ensure_freq_in_range  # noqa: E402
from .pole_residue import DispersiveMedium  # noqa: E402


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

    _traced_indexed_root: ClassVar[str] = "coeffs"
    _traced_supported_paths: ClassVar[tuple[PathType, ...]] = traced_paths("eps_inf")

    eps_inf: PositiveFloat = Field(
        default=1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        json_schema_extra={"units": PERMITTIVITY},
    )

    coeffs: tuple[tuple[float, PositiveFloat], ...] = Field(
        title="Coefficients",
        description="List of (:math:`\\Delta\\epsilon_i, \\tau_i`) values for model.",
        json_schema_extra={"units": (PERMITTIVITY, SECOND)},
    )

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        super()._run_after_validators()
        self._validate_coeffs_shape()
        self._passivity_validation()
        return self

    def _passivity_validation(self) -> Self:
        """Assert passive medium if `allow_gain` is False."""
        val = self.coeffs
        if self.allow_gain:
            return self
        for del_ep, _ in val:
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

        return _medium_numerics().debye_eps_model(self.eps_inf, self.coeffs, frequency)

    # --- unified helpers for autograd + tests ---

    def _pole_residue_dict(
        self,
    ) -> dict[str, PositiveFloat | list[tuple[complex, complex]] | FreqBound | None | str | None]:
        """Dict representation of Medium as a pole-residue model."""

        poles = []
        eps_inf = self.eps_inf
        for de, tau in self.coeffs:
            # for |tau| close to 0, it's equivalent to modifying eps_inf
            if np.any(abs(_get_numpy_array(tau)) < 1 / 2 / np.pi / LARGEST_FP_NUMBER):
                eps_inf = eps_inf + de
            else:
                a = -2 * np.pi / tau + 0j
                c = -0.5 * de * a

                poles.append((a, c))

        return {
            "eps_inf": eps_inf,
            "poles": poles,
            "frequency_range": self.frequency_range,
            "name": self.name,
        }

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Adjoint derivatives for Debye params via TJP through eps_model()."""

        f, vec = self._tjp_inputs(derivative_info)

        N = len(self.coeffs)
        if N == 0 and ("eps_inf",) not in derivative_info.paths:
            return {}

        # pack into flat [eps_inf, de..., tau...]
        eps_inf0 = self.eps_inf
        de0 = np.array([de for (de, _t) in self.coeffs]) if N else np.array([])
        tau0 = np.array([t for (_de, t) in self.coeffs]) if N else np.array([])
        theta0 = np.concatenate([np.array([eps_inf0]), de0, tau0])

        def _eps_vec(theta: Sequence[PositiveFloat]) -> NDArray | ArrayBox:
            eps_inf = theta[0]
            de = theta[1 : 1 + N]
            tau = theta[1 + N : 1 + 2 * N]
            coeffs = tuple((de[i], tau[i]) for i in range(N))
            eps = self.updated_copy(eps_inf=eps_inf, coeffs=coeffs, validate=False).eps_model(f)
            return pack_complex_vec(eps)

        g = self._tjp_grad(theta0, _eps_vec, vec)

        mapping = [(("eps_inf",), 0)]
        base = 1
        mapping += [(("coeffs", i, 0), base + i) for i in range(N)]
        mapping += [(("coeffs", i, 1), base + N + i) for i in range(N)]
        return self._map_grad_real(g, derivative_info.paths, mapping)

    @staticmethod
    def _den(
        freq: float | ArrayFloat,
        tau: float | ArrayFloat,
    ) -> complex | ArrayComplex:
        return 1 - 1j * (freq * tau)

    # frequency weights for custom Debye
    @staticmethod
    def _w_de(
        freq: float | ArrayFloat,
        tau: float | ArrayFloat,
    ) -> complex | ArrayComplex:
        return 1.0 / Debye._den(freq, tau)

    @staticmethod
    def _w_tau(
        freq: float | ArrayFloat,
        de: float | ArrayFloat,
        tau: float | ArrayFloat,
    ) -> complex | ArrayComplex:
        den = Debye._den(freq, tau)
        return (1j * freq * de) / (den**2)
