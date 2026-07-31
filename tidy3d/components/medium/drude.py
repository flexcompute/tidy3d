"""Drude medium models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import autograd.numpy as np
from pydantic import (
    Field,
    PositiveFloat,
)

from tidy3d.components.autograd.path_utils import (
    traced_paths,
)
from tidy3d.components.autograd.types import (
    PathType,
)
from tidy3d.components.autograd.utils import pack_complex_vec
from tidy3d.constants import (
    HERTZ,
    PERMITTIVITY,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from autograd.numpy.numpy_boxes import ArrayBox
    from numpy.typing import NDArray

    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.autograd.types import AutogradFieldMap

    from .base import ArrayComplex, ArrayFloat


def _medium_numerics() -> Any:
    """Import shared medium kernels after Pydantic model rebuilds finish."""
    from flex_em.numerical.raw import medium as medium_numerics

    return medium_numerics


from .base import ensure_freq_in_range  # noqa: E402
from .pole_residue import DispersiveMedium  # noqa: E402


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

    _traced_indexed_root: ClassVar[str] = "coeffs"
    _traced_supported_paths: ClassVar[tuple[PathType, ...]] = traced_paths("eps_inf")

    eps_inf: PositiveFloat = Field(
        1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        json_schema_extra={"units": PERMITTIVITY},
    )

    coeffs: tuple[tuple[float, PositiveFloat], ...] = Field(
        title="Coefficients",
        description="List of (:math:`f_i, \\delta_i`) values for model.",
        json_schema_extra={"units": (HERTZ, HERTZ)},
    )

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        return _medium_numerics().drude_eps_model(self.eps_inf, self.coeffs, frequency)

    # --- unified helpers for autograd + tests ---

    def _pole_residue_dict(self) -> dict:
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

        return {
            "eps_inf": self.eps_inf,
            "poles": poles,
            "frequency_range": self.frequency_range,
            "name": self.name,
        }

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Adjoint derivatives for Drude params via TJP through eps_model()."""

        f, vec = self._tjp_inputs(derivative_info)

        N = len(self.coeffs)
        if N == 0 and ("eps_inf",) not in derivative_info.paths:
            return {}

        # pack into flat [eps_inf, fp..., delta...]
        eps_inf0 = self.eps_inf
        fp0 = np.array([fp for (fp, _d) in self.coeffs]) if N else np.array([])
        d0 = np.array([dd for (_fp, dd) in self.coeffs]) if N else np.array([])
        theta0 = np.concatenate([np.array([eps_inf0]), fp0, d0])

        def _eps_vec(theta: Sequence[PositiveFloat]) -> NDArray | ArrayBox:
            eps_inf = theta[0]
            fp = theta[1 : 1 + N]
            dd = theta[1 + N : 1 + 2 * N]
            coeffs = tuple((fp[i], dd[i]) for i in range(N))
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
        delta: float | ArrayFloat,
    ) -> complex | ArrayComplex:
        return (freq**2) + 1j * (freq * delta)

    # frequency weights for custom Drude
    @staticmethod
    def _w_fp(
        freq: float | ArrayFloat,
        fp: float | ArrayFloat,
        delta: float | ArrayFloat,
    ) -> complex | ArrayComplex:
        return -(2.0 * fp) / Drude._den(freq, delta)

    @staticmethod
    def _w_delta(
        freq: float | ArrayFloat,
        fp: float | ArrayFloat,
        delta: float | ArrayFloat,
    ) -> complex | ArrayComplex:
        den = Drude._den(freq, delta)
        return (1j * freq * (fp**2)) / (den**2)
