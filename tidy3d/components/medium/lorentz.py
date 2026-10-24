"""Lorentz medium models."""

from __future__ import annotations

from math import isclose
from typing import TYPE_CHECKING, Any, ClassVar

import autograd.numpy as np
from pydantic import Field

from tidy3d.components.autograd.path_utils import (
    traced_paths,
)
from tidy3d.components.autograd.types import (
    PathType,
    TracedFloat,
    TracedNonNegativeFloat,
    TracedPositiveFloat,
)
from tidy3d.components.data.utils import (
    CustomSpatialDataType,
    _get_numpy_array,
)
from tidy3d.constants import (
    HERTZ,
    PERMITTIVITY,
)
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.compat import Self


def _medium_numerics() -> Any:
    """Import shared medium kernels after Pydantic model rebuilds finish."""
    from flexcompute.core._migration.em.numerical.raw import medium as medium_numerics

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
    _derivative_model: ClassVar[str] = "lorentz"
    _traced_supported_paths: ClassVar[tuple[PathType, ...]] = traced_paths("eps_inf")

    eps_inf: TracedPositiveFloat = Field(
        default=1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        json_schema_extra={"units": PERMITTIVITY},
    )

    coeffs: tuple[tuple[TracedFloat, TracedFloat, TracedNonNegativeFloat], ...] = Field(
        title="Coefficients",
        description="List of (:math:`\\Delta\\epsilon_i, f_i, \\delta_i`) values for model.",
        json_schema_extra={"units": (PERMITTIVITY, HERTZ, HERTZ)},
    )

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
