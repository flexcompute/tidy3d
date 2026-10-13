"""Drude medium models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import autograd.numpy as np
from pydantic import Field

from tidy3d.components.autograd.path_utils import (
    traced_paths,
)
from tidy3d.components.autograd.types import (
    PathType,
    TracedFloat,
    TracedPositiveFloat,
)
from tidy3d.constants import (
    HERTZ,
    PERMITTIVITY,
)

if TYPE_CHECKING:
    pass


def _medium_numerics() -> Any:
    """Import shared medium kernels after Pydantic model rebuilds finish."""
    from flexcompute.core._migration.em.numerical.raw import medium as medium_numerics

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
    _derivative_model: ClassVar[str] = "drude"
    _traced_supported_paths: ClassVar[tuple[PathType, ...]] = traced_paths("eps_inf")

    eps_inf: TracedPositiveFloat = Field(
        default=1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        json_schema_extra={"units": PERMITTIVITY},
    )

    coeffs: tuple[tuple[TracedFloat, TracedPositiveFloat], ...] = Field(
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
