"""Debye medium models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import autograd.numpy as np
from flexcompute.core._migration.em.material.medium.limits import DEBYE_TAU_MINIMUM
from pydantic import Field

from tidy3d.components.autograd.path_utils import (
    traced_paths,
)
from tidy3d.components.autograd.types import (
    PathType,
    TracedFloat,
    TracedPositiveFloat,
)
from tidy3d.components.data.utils import (
    _get_numpy_array,
)
from tidy3d.constants import (
    PERMITTIVITY,
    SECOND,
)

if TYPE_CHECKING:
    from pydantic import PositiveFloat

    from tidy3d.components.types import FreqBound


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
    _derivative_model: ClassVar[str] = "debye"
    _traced_supported_paths: ClassVar[tuple[PathType, ...]] = traced_paths("eps_inf")

    eps_inf: TracedPositiveFloat = Field(
        default=1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        json_schema_extra={"units": PERMITTIVITY},
    )

    coeffs: tuple[tuple[TracedFloat, TracedPositiveFloat], ...] = Field(
        title="Coefficients",
        description="List of (:math:`\\Delta\\epsilon_i, \\tau_i`) values for model.",
        json_schema_extra={"units": (PERMITTIVITY, SECOND)},
    )

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
            if np.any(abs(_get_numpy_array(tau)) < DEBYE_TAU_MINIMUM):
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
