"""Sellmeier medium models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import autograd.numpy as np
from pydantic import Field, field_validator, model_validator

from tidy3d.components.autograd.types import TracedFloat, TracedPositiveFloat
from tidy3d.components.data.utils import (
    _get_numpy_array,
    _ones_like,
)
from tidy3d.constants import (
    C_0,
    MICROMETER,
)
from tidy3d.exceptions import ValidationError

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.time_modulation import ModulationSpec

    from .base import ArrayFloat


def _medium_numerics() -> Any:
    """Import shared medium kernels after Pydantic model rebuilds finish."""
    from flexcompute.core._migration.em.numerical.raw import medium as medium_numerics

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
    _derivative_model: ClassVar[str] = "sellmeier"

    coeffs: tuple[tuple[TracedFloat, TracedPositiveFloat], ...] = Field(
        title="Coefficients",
        description="List of Sellmeier (:math:`B_i, C_i`) coefficients.",
        json_schema_extra={"units": (None, MICROMETER + "^2")},
    )

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        AbstractMedium._run_after_validators(self)
        self._passivity_validation()
        return self._modulation_validation()

    def _passivity_validation(self) -> Self:
        """Assert passive medium if `allow_gain` is False."""
        # Thin check that calls the core rule. ``import tidy3d`` builds ``Sellmeier`` (built at
        # import), so it stays off the construction path.
        from flexcompute.core._migration.em.material.medium.validation import (
            validate_sellmeier_passivity,
        )

        from tidy3d.em.translate.components import _present

        _present(self, validate_sellmeier_passivity([b for b, _ in self.coeffs], self.allow_gain))
        return self

    @field_validator("modulation_spec")
    @classmethod
    def _validate_permittivity_modulation(cls, val: ModulationSpec | None) -> ModulationSpec | None:
        """Assert modulated permittivity cannot be <= 0."""
        # Thin field adapter that calls the core rule (built at import: ``import tidy3d``
        # builds ``Sellmeier``).
        from flexcompute.core._migration.em.material.medium.validation import (
            max_modulations,
            validate_sellmeier_modulation,
        )
        from flexcompute.core.validation import raise_first_error

        raise_first_error(validate_sellmeier_modulation(max_modulations(val)[0]), ValidationError)
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
