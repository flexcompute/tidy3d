"""Dispersionless isotropic medium models."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import autograd.numpy as np
from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from tidy3d.components.autograd.types import (
    PathType,
    TracedFloat,
)
from tidy3d.components.base import cached_property
from tidy3d.components.data.utils import (
    _get_numpy_array,
)
from tidy3d.constants import (
    CONDUCTIVITY,
    EPSILON_0,
    PERMITTIVITY,
    pec_val,
)
from tidy3d.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from tidy3d.compat import Self
    from tidy3d.components.time_modulation import ModulationSpec

    from .lorentz import Lorentz


def _medium_numerics() -> Any:
    """Import shared medium kernels after Pydantic model rebuilds finish."""
    from flexcompute.core._migration.em.numerical.raw import medium as medium_numerics

    return medium_numerics


from .base import (  # noqa: E402
    _EPS_SIGMA_TRACED_PATHS,
    AbstractMedium,
    _constant_over_frequency,
    ensure_freq_in_range,
)


def _reject_modulation_spec(medium_name: str, val: ModulationSpec | None) -> ModulationSpec | None:
    """Thin field adapter of ``PECMedium`` and ``PMCMedium`` that calls the core rule."""
    # ``import tidy3d`` builds both classes (built at import), so they stay off the construction
    # path.
    from flexcompute.core._migration.em.material.medium.validation import (
        validate_unsupported_modulation_spec,
    )
    from flexcompute.core.validation import raise_first_error

    raise_first_error(
        validate_unsupported_modulation_spec(
            medium_name, None if val is None else type(val).__name__
        ),
        ValidationError,
    )
    return val


class PECMedium(AbstractMedium):
    """Perfect electrical conductor class.

    Note
    ----

        To avoid confusion from duplicate PECs, must import ``tidy3d.PEC`` instance directly.



    """

    @field_validator("modulation_spec")
    @classmethod
    def _validate_modulation_spec(cls, val: ModulationSpec | None) -> ModulationSpec | None:
        """Check compatibility with modulation_spec."""
        return _reject_modulation_spec(cls.__name__, val)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        return _constant_over_frequency(pec_val + 0j, frequency)

    @cached_property
    def n_cfl(self) -> float:
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.
        """
        return 1.0

    @cached_property
    def is_pec(self) -> bool:
        """Whether the medium is a PEC."""
        return True


# PEC builtin instance
PEC = PECMedium(name="PEC")


# PMC keyword
class PMCMedium(AbstractMedium):
    """Perfect magnetic conductor class.

    Note
    ----

        To avoid confusion from duplicate PMCs, must import ``tidy3d.PMC`` instance directly.



    """

    @field_validator("modulation_spec")
    @classmethod
    def _validate_modulation_spec(cls, val: ModulationSpec | None) -> ModulationSpec | None:
        """Check compatibility with modulation_spec."""
        return _reject_modulation_spec(cls.__name__, val)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        return _constant_over_frequency(1.0 + 0j, frequency)

    @cached_property
    def n_cfl(self) -> float:
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.
        """
        return 1.0

    @cached_property
    def is_pmc(self) -> bool:
        """Whether the medium is a PMC."""
        return True


# PEC builtin instance
PMC = PMCMedium(name="PMC")


class Medium(AbstractMedium):
    """Dispersionless medium. Mediums define the optical properties of the materials within the simulation.

    Notes
    -----

        In a dispersion-less medium, the displacement field :math:`D(t)` reacts instantaneously to the applied
        electric field :math:`E(t)`.

        .. math::

            D(t) = \\epsilon E(t)

        The ``permittivity`` parameter is the relative permittivity (dimensionless). The ``conductivity``
        parameter has units of S/μm (siemens per micrometer), consistent with Tidy3D's micrometer-based unit
        system. To convert from standard S/m, divide by 1e6.

        **Practical Advice**

        **Choosing a Material Type**

        - Material is in ``td.material_library``? → Use it directly
          (e.g. ``td.material_library['cSi']['Li1993_293K']``).
        - Lossless, wavelength-independent refractive index? → ``td.Medium(permittivity=n**2)``.
        - Known n and k at a specific frequency? → ``td.Medium.from_nk(n=2.4, k=0.01, freq=freq0)``.
          Note: when k > 0, the resulting medium has wavelength-independent n but wavelength-dependent k.
        - You have n,k data vs wavelength? → Use ``FastDispersionFitter`` from
          ``tidy3d.plugins.dispersion`` to fit a pole-residue model.
        - Permittivity varies spatially? → Use ``CustomMedium`` with a ``SpatialDataArray``.
        - Need an analytical dispersive model? → Use ``Sellmeier``, ``Lorentz``, ``Drude``,
          ``Debye``, or ``PoleResidue`` directly.

        **Common Library Materials (telecom, ~1.55 μm)**

        - Silicon: ``td.material_library['cSi']['Li1993_293K']`` (n ≈ 3.48)
        - SiO2: ``td.material_library['SiO2']['Palik_NoLoss']`` (n ≈ 1.44)
        - Si3N4: ``td.material_library['Si3N4']['Luke2015PMLStable']`` (n ≈ 2.0)
        - Gold: ``td.material_library['Au']['JohnsonChristy1972']``
        - Silver: ``td.material_library['Ag']['JohnsonChristy1972']``

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

    _traced_supported_paths: ClassVar[tuple[PathType, ...]] = _EPS_SIGMA_TRACED_PATHS

    permittivity: TracedFloat = Field(
        default=1.0,
        ge=1.0,
        title="Permittivity",
        description="Relative permittivity.",
        json_schema_extra={"units": PERMITTIVITY},
    )

    conductivity: TracedFloat = Field(
        default=0.0,
        title="Conductivity",
        description="Electric conductivity. Defined such that the imaginary part of the complex "
        "permittivity at angular frequency omega is given by conductivity/omega.",
        json_schema_extra={"units": CONDUCTIVITY},
    )

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        super()._run_after_validators()
        self._passivity_validation()
        self._permittivity_modulation_validation()
        self._passivity_modulation_validation()
        return self

    # Thin checks that call the core rules: ``import tidy3d`` builds ``Medium``, so it
    # stays off the construction path. Subclasses on the path also run them here.
    def _passivity_validation(self) -> Self:
        """Assert passive medium if ``allow_gain`` is False."""
        from flexcompute.core._migration.em.material.medium.validation import (
            validate_conductivity_passivity,
        )

        from tidy3d.em.translate.components import _present

        _present(self, validate_conductivity_passivity(self.conductivity, self.allow_gain))
        return self

    def _permittivity_modulation_validation(self) -> Self:
        """Assert modulated permittivity cannot be <= 0."""
        from flexcompute.core._migration.em.material.medium.validation import (
            validate_modulated_permittivity,
        )

        from tidy3d.em.translate.components import _present

        modulation = self.modulation_spec
        if modulation is not None and modulation.permittivity is not None:
            values = _get_numpy_array(self.permittivity)
            _present(
                self,
                validate_modulated_permittivity(values, modulation.permittivity.max_modulation),
            )
        return self

    def _passivity_modulation_validation(self) -> Self:
        """Assert passive medium if ``allow_gain`` is False."""
        from flexcompute.core._migration.em.material.medium.validation import (
            validate_modulated_conductivity,
        )

        from tidy3d.em.translate.components import _present

        modulation = self.modulation_spec
        if modulation is not None and modulation.conductivity is not None:
            values = _get_numpy_array(self.conductivity)
            diagnostics = validate_modulated_conductivity(
                values, modulation.conductivity.max_modulation, self.allow_gain
            )
            _present(self, diagnostics)
        return self

    @cached_property
    def n_cfl(self) -> float:
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

        return _medium_numerics().medium_eps_model(permittivity, conductivity, frequency)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        return self._eps_model(self.permittivity, self.conductivity, frequency)

    @classmethod
    def from_nk(cls, n: float, k: float, freq: float, **kwargs: Any) -> Self:
        """Convert ``n`` and ``k`` values at frequency ``freq`` to :class:`.Medium`.

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
        :class:`.Medium`
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

    def _derivative_rule(self, path: PathType) -> Callable[[np.ndarray, np.ndarray], Any]:
        """Permittivity or conductivity from the component-summed sensitivity."""

        if path not in (("permittivity",), ("conductivity",)):
            return super()._derivative_rule(path)
        return partial(_medium_vjp, conductivity=path == ("conductivity",))


def _medium_vjp(sensitivity: Any, frequencies: Any, *, conductivity: bool) -> Any:
    """``sum_f Re S`` for permittivity or ``sum_f -Im S / (w eps0)`` for conductivity."""

    by_frequency = np.sum(np.asarray(sensitivity, dtype=complex), axis=0)
    if not conductivity:
        return np.sum(np.real(by_frequency))
    return np.sum(-np.imag(by_frequency) / (2 * np.pi * frequencies) / EPSILON_0)


def medium_from_nk(n: float, k: float, freq: float, **kwargs: Any) -> Medium | Lorentz:
    """Construct a dispersionless or Lorentz medium from refractive-index data.

    Parameters
    ----------
    n : float
        Real part of refractive index.
    k : float
        Imaginary part of refractive index.
    freq : float
        Frequency at which ``n`` and ``k`` are specified, in Hz.
    **kwargs
        Additional arguments forwarded to the selected medium constructor.

    Returns
    -------
    Medium | Lorentz
        A dispersionless medium when ``Re[epsilon] >= 1``; otherwise a Lorentz medium.
    """
    # Local import avoids the isotropic -> Lorentz -> pole-residue -> isotropic cycle.
    from .lorentz import Lorentz

    eps_complex = AbstractMedium.nk_to_eps_complex(n, k)
    if eps_complex.real >= 1:
        return Medium.from_nk(n, k, freq, **kwargs)
    return Lorentz.from_nk(n, k, freq, **kwargs)
