"""Surface-impedance lossy-metal medium models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import autograd.numpy as np
from pydantic import (
    Field,
    PositiveFloat,
    field_validator,
)

from tidy3d.components.base import cached_property
from tidy3d.components.dispersion_fitter import (
    fit,
)
from tidy3d.components.types import TYPE_TAG_STR, FreqBound
from tidy3d.components.viz import add_ax_if_none
from tidy3d.constants import (
    C_0,
    CONDUCTIVITY,
    ETA_0,
    HERTZ,
    MICROMETER,
    MU_0,
    PERMITTIVITY,
)
from tidy3d.exceptions import SetupError, ValidationError

if TYPE_CHECKING:
    from tidy3d.components.types import (
        ArrayFloat1D,
        Ax,
    )

    from .base import ArrayComplex


from .isotropic import Medium
from .pole_residue import PoleResidue
from .roughness import SurfaceImpedanceFitterParam, SurfaceRoughnessType

LOSSY_METAL_SCALED_REAL_PART = 10.0


class LossyMetalMedium(Medium):
    """Lossy metal that can be modeled with a surface impedance boundary condition (SIBC).

    Notes
    -----

        SIBC is most accurate when the skin depth is much smaller than the structure feature size.
        If not the case, please use a regular medium instead, or set ``simulation.subpixel.lossy_metal``
        to ``td.VolumetricAveraging()`` or ``td.Staircasing()``.

    Example
    -------
    >>> lossy_metal = LossyMetalMedium(conductivity=10, frequency_range=(9e9, 10e9))

    """

    allow_gain: Literal[False] = Field(
        False,
        title="Allow gain medium",
        description="Allow the medium to be active. Caution: "
        "simulations with a gain medium are unstable, and are likely to diverge."
        "Simulations where ``allow_gain`` is set to ``True`` will still be charged even if "
        "diverged. Monitor data up to the divergence point will still be returned and can be "
        "useful in some cases.",
    )

    permittivity: Literal[1.0] = Field(  # pyrefly: ignore[invalid-literal]
        1.0,
        title="Permittivity",
        description="Relative permittivity.",
        json_schema_extra={"units": PERMITTIVITY},
    )

    conductivity: PositiveFloat = Field(
        title="Conductivity",
        description="Electric conductivity. Defined such that the imaginary part of the complex "
        "permittivity at angular frequency omega is given by conductivity/omega.",
        json_schema_extra={"units": CONDUCTIVITY},
    )

    roughness: SurfaceRoughnessType | None = Field(
        None,
        title="Surface Roughness Model",
        description="Surface roughness model that applies a frequency-dependent scaling "
        "factor to surface impedance. Takes effect only when ``penetrable=False`` and "
        "``Simulation.subpixel.lossy_metal`` is ``SurfaceImpedance``.",
        discriminator=TYPE_TAG_STR,
    )

    thickness: PositiveFloat | None = Field(
        None,
        title="Conductor Thickness",
        description="When the thickness of the conductor is not much greater than skin depth, "
        "1D transmission line model is applied to compute the surface impedance of the thin conductor. "
        "Takes effect only when ``penetrable=False`` and ``Simulation.subpixel.lossy_metal`` is "
        "``SurfaceImpedance``.",
        json_schema_extra={"units": MICROMETER},
    )

    frequency_range: FreqBound = Field(
        title="Frequency Range",
        description="Frequency range of validity for the medium.",
        json_schema_extra={"units": (HERTZ, HERTZ)},
    )

    fit_param: SurfaceImpedanceFitterParam = Field(
        default_factory=SurfaceImpedanceFitterParam,
        title="Fitting Parameters For Surface Impedance",
        description="Parameters for fitting surface impedance divided by (-1j * omega) over "
        "the frequency range using pole-residue pair model. Takes effect only when "
        "``penetrable=False`` and ``Simulation.subpixel.lossy_metal`` is ``SurfaceImpedance``.",
    )

    penetrable: bool = Field(
        False,
        title="Penetrable",
        description="If ``True``, the metal is solved as a regular conductive medium with the "
        "given ``conductivity`` (and ``permittivity = 1``), and subpixel averaging on this "
        "material follows ``Simulation.subpixel.dielectric``. If ``False`` (default), the metal "
        "uses the lossy-metal handling selected by ``Simulation.subpixel.lossy_metal`` (e.g. a "
        "surface impedance boundary condition).",
    )

    @field_validator("frequency_range")
    @classmethod
    def _validate_frequency_range(cls, val: FreqBound) -> FreqBound:
        """Validate that frequency range is finite and non-zero."""
        for freq in val:
            if not np.isfinite(freq):
                raise ValidationError("Values in 'frequency_range' must be finite.")
            if freq <= 0:
                raise ValidationError("Values in 'frequency_range' must be positive.")
        return val

    @cached_property
    def is_pec_like(self) -> bool:
        """Whether the medium is treated as a PEC medium in surface monitors."""
        return not self.penetrable

    @cached_property
    def _fitting_result(self) -> tuple[PoleResidue, float]:
        """Fitted scaled surface impedance and residue."""

        omega_data = self.Hz_to_angular_freq(self.sampling_frequencies)
        surface_impedance = self.surface_impedance(self.sampling_frequencies)
        scaled_impedance = surface_impedance / (-1j * omega_data)

        # let's use scaled quantity in fitting: minimal real part equals ``SCALED_REAL_PART``
        min_real = np.min(scaled_impedance.real)
        if min_real <= 0:
            raise SetupError(
                "The real part of scaled surface impedance must be positive. "
                "Please create a github issue so that the problem can be investigated. "
                "In the meantime, make sure the material is passive."
            )

        scaling_factor = LOSSY_METAL_SCALED_REAL_PART / min_real
        scaled_impedance *= scaling_factor

        (res_inf, poles, residues), error = fit(
            omega_data=omega_data,
            resp_data=scaled_impedance,
            min_num_poles=0,
            max_num_poles=self.fit_param.max_num_poles,
            resp_inf=None,
            tolerance_rms=self.fit_param.tolerance_rms,
            scale_factor=1.0 / np.max(omega_data),
        )

        res_inf /= scaling_factor
        residues /= scaling_factor
        return PoleResidue(eps_inf=res_inf, poles=list(zip(poles, residues))), error

    @cached_property
    def scaled_surface_impedance_model(self) -> PoleResidue:
        """Fitted surface impedance divided by (-j \\omega) using pole-residue pair model within ``frequency_range``."""
        return self._fitting_result[0]

    @cached_property
    def num_poles(self) -> int:
        """Number of poles in the fitted model."""
        return len(self.scaled_surface_impedance_model.poles)

    def surface_impedance(self, frequencies: ArrayFloat1D) -> ArrayComplex:
        """Computing surface impedance including surface roughness effects."""
        # compute complex-valued skin depth
        n, k = self.nk_model(frequencies)

        # with surface roughness effects
        correction = 1.0
        if self.roughness is not None:
            skin_depths = 1 / np.sqrt(np.pi * frequencies * MU_0 * self.conductivity)
            correction = self.roughness.roughness_correction_factor(frequencies, skin_depths)

        if self.thickness is not None:
            k_wave = self.Hz_to_angular_freq(frequencies) / C_0 * (n + 1j * k)
            correction /= -np.tanh(1j * k_wave * self.thickness)

        return correction * ETA_0 / (n + 1j * k)

    @cached_property
    def sampling_frequencies(self) -> ArrayFloat1D:
        """Sampling frequencies used in fitting."""
        if self.fit_param.frequency_sampling_points < 2:
            return np.array([np.mean(self.frequency_range)])

        if self.fit_param.log_sampling:
            return np.logspace(
                np.log10(self.frequency_range[0]),
                np.log10(self.frequency_range[1]),
                self.fit_param.frequency_sampling_points,
            )
        return np.linspace(
            self.frequency_range[0],
            self.frequency_range[1],
            self.fit_param.frequency_sampling_points,
        )

    def eps_diagonal_numerical(self, frequency: float) -> tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor for numerical considerations
        such as meshing and runtime estimation.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        tuple[complex, complex, complex]
            The diagonal elements of relative permittivity tensor relevant for numerical
            considerations evaluated at ``frequency``.
        """
        return (1.0 + 0j,) * 3

    @add_ax_if_none
    def plot(
        self,
        ax: Ax = None,
    ) -> Ax:
        """Make plot of complex-valued surface imepdance model vs fitted model, at sampling frequencies.
        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes = None
            Axes to plot the data on, if None, a new one is created.
        Returns
        -------
        matplotlib.axis.Axes
            Matplotlib axis corresponding to plot.
        """
        frequencies = self.sampling_frequencies
        surface_impedance = self.surface_impedance(frequencies)

        ax.plot(frequencies, surface_impedance.real, "x", label="Real")
        ax.plot(frequencies, surface_impedance.imag, "+", label="Imag")

        surface_impedance_model = (
            -1j
            * self.Hz_to_angular_freq(frequencies)
            * self.scaled_surface_impedance_model.eps_model(frequencies)
        )
        ax.plot(frequencies, surface_impedance_model.real, label="Real (model)")
        ax.plot(frequencies, surface_impedance_model.imag, label="Imag (model)")

        ax.set_ylabel(r"Surface impedance ($\Omega$)")
        ax.set_xlabel("Frequency (Hz)")
        ax.legend()

        return ax
