from __future__ import annotations

from abc import abstractmethod
from typing import Tuple, Union

import autograd.numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.dispersion_fitter import fit
from tidy3d.components.types import (
    TYPE_TAG_STR,
    ArrayComplex1D,
    ArrayFloat1D,
    Ax,
    FreqBound,
    Literal,
)
from tidy3d.components.viz import add_ax_if_none
from tidy3d.constants import ETA_0, HERTZ, MICROMETER, MU_0, PERMITTIVITY
from tidy3d.exceptions import SetupError, ValidationError

from .constants import (
    LOSSY_METAL_DEFAULT_MAX_POLES,
    LOSSY_METAL_DEFAULT_SAMPLING_FREQUENCY,
    LOSSY_METAL_DEFAULT_TOLERANCE_RMS,
    LOSSY_METAL_SCALED_REAL_PART,
)
from .dispersionless import Medium
from .dispersive import PoleResidue


class SurfaceImpedanceFitterParam(Tidy3dBaseModel):
    """Advanced parameters for fitting surface impedance of a :class:`.LossyMetalMedium`.
    Internally, the quantity to be fitted is surface impedance divided by ``-1j * \\omega``.
    """

    max_num_poles: pd.PositiveInt = pd.Field(
        LOSSY_METAL_DEFAULT_MAX_POLES,
        title="Maximal Number Of Poles",
        description="Maximal number of poles in complex-conjugate pole residue model for "
        "fitting surface impedance.",
    )

    tolerance_rms: pd.NonNegativeFloat = pd.Field(
        LOSSY_METAL_DEFAULT_TOLERANCE_RMS,
        title="Tolerance In Fitting",
        description="Tolerance in fitting.",
    )

    frequency_sampling_points: pd.PositiveInt = pd.Field(
        LOSSY_METAL_DEFAULT_SAMPLING_FREQUENCY,
        title="Number Of Sampling Frequencies",
        description="Number of sampling frequencies used in fitting.",
    )

    log_sampling: bool = pd.Field(
        True,
        title="Frequencies Sampling In Log Scale",
        description="Whether to sample frequencies logarithmically (``True``),  "
        "or linearly (``False``).",
    )


class AbstractSurfaceRoughness(Tidy3dBaseModel):
    """Abstract class for modeling surface roughness of lossy metal."""

    @abstractmethod
    def roughness_correction_factor(
        self, frequency: ArrayFloat1D, skin_depths: ArrayFloat1D
    ) -> ArrayComplex1D:
        """Complex-valued roughness correction factor applied to surface impedance.

        Notes
        -----
            The roughness correction factor should be causal. It is multiplied to the
            surface impedance of the lossy metal to account for the effects of surface roughness.

        Parameters
        ----------
        frequency : ArrayFloat1D
            Frequency to evaluate roughness correction factor at (Hz).
        skin_depths : ArrayFloat1D
            Skin depths of the lossy metal that is frequency-dependent.

        Returns
        -------
        ArrayComplex1D
            The causal roughness correction factor evaluated at ``frequency``.
        """


class HammerstadSurfaceRoughness(AbstractSurfaceRoughness):
    """Modified Hammerstad surface roughness model. It's a popular model that works well
    under 5 GHz for surface roughness below 2 micrometer RMS.

    Note
    ----

        The power loss compared to smooth surface is described by:

        .. math::

            1 + (RF-1) \\frac{2}{\\pi}\\arctan(1.4\\frac{R_q^2}{\\delta^2})

        where :math:`\\delta` is skin depth, :math:`R_q` the RMS peak-to-vally height, and RF
        roughness factor.

    Note
    ----
    This model is based on:

        Y. Shlepnev, C. Nwachukwu, "Roughness characterization for interconnect analysis",
        2011 IEEE International Symposium on Electromagnetic Compatibility,
        (DOI: 10.1109/ISEMC.2011.6038367), 2011.

        V. Dmitriev-Zdorov, B. Simonovich, I. Kochikov, "A Causal Conductor Roughness Model
        and its Effect on Transmission Line Characteristics", Signal Integrity Journal, 2018.
    """

    rq: pd.PositiveFloat = pd.Field(
        ...,
        title="RMS Peak-to-Valley Height",
        description="RMS peak-to-valley height (Rq) of the surface roughness.",
        units=MICROMETER,
    )

    roughness_factor: float = pd.Field(
        2.0,
        title="Roughness Factor",
        description="Expected maximal increase in conductor losses due to roughness effect. "
        "Value 2 gives the classic Hammerstad equation.",
        gt=1.0,
    )

    def roughness_correction_factor(
        self, frequency: ArrayFloat1D, skin_depths: ArrayFloat1D
    ) -> ArrayComplex1D:
        """Complex-valued roughness correction factor applied to surface impedance.

        Notes
        -----
            The roughness correction factor should be causal. It is multiplied to the
            surface impedance of the lossy metal to account for the effects of surface roughness.

        Parameters
        ----------
        frequency : ArrayFloat1D
            Frequency to evaluate roughness correction factor at (Hz).
        skin_depths : ArrayFloat1D
            Skin depths of the lossy metal that is frequency-dependent.

        Returns
        -------
        ArrayComplex1D
            The causal roughness correction factor evaluated at ``frequency``.
        """
        normalized_laplace = -1.4j * (self.rq / skin_depths) ** 2
        sqrt_normalized_laplace = np.sqrt(normalized_laplace)
        causal_response = np.log(
            1 + 2 * sqrt_normalized_laplace / (1 + normalized_laplace)
        ) + 2 * np.arctan(sqrt_normalized_laplace)
        return 1 + (self.roughness_factor - 1) / np.pi * causal_response


class HuraySurfaceRoughness(AbstractSurfaceRoughness):
    """Huray surface roughness model.

    Note
    ----

        The power loss compared to smooth surface is described by:

        .. math::

            \\frac{A_{matte}}{A_{flat}} + \\frac{3}{2}\\sum_i f_i/[1+\\frac{\\delta}{r_i}+\\frac{\\delta^2}{2r_i^2}]

        where :math:`\\delta` is skin depth, :math:`r_i` the radius of sphere,
        :math:`\\frac{A_{matte}}{A_{flat}}` the relative area of the matte compared to flat surface,
        and :math:`f_i=N_i4\\pi r_i^2/A_{flat}` the ratio of total sphere
        surface area (number of spheres :math:`N_i` times the individual sphere surface area)
        to the flat surface area.

    Note
    ----
    This model is based on:

        J. Eric Bracken, "A Causal Huray Model for Surface Roughness", DesignCon, 2012.
    """

    relative_area: pd.PositiveFloat = pd.Field(
        1,
        title="Relative Area",
        description="Relative area of the matte base compared to a flat surface",
    )

    coeffs: Tuple[Tuple[pd.PositiveFloat, pd.PositiveFloat], ...] = pd.Field(
        ...,
        title="Coefficients for surface ratio and sphere radius",
        description="List of (:math:`f_i, r_i`) values for model, where :math:`f_i` is "
        "the ratio of total sphere surface area to the flat surface area, and :math:`r_i` "
        "the radius of the sphere.",
        units=(None, MICROMETER),
    )

    @classmethod
    def from_cannonball_huray(cls, radius: float) -> HuraySurfaceRoughness:
        """Construct a Cannonball-Huray model.

        Note
        ----

            The power loss compared to smooth surface is described by:

            .. math::

                1 + \\frac{7\\pi}{3} \\frac{1}{1+\\frac{\\delta}{r}+\\frac{\\delta^2}{2r^2}}

        Parameters
        ----------
        radius : float
            Radius of the sphere.

        Returns
        -------
        HuraySurfaceRoughness
            The Huray surface roughness model.
        """
        return cls(relative_area=1, coeffs=[(14.0 / 9 * np.pi, radius)])

    def roughness_correction_factor(
        self, frequency: ArrayFloat1D, skin_depths: ArrayFloat1D
    ) -> ArrayComplex1D:
        """Complex-valued roughness correction factor applied to surface impedance.

        Notes
        -----
            The roughness correction factor should be causal. It is multiplied to the
            surface impedance of the lossy metal to account for the effects of surface roughness.

        Parameters
        ----------
        frequency : ArrayFloat1D
            Frequency to evaluate roughness correction factor at (Hz).
        skin_depths : ArrayFloat1D
            Skin depths of the lossy metal that is frequency-dependent.

        Returns
        -------
        ArrayComplex1D
            The causal roughness correction factor evaluated at ``frequency``.
        """

        correction = self.relative_area
        for f, r in self.coeffs:
            normalized_laplace = -2j * (r / skin_depths) ** 2
            sqrt_normalized_laplace = np.sqrt(normalized_laplace)
            correction += 1.5 * f / (1 + 1 / sqrt_normalized_laplace)
        return correction


SurfaceRoughnessType = Union[HammerstadSurfaceRoughness, HuraySurfaceRoughness]


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

    allow_gain: Literal[False] = pd.Field(
        False,
        title="Allow gain medium",
        description="Allow the medium to be active. Caution: "
        "simulations with a gain medium are unstable, and are likely to diverge."
        "Simulations where 'allow_gain' is set to 'True' will still be charged even if "
        "diverged. Monitor data up to the divergence point will still be returned and can be "
        "useful in some cases.",
    )

    permittivity: Literal[1] = pd.Field(
        1.0, title="Permittivity", description="Relative permittivity.", units=PERMITTIVITY
    )

    roughness: SurfaceRoughnessType = pd.Field(
        None,
        title="Surface Roughness Model",
        description="Surface roughness model that applies a frequency-dependent scaling "
        "factor to surface impedance.",
        discriminator=TYPE_TAG_STR,
    )

    frequency_range: FreqBound = pd.Field(
        ...,
        title="Frequency Range",
        description="Frequency range of validity for the medium.",
        units=(HERTZ, HERTZ),
    )

    fit_param: SurfaceImpedanceFitterParam = pd.Field(
        SurfaceImpedanceFitterParam(),
        title="Fitting Parameters For Surface Impedance",
        description="Parameters for fitting surface impedance divided by (-1j * omega) over "
        "the frequency range using pole-residue pair model.",
    )

    @pd.validator("frequency_range")
    def _validate_frequency_range(cls, val):
        """Validate that frequency range is finite and non-zero."""
        for freq in val:
            if not np.isfinite(freq):
                raise ValidationError("Values in 'frequency_range' must be finite.")
            if freq <= 0:
                raise ValidationError("Values in 'frequency_range' must be positive.")
        return val

    @pd.validator("conductivity", always=True)
    def _positive_conductivity(cls, val):
        """Assert conductivity>0."""
        if val <= 0:
            raise ValidationError("For lossy metal, 'conductivity' must be positive. ")
        return val

    @cached_property
    def _fitting_result(self) -> Tuple[PoleResidue, float]:
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

    def surface_impedance(self, frequencies: ArrayFloat1D):
        """Computing surface impedance including surface roughness effects."""
        # compute complex-valued skin depth
        n, k = self.nk_model(frequencies)

        # with surface roughness effects
        correction = 1.0
        if self.roughness is not None:
            skin_depths = 1 / np.sqrt(np.pi * frequencies * MU_0 * self.conductivity)
            correction = self.roughness.roughness_correction_factor(frequencies, skin_depths)

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

    def eps_diagonal_numerical(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor for numerical considerations
        such as meshing and runtime estimation.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[complex, complex, complex]
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
