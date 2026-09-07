"""Surface-roughness models used by lossy conductors."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import autograd.numpy as np
from pydantic import (
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import (
    MICROMETER,
)

LOSSY_METAL_DEFAULT_SAMPLING_FREQUENCY = 20
LOSSY_METAL_DEFAULT_MAX_POLES = 5
LOSSY_METAL_DEFAULT_TOLERANCE_RMS = 1e-3

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.types import (
        ArrayComplex1D,
        ArrayFloat1D,
    )


class SurfaceImpedanceFitterParam(Tidy3dBaseModel):
    """Advanced parameters for fitting surface impedance of a :class:`.LossyMetalMedium`.
    Internally, the quantity to be fitted is surface impedance divided by ``-1j * \\omega``.
    """

    max_num_poles: PositiveInt = Field(
        default=LOSSY_METAL_DEFAULT_MAX_POLES,
        title="Maximal Number Of Poles",
        description="Maximal number of poles in complex-conjugate pole residue model for "
        "fitting surface impedance.",
    )

    tolerance_rms: NonNegativeFloat = Field(
        default=LOSSY_METAL_DEFAULT_TOLERANCE_RMS,
        title="Tolerance In Fitting",
        description="Tolerance in fitting.",
    )

    frequency_sampling_points: PositiveInt = Field(
        default=LOSSY_METAL_DEFAULT_SAMPLING_FREQUENCY,
        title="Number Of Sampling Frequencies",
        description="Number of sampling frequencies used in fitting.",
    )

    log_sampling: bool = Field(
        default=True,
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

    rq: PositiveFloat = Field(
        title="RMS Peak-to-Valley Height",
        description="RMS peak-to-valley height (Rq) of the surface roughness.",
        json_schema_extra={"units": MICROMETER},
    )

    roughness_factor: float = Field(
        default=2.0,
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

    relative_area: PositiveFloat = Field(
        default=1,
        title="Relative Area",
        description="Relative area of the matte base compared to a flat surface",
    )

    coeffs: tuple[tuple[PositiveFloat, PositiveFloat], ...] = Field(
        title="Coefficients for surface ratio and sphere radius",
        description="List of (:math:`f_i, r_i`) values for model, where :math:`f_i` is "
        "the ratio of total sphere surface area to the flat surface area, and :math:`r_i` "
        "the radius of the sphere.",
        json_schema_extra={"units": (None, MICROMETER)},
    )

    @classmethod
    def from_cannonball_huray(cls, radius: float) -> Self:
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


SurfaceRoughnessType = HammerstadSurfaceRoughness | HuraySurfaceRoughness
