from __future__ import annotations

from typing import Dict, List, Tuple, Union

import autograd.numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import cached_property, skip_if_fields_missing
from tidy3d.components.geometry.base import Geometry
from tidy3d.components.types import TYPE_TAG_STR, Ax, Axis
from tidy3d.components.viz import add_ax_if_none
from tidy3d.exceptions import ValidationError
from tidy3d.log import log

from .anisotropic import (
    AnisotropicMedium,
    AnisotropicMediumFromMedium2D,
    CustomAnisotropicMedium,
    FullyAnisotropicMedium,
)
from .base import AbstractMedium
from .dispersionless import PEC, CustomMedium, Medium, PECMedium
from .dispersive import (
    CustomDebye,
    CustomDrude,
    CustomLorentz,
    CustomPoleResidue,
    CustomSellmeier,
    Debye,
    Drude,
    Lorentz,
    PoleResidue,
    Sellmeier,
)
from .dispersive_abc import DispersiveMedium
from .lossy_metal import LossyMetalMedium
from .perturbation import PerturbationMedium, PerturbationPoleResidue
from .types import IsotropicUniformMediumType
from .utils import ensure_freq_in_range

MediumType3D = Union[
    Medium,
    AnisotropicMedium,
    PECMedium,
    PoleResidue,
    Sellmeier,
    Lorentz,
    Debye,
    Drude,
    FullyAnisotropicMedium,
    CustomMedium,
    CustomPoleResidue,
    CustomSellmeier,
    CustomLorentz,
    CustomDebye,
    CustomDrude,
    CustomAnisotropicMedium,
    PerturbationMedium,
    PerturbationPoleResidue,
    LossyMetalMedium,
]


class Medium2D(AbstractMedium):
    """2D diagonally anisotropic medium.

    Notes
    -----

        Only diagonal anisotropy is currently supported.

    Example
    -------
    >>> drude_medium = Drude(eps_inf=2.0, coeffs=[(1,2), (3,4)])
    >>> medium2d = Medium2D(ss=drude_medium, tt=drude_medium)

    """

    ss: IsotropicUniformMediumType = pd.Field(
        ...,
        title="SS Component",
        description="Medium describing the ss-component of the diagonal permittivity tensor. "
        "The ss-component refers to the in-plane dimension of the medium that is the first "
        "component in order of 'x', 'y', 'z'. "
        "If the 2D material is normal to the y-axis, for example, then this determines the "
        "xx-component of the corresponding 3D medium.",
        discriminator=TYPE_TAG_STR,
    )

    tt: IsotropicUniformMediumType = pd.Field(
        ...,
        title="TT Component",
        description="Medium describing the tt-component of the diagonal permittivity tensor. "
        "The tt-component refers to the in-plane dimension of the medium that is the second "
        "component in order of 'x', 'y', 'z'. "
        "If the 2D material is normal to the y-axis, for example, then this determines the "
        "zz-component of the corresponding 3D medium.",
        discriminator=TYPE_TAG_STR,
    )

    @pd.validator("modulation_spec", always=True)
    def _validate_modulation_spec(cls, val):
        """Check compatibility with modulation_spec."""
        if val is not None:
            raise ValidationError(
                f"A 'modulation_spec' of class {type(val)} is not "
                f"currently supported for medium class {cls}."
            )
        return val

    @skip_if_fields_missing(["ss"])
    @pd.validator("tt", always=True)
    def _validate_inplane_pec(cls, val, values):
        """ss/tt components must be both PEC or non-PEC."""
        if isinstance(val, PECMedium) != isinstance(values["ss"], PECMedium):
            raise ValidationError(
                "Materials describing ss- and tt-components must be "
                "either both 'PECMedium', or non-'PECMedium'."
            )
        return val

    @classmethod
    def _weighted_avg(
        cls, meds: List[IsotropicUniformMediumType], weights: List[float]
    ) -> Union[PoleResidue, PECMedium]:
        """Average ``meds`` with weights ``weights``."""
        eps_inf = 1
        poles = []
        for med, weight in zip(meds, weights):
            if isinstance(med, DispersiveMedium):
                pole_res = med.pole_residue
                eps_inf += weight * (med.pole_residue.eps_inf - 1)
            elif isinstance(med, Medium):
                pole_res = PoleResidue.from_medium(med)
                eps_inf += weight * (med.permittivity - 1)
            elif isinstance(med, PECMedium):
                # special treatment for PEC
                return med
            else:
                raise ValidationError("Invalid medium type for the components of 'Medium2D'.")
            poles += [(a, weight * c) for (a, c) in pole_res.poles if c != 0.0]
        return PoleResidue(eps_inf=np.real(eps_inf), poles=poles)

    def volumetric_equivalent(
        self,
        axis: Axis,
        adjacent_media: Tuple[MediumType3D, MediumType3D],
        adjacent_dls: Tuple[float, float],
    ) -> AnisotropicMedium:
        """Produces a 3D volumetric equivalent medium. The new medium has thickness equal to
        the average of the ``dls`` in the ``axis`` direction.
        The ss and tt components of the 2D material are mapped in order onto the xx, yy, and
        zz components of the 3D material, excluding the ``axis`` component. The conductivity
        and residues (in the case of a dispersive 2D material) are rescaled by ``1/dl``.
        The neighboring media ``neighbors`` enter in as a background for the resulting
        volumetric equivalent.


        Parameters
        ----------
        axis : Axis
            Index (0, 1, or 2 for x, y, or z respectively) of the normal direction to the
            2D material.
        adjacent_media : Tuple[MediumType3D, MediumType3D]
            The neighboring media on either side of the 2D material.
            The first element is directly on the - side of the 2D material in the supplied axis,
            and the second element is directly on the + side.
        adjacent_dls : Tuple[float, float]
            Each dl represents twice the thickness of the desired volumetric model on the
            respective side of the 2D material.

        Returns
        -------
        :class:`.AnisotropicMedium`
            The 3D material corresponding to this 2D material.
        """

        def get_component(med: MediumType3D, comp: Axis) -> IsotropicUniformMediumType:
            """Extract the ``comp`` component of ``med``."""
            if isinstance(med, AnisotropicMedium):
                dim = "xyz"[comp]
                element_name = dim + dim
                return med.elements[element_name]
            return med

        def get_background(comp: Axis) -> PoleResidue:
            """Get the background medium appropriate for the ``comp`` component."""
            meds = [get_component(med=med, comp=comp) for med in adjacent_media]
            # the Yee site for the E field in the normal direction is fully contained
            # in the medium on the + side
            if comp == axis:
                return meds[1]
            weights = np.array(adjacent_dls) / np.sum(adjacent_dls)
            return self._weighted_avg(meds, weights)

        dl = (adjacent_dls[0] + adjacent_dls[1]) / 2
        media_bg = [get_background(comp=i) for i in range(3)]

        # perform weighted average of planar media transverse dimensions with the
        # respective background media
        media_fg_plane = list(self.elements.values())
        _, media_bg_plane = Geometry.pop_axis(media_bg, axis=axis)
        media_fg_weighted = [
            self._weighted_avg([media_bg, media_fg], [1, 1 / dl])
            for media_bg, media_fg in zip(media_bg_plane, media_fg_plane)
        ]

        # combine the two weighted, planar media with the background medium and put in the xyz basis
        media_3d = Geometry.unpop_axis(
            ax_coord=media_bg[axis], plane_coords=media_fg_weighted, axis=axis
        )
        media_3d_kwargs = {dim + dim: medium for dim, medium in zip("xyz", media_3d)}
        return AnisotropicMediumFromMedium2D(
            **media_3d_kwargs, frequency_range=self.frequency_range
        )

    def to_anisotropic_medium(self, axis: Axis, thickness: float) -> AnisotropicMedium:
        """Generate a 3D :class:`.AnisotropicMedium` equivalent of a given thickness.

        Parameters
        ----------
        axis: Axis
            The normal axis to the 2D medium.
        thickness: float
            The thickness of the desired 3D medium.

        Returns
        -------
        :class:`.AnisotropicMedium`
            The 3D equivalent of this 2D medium.
        """
        media = list(self.elements.values())
        media_weighted = [self._weighted_avg([medium], [1 / thickness]) for medium in media]
        media_3d = Geometry.unpop_axis(ax_coord=Medium(), plane_coords=media_weighted, axis=axis)
        media_3d_kwargs = {dim + dim: medium for dim, medium in zip("xyz", media_3d)}
        return AnisotropicMedium(**media_3d_kwargs, frequency_range=self.frequency_range)

    def to_pole_residue(self, thickness: float) -> PoleResidue:
        """Generate a :class:`.PoleResidue` equivalent of a given thickness.
        The 2D medium to be isotropic in-plane (otherwise the components are averaged).

        Parameters
        ----------
        thickness: float
            The thickness of the desired 3D medium.

        Returns
        -------
        :class:`.PoleResidue`
            The 3D equivalent pole residue model of this 2D medium.
        """
        return self._weighted_avg(
            [self.ss, self.tt], [1 / (2 * thickness), 1 / (2 * thickness)]
        ).updated_copy(frequency_range=self.frequency_range)

    def to_medium(self, thickness: float) -> Medium:
        """Generate a :class:`.Medium` equivalent of a given thickness.
        The 2D medium must be isotropic in-plane (otherwise the components are averaged)
        and non-dispersive besides a constant conductivity.

        Parameters
        ----------
        thickness: float
            The thickness of the desired 3D medium.

        Returns
        -------
        :class:`.Medium`
            The 3D equivalent of this 2D medium.
        """
        from . import PEC

        if self.is_pec:
            return PEC
        return self.to_pole_residue(thickness=thickness).to_medium()

    @classmethod
    def from_medium(cls, medium: Medium, thickness: float) -> Medium2D:
        """Generate a :class:`.Medium2D` equivalent of a :class:`.Medium`
        with a given thickness.

        Parameters
        ----------
        medium: :class:`.Medium`
            The 3D medium to convert.
        thickness : float
            The thickness of the 3D material.

        Returns
        -------
        :class:`.Medium2D`
            The 2D equivalent of the given 3D medium.
        """
        med = cls._weighted_avg([medium], [thickness])
        return Medium2D(ss=med, tt=med, frequency_range=medium.frequency_range)

    @classmethod
    def from_dispersive_medium(cls, medium: DispersiveMedium, thickness: float) -> Medium2D:
        """Generate a :class:`.Medium2D` equivalent of a :class:`.DispersiveMedium`
        with a given thickness.

        Parameters
        ----------
        medium: :class:`.DispersiveMedium`
            The 3D dispersive medium to convert.
        thickness : float
            The thickness of the 3D material.

        Returns
        -------
        :class:`.Medium2D`
            The 2D equivalent of the given 3D medium.
        """
        med = cls._weighted_avg([medium], [thickness])
        return Medium2D(ss=med, tt=med, frequency_range=medium.frequency_range, name=medium.name)

    @classmethod
    def from_anisotropic_medium(
        cls, medium: AnisotropicMedium, axis: Axis, thickness: float
    ) -> Medium2D:
        """Generate a :class:`.Medium2D` equivalent of a :class:`.AnisotropicMedium`
        with given normal axis and thickness. The ``ss`` and ``tt`` components of the resulting
        2D medium correspond to the first of the ``xx``, ``yy``, and ``zz`` components of
        the 3D medium, with the ``axis`` component removed.

        Parameters
        ----------
        medium: :class:`.AnisotropicMedium`
            The 3D anisotropic medium to convert.
        axis: :class:`.Axis`
            The normal axis to the 2D material.
        thickness : float
            The thickness of the 3D material.

        Returns
        -------
        :class:`.Medium2D`
            The 2D equivalent of the given 3D medium.
        """
        media = list(medium.elements.values())
        _, media_plane = Geometry.pop_axis(media, axis=axis)
        media_plane_scaled = []
        for _, med in enumerate(media_plane):
            media_plane_scaled.append(cls._weighted_avg([med], [thickness]))
        media_kwargs = {dim + dim: medium for dim, medium in zip("st", media_plane_scaled)}
        return Medium2D(**media_kwargs, frequency_range=medium.frequency_range)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""
        return np.mean(self.eps_diagonal(frequency=frequency), axis=0)

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor as a function of frequency."""
        log.warning(
            "The permittivity of a 'Medium2D' is unphysical. "
            "Use 'Medium2D.to_anisotropic_medium' or 'Medium2D.to_pole_residue' first "
            "to obtain the physical refractive index."
        )

        eps_ss = self.ss.eps_model(frequency)
        eps_tt = self.tt.eps_model(frequency)
        return (eps_ss, eps_tt)

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
    def plot(self, freqs: float, ax: Ax = None) -> Ax:
        """Plot n, k of a :class:`.Medium` as a function of frequency."""
        log.warning(
            "The refractive index of a 'Medium2D' is unphysical. "
            "Use 'Medium2D.plot_sigma' instead to plot surface conductivity, or call "
            "'Medium2D.to_anisotropic_medium' or 'Medium2D.to_pole_residue' first "
            "to obtain the physical refractive index."
        )

        freqs = np.array(freqs)
        freqs_thz = freqs / 1e12

        for label, medium_component in self.elements.items():
            eps_complex = medium_component.eps_model(freqs)
            n, k = AbstractMedium.eps_complex_to_nk(eps_complex)
            ax.plot(freqs_thz, n, label=f"n, eps_{label}")
            ax.plot(freqs_thz, k, label=f"k, eps_{label}")

        ax.set_xlabel("frequency (THz)")
        ax.set_title("medium dispersion")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    @add_ax_if_none
    def plot_sigma(self, freqs: float, ax: Ax = None) -> Ax:
        """Plot the surface conductivity of the 2D material."""
        freqs = np.array(freqs)
        freqs_thz = freqs / 1e12

        for label, medium_component in self.elements.items():
            sigma = medium_component.sigma_model(freqs)
            ax.plot(freqs_thz, np.real(sigma) * 1e6, label=f"Re($\\sigma$) ($\\mu$S), eps_{label}")
            ax.plot(freqs_thz, np.imag(sigma) * 1e6, label=f"Im($\\sigma$) ($\\mu$S), eps_{label}")

        ax.set_xlabel("frequency (THz)")
        ax.set_title("surface conductivity")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    @ensure_freq_in_range
    def sigma_model(self, freq: float) -> complex:
        """Complex-valued conductivity as a function of frequency.

        Parameters
        ----------
        freq: float
            Frequency to evaluate conductivity at (Hz).

        Returns
        -------
        complex
            Complex conductivity at this frequency.
        """
        return np.mean([self.ss.sigma_model(freq), self.tt.sigma_model(freq)], axis=0)

    @property
    def elements(self) -> Dict[str, IsotropicUniformMediumType]:
        """The diagonal elements of the 2D medium as a dictionary."""
        return dict(ss=self.ss, tt=self.tt)

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.
        """
        return 1.0

    @cached_property
    def is_pec(self):
        """Whether the medium is a PEC."""
        return any(isinstance(comp, PECMedium) for comp in self.elements.values())

    def is_comp_pec_2d(self, comp: Axis, axis: Axis):
        """Whether the medium is a PEC."""
        elements_3d = Geometry.unpop_axis(
            ax_coord=Medium(), plane_coords=self.elements.values(), axis=axis
        )
        return isinstance(elements_3d[comp], PECMedium)


MediumType = Union[MediumType3D, Medium2D, AnisotropicMediumFromMedium2D]
PEC2D = Medium2D(ss=PEC, tt=PEC)
