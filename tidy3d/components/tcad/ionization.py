"""Incomplete ionization of the dopants of a semiconductor in the charge solver."""

from __future__ import annotations

from pydantic import Field, NonNegativeFloat, PositiveFloat

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.constants import ELECTRON_VOLT, PERCMCUBE

# Sites a MultiSiteIonization may hold, as the solver evaluates them.
MAX_IONIZATION_SITES = 4


class PearsonBardeenIonization(Tidy3dBaseModel):
    """Incomplete ionization of a dopant level whose depth can decrease with the doping.

    Notes
    -----
        The ionized concentration of donors of concentration :math:`N_d` is

        .. math::

            N_d^+ = \\frac{N_d}{1 + g \\exp\\left(\\frac{E_{Fn} - E_c + \\Delta E}{k_B T}\\right)},
            \\qquad \\Delta E = \\max\\left(\\Delta E_0 - \\alpha N_d^{1/3}, 0\\right),

        and that of acceptors of concentration :math:`N_a` is

        .. math::

            N_a^- = \\frac{N_a}{1 + g \\exp\\left(\\frac{E_v - E_{Fp} + \\Delta E}{k_B T}\\right)},
            \\qquad \\Delta E = \\max\\left(\\Delta E_0 - \\alpha N_a^{1/3}, 0\\right),

        where :math:`\\Delta E_0` is the depth of the level below the conduction band edge
        :math:`E_c` (donors) or above the valence band edge :math:`E_v` (acceptors) at low doping,
        :math:`g` its degeneracy factor and :math:`T` the lattice temperature. The lowering
        :math:`\\alpha N^{1/3}` [1]_ describes the screening of the impurity potential as the
        doping rises; ``alpha=0`` gives a level of constant depth. A level at the band edge stays
        largely occupied in degenerately doped material, so this law does not make heavily doped
        material fully ionized; :class:`AltermattSchenkIonization` does, for silicon.

        How the level exchanges carriers and where the ionized concentrations enter are
        described in :class:`SemiconductorMedium`.

        .. [1] G. L. Pearson and J. Bardeen, "Electrical properties of pure silicon and silicon
               alloys containing boron and phosphorus," Physical Review 75, 865 (1949).

    Note
    ----
    Supported only by the accelerated charge solver.

    Example
    -------
        >>> import tidy3d as td
        >>> constant_level = td.PearsonBardeenIonization(energy=0.045, degeneracy=2)
        >>> lowered_level = td.PearsonBardeenIonization(energy=0.220, degeneracy=4, alpha=2.32e-8)
    """

    energy: NonNegativeFloat = Field(
        allow_inf_nan=False,
        title="Ionization energy",
        description="Depth :math:`\\Delta E_0` of the dopant level below the conduction band "
        "edge for donors, or above the valence band edge for acceptors, at low doping.",
        json_schema_extra={"units": ELECTRON_VOLT},
    )

    degeneracy: PositiveFloat = Field(
        allow_inf_nan=False,
        title="Degeneracy factor",
        description="Degeneracy factor :math:`g` of the dopant level, commonly 2 for donors "
        "and 4 for acceptors.",
    )

    alpha: NonNegativeFloat = Field(
        default=0.0,
        allow_inf_nan=False,
        title="Lowering coefficient",
        description="Coefficient :math:`\\alpha` of the lowering of the level depth with the "
        "cube root of the dopant concentration. Zero gives a level of constant depth.",
        json_schema_extra={"units": "eV*cm"},
    )


class AltermattSchenkIonization(Tidy3dBaseModel):
    """Incomplete ionization of a dopant in silicon, fully ionized above the Mott transition.

    Notes
    -----
        The ionized concentration of donors of concentration :math:`N_d` is [1]_

        .. math::

            N_d^+ = N_d \\left(1 - \\frac{b\\, x}{1 + x}\\right), \\qquad
            x = g \\exp\\left(\\frac{E_{Fn} - E_c + \\Delta E}{k_B T}\\right),

        with the level depth below the conduction band edge :math:`E_c` and the share of the
        dopant states that are still bound

        .. math::

            \\Delta E = \\frac{\\Delta E_0}{1 + (N_d / N_{ref})^c}, \\qquad
            b = \\frac{1}{1 + (N_d / N_b)^d}.

        Acceptors of concentration :math:`N_a` follow the same law above the valence band edge
        :math:`E_v`, with :math:`x = g \\exp((E_v - E_{Fp} + \\Delta E)/k_B T)` and :math:`N_a` in
        :math:`\\Delta E` and :math:`b`. As the doping passes the Mott transition, the bound states
        merge with the band and :math:`b` falls to zero, so heavily doped silicon is fully
        ionized; ionization is least complete near the Mott transition. With Boltzmann statistics
        :math:`x = g\\, n\\, e^{\\Delta E / k_B T} / N_c`, which is Eq. (3a) of [1]_. The
        parameters of Table I there were fitted to measurements from 30 K to 300 K on
        uncompensated or weakly compensated silicon, and are:

        .. list-table::
           :widths: 16 16 16 12 16 12 12
           :header-rows: 1

           * - Dopant
             - ``energy`` [eV]
             - ``N_ref`` [cm⁻³]
             - ``c``
             - ``N_b`` [cm⁻³]
             - ``d``
             - ``degeneracy``
           * - P
             - 0.0455
             - 2.2e18
             - 2
             - 6e18
             - 2.3
             - 2
           * - As
             - 0.0537
             - 3e18
             - 1.5
             - 9e18
             - 1.8
             - 2
           * - B
             - 0.04439
             - 1.3e18
             - 1.4
             - 4.5e18
             - 2.4
             - 4

        The degeneracy factor here is the inverse of the :math:`g` of [1]_, which lists 1/2 for
        P and As and 1/4 for B.

        How the level exchanges carriers and where the ionized concentrations enter are
        described in :class:`SemiconductorMedium`.

        .. [1] A. Schenk, P. P. Altermatt, and B. Schmithüsen, "Physical model of incomplete
               ionization for silicon device simulation," Proceedings of SISPAD 2006, pp. 51-54.

    Note
    ----
    Supported only by the accelerated charge solver.

    Example
    -------
        >>> import tidy3d as td
        >>> phosphorus = td.AltermattSchenkIonization(
        ...     energy=0.0455, degeneracy=2, N_ref=2.2e18, c=2, N_b=6e18, d=2.3
        ... )
    """

    energy: NonNegativeFloat = Field(
        allow_inf_nan=False,
        title="Ionization energy",
        description="Depth :math:`\\Delta E_0` of the dopant level below the conduction band "
        "edge for donors, or above the valence band edge for acceptors, at low doping.",
        json_schema_extra={"units": ELECTRON_VOLT},
    )

    degeneracy: PositiveFloat = Field(
        allow_inf_nan=False,
        title="Degeneracy factor",
        description="Degeneracy factor :math:`g` of the dopant level, commonly 2 for donors "
        "and 4 for acceptors.",
    )

    N_ref: PositiveFloat = Field(
        allow_inf_nan=False,
        title="Screening concentration",
        description="Dopant concentration :math:`N_{ref}` of the decrease of the level depth.",
        json_schema_extra={"units": PERCMCUBE},
    )

    c: PositiveFloat = Field(
        allow_inf_nan=False,
        title="Screening exponent",
        description="Exponent :math:`c` of the decrease of the level depth.",
    )

    N_b: PositiveFloat = Field(
        allow_inf_nan=False,
        title="Delocalization concentration",
        description="Dopant concentration :math:`N_b` at which half of the dopant states have "
        "merged with the band.",
        json_schema_extra={"units": PERCMCUBE},
    )

    d: PositiveFloat = Field(
        allow_inf_nan=False,
        title="Delocalization exponent",
        description="Exponent :math:`d` of the share of the dopant states that are still bound.",
    )


SingleLevelIonizationType = PearsonBardeenIonization | AltermattSchenkIonization


class IonizationSite(Tidy3dBaseModel):
    """A lattice site of a dopant, with the share of the dopant on it and its level.

    Notes
    -----
        The example is nitrogen on the hexagonal site of 4H-SiC. Its level depth and lowering
        coefficient are those of Kagamihara et al. [1]_, as tabulated in Table XXV of [2]_.

        .. [1] S. Kagamihara, H. Matsuura, T. Hatakeyama, T. Watanabe, M. Kushibe, T. Shinohe and
               K. Arai, "Parameters required to simulate electric characteristics of SiC devices
               for n-type 4H-SiC," Journal of Applied Physics 96, 5601 (2004).

        .. [2] J. Burin et al., "TCAD parameters for 4H-SiC: a review," arXiv:2410.06798 (2024).

    Example
    -------
        >>> import tidy3d as td
        >>> hexagonal = td.IonizationSite(
        ...     fraction=0.5,
        ...     level=td.PearsonBardeenIonization(energy=0.0709, degeneracy=2, alpha=3.38e-8),
        ... )
    """

    fraction: float = Field(
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
        title="Fraction",
        description="Share of the dopant concentration on this site.",
    )

    level: SingleLevelIonizationType = Field(
        discriminator=TYPE_TAG_STR,
        title="Level",
        description="Ionization law of the dopant level on this site.",
    )


class MultiSiteIonization(Tidy3dBaseModel):
    """Incomplete ionization of a dopant spread over inequivalent lattice sites.

    Notes
    -----
        Each site holds the share ``fraction`` of the dopant, and its level ionizes by its own
        law, evaluated at the full dopant concentration. The ionized concentration is the sum
        over the sites, so for donors

        .. math::

            N_d^+ = \\sum_k f_k\\, N_{d,k}^+(N_d),

        where :math:`N_{d,k}^+(N_d)` is the ionized concentration that the law of site :math:`k`
        gives for the whole donor concentration :math:`N_d`. The fractions must add up to 1. An
        example is nitrogen in 4H-SiC, which sits on hexagonal and cubic sites with different
        ionization energies. The example below takes equal shares of the two sites, a
        degeneracy of 2 and the sum over sites of Eq. (88) of [2]_, with the level depths
        0.0709 eV and 0.1237 eV and the lowering coefficients 3.38e-8 eV cm and
        4.65e-8 eV cm of Kagamihara et al. [1]_, as tabulated in Table XXV of [2]_.

        .. [1] S. Kagamihara, H. Matsuura, T. Hatakeyama, T. Watanabe, M. Kushibe, T. Shinohe and
               K. Arai, "Parameters required to simulate electric characteristics of SiC devices
               for n-type 4H-SiC," Journal of Applied Physics 96, 5601 (2004).

        .. [2] J. Burin et al., "TCAD parameters for 4H-SiC: a review," arXiv:2410.06798 (2024).

    Note
    ----
    Supported only by the accelerated charge solver.

    Example
    -------
        >>> import tidy3d as td
        >>> nitrogen = td.MultiSiteIonization(
        ...     sites=(
        ...         td.IonizationSite(
        ...             fraction=0.5,
        ...             level=td.PearsonBardeenIonization(
        ...                 energy=0.0709, degeneracy=2, alpha=3.38e-8
        ...             ),
        ...         ),
        ...         td.IonizationSite(
        ...             fraction=0.5,
        ...             level=td.PearsonBardeenIonization(
        ...                 energy=0.1237, degeneracy=2, alpha=4.65e-8
        ...             ),
        ...         ),
        ...     )
        ... )
    """

    sites: tuple[IonizationSite, ...] = Field(
        min_length=1,
        max_length=MAX_IONIZATION_SITES,
        title="Sites",
        description="Lattice sites of the dopant, whose fractions add up to 1. At most "
        f"{MAX_IONIZATION_SITES} sites.",
    )


DopantIonizationModelType = (
    PearsonBardeenIonization | AltermattSchenkIonization | MultiSiteIonization
)
