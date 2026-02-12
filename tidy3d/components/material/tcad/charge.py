"""Defines heat material specifications"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from pydantic import Field, NonNegativeFloat, PositiveFloat, field_validator

from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.components.medium import AbstractMedium
from tidy3d.components.tcad.doping import ConstantDoping, DopingBoxType
from tidy3d.components.tcad.types import (
    BandGapNarrowingModelType,
    ConstantEffectiveDOS,
    ConstantEnergyBandGap,
    EffectiveDOSModelType,
    EnergyBandGapModelType,
    MobilityModelType,
    RecombinationModelType,
)
from tidy3d.constants import CONDUCTIVITY, ELECTRON_VOLT, PERCMCUBE, PERMITTIVITY
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.compat import Self


class AbstractChargeMedium(AbstractMedium):
    """Abstract class for Charge specifications
    Currently, permittivity is treated as a constant."""

    permittivity: float = Field(
        1.0,
        ge=1.0,
        title="Permittivity",
        description="Relative permittivity.",
        json_schema_extra={"units": PERMITTIVITY},
    )

    @property
    def charge(self) -> Self:
        """
        This means that a charge medium has been defined inherently within this solver medium.
        This provides interconnection with the :class:`~tidy3d.MultiPhysicsMedium` higher-dimensional classes.
        """
        return self

    def eps_model(self, frequency: float) -> complex:
        return self.permittivity

    def n_cfl(self) -> None:
        return None


class ChargeInsulatorMedium(AbstractChargeMedium):
    """
    Insulating medium. Conduction simulations will not solve for electric
    potential in a structure that has a medium with this ``charge``.

    Example
    -------
    >>> import tidy3d as td
    >>> solid = td.ChargeInsulatorMedium()
    >>> solid2 = td.ChargeInsulatorMedium(permittivity=1.1)

    Note
    ----
        A relative permittivity :math:`\\varepsilon` will be assumed 1 if no value is specified.
    """


class ChargeConductorMedium(AbstractChargeMedium):
    """Conductor medium for conduction simulations.

    Example
    -------
    >>> import tidy3d as td
    >>> solid = td.ChargeConductorMedium(conductivity=3)

    Note
    ----
        A relative permittivity will be assumed 1 if no value is specified.
    """

    conductivity: PositiveFloat = Field(
        title="Electric conductivity",
        description="Electric conductivity of material.",
        json_schema_extra={"units": CONDUCTIVITY},
    )


class SemiconductorMedium(AbstractChargeMedium):
    """
    This class is used to define semiconductors.

    Notes
    -----
    Semiconductors are associated with ``Charge`` simulations. During these simulations
    the Drift-Diffusion (DD) equations will be solved in semiconductors. In what follows, a
    description of the assumptions taken and its limitations is put forward.

    The iso-thermal DD equations are summarized here

    .. math::

        \\begin{equation}
                - \\nabla \\cdot \\left( \\varepsilon_0 \\varepsilon_r \\nabla \\psi \\right) = q
            \\left( p - n + N_d^+ - N_a^- \\right)
        \\end{equation}

    .. math::

        \\begin{equation}
            q \\frac{\\partial n}{\\partial t} = \\nabla \\cdot \\mathbf{J_n} - qR
        \\end{equation}

    .. math::

        \\begin{equation}
            q \\frac{\\partial p}{\\partial t} = -\\nabla \\cdot \\mathbf{J_p} - qR
        \\end{equation}

    The system defaults to isothermal conditions at :math:`T=300 \\text{K}`.
    Non-isothermal (self-heating) simulations can be enabled via appropriate
    analysis specifications (e.g., ``SteadyChargeDCAnalysis``).

    The above system requires the definition of the flux functions (free carrier current density), :math:`\\mathbf{J_n}` and
    :math:`\\mathbf{J_p}`. We consider the usual form

    .. math::

        \\begin{equation}
             \\mathbf{J_n} = q \\mu_n \\mathbf{F_{n}} + q D_n \\nabla n
        \\end{equation}


    .. math::

        \\begin{equation}
             \\mathbf{J_p} = q \\mu_p \\mathbf{F_{p}} - q D_p \\nabla p
        \\end{equation}


    where we simplify the effective field defined in [1]_ to

    .. math::

        \\begin{equation}
            \\mathbf{F_{n,p}} = \\nabla \\psi
        \\end{equation}

    i.e., we are not considering the effect of band-gap narrowing and degeneracy on the effective
    electric field :math:`\\mathbf{F_{n,p}}`. This is a good approximation for non-degenerate semiconductors.

    Let's explore how material properties are defined as class parameters or other classes.

     .. list-table::
       :widths: 25 25 75
       :header-rows: 1

       * - Symbol
         - Parameter Name
         - Description
       * - :math:`N_a`
         - ``N_a``
         - Ionized acceptors density
       * - :math:`N_d`
         - ``N_d``
         - Ionized donors density
       * - :math:`N_c`
         - ``N_c``
         - Effective density of states in the conduction band.
       * - :math:`N_v`
         - ``N_v``
         - Effective density of states in valence band.
       * - :math:`R`
         - ``R``
         - Generation-Recombination term.
       * - :math:`E_g`
         - ``E_g``
         - Bandgap Energy.
       * - :math:`\\Delta E_g`
         - ``delta_E_g``
         - Bandgap Narrowing.
       * - :math:`\\sigma`
         - ``conductivity``
         - Electrical conductivity.
       * - :math:`\\varepsilon_r`
         - ``permittivity``
         - Relative permittivity.
       * - :math:`q`
         - ``tidy3d.constants.Q_e``
         - Fundamental electron charge.

    Example
    -------
        >>> import tidy3d as td
        >>> default_Si = td.SemiconductorMedium(
        ...     N_c=td.ConstantEffectiveDOS(N=2.86e19),
        ...     N_v=td.ConstantEffectiveDOS(N=3.1e19),
        ...     E_g=td.ConstantEnergyBandGap(eg=1.11),
        ...     mobility_n=td.CaugheyThomasMobility(
        ...         mu_min=52.2,
        ...         mu=1471.0,
        ...         ref_N=9.68e16,
        ...         exp_N=0.68,
        ...         exp_1=-0.57,
        ...         exp_2=-2.33,
        ...         exp_3=2.4,
        ...         exp_4=-0.146,
        ...     ),
        ...     mobility_p=td.CaugheyThomasMobility(
        ...         mu_min=44.9,
        ...         mu=470.5,
        ...         ref_N=2.23e17,
        ...         exp_N=0.719,
        ...         exp_1=-0.57,
        ...         exp_2=-2.33,
        ...         exp_3=2.4,
        ...         exp_4=-0.146,
        ...     ),
        ...     R=([
        ...         td.ShockleyReedHallRecombination(
        ...             tau_n=3.3e-6,
        ...             tau_p=4e-6
        ...         ),
        ...         td.RadiativeRecombination(
        ...             r_const=1.6e-14
        ...         ),
        ...         td.AugerRecombination(
        ...             c_n=2.8e-31,
        ...             c_p=9.9e-32
        ...         ),
        ...     ]),
        ...     delta_E_g=td.SlotboomBandGapNarrowing(
        ...         v1=6.92 * 1e-3,
        ...         n2=1.3e17,
        ...         c2=0.5,
        ...         min_N=1e15,
        ...     ),
        ...     N_a=[td.ConstantDoping(concentration=1e15)],
        ...     N_d=[td.ConstantDoping(concentration=1e15)]
        ... )


    Warning
    -------
        Current limitations of the formulation include:

        - Boltzmann statistics by default; Fermi-Dirac statistics available
        - Isothermal by default at T=300K; self-heating available through analysis spec
        - Steady-state DC and small-signal AC analyses supported
        - Dopants are considered to be fully ionized

    Note
    ----
        - Both :math:`N_a` and :math:`N_d` can be either a positive number or an ``xarray.DataArray``.
        - Default values for parameters and models are those appropriate for Silicon.
        - The current implementation is a good approximation for non-degenerate semiconductors.


    .. [1] Schroeder, D., T. Ostermann, and O. Kalz. "Comparison of transport models far the simulation of degenerate semiconductors." Semiconductor science and technology 9.4 (1994): 364.

    """

    N_c: Union[EffectiveDOSModelType, PositiveFloat] = Field(
        title="Effective density of electron states",
        description=":math:`N_c` Effective density of states in the conduction band.",
        json_schema_extra={"units": PERCMCUBE},
    )

    N_v: Union[EffectiveDOSModelType, PositiveFloat] = Field(
        title="Effective density of hole states",
        description=":math:`N_v` Effective density of states in the valence band.",
        json_schema_extra={"units": PERCMCUBE},
    )

    E_g: Union[EnergyBandGapModelType, PositiveFloat] = Field(
        title="Band-gap energy",
        description=":math:`E_g` Band-gap energy",
        json_schema_extra={"units": ELECTRON_VOLT},
    )

    mobility_n: MobilityModelType = Field(
        title="Mobility model for electrons",
        description="Mobility model for electrons",
    )

    mobility_p: MobilityModelType = Field(
        title="Mobility model for holes",
        description="Mobility model for holes",
    )

    R: tuple[RecombinationModelType, ...] = Field(
        (),
        title="Generation-Recombination models",
        description="Array containing the R models to be applied to the material.",
    )

    delta_E_g: Optional[BandGapNarrowingModelType] = Field(
        None,
        title="Bandgap narrowing model.",
        description=":math:`\\Delta E_g` Bandgap narrowing model.",
        json_schema_extra={"units": ELECTRON_VOLT},
    )

    N_a: Union[
        tuple[DopingBoxType, ...],
        list[DopingBoxType],
        SpatialDataArray,
        NonNegativeFloat,
    ] = Field(
        (),
        title="Doping: Acceptor concentration",
        description="Concentration of acceptor impurities, which create mobile holes, resulting in p-type material. "
        "Can be specified as a single float for uniform doping, a :class:`SpatialDataArray` for a custom profile, "
        "or a tuple/list of geometric shapes to define specific doped regions.",
        json_schema_extra={"units": PERCMCUBE},
    )

    N_d: Union[
        tuple[DopingBoxType, ...],
        list[DopingBoxType],
        SpatialDataArray,
        NonNegativeFloat,
    ] = Field(
        (),
        title="Doping: Donor concentration",
        description="Concentration of donor impurities, which create mobile electrons, resulting in n-type material. "
        "Can be specified as a single float for uniform doping, a :class:`SpatialDataArray` for a custom profile, "
        "or a tuple/list of geometric shapes to define specific doped regions.",
        json_schema_extra={"units": PERCMCUBE},
    )

    # DEPRECATION VALIDATORS
    @field_validator("N_c")
    @classmethod
    def check_nc_uses_model(cls, val: Union[EffectiveDOSModelType, float]) -> EffectiveDOSModelType:
        """Issue deprecation warning if float is provided"""
        if isinstance(val, (float, int)):
            log.warning(
                "Passing a float to 'N_c' is deprecated and will be removed in future versions. "
                "Please use 'ConstantEffectiveDOS' instead."
            )
            return ConstantEffectiveDOS(N=val)
        return val

    @field_validator("N_v")
    @classmethod
    def check_nv_uses_model(cls, val: Union[EffectiveDOSModelType, float]) -> EffectiveDOSModelType:
        """Issue deprecation warning if float is provided"""
        if isinstance(val, (float, int)):
            log.warning(
                "Passing a float to 'N_v' is deprecated and will be removed in future versions. "
                "Please use 'ConstantEffectiveDOS' instead."
            )
            return ConstantEffectiveDOS(N=val)
        return val

    @field_validator("E_g")
    @classmethod
    def check_eg_uses_model(
        cls, val: Union[EnergyBandGapModelType, float]
    ) -> EnergyBandGapModelType:
        """Issue deprecation warning if float is provided"""
        if isinstance(val, (float, int)):
            log.warning(
                "Passing a float to 'E_g' is deprecated and will be removed in future versions. "
                "Please use 'ConstantEnergyBandGap' instead."
            )
            return ConstantEnergyBandGap(eg=val)
        return val

    @field_validator("N_d")
    @classmethod
    def check_nd_uses_model(
        cls,
        val: Union[
            NonNegativeFloat, SpatialDataArray, tuple[DopingBoxType, ...], list[DopingBoxType]
        ],
    ) -> Union[SpatialDataArray, tuple[DopingBoxType, ...]]:
        """Issue deprecation warning if float is provided"""
        if isinstance(val, list):
            return tuple(val)
        if isinstance(val, (float, int)):
            log.warning(
                "Passing a float to 'N_d' is deprecated and will be removed in future versions. "
                f"Please use a list of 'DopingBoxType' instead, e.g., [ConstantDoping(concentration={val})]."
            )
            return (ConstantDoping(concentration=val),)
        return val

    @field_validator("N_a")
    @classmethod
    def check_na_uses_model(
        cls,
        val: Union[
            NonNegativeFloat, SpatialDataArray, tuple[DopingBoxType, ...], list[DopingBoxType]
        ],
    ) -> Union[SpatialDataArray, tuple[DopingBoxType, ...]]:
        """Issue deprecation warning if float is provided"""
        if isinstance(val, list):
            return tuple(val)
        if isinstance(val, (float, int)):
            log.warning(
                "Passing a float to 'N_a' is deprecated and will be removed in future versions. "
                f"Please use a list of 'DopingBoxType' instead, e.g., [ConstantDoping(concentration={val})]."
            )
            return (ConstantDoping(concentration=val),)
        return val
