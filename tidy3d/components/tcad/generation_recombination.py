from __future__ import annotations

from typing import Literal, Union

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.constants import PERCMCUBE, SECOND


class FossumCarrierLifetime(Tidy3dBaseModel):
    """
    Parameters for the Fossum carrier lifetime model

    Notes
    -----

        This model expresses the carrier lifetime as a function of the temperature and doping concentration.

        .. math::

            \\tau = \\frac{\\tau_{300} \\left( T/300 \\right)^\\alpha_T}{A + B (N/N_0) + C (N/N_0)^\\alpha}

    Example
    -------
        >>> import tidy3d as td
        >>> default_Si = td.FossumCarrierLifetime(
        ...   tau_300=3.3e-6,
        ...   alpha_T=-0.5,
        ...   N0=7.1e15,
        ...   A=1,
        ...   B=0,
        ...   C=1,
        ...   alpha=1
        ... )

    References
    ----------

        Fossum, J. G., and D. S. Lee. "A physical model for the dependence of carrier lifetime on doping density in nondegenerate silicon." Solid-State Electronics 25.8 (1982): 741-747.

    """

    tau_300: pd.PositiveFloat = pd.Field(
        ..., title="Tau at 300K", description="Carrier lifetime at 300K", units=SECOND
    )

    alpha_T: float = pd.Field(
        ..., title="Exponent for thermal dependence", description="Exponent for thermal dependence"
    )

    N0: pd.PositiveFloat = pd.Field(
        ..., title="Reference concentration", description="Reference concentration", units=PERCMCUBE
    )

    A: float = pd.Field(..., title="Constant A", description="Constant A")

    B: float = pd.Field(..., title="Constant B", description="Constant B")

    C: float = pd.Field(..., title="Constant C", description="Constant C")

    alpha: float = pd.Field(..., title="Exponent constant", description="Exponent constant")


CarrierLifetimeType = Union[FossumCarrierLifetime]


class AugerRecombination(Tidy3dBaseModel):
    """
    Parameters for the Auger recombination model.

    Notes
    -----

        The Auger recombination rate ``R_A`` is primarily defined by the electrons and holes Auger recombination
        coefficients, :math:`C_n` and :math:`C_p`, respectively.

        .. math::

            R_A = \\left( C_n n + C_p p \\right) \\left( np - n_0 p_0 \\right)

    Example
    -------
        >>> import tidy3d as td
        >>> default_Si = td.AugerRecombination(
        ...   c_n=2.8e-31,
        ...   c_p=9.9e-32,
        ... )
    """

    c_n: pd.PositiveFloat = pd.Field(
        ...,
        title="Constant for electrons",
        description="Constant for electrons.",
        units="cm^6/s",
    )

    c_p: pd.PositiveFloat = pd.Field(
        ...,
        title="Constant for holes",
        description="Constant for holes.",
        units="cm^6/s",
    )


class RadiativeRecombination(Tidy3dBaseModel):
    """
    Defines the parameters for the radiative recombination model.

    Notes
    -----

        This is a direct recombination model primarily defined by a radiative recombination coefficient :math:`R_{\\text{rad}}`.

        .. math::

            R_{\\text{rad}} = C \\left( np - n_0 p_0 \\right)

    Example
    -------
        >>> import tidy3d as td
        >>> default_Si = td.RadiativeRecombination(
        ...   r_const=1.6e-14
        ... )
    """

    r_const: float = pd.Field(
        ...,
        title="Radiation constant",
        description="Radiation constant of the radiative recombination model.",
        units="cm^3/s",
    )


class ShockleyReedHallRecombination(Tidy3dBaseModel):
    """Defines the parameters for the Shockley-Reed-Hall (SRH) recombination model.

    Notes
    -----

        The recombination rate parameter from this model is defined from [1]_ as follows:

        .. math::

           R_{SRH} = \\frac{n p - n_0 p_0}{\\tau_p \\left(n + \\sqrt{n_0 p_0}\\right) + \\tau_n \\left(p + \\sqrt{n_0 p_0}\\right)}.

        Note that the electron and holes densities are defined within the :class:`SemiconductorMedium`. The electron
        lifetime :math:`\\tau_n` and hole lifetimes :math:`\\tau_p` need to be defined.


        .. [1] Schenk. A model for the field and temperature dependence of shockley-read-hall
               lifetimes in silicon. Solid-State Electronics, 35:1585–1596, 1992.

    Example
    -------
        >>> import tidy3d as td
        >>> default_Si = td.ShockleyReedHallRecombination(
        ...   tau_n=3.3e-6,
        ...   tau_p=4e-6,
        ... )

    Note
    ----
    Important considerations when using this model:

    - This model represents mid-gap traps Shockley-Reed-Hall recombination.
    """

    tau_n: Union[pd.PositiveFloat, CarrierLifetimeType] = pd.Field(
        ..., title="Electron lifetime", description="Electron lifetime", units=SECOND
    )

    tau_p: Union[pd.PositiveFloat, CarrierLifetimeType] = pd.Field(
        ..., title="Hole lifetime", description="Hole lifetime", units=SECOND
    )


class DistributedGeneration(Tidy3dBaseModel):
    """Class that allows to add a distributed generation model.

    Notes
    -----
    The generation rate will be interpolated to the simulation mesh during the setup phase.
    In places where the generation rate is not defined, it will be filled with zeros.

    Example
    -------
    >>> import tidy3d as td
    >>> import numpy as np
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> coords = dict(x=x, y=y, z=z)
    >>> fd = td.SpatialDataArray(np.random.random((2,3,4)), coords=coords)
    >>> dist_g = td.DistributedGeneration(rate=fd)
    """

    rate: SpatialDataArray = pd.Field(
        ...,
        title="Generation rate",
        description="Spatially varying generation rate.",
        units="1/(cm^3 s^1)",
    )

    @classmethod
    def from_rate_um3(cls, gen_um3: SpatialDataArray) -> DistributedGeneration:
        """Creates a DistributedGeneration from a SpatialDataArray in um^-3 s^-1."""
        gen_cm3 = np.array(gen_um3.data) * 1e12  # Convert from um^-3 to cm^-3
        new_gen = SpatialDataArray(gen_cm3, coords=gen_um3.coords)
        return cls(rate=new_gen)

    @pd.root_validator(skip_on_failure=True)
    def check_spatialdataarray_dimensions(cls, values):
        """Check that the SpatialDataArray is at least 2D:"""

        rate = values.get("rate")

        zero_dims = [d for d in ["x", "y", "z"] if len(rate.coords[d]) <= 1]

        if len(zero_dims) > 1:
            raise ValueError("SpatialDataArray must be at least 2D.")

        return values


class HurkxDirectBandToBandTunneling(Tidy3dBaseModel):
    """
    This class defines a direct band-to-band tunneling recombination model based on the Hurkx model
    as described in [1].

    Notes
    -----

    The direct band-to-band tunneling recombination rate :math:`R^{\\text{BTBT}}` is primarily defined by the
    material's bandgap energy :math:`E_g` and the electric field :math:`F`.

    Default values are provided for silicon.

    .. math::

        R^{\\text{BTBT}} = A \\cdot \\frac{n \\cdot p - n_i^2}{(n + n_i) \\cdot (p + n_i)} \\cdot \\left( \\frac{|\\mathbf{E}|}{E_0} \\right)^{\\sigma} \\cdot \\exp \\left(-\\frac{B}{|\\mathbf{E}|} \\cdot \\left( \\frac{E_g}{E_{g, 300}} \\right)^{3/2} \\right)

    where :math:`A`, :math:`B`, :math:`E_0`, and :math:`\\sigma` are material-dependent parameters.

    Example
    -------
    >>> import tidy3d as td
    >>> default_Si = td.HurkxDirectBandToBandTunneling(
    ...   A=1e19,
    ...   B=1.9e6,
    ...   E_0=1,
    ...   sigma=2.5
    ... )

    References
    ----------
        .. [1] Palankovski, Vassil, and Rüdiger Quay. Analysis and simulation of heterostructure devices. Springer Science & Business Media, 2004.
    """

    A: pd.PositiveFloat = pd.Field(
        4e14,
        title="Parameter :math:`A`",
        description="Parameter :math:`A` in the direct BTBT Hurkx model.",
        units="1/(cm^3 s)",
    )
    B: float = pd.Field(
        1.9e6,
        title="Parameter :math:`B`",
        description="Parameter :math:`B` in the direct BTBT Hurkx model.",
        units="V/cm",
    )
    E_0: pd.PositiveFloat = pd.Field(
        1,
        title="Reference electric field :math:`E_0`",
        description="Reference electric field :math:`E_0` in the direct BTBT Hurkx model.",
        units="V/cm",
    )
    sigma: float = pd.Field(
        2.5,
        title="Exponent parameter",
        description="Exponent :math:`\\sigma` in the direct BTBT Hurkx model. For direct "
        "semiconductors :math:`\\sigma` is typically 2.0, while for indirect "
        "semiconductors :math:`\\sigma` is typically 2.5.",
    )


class SelberherrImpactIonization(Tidy3dBaseModel):
    """
    This class defines the parameters for the Selberherr impact ionization model. Two formulations are available that
    depend on the driving field, as described in [1]_ (:math:`\\| E \\|`) and [2]_ (:math:`E \\cdot J_{\\nu} / \\| E \\|` for :math:`\\nu = n,p`).

    Notes
    -----

        The impact ionization rate ``\\alpha_{\\nu}`` (for :math:`\\nu = p` (holes) and :math:`\\nu = n` (electrons)) is defined by:

        .. math::

            \\alpha_{\\nu} = \\alpha_{\\nu}^\\infty \\cdot \\exp \\left( - \\left( \\frac{E_{\\nu}^{\\text{crit}} \\cdot |\\mathbf{J}_{\\nu}|}{E \\cdot \\mathbf{J}_{\\nu}} \\right)^{\\beta_{\\nu}} \\right)

        where :math:`\\alpha_{\\nu}^\\infty`, :math:`E_{\\nu}^{\\text{crit}}`, and :math:`\\beta_{\\nu}` are material-dependent parameters.

    Example
    -------
        >>> import tidy3d as td
        >>> default_Si = td.SelberherrImpactIonization(
        ...   alpha_n_inf=7.03e5,
        ...   alpha_p_inf=1.582e6,
        ...   E_n_crit=1.23e6,
        ...   E_p_crit=2.03e6,
        ...   beta_n=1,
        ...   beta_p=1,
        ...   formulation='PQ'
        ... )

    References
    ----------
        .. [1] Selberherr, Siegfried. Analysis and simulation of semiconductor devices. Springer Science & Business Media, 1984.
        .. [2] Vassil Palankovski and Rüdiger Quay. Analysis and simulation of heterostructure devices. Springer Science & Business Media, 2004.
    """

    alpha_n_inf: pd.PositiveFloat = pd.Field(
        ...,
        title="Electron ionization coefficient at infinite field",
        description="Electron ionization coefficient at infinite field.",
        units="1/cm",
    )
    alpha_p_inf: pd.PositiveFloat = pd.Field(
        ...,
        title="Hole ionization coefficient at infinite field",
        description="Hole ionization coefficient at infinite field.",
        units="1/cm",
    )
    E_n_crit: pd.PositiveFloat = pd.Field(
        ...,
        title="Critical electric field for electrons",
        description="Critical electric field for electrons.",
        units="V/cm",
    )
    E_p_crit: pd.PositiveFloat = pd.Field(
        ...,
        title="Critical electric field for holes",
        description="Critical electric field for holes.",
        units="V/cm",
    )
    beta_n: pd.PositiveFloat = pd.Field(
        ...,
        title="Exponent for electrons",
        description="Exponent for electrons.",
    )
    beta_p: pd.PositiveFloat = pd.Field(
        ...,
        title="Exponent for holes",
        description="Exponent for holes.",
    )

    formulation: Literal["Selberherr", "PQ"] = pd.Field(
        "PQ",
        title="Formulation",
        description="Formulation used for impact ionization. Options are 'Selberherr' "
        "or 'PQ' for Selberherr and Palankovski and Quay formulations, respectively.",
    )
