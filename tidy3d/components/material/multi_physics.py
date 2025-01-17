from typing import Optional

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.material.solver_types import (
    ChargeMediumType,
    ElectricalMediumType,
    HeatMediumType,
    OpticalMediumType,
)


class MultiPhysicsMedium(Tidy3dBaseModel):
    """
    Contains multiple multi-physical properties as defined for each solver medium.

    Examples
    --------
    For *silica* (:math:`SiO_2`):
        >>> import tidy3d as td
        >>> SiO2 = td.MultiPhysicsMedium(
        ...   optical=td.Medium(permittivity=3.9),
        ...   charge=td.ChargeInsulatorMedium(permittivity=3.9), # redefining permittivity
        ...   name="SiO2",
        ... )

    For a silicon ``MultiPhysicsMedium`` composed of an optical model
    from the material library and custom charge :class:`SemiconductorMedium`:
        >>> import tidy3d as td
        >>> default_multiphysics_Si = td.MultiPhysicsMedium(
        ...     optical=td.material_library['cSi']['Green2008'],
        ...     charge=td.SemiconductorMedium(
        ...         N_c=2.86e19,
        ...         N_v=3.1e19,
        ...         E_g=1.11,
        ...         mobility=td.CaugheyThomasMobility(
        ...             mu_n_min=52.2,
        ...             mu_n=1471.0,
        ...             mu_p_min=44.9,
        ...             mu_p=470.5,
        ...             exp_t_mu=-2.33,
        ...             exp_d_n=0.68,
        ...             exp_d_p=0.719,
        ...             ref_N=2.23e17,
        ...             exp_t_mu_min=-0.57,
        ...             exp_t_d=2.4,
        ...             exp_t_d_exp=-0.146,
        ...         ),
        ...         R=[
        ...             td.ShockleyReedHallRecombination(
        ...                 tau_n=3.3e-6,
        ...                 tau_p=4e-6
        ...             ),
        ...             td.RadiativeRecombination(
        ...                 r_const=1.6e-14
        ...             ),
        ...             td.AugerRecombination(
        ...                 c_n=2.8e-31,
        ...                 c_p=9.9e-32
        ...             ),
        ...         ],
        ...         delta_E_g=td.SlotboomBandGapNarrowing(
        ...             v1=6.92 * 1e-3,
        ...             n2=1.3e17,
        ...             c2=0.5,
        ...         ),
        ...         N_a=0,
        ...         N_d=0
        ...     )
        ... )
    """

    name: Optional[str] = None
    optical: Optional[OpticalMediumType] = None
    electrical: Optional[ElectricalMediumType] = None
    heat: Optional[HeatMediumType] = None
    charge: Optional[ChargeMediumType] = None

    @property
    def heat_spec(self):
        if self.heat is not None:
            return self.heat
        elif self.optical is not None:
            return self.optical.heat_spec
        else:
            return None
