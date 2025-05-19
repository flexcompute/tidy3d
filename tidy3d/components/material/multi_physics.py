from __future__ import annotations

from typing import Optional

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.material.solver_types import (
    ChargeMediumType,
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
        ...         mobility_n=td.CaugheyThomasMobility(
        ...             mu_min=52.2,
        ...             mu=1471.0,
        ...             ref_N=9.68e16,
        ...             exp_N=0.68,
        ...             exp_1=-0.57,
        ...             exp_2=-2.33,
        ...             exp_3=2.4,
        ...             exp_4=-0.146,
        ...         ),
        ...         mobility_p=td.CaugheyThomasMobility(
        ...             mu_min=44.9,
        ...             mu=470.5,
        ...             ref_N=2.23e17,
        ...             exp_N=0.719,
        ...             exp_1=-0.57,
        ...             exp_2=-2.33,
        ...             exp_3=2.4,
        ...             exp_4=-0.146,
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
        ...             min_N=1e15,
        ...         ),
        ...         N_a=0,
        ...         N_d=0
        ...     )
        ... )
    """

    name: Optional[str] = pd.Field(None, title="Name", description="Medium name")

    optical: Optional[OpticalMediumType] = pd.Field(
        None, title="Optical properties", description="Specifies optical properties."
    )

    # electrical: Optional[ElectricalMediumType] = pd.Field(
    #     None,
    #     title="Electrical properties",
    #     description="Specifies electrical properties for RF simulations. This is currently not in use.",
    # )

    heat: Optional[HeatMediumType] = pd.Field(
        None, title="Heat properties", description="Specifies properties for Heat simulations."
    )

    charge: Optional[ChargeMediumType] = pd.Field(
        None, title="Charge properties", description="Specifies properties for Charge simulations."
    )

    def __getattr__(self, name: str):
        """
        Delegate attribute lookup to inner media or fail fast.

        Parameters
        ----------
        name : str
            The attribute that could not be found on the ``MultiPhysicsMedium`` itself.

        Returns
        -------
        Any
            * The attribute value obtained from a delegated sub-medium when
            ``name`` is listed in ``DELEGATED_ATTRIBUTES``.
            * ``None`` when ``name`` is explicitly ignored (e.g. ``"__deepcopy__"``).

        Raises
        ------
        ValueError
            If ``name`` is neither ignored nor in the delegation map, signalling that
            the caller may have intended to access ``optical``, ``heat``, or
            ``charge`` directly.

        Notes
        -----
        Only the attributes enumerated in the local ``DELEGATED_ATTRIBUTES`` dict are
        forwarded.
        Extend that mapping as additional cross-medium shim behaviour becomes
        necessary.
        """
        # first check whether the attribute is already present
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        IGNORED_ATTRIBUTES = ["__deepcopy__"]
        if name in IGNORED_ATTRIBUTES:
            return None

        DELEGATED_ATTRIBUTES = {
            "_eps_plot": self.optical,
            "viz_spec": self.optical,
            "eps_diagonal_numerical": self.optical,
            "eps_complex_to_nk": self.optical,
            "nonlinear_spec": self.optical,
            "is_pec": self.optical,
            "is_time_modulated": self.optical,
            "is_nonlinear": self.optical,
            "is_fully_anisotropic": self.optical,
            "is_custom": self.optical,
            "is_isotropic": self.optical,
            "is_spatially_uniform": self.optical,
            "_incompatible_material_types": self.optical,
            "frequency_range": self.optical,
            "eps_model": self.optical,
        }

        if name == "_has_incompatibilities":
            return (self.optical and self.optical._has_incompatibilities) or (
                self.charge and self.charge._has_incompatibilities
            )

        if name in DELEGATED_ATTRIBUTES:
            sub = DELEGATED_ATTRIBUTES[name]
            if sub is None:
                raise AttributeError(
                    f"Requested attribute {name!r}, but the optical medium is 'None' "
                    " on this 'MultiPhysicsMedium' instance."
                )
            return getattr(sub, name)

        raise AttributeError(
            f"MultiPhysicsMedium has no attribute called {name}. "
            "Did you mean to access the attribute of one of the optical, heat or charge media?"
        )

    @property
    def heat_spec(self):
        if self.heat is not None:
            return self.heat

        if self.optical is not None:
            return self.optical.heat_spec
        return None
